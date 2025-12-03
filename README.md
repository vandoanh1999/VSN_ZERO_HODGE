"""
VSN Module 1 -- 3D Navier–Stokes Pseudospectral Solver (Real Experimental Edition)

Purpose:
- Full 3D incompressible Navier–Stokes pseudospectral solver on periodic cube [0,2π]^3.
- Designed as a **research-grade numerical experiment**: no theoretical claims, purely computational.
- Implements velocity-field formulation with divergence-free projection (Chorin-Temam style) in spectral space.

Features:
- 3D pseudospectral spatial discretization using FFTs (numpy or cupy backend).
- 2/3 Orszag de-aliasing.
- Explicit 4th-order Runge–Kutta time stepping with pressure projection done spectrally.
- Viscosity control, optional hyperviscosity, optional deterministic/ABC forcing.
- Diagnostics: kinetic energy, enstrophy (vorticity squared), energy spectrum, spectra time series.
- Snapshot saving (.npy) for velocity and vorticity fields, logging diagnostics as JSON.
- GPU-ready (cupy) and pyfftw-friendly where available.

Notes and limitations:
- This solver is intended for experimental exploration of 3D flow behavior. It does not constitute a proof of global regularity.
- Use moderate grid sizes (e.g., N=64 or N=128) for local machines; N>=256 requires significant memory and GPU.
- To study potential singularity formation, use adaptive strategies, resolution continuation, and extensive verification.

Run:
    python vsn_module1_navier_stokes_3d.py

Dependencies:
    numpy, scipy, matplotlib
    optional: cupy, pyfftw, numba

Author: VSN Real Experimental Edition - Module 1 (3D)
"""

import os
import time
import json
from typing import Optional, Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

# GPU backend detection
try:
    import cupy as cp
    XP = cp
    USING_CUPY = True
except Exception:
    XP = np
    USING_CUPY = False

# Optional pyfftw for faster FFTs on CPU
_USE_PYFFTW = False
try:
    import pyfftw
    _USE_PYFFTW = True
except Exception:
    _USE_PYFFTW = False

# FFT wrappers
if USING_CUPY:
    fftn = cp.fft.fftn
    ifftn = cp.fft.ifftn
else:
    if _USE_PYFFTW:
        from pyfftw.interfaces.numpy_fft import fftn as fftn_py, ifftn as ifftn_py
        fftn = fftn_py
        ifftn = ifftn_py
    else:
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn


class NavierStokes3D:
    """3D pseudospectral incompressible Navier–Stokes solver.

    Equations (dimensionless):
      u_t + (u · ∇) u = -∇p + ν ∇^2 u + f
      ∇·u = 0

    We evolve velocity u in spectral space and enforce incompressibility via projection.
    """

    def __init__(self,
                 N: int = 64,
                 viscosity: float = 1e-3,
                 dt: float = 1e-3,
                 t_end: float = 1.0,
                 dealias: bool = True,
                 hypervisc_order: int = 0,
                 hypervisc_coeff: float = 0.0,
                 forcing: Optional[Callable[[float, XP.ndarray], XP.ndarray]] = None,
                 output_dir: str = "vsn_ns3d_output"):
        self.N = N
        self.nu = viscosity
        self.dt = dt
        self.t_end = t_end
        self.dealias = dealias
        self.hypervisc_order = hypervisc_order
        self.hypervisc_coeff = hypervisc_coeff
        self.forcing = forcing
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # spatial grid (physical) - cube [0, 2π)^3
        L = 2.0 * np.pi
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.z = np.linspace(0, L, N, endpoint=False)

        # spectral wavenumbers (integer grid) - unshifted for multiplication
        k = np.fft.fftfreq(N, 1.0 / N) * 1.0
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        self.kx = XP.array(KX, dtype=XP.float64)
        self.ky = XP.array(KY, dtype=XP.float64)
        self.kz = XP.array(KZ, dtype=XP.float64)
        self.ksq = self.kx**2 + self.ky**2 + self.kz**2
        # avoid division by zero at k=0
        if USING_CUPY:
            self.ksq[0,0,0] = 1.0
        else:
            self.ksq[0,0,0] = 1.0

        # projection operator P_ij(k) = delta_ij - k_i k_j / |k|^2
        # in spectral space we will apply projection to velocity field mode-wise
        self.inv_ksq = 1.0 / self.ksq
        self.inv_ksq[0,0,0] = 0.0

        # Dealias mask (2/3 rule) in spectral integer wavenumber space
        if self.dealias:
            cutoff = int(np.floor((2.0 / 3.0) * (N // 2)))
            kgrid = np.fft.fftfreq(N) * N
            KXg, KYg, KZg = np.meshgrid(kgrid, kgrid, kgrid, indexing='ij')
            mask = (np.abs(KXg) <= cutoff) & (np.abs(KYg) <= cutoff) & (np.abs(KZg) <= cutoff)
            self.dealias_mask = XP.array(mask)
        else:
            self.dealias_mask = XP.ones((N,N,N), dtype=bool)

    def _to_spectral(self, f_phys: XP.ndarray) -> XP.ndarray:
        return fftn(f_phys)

    def _to_physical(self, f_hat: XP.ndarray) -> XP.ndarray:
        return ifftn(f_hat)

    def _project(self, u_hat: XP.ndarray) -> XP.ndarray:
        """Project velocity field in spectral space to divergence-free subspace."""
        # u_hat shape: (3, N, N, N)
        # compute k·u_hat
        kdotuhat = self.kx * u_hat[0] + self.ky * u_hat[1] + self.kz * u_hat[2]
        # subtract k * (k·u)/|k|^2
        u_hat[0] = u_hat[0] - self.kx * (kdotuhat * self.inv_ksq)
        u_hat[1] = u_hat[1] - self.ky * (kdotuhat * self.inv_ksq)
        u_hat[2] = u_hat[2] - self.kz * (kdotuhat * self.inv_ksq)
        return u_hat

    def _compute_nonlinear_hat(self, u_hat: XP.ndarray) -> XP.ndarray:
        """Compute nonlinear term N = FFT( (u·∇) u ) with dealiasing applied.
        Returns N_hat with shape (3, N, N, N).
        """
        # transform to physical
        u_phys = XP.real(self._to_physical(u_hat))
        ux = u_phys[0]
        uy = u_phys[1]
        uz = u_phys[2]
        # compute derivatives in spectral space for each velocity component
        # ∂_x u_j = i*kx * u_j_hat -> transform of derivative
        ux_hat = self._to_spectral(ux)
        uy_hat = self._to_spectral(uy)
        uz_hat = self._to_spectral(uz)
        ux_x = XP.real(self._to_physical(1j * self.kx * ux_hat))
        ux_y = XP.real(self._to_physical(1j * self.ky * ux_hat))
        ux_z = XP.real(self._to_physical(1j * self.kz * ux_hat))
        uy_x = XP.real(self._to_physical(1j * self.kx * uy_hat))
        uy_y = XP.real(self._to_physical(1j * self.ky * uy_hat))
        uy_z = XP.real(self._to_physical(1j * self.kz * uy_hat))
        uz_x = XP.real(self._to_physical(1j * self.kx * uz_hat))
        uz_y = XP.real(self._to_physical(1j * self.ky * uz_hat))
        uz_z = XP.real(self._to_physical(1j * self.kz * uz_hat))

        # compute convective terms (u·∇)u for each component
        N1 = ux * ux_x + uy * ux_y + uz * ux_z
        N2 = ux * uy_x + uy * uy_y + uz * uy_z
        N3 = ux * uz_x + uy * uz_y + uz * uz_z

        # transform back to spectral and apply dealias mask
        N1_hat = self._to_spectral(N1) * self.dealias_mask
        N2_hat = self._to_spectral(N2) * self.dealias_mask
        N3_hat = self._to_spectral(N3) * self.dealias_mask

        N_hat = XP.stack([N1_hat, N2_hat, N3_hat], axis=0)
        return N_hat

    def _rhs(self, u_hat: XP.ndarray, t: float) -> XP.ndarray:
        # nonlinear term
        N_hat = self._compute_nonlinear_hat(u_hat)
        # viscous term: nu * k^2 * u_hat
        visc = -self.nu * (self.ksq) * u_hat
        if self.hypervisc_order > 0 and self.hypervisc_coeff != 0.0:
            hv = -self.hypervisc_coeff * (self.ksq ** (self.hypervisc_order/2.0)) * u_hat
        else:
            hv = 0.0
        # forcing in physical space -> spectral
        F_hat = 0.0
        if self.forcing is not None:
            f_phys = self.forcing(t, None)
            # forcing should return shape (3, N, N, N)
            F_hat = self._to_spectral(f_phys) * self.dealias_mask
        # total RHS = -N_hat + visc + hv + F_hat
        rhs = -N_hat + visc + hv + F_hat
        # ensure divergence-free by projecting rhs if needed (we will project after RK stages)
        return rhs

    def _rk4_step(self, u_hat: XP.ndarray, t: float) -> XP.ndarray:
        dt = self.dt
        k1 = self._rhs(u_hat, t)
        k2 = self._rhs(u_hat + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._rhs(u_hat + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._rhs(u_hat + dt * k3, t + dt)
        u_hat_new = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # project to divergence-free
        u_hat_new = self._project(u_hat_new)
        return u_hat_new

    def kinetic_energy_enstrophy(self, u_hat: XP.ndarray) -> Dict[str, float]:
        # u_hat shape (3, N, N, N)
        u_phys = XP.real(self._to_physical(u_hat))
        ux = u_phys[0]
        uy = u_phys[1]
        uz = u_phys[2]
        E = 0.5 * XP.mean(ux**2 + uy**2 + uz**2)
        # enstrophy = 0.5 * mean(|ω|^2), ω = curl u
        # compute vorticity via spectral derivatives
        ux_hat = self._to_spectral(ux)
        uy_hat = self._to_spectral(uy)
        uz_hat = self._to_spectral(uz)
        wx = XP.real(self._to_physical(1j * (self.ky * uz_hat - self.kz * uy_hat)))
        wy = XP.real(self._to_physical(1j * (self.kz * ux_hat - self.kx * uz_hat)))
        wz = XP.real(self._to_physical(1j * (self.kx * uy_hat - self.ky * ux_hat)))
        Omega2 = 0.5 * XP.mean(wx**2 + wy**2 + wz**2)
        if USING_CUPY:
            return {'energy': float(E.get()), 'enstrophy': float(Omega2.get())}
        else:
            return {'energy': float(E), 'enstrophy': float(Omega2)}

    def energy_spectrum(self, u_hat: XP.ndarray, nbins: int = 50) -> Dict[str, XP.ndarray]:
        # isotropic energy spectrum via shell averaging in k-space
        # compute energy per mode from u_hat
        E_mode = 0.5 * (XP.abs(u_hat[0])**2 + XP.abs(u_hat[1])**2 + XP.abs(u_hat[2])**2)
        k_mag = XP.sqrt(self.ksq)
        k_flat = k_mag.ravel()
        E_flat = E_mode.ravel()
        kmax = float(XP.max(k_mag))
        kbins = XP.linspace(0.0, kmax, nbins+1)
        Eb = XP.zeros(nbins)
        kb = XP.zeros(nbins)
        for i in range(nbins):
            mask = (k_flat >= kbins[i]) & (k_flat < kbins[i+1])
            if USING_CUPY:
                s = float(XP.sum(E_flat[mask]).get()) if XP.sum(mask) > 0 else 0.0
            else:
                s = float(XP.sum(E_flat[mask])) if XP.sum(mask) > 0 else 0.0
            Eb[i] = s
            kb[i] = 0.5 * (kbins[i] + kbins[i+1])
        return {'k': kb, 'E': Eb}

    def run(self, u0_phys: Optional[XP.ndarray] = None, save_every: int = 50):
        # initialize velocity field
        if u0_phys is None:
            # Taylor-Green like 3D initial condition (divergence-free)
            X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
            u = np.empty((3, self.N, self.N, self.N), dtype=float)
            u[0] = np.sin(X) * np.cos(Y) * np.cos(Z)
            u[1] = -np.cos(X) * np.sin(Y) * np.cos(Z)
            u[2] = 0.0 * Z
            u0_phys = XP.array(u)
        # spectral transform and projection
        u_hat = self._to_spectral(u0_phys)
        u_hat = self._project(u_hat)

        t = 0.0
        nsteps = int(np.ceil(self.t_end / self.dt))
        snapshots = []
        diagnostics = []
        start_time = time.time()
        print(f"Starting 3D Navier–Stokes: N={self.N}, dt={self.dt}, steps={nsteps}, nu={self.nu}")
        for step in range(nsteps):
            u_hat = self._rk4_step(u_hat, t)
            t += self.dt
            if step % save_every == 0 or step == nsteps - 1:
                diag = self.kinetic_energy_enstrophy(u_hat)
                diagnostics.append({'t': t, 'energy': diag['energy'], 'enstrophy': diag['enstrophy']})
                # save snapshot (physical u and vorticity)
                u_phys = XP.real(self._to_physical(u_hat))
                ux = u_phys[0]
                uy = u_phys[1]
                uz = u_phys[2]
                # compute vorticity
                ux_hat = self._to_spectral(ux)
                uy_hat = self._to_spectral(uy)
                uz_hat = self._to_spectral(uz)
                wx = XP.real(self._to_physical(1j * (self.ky * uz_hat - self.kz * uy_hat)))
                wy = XP.real(self._to_physical(1j * (self.kz * ux_hat - self.kx * uz_hat)))
                wz = XP.real(self._to_physical(1j * (self.kx * uy_hat - self.ky * ux_hat)))

                # convert to numpy for saving
                if USING_CUPY:
                    ux_n = XP.asnumpy(ux)
                    uy_n = XP.asnumpy(uy)
                    uz_n = XP.asnumpy(uz)
                    wx_n = XP.asnumpy(wx)
                    wy_n = XP.asnumpy(wy)
                    wz_n = XP.asnumpy(wz)
                else:
                    ux_n = ux
                    uy_n = uy
                    uz_n = uz
                    wx_n = wx
                    wy_n = wy
                    wz_n = wz

                fname_u = os.path.join(self.output_dir, f"u_t_{step:06d}.npz")
                np.savez_compressed(fname_u, ux=ux_n, uy=uy_n, uz=uz_n, wx=wx_n, wy=wy_n, wz=wz_n)
                snapshots.append(fname_u)
                elapsed = time.time() - start_time
                print(f" step {step}/{nsteps} t={t:.4f} E={diag['energy']:.6e} Omega2={diag['enstrophy']:.6e} elapsed={elapsed:.1f}s")
        total_time = time.time() - start_time
        print(f"Simulation finished in {total_time:.2f}s")

        # save diagnostics
        diag_file = os.path.join(self.output_dir, "diagnostics.json")
        with open(diag_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)

        return {'snapshots': snapshots, 'diagnostics': diagnostics}


# -----------------------------
# Utilities: backend info, benchmarks, and demo run
# -----------------------------

def backend_info():
    info = {'using_cupy': USING_CUPY, 'use_pyfftw': _USE_PYFFTW}
    if USING_CUPY:
        try:
            dev = cp.cuda.Device()
            info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(dev.id)['name']
        except Exception:
            info['gpu_name'] = None
    return info


def benchmark_fft(N=128, n_iters=5):
    x = XP.random.randn(N, N, N)
    # warmup
    for _ in range(2):
        _ = fftn(x)
        _ = ifftn(_)
    t0 = time.time()
    for _ in range(n_iters):
        fh = fftn(x)
        _ = ifftn(fh)
    return {'backend': 'cupy' if USING_CUPY else 'numpy/pyfftw', 'N': N, 'n_iters': n_iters, 'time_s': time.time()-t0}


def run_demo(N=64, dt=1e-3, t_end=0.1, use_gpu=False):
    print("Backend:", backend_info())
    if use_gpu and not USING_CUPY:
        print("Warning: cupy not available; running on CPU")
    print("FFT benchmark:", benchmark_fft(min(128, N), n_iters=3))
    solver = NavierStokes3D(N=N, viscosity=1e-3, dt=dt, t_end=t_end, dealias=True, output_dir='vsn_ns3d_output')
    res = solver.run(save_every=max(1, int((t_end/dt)//4)))
    # plot diagnostics
    diag_file = os.path.join(solver.output_dir, 'diagnostics.json')
    with open(diag_file, 'r') as f:
        diagnostics = json.load(f)
    times = [d['t'] for d in diagnostics]
    energies = [d['energy'] for d in diagnostics]
    enstrophies = [d['enstrophy'] for d in diagnostics]
    plt.figure(figsize=(8,4))
    plt.plot(times, energies, label='Kinetic Energy')
    plt.plot(times, enstrophies, label='Enstrophy')
    plt.xlabel('t')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(solver.output_dir, 'energy_enstrophy_3d.png'), dpi=300)
    plt.show()
    return res


if __name__ == '__main__':
    # quick demo for small grid
    run_demo(N=48, dt=2e-3, t_end=0.06, use_gpu=False)
