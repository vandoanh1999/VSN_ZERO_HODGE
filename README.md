# VSN Zero: A Quantum--Functorial Framework for Algebraic Cycles

This repository contains the LaTeX source code and computational framework for the VSN (Variational Symplectic Normalization) project. The work introduces a novel Hodge–VSN operator to constructively and computationally address key problems in algebraic geometry, particularly related to the Hodge Conjecture.

## Key Features
- **Hodge–VSN Operator:** A new tool acting on de Rham cohomology, defined via period orthogonality.
- **Functorial Bridge:** A theoretical framework connecting the variation of Hodge structures (VHS) to the construction of algebraic cycles.
- **Computational Efficiency:** Implementation of the Ω–Blade algorithm for efficient cycle testing and construction.
- **Reproducibility:** All code and benchmarks are provided for full transparency and verification.

## Project Structure
- `vsn_zero_hodge.tex`: The main LaTeX source file for the paper.
- `src/oblade.py`: The core computational engine for the VSN framework.
- `benchmarks/`: Data and results from computational benchmarks.
- `docs/`: Documentation including a log of our peer review process and a summary of the framework's theory.

## Getting Started
To compile the paper, you need a standard LaTeX distribution.
```bash
pdflatex vsn_zero_hodge.tex

