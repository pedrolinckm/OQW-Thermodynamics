<!-- .github/copilot-instructions.md: Guidance for AI coding agents working on this repo -->

# OQW-Thermodynamics — Copilot instructions

Purpose: Quickly orient coding agents (Copilot-style) to be productive in this repository.

High level
- This repo is research code for open quantum walks (OQW) thermodynamics. The main driver is the Jupyter notebook `thermodynamics_oqw.ipynb` which imports helper functions from `qc_module.py` (thermodynamics math + small Qiskit helpers) and `quantum_computation_module.py`.
- Two related concerns coexist: (1) numerical/statistical thermodynamics (probabilities, entropy, approximations, plotting) and (2) quantum circuit utilities (Qiskit adders). The notebook is focused on (1); Qiskit code is present but lightly integrated.

Key files to inspect
- `thermodynamics_oqw.ipynb` — primary analysis and plotting: data generation, plotting (.pgf outputs), and evaluation of analytic vs numerical formulas.
- `qc_module.py` — main library used by the notebook: `Prob`, `S_entropy`, `S_corrected`, `S_G`, `mean_energy`, `standart_deviation_energy`, `t_start`, `t_end`, `S_a_2`, `Prob_approximation`, `erf_approximation`.
- `quantum_computation_module.py` — overlapping Qiskit helper utilities (some duplication with `qc_module.py`) used for building arithmetic gates.

Important patterns & conventions (project-specific)
- Array shapes and indexing: `Prob(N,omega,t)` returns an array of shape `(N, t+1)` with indexing `P[m, n]` where `n` is time. Many routines expect `P[m, t]` for time `t` (single-step entropy calls use last index).
- Numerical stability: functions intentionally clip or add tiny epsilons before taking logs (`+1e-15` or `np.clip(p, 1e-300, 1.0)`); replicate this style when adding new code that computes entropies or probabilities.
- Use of specialized functions for cancellation: prefer `np.expm1` where present (see robust `probs_p` in the notebook). When vectorizing, preserve numerically-stable formulas.
- Parameters naming: `N` (system size), `omega` (transition probability in (0,1)), `t` (discrete time). `epsilon` is used in energy formulas (frequently set to 1).
- Plots are saved to PGF (`.pgf`) and rely on matplotlib; keep plot code tidy and call `plt.savefig("name.pgf")` like existing cells.

Workflows and commands
- Interactive development: open `thermodynamics_oqw.ipynb` in Jupyter / JupyterLab and run cells. Notebook is the canonical way to reproduce figures.
- To run the notebook headlessly (CI or batch):
  - Create a virtual env and install deps (inferred):
    - `python -m venv .venv && source .venv/bin/activate`
    - `pip install numpy scipy matplotlib sympy pandas pylatexenc qiskit jupyter`
  - Execute: `jupyter nbconvert --to notebook --execute thermodynamics_oqw.ipynb --ExecutePreprocessor.timeout=600`
- No test suite detected. For small checks, run individual notebook cells or small script wrappers that import `qc_module.py` and call functions like `Prob` and `S_entropy`.

Integration points & dependencies
- Qiskit is used for circuit construction (`CDKMRippleCarryAdder`) — only use Qiskit imports where required. Building/executing circuits is optional for thermodynamics analysis.
- Heavy deps: `numpy`, `scipy`, `matplotlib`, `sympy`, `pandas`, `pylatexenc`, `qiskit`. The repo contains no pinned `requirements.txt`, so ensure reproducible installs if adding automation.

Practical examples for agents (copyable guidance)
- Compute entropy at time t: `S_entropy(N=100, omega=2/3, t=50)` (uses `Prob` internally and clips probabilities).
- Use analytic estimate: `S_corrected(N, omega, t)` and compare to `S_entropy` and `S_a_2` using `t_start(N,omega)` / `t_end(N,omega)` to pick the valid regime.
- Probabilities shape: `P = Prob(N, omega, t_max); P.shape == (N, t_max+1)` and probability at site m and time t is `P[m, t]`.

Editing & PR guidance
- Preserve numeric stability patterns: avoid naive `np.log` on small arrays; maintain clipping and `expm1` use. When changing probability formulas, ensure normalization `p /= p.sum()` where present.
- When refactoring, prefer cleaning duplicate helpers (`quantum_computation_module.py` vs `qc_module.py`) only with an explicit PR note — the functions are duplicated and researchers may rely on the two separate modules.

When uncertain, ask the user for:
- Which functions should be considered authoritative (there are duplicates for some Qiskit helpers).
- Whether to add a pinned `requirements.txt` or CI notebook execution.

If you'd like, I can: (a) add a `requirements.txt`, (b) deduplicate the Qiskit helpers, or (c) open PR-ready edits to the notebook to make numeric routines more vectorized.
