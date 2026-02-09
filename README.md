# SyntheticGem_J — Decoupled LSGP / Flow BO on a Synthetic Sequence Landscape

`SyntheticGem_J.py` runs a synthetic benchmark for sequence optimization using Bayesian Optimization (BO) with:
- **Sequence-space BO** (baseline)
- **Flow-encoded latent BO** (“Flow BO”)
- **Flow BO + PoV gating** (Probability-of-Viability reweighting)

The underlying objective is a simple **Ehrlich-style occupancy** proxy derived from **Hamming similarity** to a hidden target sequence, with optional viability filtering (non-viable points return `NaN` and are excluded from training).

The script produces publication-style plots and CSVs summarizing convergence and method comparisons.

---

## What this script does

### Objective: Ehrlich Occupancy Time (synthetic)
- Samples a hidden target sequence of length `L`.
- Measures similarity using normalized Hamming similarity.
- Converts similarity to an occupancy-like score:
  - `kd = 1 - similarity`
  - `occupancy = 1 / (kd + 1)`
- Enforces a **viability threshold**:
  - if `similarity < viability_threshold` → returns `NaN` (treated as non-viable)

### Methods compared
1. **Sequence BO**
   - GP surrogate on raw sequence vectors (integer-coded)  
   - Acquisition mixes EI and UCB

2. **Flow BO**
   - Uses a simple “bijective flow” module for encoding/decoding (linear map)
   - GP is trained in latent space

3. **Flow + PoV**
   - Same as Flow BO, but acquisition score is multiplied by a PoV classifier output
   - Intended to downweight non-viable candidates

### Candidate generation strategy
- **Global random candidates**
- **Local mutations** from elite sequences (GA-like exploitation)
- **Random injection** to maintain diversity

### Outputs
- `figure_2_convergence.png` — single-seed convergence comparison
- `figure_2_convergence_mean_std.png` — multi-seed mean ± std convergence
- `figure_5_method_comparison.png` — bar chart comparing mean vs final best (single-seed)
- `figure_3_efficiency.png` — synthetic “real-world efficiency” bar chart (benchmark-style)
- `figure_6_flow_manifold.png` — latent scatter/hexbin (best method) + viability visualization
- `figure_4_benchmark.png` — synthetic 2D manifold + viability boundary illustration
- `figure_2_results.csv` — per-iteration convergence for single-seed + summary stats
- `figure_2_results_aggregate.csv` — per-iteration mean/std across seeds

---

## Requirements

Python 3.9+ recommended.

Install dependencies:
```bash
pip install numpy torch matplotlib seaborn scikit-learn scipy
