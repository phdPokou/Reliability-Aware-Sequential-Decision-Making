# Reliability-Aware Sequential Decision-Making under Heavy-Tailed Uncertainty

<p align="center">
  <strong>Replication Data and Reproducibility Package</strong>
</p>

<p align="center">
  <em>A Localized Distributionally Robust Approach</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-GPU%20Computing-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Reproducibility-TRAIN%20%E2%86%92%20VALIDATION%20%E2%86%92%20TEST-success" alt="Reproducibility">
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey" alt="License">
</p>

---

## Overview

This repository contains the computational materials associated with the study:

> **Reliability-Aware Sequential Decision-Making under Heavy-Tailed Uncertainty: A Localized Distributionally Robust Approach**

The study investigates reliability-aware sequential decision-making when heavy-tailed uncertainty is represented through a hierarchical latent model. The proposed **Localized Kullback–Leibler (Localized-KL)** construction places distributional ambiguity on the latent mixing law while maintaining the conditional disturbance mechanism and the observable information structure of the controller.

The computational experiment compares four policy-selection criteria:

- **Expected Cost**
- **Nominal Conditional Value-at-Risk (CVaR)**
- **Global-KL**
- **Localized-KL**

The empirical application uses **Bitcoin (BTCUSDT)** and **Ether (ETHUSDT)** spot-market data and preserves a strictly chronological experimental protocol.

This repository is intended to support verification and computational reproducibility of the numerical and empirical results reported in the paper. It does not imply that the maintained stochastic model exhausts all forms of market or model uncertainty.

---

## Reproducibility Design

The experimental protocol is deliberately sequential:

```text
┌─────────────────┐
│      TRAIN      │
│                 │
│ Fit / freeze    │
│ environment     │
│       ↓         │
│ Numerical gates │
│       ↓         │
│ Select policy   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   VALIDATION    │
│                 │
│ Confirm frozen  │
│ TRAIN choices   │
│                 │
│   NO RETUNING   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      TEST       │
│                 │
│ Frozen policies │
│       ↓         │
│ Final held-out  │
│ evaluation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ POST-TEST I.7   │
│                 │
│ Integrity audit │
│ Paired inference│
│ Tables / figures│
└─────────────────┘
```

The protocol prevents TEST information from entering policy selection. Policies are selected on TRAIN, frozen before subsequent evaluation, and VALIDATION is used without retuning.

The final statistical analysis is performed only after completion of the locked TEST stage.

---

## Experimental Design

| Component | Specification |
|---|---|
| Assets | BTCUSDT, ETHUSDT |
| Decision horizon | \(T=120\) |
| Temporal grid | 5 seconds |
| Tail level | \(\alpha=0.95\) |
| Policy grid | \(\gamma \in \{0.0,0.1,\ldots,1.0\}\) |
| Objectives | Expected Cost, Nominal CVaR, Global-KL, Localized-KL |
| Localized-KL sensitivity grid | \(\epsilon \in \{0,0.02,0.04,0.08,0.12,0.16\}\) |
| Evaluation seeds | 15 predeclared seeds |
| TEST simulations | \(\geq 20,000\) trajectories per selected policy and seed |
| Asset pooling | None |
| Variance reduction | Common random numbers within admissible comparisons |
| Policy information | Observable state/history only |

BTC and ETH are analyzed separately throughout the experimental protocol.

The latent variable is part of the probabilistic representation used for robust evaluation and is **never supplied as an input to the policy**.

---

## Repository Structure

The replication package is organized around the successive numerical stages of the study.

```text
.
├── ress_full_pipeline_fast_v1i6.py
├── ress_full_pipeline_worker_turbo1_v1i6.py
├── ress_final_statistical_analysis_v1i7.py
│
├── ress_asset_specific_train_convergence_v1i5d3.py
├── ress_coupled_execution_v1h33.py
│
├── Results_RESS_V1D/
│   └── panels/
│
├── Results_RESS_V1F53/
├── Results_RESS_V1G55/
├── Results_RESS_V1H22/
│
├── Results_RESS_V1I5D2/
│   └── V1I5D2_PROTOCOL.json
│
├── Results_RESS_V1I5D5/
│   └── V1I5D5_SUMMARY.json
│
└── Results_RESS_V1I6_TURBO/
    ├── TRAIN_LOCK.json
    ├── VALIDATION_LOCK.json
    ├── TEST_COMPLETE.json
    ├── I6_MANIFEST.json
    └── ...
```

Some intermediate directories contain fitted models, processed panels, numerical diagnostics, or locked outputs required by later stages. Their names are retained to preserve correspondence with the computational audit trail used during the study.

---

## Computational Requirements

The main pipeline is written in **Python** and supports GPU execution through **PyTorch** and **CUDA**.

### Core software

```text
Python
PyTorch
CUDA-enabled PyTorch installation
NumPy
pandas
SciPy
```

Additional Python packages may be required by upstream data-processing and figure-generation stages.

A CUDA-capable NVIDIA GPU is recommended for reproducing the complete simulation pipeline. CPU execution may be possible for individual components but is not the reference configuration for the computationally intensive experiment.

### Verify CUDA availability

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

A successful CUDA configuration should return:

```text
CUDA available: True
```

---

# Reproducing the Main Experiment

## 1. TRAIN

TRAIN fits or loads the frozen environment components, performs the required numerical evaluations, and selects criterion-specific policy parameters without accessing TEST.

Run:

```bash
python ress_full_pipeline_fast_v1i6.py ^
  --stage train ^
  --worker ress_full_pipeline_worker_turbo1_v1i6.py ^
  --protocol Results_RESS_V1I5D2/V1I5D2_PROTOCOL.json ^
  --d5 Results_RESS_V1I5D5/V1I5D5_SUMMARY.json ^
  --results Results_RESS_V1I6_TURBO ^
  --d3-script ress_asset_specific_train_convergence_v1i5d3.py ^
  --h33-script ress_coupled_execution_v1h33.py ^
  --panel Results_RESS_V1D/panels ^
  --vep-results Results_RESS_V1F53 ^
  --g55-results Results_RESS_V1G55 ^
  --h22-results Results_RESS_V1H22 ^
  --device cuda ^
  --jobs 4
```

Successful completion creates the TRAIN lock required by the subsequent stages.

---

## 2. VALIDATION

VALIDATION evaluates the choices inherited from TRAIN.

**No policy retuning is permitted at this stage.**

Run:

```bash
python ress_full_pipeline_fast_v1i6.py ^
  --stage validation ^
  --worker ress_full_pipeline_worker_turbo1_v1i6.py ^
  --protocol Results_RESS_V1I5D2/V1I5D2_PROTOCOL.json ^
  --d5 Results_RESS_V1I5D5/V1I5D5_SUMMARY.json ^
  --results Results_RESS_V1I6_TURBO ^
  --d3-script ress_asset_specific_train_convergence_v1i5d3.py ^
  --h33-script ress_coupled_execution_v1h33.py ^
  --panel Results_RESS_V1D/panels ^
  --vep-results Results_RESS_V1F53 ^
  --g55-results Results_RESS_V1G55 ^
  --h22-results Results_RESS_V1H22 ^
  --device cuda ^
  --jobs 4
```

The pipeline requires a valid TRAIN lock before VALIDATION can proceed.

---

## 3. TEST

TEST is reserved for final evaluation of the policies frozen by the preceding stages.

Run:

```bash
python ress_full_pipeline_fast_v1i6.py ^
  --stage test ^
  --worker ress_full_pipeline_worker_turbo1_v1i6.py ^
  --protocol Results_RESS_V1I5D2/V1I5D2_PROTOCOL.json ^
  --d5 Results_RESS_V1I5D5/V1I5D5_SUMMARY.json ^
  --results Results_RESS_V1I6_TURBO ^
  --d3-script ress_asset_specific_train_convergence_v1i5d3.py ^
  --h33-script ress_coupled_execution_v1h33.py ^
  --panel Results_RESS_V1D/panels ^
  --vep-results Results_RESS_V1F53 ^
  --g55-results Results_RESS_V1G55 ^
  --h22-results Results_RESS_V1H22 ^
  --device cuda ^
  --jobs 4
```

The pipeline enforces the following ordering:

```text
TRAIN_LOCK.json
       ↓
VALIDATION_LOCK.json
       ↓
TEST
       ↓
TEST_COMPLETE.json
```

TEST cannot be executed through the controller without the required TRAIN and VALIDATION locks.

The reference protocol requires at least **20,000 trajectories per selected policy and evaluation seed**.

---

## 4. Final Statistical Analysis

After TEST has completed, run the final statistical analysis:

### Windows

```powershell
python ress_final_statistical_analysis_v1i7.py --input "C:\Users\fredy\Downloads\Results_RESS_V1I6_TURBO"
```

### Linux / macOS

```bash
python ress_final_statistical_analysis_v1i7.py ^
  --input /path/to/Results_RESS_V1I6_TURBO
```

This stage is **post-TEST only**.

It does not simulate new trajectories and does not retune the selected policies. It consumes the locked TEST outputs and performs:

1. integrity and reproducibility checks;
2. paired seed-level policy comparisons;
3. reliability decomposition;
4. bootstrap inference;
5. secondary nonparametric diagnostics;
6. generation of publication-ready tables and figures;
7. generation of the final machine-readable summary.

The primary statistical comparison is:

```text
Localized-KL
     │
     ├── vs Expected Cost
     ├── vs Nominal CVaR
     └── vs Global-KL
```

The inference unit is the **evaluation seed**, rather than individual simulated trajectories, preserving the paired common-random-number design.

---

## Statistical Inference

The final analysis uses the 15 predeclared evaluation seeds:

```text
101, 202, 303, 404, 505,
606, 707, 808, 909, 1001,
1111, 1212, 1313, 1414, 1515
```

Primary uncertainty quantification is based on a **paired nonparametric bootstrap over seeds**.

The final post-TEST analysis uses:

```text
50,000 bootstrap replications
```

with a fixed bootstrap seed.

Exact two-sided sign tests and Wilcoxon signed-rank tests, when available, are reported as secondary diagnostics rather than substitutes for the paired effect estimates and bootstrap confidence intervals.

---

## Reliability Metrics

The TEST evaluation records multiple dimensions of the policy-induced cost distribution, including:

- mean cost;
- Value-at-Risk at the 0.95 level;
- Conditional Value-at-Risk at the 0.95 level;
- secondary Conditional Value-at-Risk at the 0.99 level;
- frozen-threshold exceedance frequency;
- conditional excess severity;
- completion or settlement diagnostics.

Mean performance, exceedance frequency, and upper-tail severity are intentionally retained as distinct empirical quantities.

---

## Common Random Numbers

Comparable policies are evaluated using **common random numbers (CRN)** within the relevant asset, seed, and simulation design.

Conceptually,

```text
same stochastic realization
        │
        ├── Expected Cost policy
        ├── Nominal CVaR policy
        ├── Global-KL policy
        └── Localized-KL policy
```

This pairing reduces Monte Carlo noise in policy contrasts without changing the marginal stochastic law of each policy evaluation.

CRN are not imposed across BTC and ETH.

---

## Localized-KL and Global-KL

The repository distinguishes two ambiguity architectures.

### Localized-KL

```text
Nominal latent law
       │
       ▼
KL perturbation of latent mixing law
       │
       ▼
Maintained conditional mechanism
       │
       ▼
Policy-induced cumulative cost
       │
       ▼
Robust tail criterion
```

### Global-KL

```text
Nominal policy-induced cost law
       │
       ▼
Direct KL perturbation
       │
       ▼
Robust tail criterion
```

Using the same numerical divergence radius in the two constructions is treated as a **matched-budget comparison**, not as evidence that the two ambiguity sets represent empirically equivalent amounts of misspecification.

---

## Important Interpretation Guards

The computational package preserves several restrictions that are important for interpreting the results:

- the latent variable is not a policy input;
- BTC and ETH are not pooled for policy optimization;
- VALIDATION does not retune TRAIN-selected policies;
- TEST is reserved for evaluation after the design lock;
- policy optimality is restricted to the specified numerical policy grid;
- the Localized-KL and Global-KL ambiguity architectures are not treated as equivalent;
- a common numerical KL radius does not imply calibration equivalence;
- finite empirical exponential moments are not interpreted as proof of population-level entropic admissibility;
- numerical clipping, when used, is not treated as innocuous because it changes the evaluated objective.

These restrictions are part of the reproducibility design rather than post hoc qualifications.

---

## Numerical Reproducibility

The main simulation worker uses shared exogenous stochastic paths when evaluating the policy grid under a fixed simulation cell.

This implementation is designed to improve computational efficiency without changing the intended scientific comparison.

For the reference configuration:

```text
BTCUSDT:
    outer paths M = 512

ETHUSDT:
    outer paths M = 256

Inner simulations:
    L = 2048

Policy grid:
    gamma = 0.0, 0.1, ..., 1.0

Tail level:
    alpha = 0.95
```

The computational acceleration concerns implementation only; it is not intended to redefine the underlying experimental design.

---

## Reproducibility Checklist

Before interpreting regenerated results, verify that:

```text
[ ] Python environment is available
[ ] PyTorch imports successfully
[ ] CUDA is available for GPU replication
[ ] Required input directories are present
[ ] V1I5D2_PROTOCOL.json is unchanged
[ ] TRAIN completes before VALIDATION
[ ] TRAIN_LOCK.json exists
[ ] VALIDATION performs no retuning
[ ] VALIDATION_LOCK.json exists
[ ] TEST is executed only after both locks
[ ] TEST uses at least 20,000 trajectories/policy/seed
[ ] TEST_COMPLETE.json is generated
[ ] Final statistical analysis is run only after TEST
[ ] BTC and ETH results are interpreted separately
```

---

## Design Integrity

The predeclared numerical protocol is identified by its SHA-256 design hash:

```text
0791be105f4afd18170c7e5b12a74078afc684815414f713ecb80f46a99bcf50
```

The pipeline checks this identifier before executing the locked experiment.

This provides a machine-verifiable connection between the declared numerical design and the computational pipeline used for the final experiment.

---

## Data and Outputs

The complete replication archive contains the data inputs permitted for redistribution together with processed data, fitted-model outputs, numerical diagnostics, simulation results, statistical outputs, tables, and figure-generation materials used in the study.

The archived results include the intermediate computational stages required to audit the progression from the fitted stochastic environment to the final TEST evaluation.

Large intermediate simulation objects are retained where they are required for numerical verification or post-TEST analysis.

---

## Reproducibility Archive

A frozen replication package is archived on **Zenodo**.

**Version:** `v1`

**Publication date:** `2026-10-04`

**Version-specific DOI:**

```text
10.5281/zenodo.23136337
```

**Concept DOI for all versions:**

```text
10.5281/zenodo.23136336
```

For exact replication of the version accompanying the current study, use the **version-specific DOI**.

---

## Scope

This repository reproduces results conditional on the maintained:

- stochastic specification;
- fitted latent representation;
- conditional disturbance mechanism;
- policy class;
- ambiguity architecture;
- numerical approximation;
- empirical data construction;
- chronological experimental protocol.

Accordingly, successful computational replication establishes reproducibility of the reported experiment under these maintained components. It should not be interpreted as criterion-independent dominance or robustness to forms of misspecification outside the ambiguity architectures considered in the study.

---



## License

The replication materials are released under the license specified in the accompanying Zenodo record.

Please consult individual files or subdirectories for any additional licensing or redistribution restrictions applicable to third-party data or software components.

---

## Contact

For questions concerning the replication package, numerical protocol, or computational implementation, please open a GitHub issue.

---

<p align="center">
  <strong>Reproducible research requires more than code availability.</strong><br>
  The repository preserves the chronology, policy locks, randomization design,
  numerical diagnostics, and post-TEST analysis used in the reported experiment.
</p>
