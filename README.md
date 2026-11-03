# Gender Recognition Grand Challenge
**Computer Vision & Pattern Recognition Lab (CVR Lab)**

A MATLAB-based face image gender recognition system that compares four classification pipelines combining PCA/LDA dimensionality reduction with Bayes/MLP classifiers.

---
## Work Distribution

M1 ~ M8 represents each member and are **not allowed to modify files outside their own assignment**.  
Members are required to insert their name in their corresponding role in order to recognize **contributions for grading**.
To add your name: open `README.md`, find your row, replace `**Mn**` with `**Mn Your Name**`, and commit with message `docs: add name for Mn`.

| Member | Role | Files Owned |
|---|---|---|
| **M1 William Hsieh** | Project Lead / Data Pipeline / Integration | `main_train.m`, `main_test.m`, `data/`, `pipelines/` |
| **M2** | PCA Specialist | `dim_reduction/pca_train.m`, `dim_reduction/pca_project.m` |
| **M3** | LDA Specialist | `dim_reduction/lda_train.m`, `dim_reduction/lda_project.m` |
| **M4** | Bayes Classifier | `classifiers/bayes_train.m`, `classifiers/bayes_predict.m` |
| **M5** | MLP Classifier | `classifiers/mlp_train.m`, `classifiers/mlp_predict.m` |
| **M6** | Evaluation Framework | `evaluation/cross_validate.m`, `evaluation/compute_roc.m`, `evaluation/plot_roc.m`, `evaluation/compute_auc.m`, `evaluation/compute_accuracy.m` |
| **M7** | Experiments & Analysis | No skeleton file — create `experiments/run_experiments.m` that calls `main_train` with varied hyperparameters and collects results for Chapter 3 of the report |
| **M8** | Report & Presentation Lead | No MATLAB files — produce the CCU-format report (`.docx`/`.pdf`), presentation slides, and coordinate the live demo rehearsal |

---

## Getting Started

### 1 — Clone the repository
```bash
git clone https://github.com/DuckSaysBruh/Pattern_Recognition_Project.git
cd Pattern_Recognition_Project
```
Or download the ZIP from GitHub → **Code → Download ZIP**.

### 2 — Set up MATLAB
1. Open MATLAB (R2016b or later).
2. In the **Current Folder** panel, navigate to `Pattern_Recognition_Project/` — **this is required** because `main_train` calls `addpath(genpath(pwd))`, which adds all subfolders relative to MATLAB's Current Folder, not the script's location.
3. Place the `train/` image folder inside the project root (200 PNG files — not committed to git).
4. *(For `main_test` only)* Place the `test/` folder once the test dataset is released.

---

## Requirements

| Requirement | Detail |
|---|---|
| Language | MATLAB R2016b or later |
| Toolboxes | Image Processing Toolbox (for `imread`, `imresize`, `rgb2gray`) |
| Dataset | 200 training images + 100 test images (PNG, face photos) |

---

## Dataset Layout

```
train/
  man_1_1.png   man_1_2.png   …   man_50_2.png     (100 male images)
  woman_1_1.png woman_1_2.png …   woman_50_2.png   (100 female images)

test/
  man_1_1.png   …   man_25_2.png                   (50 male images)
  woman_1_1.png …   woman_25_2.png                 (50 female images)
```

Labels are inferred from file names: filenames containing `"woman"` → **1 (female)**, others → **0 (male)**.

---

## File Tree

```
Pattern_Recognition_Project/
│
├── main_train.m              ← [M1] Entry point: loads data, runs 5-fold CV on all pipelines,
│                                    prints summary table, saves ROC plot
├── main_test.m               ← [M1] Entry point: trains on full train set, evaluates test set
│
├── data/
│   ├── load_data.m           ← [M1] Read PNG images → [N×D] feature matrix + labels
│   └── normalize_data.m      ← [M1] Zero-mean / unit-variance normalisation
│
├── pipelines/                        (Integration glue — Member 1)
│   ├── pipeline_pca_bayes.m  ← [M1] PCA → Bayes
│   ├── pipeline_lda_bayes.m  ← [M1] LDA → Bayes
│   ├── pipeline_pca_mlp.m    ← [M1] PCA → MLP
│   └── pipeline_lda_mlp.m    ← [M1] LDA → MLP
│
├── dim_reduction/
│   ├── pca_train.m           ← [M2] Fit PCA model (eigendecomposition)
│   ├── pca_project.m         ← [M2] Project data onto principal components
│   ├── lda_train.m           ← [M3] Fit LDA model (scatter matrices + gen. eigen)
│   └── lda_project.m         ← [M3] Project data onto discriminant directions
│
├── classifiers/
│   ├── bayes_train.m         ← [M4] Fit Gaussian Bayes classifier (means, covs, priors)
│   ├── bayes_predict.m       ← [M4] Bayes inference: hard labels + posterior scores
│   ├── mlp_train.m           ← [M5] Train single-hidden-layer MLP via backpropagation
│   └── mlp_predict.m         ← [M5] MLP inference: hard labels + sigmoid scores
│
├── evaluation/
│   ├── cross_validate.m      ← [M6] k-fold cross-validation wrapper (any pipeline)
│   ├── compute_roc.m         ← [M6] Compute (FPR, TPR) curve from scores
│   ├── plot_roc.m            ← [M6] Plot multiple ROC curves on one figure
│   ├── compute_auc.m         ← [M6] AUC via trapezoidal rule
│   └── compute_accuracy.m    ← [M6] Classification accuracy
│
├── results/                  (auto-generated — ignored by git)
│   ├── roc_cv.png
│   ├── roc_test.png
│   ├── cv_results.mat
│   └── test_results.mat
│
├── train/                    (training dataset — 200 PNG images)
├── experiments/              (M7 — create run_experiments.m here)
│
├── train/                    (training dataset — 200 PNG images, NOT in git)
└── test/                     (test dataset — 100 PNG images, NOT in git; place here when released)
```

---

## How to Run

> **Before running:** MATLAB's **Current Folder** must be set to the project root (`Pattern_Recognition_Project/`). Running from any other directory will cause `Undefined function` errors because `addpath(genpath(pwd))` adds paths relative to wherever the Current Folder is set.

### Training + Cross-Validation
```matlab
>> main_train
```
Outputs accuracy, AUC, feature dim, parameter count, and timing for all four methods. Saves `results/roc_cv.png` and `results/cv_results.mat`.

> **Prerequisite:** M2–M6 must all have removed their `error('TODO...')` line before this runs end-to-end.

### Final Test Evaluation
```matlab
>> main_test
```
Trains on the full training set and evaluates on the `test/` folder. Saves `results/roc_test.png` and `results/test_results.mat`.

> **Prerequisite:** `test/` folder must exist in the project root. `main_test` will print a clear error message if it is missing.

---

## Four Pipelines

| # | Method | Dim Reduction | Classifier |
|---|---|---|---|
| 1 | PCA + Bayes | PCA (`pca_train`, `pca_project`) | Gaussian Bayes (`bayes_train`, `bayes_predict`) |
| 2 | LDA + Bayes | LDA (`lda_train`, `lda_project`) | Gaussian Bayes (`bayes_train`, `bayes_predict`) |
| 3 | PCA + MLP | PCA (`pca_train`, `pca_project`) | MLP (`mlp_train`, `mlp_predict`) |
| 4 | LDA + MLP | LDA (`lda_train`, `lda_project`) | MLP (`mlp_train`, `mlp_predict`) |

Each pipeline is evaluated with **5-fold cross-validation** reporting:
accuracy · AUC · feature dimensionality · parameter count · training time · inference time per sample

---

## Implementation Rules for Members 2–6

> **Language: MATLAB** (R2016b or later)

Each skeleton file contains the complete algorithm as **commented-out pseudocode** — every pseudocode line starts with `% ` (MATLAB's comment character followed by a space).

**To complete your task:**
1. Open your `.m` file in MATLAB or any text editor.
2. Find the `% ── TODO:` block inside the function body.
3. **Remove the `% ` prefix (percent sign + space) from the start of each pseudocode line** to activate the code.
4. Delete the `error('TODO...')` line at the bottom of the block.
5. Set MATLAB's Current Folder to the project root, then run `main_train` to test.

### Shared rules all members must follow

1. **Only edit files listed under your name.** The pipeline files in `pipelines/` belong to M1 — do not modify them. If you spot a bug in the interface, report it to M1.
2. **Remove the `error('TODO...')` guard** when your implementation is ready — it is what prevents `main_train` from running end-to-end.
3. **Classifiers (M4, M5) must set `model.param_count`** (integer) on the returned model struct — the pipeline glue code reads this field to report parameter counts.
4. **`bayes_predict` and `mlp_predict` must return `scores` as P(y=1) ∈ [0,1]** (real-valued, not just 0/1) — `cross_validate` and `compute_roc` need continuous probability scores to build the ROC curve.

---

### Member 2 — PCA (`dim_reduction/`)

Files: `pca_train.m`, `pca_project.m`

- Remove the `% ` prefix from each pseudocode line in both files, then delete `error(...)`.
- In `pca_train`, use the **compact trick**: eigen-decompose `X_c * X_c'` (200×200) instead of `X_c' * X_c` (4096×4096) since D >> N. After obtaining N-dimensional eigenvectors `U`, recover D-dimensional vectors with `V = X_c' * U`, then normalise: `V = V ./ sqrt(sum(V.^2, 1));` — **do not use `vecnorm`**, it requires R2017b and will error on R2016b.
- `pca_project` is two lines: subtract `mean_vec`, then multiply by `W`.

---

### Member 3 — LDA (`dim_reduction/`)

Files: `lda_train.m`, `lda_project.m`

- Remove the `% ` prefix from each pseudocode line in both files, then delete `error(...)`.
- Binary classification → LDA produces exactly **1** discriminant direction (K−1 = 1); `lda_model.W` will be `[D × 1]`.
- **Do not skip the regularisation line** (`S_W = S_W + 1e-6 * trace(S_W)/D * eye(D)`) — with D=4096 and ~160 training samples, S_W is rank-deficient; without regularisation, `eig(S_B, S_W)` returns `Inf`/`NaN`. If you still see `Inf`/`NaN` results after adding regularisation, increase the coefficient from `1e-6` to `1e-3`.
- `lda_project` is one line: `X_red = X * lda_model.W;`

---

### Member 4 — Bayes Classifier (`classifiers/`)

Files: `bayes_train.m`, `bayes_predict.m`

- Remove the `% ` prefix from each pseudocode line in both files, then delete `error(...)`.
- In `bayes_train`, set `model.param_count = K * (D + D^2) + K` before returning.
- In `bayes_predict`, the output `scores` must be the **posterior probability of class 1** computed via log-posterior softmax (the shifted-exp trick for numerical stability is shown in the comments).

---

### Member 5 — MLP (`classifiers/`)

Files: `mlp_train.m`, `mlp_predict.m`

- Remove the `% ` prefix from each pseudocode line in both files, then delete `error(...)`.
- `sigmoid_fn` and `sigmoid_grad_fn` at the bottom of `mlp_train.m` are already fully implemented — call them directly inside the training loop.
- Set `model.param_count = D*H + H + H + 1` before returning the model.
- `mlp_predict` is a 4-line forward pass followed by a threshold at 0.5.

---

### Member 6 — Evaluation (`evaluation/`)

Files: `compute_accuracy.m`, `compute_auc.m`, `compute_roc.m`, `cross_validate.m`, `plot_roc.m`

Implement in this order (each builds on the previous):

1. **`compute_accuracy`** — one line: `mean(y_pred(:) == y_true(:))`
2. **`compute_auc`** — two lines: sort by FPR, then `trapz(fpr, tpr)`
3. **`compute_roc`** — threshold sweep loop (pseudocode provided)
4. **`cross_validate`** — coordinates everything; averages metrics across k folds and interpolates ROC curves to a common FPR grid using `interp1` (pseudocode provided). Use `cvpartition(y, 'KFold', k)` for stratified splits — this guarantees balanced class counts per fold, unlike plain `randperm`.
5. **`plot_roc`** — optional helper; `main_train.m` already has inline plotting, but implement this for cleaner modular use

---

## Module Interface Contract

All pipeline functions must follow this signature:

```matlab
% pipeline_fn(X_train, y_train, X_test) → (y_pred, scores, info)
%
% X_train  [N_tr × D]  normalised training features
% y_train  [N_tr × 1]  labels: 0 = male, 1 = female
% X_test   [N_te × D]  normalised test features
%
% y_pred   [N_te × 1]  hard predictions (0 or 1)
% scores   [N_te × 1]  P(y=1|x) in [0,1]  — needed for ROC / AUC
% info     struct       .feature_dim  (int)
%                       .param_count  (int)
%                       .train_time   (seconds)
%                       .infer_time   (seconds per sample)
```

Classifiers must return a **model struct** that includes:
- `model.param_count` — used by pipeline glue code to report parameter count

> **Members 2–5** implement only their individual module functions. Do **not** modify anything in `pipelines/` — those are M1's files. If you find an interface bug, report it to M1 rather than editing the pipeline file yourself.

---

## Project Schedule

| Milestone | Date |
|---|---|
| Group Member List | September 29 |
| Oral Presentation + Live Demo | **December 15** |
| Written Report Submission | **December 22** |

---

## Grading

| Item | Weight |
|---|---|
| Oral Presentation & Live Demo | 60% |
| Written Report | 30% |
| Test Set AUC | 10% |

---

