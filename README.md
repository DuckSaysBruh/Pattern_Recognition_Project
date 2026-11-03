# Gender Recognition Grand Challenge
**Computer Vision & Pattern Recognition Lab (CVR Lab)**

A MATLAB-based face image gender recognition system that compares four classification pipelines combining PCA/LDA dimensionality reduction with Bayes/MLP classifiers.

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
└── test/                     (test dataset — 100 PNG images, add when available)
```

---

## How to Run

### Training + Cross-Validation
```matlab
>> main_train
```
Outputs accuracy, AUC, feature dim, parameter count, and timing for all four methods. Saves `results/roc_cv.png` and `results/cv_results.mat`.

### Final Test Evaluation
```matlab
>> main_test
```
Trains on the full training set and evaluates on the `test/` folder. Saves `results/roc_test.png` and `results/test_results.mat`.

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

## Work Distribution

| Member | Role | Files Owned |
|---|---|---|
| **M1** | Project Lead / Data Pipeline / Integration | `main_train.m`, `main_test.m`, `data/`, `pipelines/` |
| **M2** | PCA Specialist | `dim_reduction/pca_train.m`, `dim_reduction/pca_project.m` |
| **M3** | LDA Specialist | `dim_reduction/lda_train.m`, `dim_reduction/lda_project.m` |
| **M4** | Bayes Classifier | `classifiers/bayes_train.m`, `classifiers/bayes_predict.m` |
| **M5** | MLP Classifier | `classifiers/mlp_train.m`, `classifiers/mlp_predict.m` |
| **M6** | Evaluation Framework | `evaluation/cross_validate.m`, `evaluation/compute_roc.m`, `evaluation/plot_roc.m`, `evaluation/compute_auc.m`, `evaluation/compute_accuracy.m` |
| **M7** | Experiments & Analysis | Runs experiments, tunes hyperparameters, writes Chapter 3 of report |
| **M8** | Report & Presentation Lead | Slides, CCU thesis report, live demo coordination |

---

## Implementation Rules for Members 2–6

> **Language: MATLAB** (R2016b or later)

Each skeleton file contains the complete algorithm as commented-out pseudocode.
To complete your task: **uncomment the provided code block and delete the `error('TODO...')` line.**

### Shared rules all members must follow

1. **Remove the `error('TODO...')` guard** when your implementation is ready — it is what prevents `main_train` from running.
2. **Classifiers must set `model.param_count`** (integer) — the pipeline glue code reads this field to report parameter counts.
3. **`bayes_predict` and `mlp_predict` must return `scores` as P(y=1) ∈ [0,1]**, not just hard 0/1 labels — `cross_validate` and `compute_roc` depend on real-valued scores to build the ROC curve.

---

### Member 2 — PCA (`dim_reduction/`)

Files: `pca_train.m`, `pca_project.m`

- Uncomment the pseudocode already in each file.
- Use the **compact trick** in `pca_train`: eigen-decompose `X*X'` (200×200) instead of `X'*X` (4096×4096) since D >> N.
- `pca_project` is a two-liner: subtract `mean_vec`, multiply by `W`.

---

### Member 3 — LDA (`dim_reduction/`)

Files: `lda_train.m`, `lda_project.m`

- Uncomment the pseudocode already in each file.
- Binary classification → LDA produces exactly **1** discriminant direction (K−1 = 1).
- **Do not skip the regularisation line** for S_W (`S_W + 1e-6 * trace(S_W)/D * eye(D)`) — without it S_W will be singular and `eig` will fail.
- `lda_project` is a one-liner: `X * W`.

---

### Member 4 — Bayes Classifier (`classifiers/`)

Files: `bayes_train.m`, `bayes_predict.m`

- Uncomment the pseudocode in both files.
- In `bayes_train`, set `model.param_count = K * (D + D^2) + K` before returning.
- In `bayes_predict`, the output `scores` must be the **posterior probability of class 1** computed via log-posterior softmax (the shifted-exp trick for numerical stability is shown in the comments).

---

### Member 5 — MLP (`classifiers/`)

Files: `mlp_train.m`, `mlp_predict.m`

- The full backpropagation equations are written as comments in `mlp_train.m` — implement them.
- `sigmoid_fn` and `sigmoid_grad_fn` at the bottom of `mlp_train.m` are already complete; call them directly.
- Set `model.param_count = D*H + H + H + 1` before returning the model.
- `mlp_predict` is a 4-line forward pass followed by a threshold at 0.5.

---

### Member 6 — Evaluation (`evaluation/`)

Files: `compute_accuracy.m`, `compute_auc.m`, `compute_roc.m`, `cross_validate.m`, `plot_roc.m`

Implement in this order (each builds on the previous):

1. **`compute_accuracy`** — one line: `mean(y_pred(:) == y_true(:))`
2. **`compute_auc`** — two lines: sort by FPR, then `trapz(fpr, tpr)`
3. **`compute_roc`** — threshold sweep loop (pseudocode provided)
4. **`cross_validate`** — coordinates everything; averages metrics across k folds and interpolates ROC curves to a common FPR grid (pseudocode provided)
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

## Contact

Teaching Assistant: cvrlab308@gmail.com
