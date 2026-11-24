# Gender Recognition Grand Challenge
Computer Vision & Pattern Recognition Lab (CVR Lab)

A MATLAB-based face image gender recognition system that compares four classification pipelines combining PCA/LDA dimensionality reduction with Bayes/MLP classifiers.

---

## Work Distribution

The member listing below follows the original project README. Module scopes have been updated to reflect the shared framework; this update does not assign new members or redistribute responsibilities.

| Member | Role | Module Scope |
|---|---|---|
| M1 William Hsieh | Project Lead / Data Pipeline / Integration | Entry points, configuration, `data/`, `pipelines/` |
| M2 Ming-Jin (Jim) Hsieh | PCA Specialist | `dim_reduction/pca_train.m`, `dim_reduction/pca_project.m` |
| M3 吳崧瑀 | LDA Specialist | `dim_reduction/lda_train.m`, `dim_reduction/lda_project.m` |
| M4 湯鈞凱 | Bayes Classifier | `classifiers/bayes_train.m`, `classifiers/bayes_predict.m` |
| M5 | MLP Classifier | `classifiers/mlp_train.m`, `classifiers/mlp_predict.m` |
| M6 Chandana Virupaksha | Evaluation Framework | `evaluation/` |
| M7 Naisha E Shelke | Experiments & Analysis | Reproducible experiments and Chapter 3 results |
| M8 | Report & Presentation Lead | CCU-format report, presentation slides, live demo preparation |

Keep module interfaces consistent when making changes. Cross-module changes should be coordinated and reviewed together.

---

## Getting Started

### 1 — Get the repository

```bash
git clone https://github.com/DuckSaysBruh/Pattern_Recognition_Project.git
cd Pattern_Recognition_Project
```

Or download the ZIP from GitHub → Code → Download ZIP.

> This README describes the updated modular framework. If the remote repository still contains the original TODO skeleton, apply the replacement project and the readable-output-name patch first. Do not mix old and new pipeline implementations.

### 2 — Set up MATLAB

1. Open MATLAB; R2020b or later is recommended. Earlier releases have not been verified.
2. In the Current Folder panel, navigate to `Pattern_Recognition_Project/`.
3. Add the project folders to the MATLAB path:

```matlab
addpath(genpath(pwd));
```

4. Prepare the training images. A `train/` folder inside the project root is recommended; the training entry point also lets you select another folder.
5. Keep datasets and generated results out of Git.

### 3 — Run the self-test

```matlab
selftest_project
```

Expected completion message:

```text
All synthetic tests passed. This does not validate the real dataset.
```

The self-test checks person-grouped folds, PCA orthogonality, sample-space LDA against a direct small-dimensional solution, ROC edge cases, and synthetic-data execution of all four methods.

---

## Requirements

| Requirement | Detail |
|---|---|
| Language | MATLAB; R2020b or later recommended |
| Toolboxes | Image Processing Toolbox |
| Training dataset | 200 images: 100 male and 100 female; 50 persons per class, two images per person |
| Test dataset | 100 held-out images, evaluated when available |
| Image size | Original 640 × 640 resolution; no resizing by default |
| GPU / Python | Not required |

PCA, LDA, Bayes, MLP, fold generation, and ROC/AUC are implemented without requiring Statistics and Machine Learning Toolbox or Deep Learning Toolbox.

A 200 × 409,600 double-precision feature matrix occupies approximately 625 MiB. Training also requires temporary arrays and data copies, so peak memory usage is higher.

---

## Dataset Layout

```text
train/
  man_1_1.jpg   man_1_2.jpg   ...   man_50_2.jpg
  woman_1_1.jpg woman_1_2.jpg ...   woman_50_2.jpg
```

Training labels and person IDs are parsed from the training naming convention:

| Prefix | Label | Example Person ID |
|---|---:|---|
| `man` / `male` | 0 | `male_1` |
| `woman` / `female` | 1 | `female_1` |

The loader supports PNG, JPG, JPEG, BMP, TIF, and TIFF images, including images in subfolders.

Fixed image preprocessing:

```text
Read image → Check 640×640 → Grayscale → Convert to [0,1] → Flatten
```

No resizing, cropping, background removal, or augmentation is applied by default. An unexpected image size raises an error instead of being silently resized.

---

## File Tree

```text
Pattern_Recognition_Project/
│
├── main_train.m              ← Training menu, shared CV, final model fitting
├── main_test.m               ← Load a saved model and evaluate new data
├── config_project.m         ← Shared experiment settings
├── create_run_dir.m         ← Readable, non-overwriting result directories
│
├── data/
│   ├── load_data.m           ← Images → feature matrix, labels, metadata
│   └── normalize_data.m      ← Fit or apply training-only normalization
│
├── pipelines/
│   ├── pipeline_train.m      ← Normalization → PCA/LDA → Bayes/MLP
│   └── pipeline_predict.m    ← Apply the complete saved pipeline
│
├── dim_reduction/
│   ├── pca_train.m           ← Economy-SVD PCA
│   ├── pca_project.m         ← Project onto principal components
│   ├── lda_train.m           ← Sample-space regularized Fisher LDA
│   └── lda_project.m         ← Project onto the discriminant direction
│
├── classifiers/
│   ├── bayes_train.m         ← Class means, covariances, priors
│   ├── bayes_predict.m       ← Hard labels and posterior scores
│   ├── mlp_train.m           ← Single-hidden-layer MLP
│   └── mlp_predict.m         ← Forward inference
│
├── evaluation/
│   ├── make_folds.m          ← Shared, person-grouped stratified folds
│   ├── cross_validate.m      ← Training and evaluation for each fold
│   ├── compute_roc.m         ← Tie-aware ROC construction
│   ├── plot_roc.m            ← Plot ROC curves
│   ├── compute_auc.m         ← Trapezoidal AUC
│   └── compute_accuracy.m    ← Classification accuracy
│
├── tests/
│   └── selftest_project.m    ← Synthetic correctness and smoke tests
│
├── results/                 ← Auto-generated; ignored by Git
├── train/                   ← Local training images; ignored by Git
└── test/                    ← Local held-out images; ignored by Git
```

The two shared pipeline functions replace the original four `pipeline_pca_*` / `pipeline_lda_*` wrappers. Create the local dataset directories as needed.

---

## How to Run

### Training + Cross-Validation

```matlab
main_train
```

Select a method from the keyboard menu:

```text
1 PCA + Bayes
2 LDA + Bayes
3 PCA + MLP
4 LDA + MLP
5 All four
0 Exit
```

Then select the training folder. For the first experiment, select `1` to run PCA + Bayes.

The entry point:

1. Loads images and person metadata.
2. Creates one shared fold assignment.
3. Runs the selected methods on the same folds.
4. Saves per-fold metrics, out-of-fold predictions, ROC plots, and configuration.
5. Fits each final model on the full training set and saves it.
6. Reloads each saved model and checks prediction consistency on a small sample.

The reload check verifies serialization consistency, not generalization performance.

### Final Test Evaluation

```matlab
main_test
```

Select a saved `final_model.mat` and the test image folder. The entry point provides an optional ground-truth CSV selection or a predictions-only mode.

No retraining is performed. Normalization, dimensionality reduction, and classification use the saved training parameters. Accuracy is reported when complete ground truth is available; ROC/AUC additionally require both classes.

### Scripted Experiments

```matlab
addpath(genpath(pwd));
cfg = config_project();
cfg.train_path = fullfile(pwd, 'train');
cfg.methods = {'pca_bayes'};
runDir = main_train(cfg);
```

Run all four methods:

```matlab
cfg.methods = {'pca_bayes', 'lda_bayes', 'pca_mlp', 'lda_mlp'};
runDir = main_train(cfg);
```

---

## Four Pipelines

| # | Method | Dimensionality Reduction | Classifier |
|---|---|---|---|
| 1 | PCA + Bayes | PCA | Gaussian Bayes |
| 2 | LDA + Bayes | Regularized Fisher LDA | Gaussian Bayes |
| 3 | PCA + MLP | PCA | Single-hidden-layer MLP |
| 4 | LDA + MLP | Regularized Fisher LDA | Single-hidden-layer MLP |

Each method reports accuracy, AUC, feature dimensionality, classifier parameter count, training time, and batch-average inference time per image.

---

## Experiment Configuration

Edit `config_project.m` to change shared settings. Method selection is handled by `main_train`; changing methods does not require editing individual modules.

| Setting | Default |
|---|---|
| Image size | `[640 640]` |
| Resize | `false` |
| CV mode | `'person_grouped'` |
| Folds / seed | `5` / `42` |
| Pixel normalization | Training-only z-score |
| PCA dimensions | `20` |
| PCA relative rank tolerance | `1e-10` |
| LDA relative / absolute ridge | `0.1` / `1e-10` |
| Bayes absolute ridge | `1e-5` |
| MLP hidden units | `64` |
| MLP learning rate | `0.01` |
| MLP epochs | `1000` |
| MLP base seed / L2 | `42` / `0` |
| Inference timing repetitions | `5` |

These are baseline settings, not a claim of optimal hyperparameters. Full settings are saved with every experiment.

---

## Cross-Validation Protocol

The default is person-grouped, class-stratified 5-fold cross-validation. All images belonging to one person stay in the same fold.

For the current dataset, each fold contains:

| Split | Persons | Images |
|---|---:|---:|
| Training | 80 | 160 |
| Validation | 20 | 40 |

Validation contains 20 male and 20 female images. The training/validation person-ID intersection must be empty.

All learned preprocessing is fitted inside each training fold:

```text
Training fold → Fit normalization → Fit PCA/LDA → Fit classifier
Validation fold → Apply saved normalization → Project → Predict
```

All methods within a run share the same folds. Across runs, keep dataset ordering, person IDs, seed, and CV mode unchanged, and verify the saved fold assignments.

`cfg.cv.mode = 'image_stratified'` enables an image-level comparison but does not enforce person-disjoint evaluation. If persons have unequal image counts, person stratification does not guarantee equal image counts in each fold.

---

## Module Implementation Notes

### PCA — `dim_reduction/`

- Uses economy SVD, not a pixel-by-pixel covariance matrix.
- Stores the training mean, projection matrix, and retained variance ratios.
- Rejects a requested dimension exceeding the numerical rank.

### LDA — `dim_reduction/`

- Uses binary ridge Fisher LDA and outputs one discriminant direction.
- Defines within-class covariance as `Sw = R' * R`, with class-centered residuals scaled by `sqrt(N-2)`.
- Solves the following in sample space using the Woodbury identity:

```matlab
w = (Sw + lambda * I) \ (mu1 - mu0);
lambda = max(ridge_absolute, ridge_relative * trace(Sw) / min(D, N-2));
```

- Never constructs the 409,600 × 409,600 scatter matrix.
- The ridge is relative to the mean nonzero eigenvalue of `Sw` (`rank(Sw) = min(D, N-2)`), so the tuned value `ridge_relative = 0.1` does not depend on image resolution. Person-grouped 5×5-fold CV on the training set: validation d′ 10.60 → 10.81 vs the old `1e-3 · trace/D`; accuracy/AUC are 100% / 1.000 for both.
- Does not introduce a preceding PCA stage. Regularization is part of the method and must be described in the report.

### Bayes — `classifiers/`

- Fits separate full Gaussian covariance matrices for the two classes.
- Uses covariance denominator `Nc-1`, not the Gaussian MLE denominator `Nc`.
- Adds an explicit diagonal ridge and estimates priors from training class counts.
- Uses Cholesky-based likelihood computation and stable posterior normalization.

### MLP — `classifiers/`

- One sigmoid hidden layer and one sigmoid output.
- Binary cross-entropy with full-batch gradient descent, not mini-batch SGD.
- Fits an additional training-only scaler on reduced features and saves it inside the classifier.
- Saves loss history. Fold initialization uses the base seed plus the fold index; final training uses the base seed.

### Evaluation — `evaluation/`

- Computes ROC points by admitting tied scores together.
- Uses trapezoidal integration without directly interpolating duplicate FPR coordinates.
- Reports fold-mean AUC separately from pooled out-of-fold AUC.
- ROC figures show the pooled curve, with its corresponding pooled AUC in the legend.

---

## Module Interface Contract

### Data

```matlab
[X, y, meta] = load_data(dataPath, cfg, true);
% X               [N × D] grayscale image vectors in [0,1]
% y               [N × 1] labels: 0 = male, 1 = female
% meta.files      image paths
% meta.person_id  person IDs
% meta.image_size image dimensions
```

### Training and Inference

```matlab
[model, info] = pipeline_train(X_train, y_train, method, cfg);
[y_pred, scores] = pipeline_predict(model, X_new);
% method: 'pca_bayes', 'lda_bayes', 'pca_mlp', or 'lda_mlp'
% y_pred [N × 1] hard predictions
% scores [N × 1] P(y=1|x), the female posterior score
```

The saved model contains the configuration, fitted normalization, dimensionality reduction model, classifier, and training information. Inference does not estimate new preprocessing statistics.

---

## Parameters and Timing

### Parameter Counts

Let `d` be the reduced feature dimension, `K` the class count, and `H` the hidden-layer width.

| Quantity | Definition |
|---|---|
| Bayes logical stored parameters | `K*(d+d^2)+K` |
| Bayes free parameters | `K*d+K*d*(d+1)/2+(K-1)` |
| MLP network parameters | `d*H+H+H+1` |
| MLP input scaler | `2*d` additional scalars |

The classifier count is separate from normalization and dimensionality-reduction storage. `PipelineLogicalStoredScalars` includes those transforms, classifier logical parameters, and the MLP input scaler. It is not a file-size or RAM estimate and excludes some cached and diagnostic arrays.

### Timing

- Training includes normalization fitting/application, dimensionality-reduction fitting/training projection, and classifier fitting.
- Inference includes saved normalization, projection, and classification.
- File reading and fixed image preprocessing are excluded.
- Inference is warmed up, repeated five times, and summarized by the median batch duration divided by image count.
- This is batch-average time per image, not single-request end-to-end latency.

---

## Results

Experiment directories use readable names:

```text
train_pca_bayes_640x640_pca20_person_grouped_20261111_144000
train_all4_640x640_pca20_person_grouped_20261111_144000
```

If a name already exists, `_02`, `_03`, etc. are appended. The name is only a settings summary; the saved configuration is authoritative. The `pca20` portion does not affect LDA-only methods.

```text
results/<training run>/
├── console_log.txt
├── config_and_folds.mat
├── dataset_and_folds.csv
├── environment.mat
├── summary.csv
├── roc_comparison.png
├── experiment.mat
└── pca_bayes/                 # one directory per selected method
    ├── fold_results.csv
    ├── oof_predictions.csv
    ├── misclassified.csv
    ├── cv_results.mat
    ├── roc.png
    ├── final_model.mat
    └── test_pca_bayes_<time>/
        ├── test_predictions.csv
        ├── test_summary.csv
        ├── test_results.mat
        └── test_roc.png       # generated when both true classes are available
```

The out-of-fold confusion matrix is stored in `r.confusion_matrix` inside `cv_results.mat`; rows are true classes and columns are predictions, ordered male/female.

---

## Reproducibility and Validation

- Keep the held-out test set separate from training, cross-validation, and hyperparameter selection.
- Retain configurations, fold assignments, software versions, and per-image predictions.
- Person grouping depends on correct person IDs; visual duplicate detection is not included.
- A perfect CV score does not guarantee perfect performance on external data.
- The package was generated without a MATLAB runtime. Run the supplied self-test and validate the real-data results locally; no blanket MATLAB verification or accuracy guarantee is claimed.
- Written analysis, slides, and demo preparation remain part of the course deliverables.

---

## Project Schedule

| Milestone | Date |
|---|---|
| Group Member List | September 29 |
| Oral Presentation + Live Demo | December 15 |
| Written Report Submission | December 22 |

---

## Grading

| Item | Weight |
|---|---|
| Oral Presentation & Live Demo | 60% |
| Written Report | 30% |
| Test Set AUC | 10% |

---
