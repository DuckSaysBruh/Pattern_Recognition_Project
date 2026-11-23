# 性別辨識大挑戰
電腦視覺與圖形識別實驗室（CVR Lab）

基於 MATLAB 的人臉影像性別辨識系統，比較 PCA／LDA 降維搭配 Bayes／MLP 分類器的四種分類流程。

---

## 工作分配

以下成員名單沿用原專案 README，模組範圍依共用框架更新；本次文件更新不重新分配成員或職責。

| 成員 | 角色 | 模組範圍 |
|---|---|---|
| M1 William Hsieh | 專案負責人／資料管線／整合 | 主程式、設定、`data/`、`pipelines/` |
| M2 Ming-Jin (Jim) Hsieh | PCA 專責 | `dim_reduction/pca_train.m`、`dim_reduction/pca_project.m` |
| M3 吳崧瑀 | LDA 專責 | `dim_reduction/lda_train.m`、`dim_reduction/lda_project.m` |
| M4 湯鈞凱 | Bayes 分類器 | `classifiers/bayes_train.m`、`classifiers/bayes_predict.m` |
| M5 | MLP 分類器 | `classifiers/mlp_train.m`、`classifiers/mlp_predict.m` |
| M6 Chandana Virupaksha | 評估框架 | `evaluation/` |
| M7 Naisha E Shelke | 實驗與分析 | 可重現實驗及報告第三章結果 |
| M8 | 報告與簡報負責人 | 中正大學格式報告、簡報與現場 Demo 準備 |

修改時應維持模組介面一致。涉及多個模組的變更，應共同協調與審查。

---

## 快速開始

### 1 — 取得儲存庫

```bash
git clone https://github.com/DuckSaysBruh/Pattern_Recognition_Project.git
cd Pattern_Recognition_Project
```

也可以從 GitHub → Code → Download ZIP 下載。

> 本文件描述更新後的模組化框架。如果遠端仍是原本的 TODO 骨架，請先套用完整替換專案及結果資料夾命名修改包，不要混用新舊 pipeline 實作。

### 2 — 設定 MATLAB

1. 開啟 MATLAB，建議使用 R2020b 或更新版本；較舊版本尚未驗證。
2. 在 Current Folder 面板切換至 `Pattern_Recognition_Project/`。
3. 將專案子資料夾加入 MATLAB 路徑：

```matlab
addpath(genpath(pwd));
```

4. 準備訓練影像，建議放在根目錄的 `train/`；訓練主程式也可以選擇其他資料夾。
5. 資料集與執行結果不納入 Git。

### 3 — 執行自測

```matlab
selftest_project
```

成功時顯示：

```text
All synthetic tests passed. This does not validate the real dataset.
```

自測包含人物分組、PCA 正交性、樣本空間 LDA 與小維度直接解的一致性、ROC 邊界情況，以及四種方法的合成資料訓練與推論。

---

## 環境需求

| 需求項目 | 說明 |
|---|---|
| 程式語言 | MATLAB，建議 R2020b 或更新版本 |
| 工具箱 | Image Processing Toolbox |
| 訓練資料 | 200 張，男性與女性各 100 張；每類 50 人，每人兩張 |
| 測試資料 | 100 張獨立測試影像，取得後進行評估 |
| 影像尺寸 | 原尺寸 640 × 640，預設不縮圖 |
| GPU／Python | 不需要 |

PCA、LDA、Bayes、MLP、分折與 ROC／AUC 實作不依賴 Statistics and Machine Learning Toolbox 或 Deep Learning Toolbox。

200 × 409,600 的 double 特徵矩陣約占 625 MiB。訓練仍會建立暫存與副本，因此實際峰值記憶體高於此數字。

---

## 資料集結構

```text
train/
  man_1_1.jpg   man_1_2.jpg   ...   man_50_2.jpg
  woman_1_1.jpg woman_1_2.jpg ...   woman_50_2.jpg
```

訓練標籤與人物 ID 依訓練命名規則解析：

| 前綴 | 標籤 | 人物 ID 範例 |
|---|---:|---|
| `man`／`male` | 0 | `male_1` |
| `woman`／`female` | 1 | `female_1` |

讀取支援 PNG、JPG、JPEG、BMP、TIF、TIFF，並包含子資料夾內影像。

固定影像前處理：

```text
讀取影像 → 檢查 640×640 → 轉灰階 → 轉成 [0,1] → 攤平成向量
```

預設不縮圖、不裁切、不移除背景、不做資料增強。尺寸不符時直接報錯，不會自動縮放。

---

## 檔案結構

```text
Pattern_Recognition_Project/
│
├── main_train.m              ← 訓練選單、共用五折驗證與最終模型訓練
├── main_test.m               ← 載入已保存模型並評估新資料
├── config_project.m         ← 集中管理實驗設定
├── create_run_dir.m         ← 建立易辨識且不覆蓋舊結果的資料夾
│
├── data/
│   ├── load_data.m           ← 影像轉特徵矩陣、標籤與中繼資料
│   └── normalize_data.m      ← 估計或套用訓練資料標準化參數
│
├── pipelines/
│   ├── pipeline_train.m      ← 標準化 → PCA/LDA → Bayes/MLP
│   └── pipeline_predict.m    ← 套用完整已保存流程
│
├── dim_reduction/
│   ├── pca_train.m           ← Economy SVD PCA
│   ├── pca_project.m         ← 主成分投影
│   ├── lda_train.m           ← 樣本空間正則化 Fisher LDA
│   └── lda_project.m         ← 判別方向投影
│
├── classifiers/
│   ├── bayes_train.m         ← 類別均值、共變異數與先驗
│   ├── bayes_predict.m       ← 分類標籤與後驗機率
│   ├── mlp_train.m           ← 單隱藏層 MLP
│   └── mlp_predict.m         ← 前向推論
│
├── evaluation/
│   ├── make_folds.m          ← 共用人物分組與類別分層
│   ├── cross_validate.m      ← 每折訓練與評估
│   ├── compute_roc.m         ← 處理同分數的 ROC 計算
│   ├── plot_roc.m            ← ROC 繪圖
│   ├── compute_auc.m         ← 梯形法 AUC
│   └── compute_accuracy.m    ← 分類準確率
│
├── tests/
│   └── selftest_project.m    ← 合成資料正確性與基本執行測試
│
├── results/                 ← 自動建立，不納入 Git
├── train/                   ← 本地訓練資料，不納入 Git
└── test/                    ← 本地獨立測試資料，不納入 Git
```

兩個共用 pipeline 函式取代原本四個 `pipeline_pca_*`／`pipeline_lda_*` 組合函式。本地資料集資料夾可依需要自行建立。

---

## 執行方式

### 訓練＋交叉驗證

```matlab
main_train
```

以鍵盤選擇方法：

```text
1 PCA + Bayes
2 LDA + Bayes
3 PCA + MLP
4 LDA + MLP
5 All four
0 Exit
```

接著選擇訓練資料夾。第一次建議選 1，先跑通 PCA + Bayes。

主程式會：

1. 讀取影像與人物資訊。
2. 建立一份共用分折。
3. 使用相同分折執行選定方法。
4. 保存每折指標、out-of-fold 預測、ROC 與設定。
5. 使用全部訓練資料建立並保存最終模型。
6. 重新載入模型，檢查少量影像的預測是否一致。

重新載入檢查驗證的是保存一致性，不是泛化效能。

### 最終測試集評估

```matlab
main_test
```

選擇已保存的 `final_model.mat` 與測試資料夾。主程式可選擇提供真實標籤 CSV，或僅產生預測。

測試不重新訓練；標準化、降維與分類全部沿用模型保存的訓練參數。真實標籤完整時計算 Accuracy；兩類都存在時才計算 ROC／AUC。

### 自動化實驗

```matlab
addpath(genpath(pwd));
cfg = config_project();
cfg.train_path = fullfile(pwd, 'train');
cfg.methods = {'pca_bayes'};
runDir = main_train(cfg);
```

一次執行四種方法：

```matlab
cfg.methods = {'pca_bayes', 'lda_bayes', 'pca_mlp', 'lda_mlp'};
runDir = main_train(cfg);
```

---

## 四種分類流程

| # | 方法 | 降維 | 分類器 |
|---|---|---|---|
| 1 | PCA + Bayes | PCA | Gaussian Bayes |
| 2 | LDA + Bayes | 正則化 Fisher LDA | Gaussian Bayes |
| 3 | PCA + MLP | PCA | 單隱藏層 MLP |
| 4 | LDA + MLP | 正則化 Fisher LDA | 單隱藏層 MLP |

每種方法回報 Accuracy、AUC、特徵維度、分類器參數量、訓練時間與批次平均每張推論時間。

---

## 實驗設定

共用設定集中於 `config_project.m`。方法由 `main_train` 選單選擇，不必修改個別模組。

| 設定 | 預設值 |
|---|---|
| 影像尺寸 | `[640 640]` |
| 縮放影像 | `false` |
| 交叉驗證模式 | `'person_grouped'` |
| 折數／seed | `5`／`42` |
| 像素標準化 | 僅以訓練部分估計的 z-score |
| PCA 維度 | `20` |
| PCA 相對秩門檻 | `1e-10` |
| LDA 相對／絕對 ridge | `1e-3`／`1e-10` |
| Bayes 絕對 ridge | `1e-5` |
| MLP 隱藏層寬度 | `64` |
| MLP learning rate | `0.01` |
| MLP epochs | `1000` |
| MLP 基準 seed／L2 | `42`／`0` |
| 推論計時次數 | `5` |

這些是基準設定，不代表最佳超參數。每次實驗會保存完整設定。

---

## 交叉驗證規則

預設採人物互斥、依人物性別分層的五折交叉驗證。同一人的全部影像放在同一折。

目前資料每折分配：

| 集合 | 人物數 | 影像數 |
|---|---:|---:|
| 訓練 | 80 | 160 |
| 驗證 | 20 | 40 |

驗證影像男性與女性各 20 張；訓練與驗證人物 ID 交集必須為空。

所有需要學習的前處理，都在每折訓練部分估計：

```text
訓練部分 → 估計標準化 → 建立 PCA/LDA → 訓練分類器
驗證部分 → 套用訓練標準化 → 投影 → 預測
```

同次執行的所有方法共用分折。跨次執行時須保持資料排序、人物 ID、seed 與驗證模式一致，並核對保存的分折紀錄。

`cfg.cv.mode = 'image_stratified'` 可進行照片層級比較，但不保證人物互斥。若各人物照片數不同，人物分層不保證每折影像數相同。

---

## 模組實作說明

### PCA — `dim_reduction/`

- 使用 economy SVD，不建立像素對像素共變異數矩陣。
- 保存訓練均值、投影矩陣與保留變異比例。
- 要求維度超過數值秩時報錯。

### LDA — `dim_reduction/`

- 二元 ridge Fisher LDA，輸出一維判別方向。
- `Sw = R' * R`，其中 R 是類內殘差除以 `sqrt(N-2)`。
- 以 Woodbury identity 在樣本空間求解：

```matlab
w = (Sw + lambda * I) \ (mu1 - mu0);
lambda = max(ridge_absolute, ridge_relative * trace(Sw) / D);
```

- 不建立 409,600 × 409,600 散佈矩陣。
- 沒有前置 PCA；正則化是方法定義的一部分，報告需說明。

### Bayes — `classifiers/`

- 兩類各有獨立完整高斯共變異數矩陣。
- 共變異數分母為 `Nc-1`，不是高斯 MLE 使用的 `Nc`。
- 加入對角 ridge，先驗依訓練類別比例估計。
- 使用 Cholesky 計算似然，並穩定地轉成後驗機率。

### MLP — `classifiers/`

- 一個 sigmoid 隱藏層與一個 sigmoid 輸出。
- Binary cross entropy 搭配 full-batch gradient descent，不是 mini-batch SGD。
- 分類器內另以訓練資料估計降維後特徵的標準化參數，隨模型保存。
- 保存 loss history；CV 每折使用基準 seed 加折號，最終訓練使用基準 seed。

### 評估 — `evaluation/`

- ROC 將相同分數的樣本同時納入。
- 使用梯形積分，不直接對重複 FPR 座標插值。
- 分開回報各折平均 AUC 與 pooled out-of-fold AUC。
- ROC 圖使用 pooled 曲線，圖例標示該曲線的 pooled AUC。

---

## 模組介面規範

### 資料

```matlab
[X, y, meta] = load_data(dataPath, cfg, true);
% X               [N × D] 灰階影像向量，範圍 [0,1]
% y               [N × 1] 標籤：男性 0、女性 1
% meta.files      影像路徑
% meta.person_id  人物 ID
% meta.image_size 影像尺寸
```

### 訓練與推論

```matlab
[model, info] = pipeline_train(X_train, y_train, method, cfg);
[y_pred, scores] = pipeline_predict(model, X_new);
% method: 'pca_bayes'、'lda_bayes'、'pca_mlp' 或 'lda_mlp'
% y_pred [N × 1] 預測標籤
% scores [N × 1] 女性後驗機率 P(y=1|x)
```

模型包含設定、標準化、降維模型、分類器與訓練資訊。推論不重新估計前處理統計量。

---

## 參數量與計時

### 參數量

d 為降維後特徵維度、K 為類別數、H 為隱藏層寬度。

| 項目 | 定義 |
|---|---|
| Bayes 邏輯儲存參數 | `K*(d+d^2)+K` |
| Bayes 自由參數 | `K*d+K*d*(d+1)/2+(K-1)` |
| MLP 網路參數 | `d*H+H+H+1` |
| MLP 輸入標準化 | 額外 `2*d` 個數值 |

分類器參數與標準化、降維儲存量分開列出。`PipelineLogicalStoredScalars` 包含這些轉換、分類器邏輯參數與 MLP 輸入標準化；它不是檔案大小或 RAM 估計，也不包含所有快取及診斷陣列。

### 計時

- 訓練：標準化估計與套用、降維建立與訓練投影、分類器建立。
- 推論：套用已保存標準化、投影與分類。
- 不包含讀檔與固定影像前處理。
- 推論先暖機，再重複五次，取批次時間中位數除以張數。
- 這是批次平均每張時間，不是單張請求的完整系統延遲。

---

## 執行結果

結果資料夾採易辨識命名：

```text
train_pca_bayes_640x640_pca20_person_grouped_20261111_144000
train_all4_640x640_pca20_person_grouped_20261111_144000
```

同名資料夾存在時加上 `_02`、`_03`。名稱只是設定摘要，完整設定以保存檔案為準；`pca20` 對僅使用 LDA 的方法不起作用。

```text
results/<本次訓練資料夾>/
├── console_log.txt
├── config_and_folds.mat
├── dataset_and_folds.csv
├── environment.mat
├── summary.csv
├── roc_comparison.png
├── experiment.mat
└── pca_bayes/                 # 每個選定方法各有一個子資料夾
    ├── fold_results.csv
    ├── oof_predictions.csv
    ├── misclassified.csv
    ├── cv_results.mat
    ├── roc.png
    ├── final_model.mat
    └── test_pca_bayes_<時間>/
        ├── test_predictions.csv
        ├── test_summary.csv
        ├── test_results.mat
        └── test_roc.png       # 兩類真實標籤可用時產生
```

OOF 混淆矩陣位於 `cv_results.mat` 的 `r.confusion_matrix`，列為真實類別、欄為預測類別，順序為男性／女性。

---

## 重現性與驗證

- 獨立測試集不參與訓練、交叉驗證或超參數選擇。
- 保存設定、分折、軟體版本與逐張預測。
- 人物分組依賴正確人物 ID，不包含視覺重複影像偵測。
- 完美的交叉驗證分數不保證外部資料也有相同表現。
- 專案生成環境沒有 MATLAB runtime。請在本機執行自測並驗證真實資料結果；不宣稱所有模組已通過 MATLAB 實機驗證，也不保證特定準確率。
- 書面分析、簡報與 Demo 準備仍屬課程交付內容。

---

## 專題時程

| 項目 | 日期 |
|---|---|
| 繳交組員名單 | 09 月 29 日 |
| 口頭報告與現場 Demo | 12 月 15 日 |
| 繳交書面報告 | 12 月 22 日 |

---

## 評分方式

| 項目 | 比例 |
|---|---|
| 口頭報告與現場 Demo | 60% |
| 書面報告 | 30% |
| 測試集 AUC | 10% |

---
