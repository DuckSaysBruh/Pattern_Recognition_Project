# 性別辨識大挑戰
**電腦視覺與圖形識別實驗室（CVR Lab）**

基於 MATLAB 的人臉影像性別辨識系統，比較 PCA/LDA 降維方法搭配 Bayes/MLP 分類器的四種分類流程。

---

## 工作分配

M1 ~ M8 代表各成員，**不得更動自身職責範圍以外的檔案**。  各成員必須在其欄位中填入自己名字以便劃分職責及計算分數。  
新增姓名方式：開啟 `README.md`（或 `README_zh-TW.md`），找到自己的欄位，將 `**Mn**` 改為 `**Mn 姓名**`，並以訊息 `docs: add name for Mn` 提交。

| 成員 | 角色 | 負責檔案 |
|---|---|---|
| **M1 William Hsieh** | 專案負責人 / 資料管線 / 整合 | `main_train.m`、`main_test.m`、`data/`、`pipelines/` |
| **M2** | PCA 專責 | `dim_reduction/pca_train.m`、`dim_reduction/pca_project.m` |
| **M3** | LDA 專責 | `dim_reduction/lda_train.m`、`dim_reduction/lda_project.m` |
| **M4** | 貝氏分類器 | `classifiers/bayes_train.m`、`classifiers/bayes_predict.m` |
| **M5** | MLP 分類器 | `classifiers/mlp_train.m`、`classifiers/mlp_predict.m` |
| **M6** | 評估框架 | `evaluation/cross_validate.m`、`evaluation/compute_roc.m`、`evaluation/plot_roc.m`、`evaluation/compute_auc.m`、`evaluation/compute_accuracy.m` |
| **M7** | 實驗與分析 | 無骨架檔案 — 自行建立 `experiments/run_experiments.m`，以不同超參數呼叫 `main_train` 並收集結果供報告第三章使用 |
| **M8** | 報告與簡報負責人 | 無 MATLAB 檔案 — 製作中正大學格式報告（`.docx`/`.pdf`）、簡報，並協調現場展示彩排 |

---

## 快速開始

### 1 — 複製儲存庫
```bash
git clone https://github.com/DuckSaysBruh/Pattern_Recognition_Project.git
cd Pattern_Recognition_Project
```
或至 GitHub 下載 ZIP：**Code → Download ZIP**。

### 2 — 設定 MATLAB
1. 開啟 MATLAB（R2016b 或更新版本）。
2. 在 **Current Folder** 面板中切換至 `Pattern_Recognition_Project/` 資料夾 — **這是必要步驟**，因為 `main_train` 使用 `addpath(genpath(pwd))`，路徑是相對於 MATLAB 目前工作資料夾，而非腳本的位置。
3. 將 `train/` 影像資料夾放入專案根目錄（200 張 PNG，未納入 git）。
4. *(僅 `main_test` 需要)* 測試資料集釋出後，將 `test/` 資料夾放入根目錄。

---

## 環境需求

| 需求項目 | 說明 |
|---|---|
| 程式語言 | MATLAB R2016b 或更新版本 |
| 工具箱 | Image Processing Toolbox（`imread`、`imresize`、`rgb2gray`） |
| 資料集 | 訓練影像 200 張 + 測試影像 100 張（PNG 格式人臉照片） |

---

## 資料集結構

```
train/
  man_1_1.png   man_1_2.png   …   man_50_2.png     （男性影像 100 張）
  woman_1_1.png woman_1_2.png …   woman_50_2.png   （女性影像 100 張）

test/
  man_1_1.png   …   man_25_2.png                   （男性影像 50 張）
  woman_1_1.png …   woman_25_2.png                 （女性影像 50 張）
```

標籤由檔名自動判斷：檔名含 `"woman"` → **1（女性）**，其餘 → **0（男性）**。

---

## 檔案結構

```
Pattern_Recognition_Project/
│
├── main_train.m              ← [M1] 主程式：載入資料、對所有流程執行 5-fold CV、
│                                    輸出比較表、儲存 ROC 圖
├── main_test.m               ← [M1] 主程式：以完整訓練集訓練後評估測試集
│
├── data/
│   ├── load_data.m           ← [M1] 讀取 PNG 影像 → [N×D] 特徵矩陣 + 標籤
│   └── normalize_data.m      ← [M1] 零均值 / 單位變異數正規化
│
├── pipelines/                        （整合膠水函式 — 成員 1）
│   ├── pipeline_pca_bayes.m  ← [M1] PCA → Bayes
│   ├── pipeline_lda_bayes.m  ← [M1] LDA → Bayes
│   ├── pipeline_pca_mlp.m    ← [M1] PCA → MLP
│   └── pipeline_lda_mlp.m    ← [M1] LDA → MLP
│
├── dim_reduction/
│   ├── pca_train.m           ← [M2] 訓練 PCA 模型（特徵分解）
│   ├── pca_project.m         ← [M2] 將資料投影至主成分空間
│   ├── lda_train.m           ← [M3] 訓練 LDA 模型（散佈矩陣 + 廣義特徵值）
│   └── lda_project.m         ← [M3] 將資料投影至判別方向
│
├── classifiers/
│   ├── bayes_train.m         ← [M4] 訓練高斯貝氏分類器（均值、共變異數、先驗機率）
│   ├── bayes_predict.m       ← [M4] 貝氏推論：硬標籤 + 後驗機率分數
│   ├── mlp_train.m           ← [M5] 以反向傳播訓練單隱藏層 MLP
│   └── mlp_predict.m         ← [M5] MLP 推論：硬標籤 + Sigmoid 分數
│
├── evaluation/
│   ├── cross_validate.m      ← [M6] k-fold 交叉驗證包裝器（適用任意流程）
│   ├── compute_roc.m         ← [M6] 由分數計算 ROC 曲線（FPR、TPR）
│   ├── plot_roc.m            ← [M6] 在同一張圖上繪製多條 ROC 曲線
│   ├── compute_auc.m         ← [M6] 以梯形法則計算 AUC
│   └── compute_accuracy.m    ← [M6] 計算分類正確率
│
├── results/                  （自動產生 — 已加入 .gitignore）
│   ├── roc_cv.png
│   ├── roc_test.png
│   ├── cv_results.mat
│   └── test_results.mat
│
├── experiments/              （M7 — 在此建立 run_experiments.m）
│
├── train/                    （訓練資料集 — 200 張 PNG，未納入 git）
└── test/                     （測試資料集 — 100 張 PNG，未納入 git；釋出後請放至此處）
```

---

## 執行方式

> **執行前提：** MATLAB 的 **Current Folder** 必須設定為專案根目錄（`Pattern_Recognition_Project/`）。從其他目錄執行會導致 `Undefined function` 錯誤，因為 `addpath(genpath(pwd))` 是相對於目前工作資料夾新增路徑。

### 訓練 + 交叉驗證
```matlab
>> main_train
```
輸出四種方法的正確率、AUC、特徵維度、參數量與時間。儲存 `results/roc_cv.png` 和 `results/cv_results.mat`。

> **前置條件：** M2–M6 必須全部移除各自的 `error('TODO...')` 那一行，此程式才能完整執行。

### 最終測試集評估
```matlab
>> main_test
```
以完整訓練集訓練後評估 `test/` 資料夾。儲存 `results/roc_test.png` 和 `results/test_results.mat`。

> **前置條件：** 專案根目錄必須存在 `test/` 資料夾。若資料夾不存在，`main_test` 會印出清楚的錯誤訊息。

---

## 四種分類流程

| # | 方法 | 降維 | 分類器 |
|---|---|---|---|
| 1 | PCA + Bayes | PCA（`pca_train`、`pca_project`） | 高斯貝氏（`bayes_train`、`bayes_predict`） |
| 2 | LDA + Bayes | LDA（`lda_train`、`lda_project`） | 高斯貝氏（`bayes_train`、`bayes_predict`） |
| 3 | PCA + MLP | PCA（`pca_train`、`pca_project`） | MLP（`mlp_train`、`mlp_predict`） |
| 4 | LDA + MLP | LDA（`lda_train`、`lda_project`） | MLP（`mlp_train`、`mlp_predict`） |

每種流程皆以 **5-fold 交叉驗證** 評估，回報項目：
正確率 · AUC · 特徵維度 · 參數量 · 訓練時間 · 每樣本推論時間

---

## 成員 2–6 的實作規則

> **程式語言：MATLAB**（R2016b 或更新版本）

每個骨架檔案都已將完整演算法寫成**注解形式的虛擬碼** — 每行虛擬碼都以 `% `（百分號加空格，MATLAB 的注解字元）開頭。

**完成步驟：**
1. 以 MATLAB 或文字編輯器開啟你的 `.m` 檔案。
2. 找到函式內的 `% ── TODO:` 區塊。
3. **移除每行虛擬碼開頭的 `% `（百分號加空格）**，即可啟用該程式碼。
4. 刪除區塊底部的 `error('TODO...')` 那一行。
5. 將 MATLAB Current Folder 設為專案根目錄後，執行 `main_train` 測試。

### 所有成員須遵守的共同規則

1. **只能編輯自己職責範圍內的檔案。** `pipelines/` 中的流程檔案屬於 M1 — 請勿更動。若發現介面有問題，請告知 M1。
2. **實作完成後必須移除 `error('TODO...')`** — 這行程式碼是用於阻止 `main_train` 完整執行。
3. **分類器（M4、M5）必須在回傳的 model struct 中設定 `model.param_count`**（整數）— 膠水程式碼會讀取此欄位來回報參數量。
4. **`bayes_predict` 與 `mlp_predict` 的 `scores` 輸出必須為 P(y=1) ∈ [0,1]**（實數值，非只有 0/1）— `cross_validate` 與 `compute_roc` 需要連續機率分數才能建立 ROC 曲線。

---

### 成員 2 — PCA（`dim_reduction/`）

負責檔案：`pca_train.m`、`pca_project.m`

- 移除兩個檔案中每行虛擬碼開頭的 `% `，再刪除 `error(...)`。
- 在 `pca_train` 中使用**緊湊技巧**：對 `X_c * X_c'`（200×200）進行特徵分解，而非對 `X_c' * X_c`（4096×4096），因為 D >> N。取得 N 維特徵向量 `U` 後，以 `V = X_c' * U` 還原 D 維向量，再正規化：`V = V ./ sqrt(sum(V.^2, 1));` — **請勿使用 `vecnorm`**，該函式需要 R2017b，在 R2016b 會發生錯誤。
- `pca_project` 只需兩行：減去 `mean_vec`，再乘以 `W`。

---

### 成員 3 — LDA（`dim_reduction/`）

負責檔案：`lda_train.m`、`lda_project.m`

- 移除兩個檔案中每行虛擬碼開頭的 `% `，再刪除 `error(...)`。
- 二元分類 → LDA 恰好產生 **1** 個判別方向（K−1 = 1）；`lda_model.W` 為 `[D × 1]`。
- **不可跳過 S_W 的正則化那一行**（`S_W = S_W + 1e-6 * trace(S_W)/D * eye(D)`）— D=4096 且訓練樣本僅約 160 筆，S_W 是秩不足矩陣；缺少正則化時 `eig(S_B, S_W)` 將回傳 `Inf`/`NaN`。若加入正則化後結果仍含 `Inf`/`NaN`，請將係數從 `1e-6` 調大至 `1e-3`。
- `lda_project` 只需一行：`X_red = X * lda_model.W;`

---

### 成員 4 — 貝氏分類器（`classifiers/`）

負責檔案：`bayes_train.m`、`bayes_predict.m`

- 移除兩個檔案中每行虛擬碼開頭的 `% `，再刪除 `error(...)`。
- 在 `bayes_train` 中，回傳前需設定 `model.param_count = K * (D + D^2) + K`。
- 在 `bayes_predict` 中，`scores` 輸出必須是透過對數後驗 Softmax 計算出的**類別 1 的後驗機率**（注解中已說明數值穩定的位移指數技巧）。

---

### 成員 5 — MLP（`classifiers/`）

負責檔案：`mlp_train.m`、`mlp_predict.m`

- 移除兩個檔案中每行虛擬碼開頭的 `% `，再刪除 `error(...)`。
- `mlp_train.m` 底部的 `sigmoid_fn` 與 `sigmoid_grad_fn` 已完整實作 — 直接在訓練迴圈中呼叫即可。
- 回傳模型前需設定 `model.param_count = D*H + H + H + 1`。
- `mlp_predict` 只需 4 行前向傳播，再以 0.5 作為閾值。

---

### 成員 6 — 評估框架（`evaluation/`）

負責檔案：`compute_accuracy.m`、`compute_auc.m`、`compute_roc.m`、`cross_validate.m`、`plot_roc.m`

建議依以下順序實作（後者依賴前者）：

1. **`compute_accuracy`** — 一行：`mean(y_pred(:) == y_true(:))`
2. **`compute_auc`** — 兩行：依 FPR 排序，再執行 `trapz(fpr, tpr)`
3. **`compute_roc`** — 閾值掃描迴圈（已提供虛擬碼）
4. **`cross_validate`** — 協調所有元件；對 k 個 fold 的指標取平均，並以 `interp1` 將 ROC 曲線插值至共同 FPR 網格（已提供虛擬碼）。請使用 `cvpartition(y, 'KFold', k)` 進行分層切割 — 可確保每個 fold 的類別分佈平衡，優於單純使用 `randperm`。
5. **`plot_roc`** — 選用輔助函式；`main_train.m` 已有內嵌繪圖，但實作此函式可使程式更模組化

---

## 模組介面規範

所有流程函式須遵循以下模板：

```matlab
% pipeline_fn(X_train, y_train, X_test) → (y_pred, scores, info)
%
% X_train  [N_tr × D]  正規化訓練特徵
% y_train  [N_tr × 1]  標籤：0 = 男性，1 = 女性
% X_test   [N_te × D]  正規化測試特徵
%
% y_pred   [N_te × 1]  硬預測（0 或 1）
% scores   [N_te × 1]  P(y=1|x) ∈ [0,1]  — ROC / AUC 所需
% info     struct       .feature_dim  （整數）
%                       .param_count  （整數）
%                       .train_time   （秒）
%                       .infer_time   （每樣本秒數）
```

分類器必須回傳包含以下欄位的 **model struct**：
- `model.param_count` — 由膠水程式碼讀取以回報參數量

> **成員 2–5** 只需實作各自負責的模組函式，**請勿**更動 `pipelines/` 中的任何檔案 — 那些是 M1 的負責範圍。若發現介面有問題，請告知 M1，由 M1 修正。

---

## 專案時程

| 里程碑 | 日期 |
|---|---|
| 組員名單繳交 | 9 月 29 日 |
| 口頭報告 + demo | **12 月 15 日** |
| 書面報告繳交 | **12 月 22 日** |

---

## 評分標準

| 項目 | 權重 |
|---|---|
| 口頭報告與現場展示 | 60% |
| 書面報告 | 30% |
| 測試集 AUC | 10% |


