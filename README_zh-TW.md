# 性別辨識大挑戰
**電腦視覺與圖形識別實驗室（CVR Lab）**

基於 MATLAB 的人臉影像性別辨識系統，比較 PCA/LDA 降維方法搭配 Bayes/MLP 分類器的四種分類流程。

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
├── train/                    （訓練資料集 — 200 張 PNG）
└── test/                     （測試資料集 — 100 張 PNG，需另行取得）
```

---

## 執行方式

### 訓練 + 交叉驗證
```matlab
>> main_train
```
輸出四種方法的正確率、AUC、特徵維度、參數量與時間。儲存 `results/roc_cv.png` 和 `results/cv_results.mat`。

### 最終測試集評估
```matlab
>> main_test
```
以完整訓練集訓練後評估 `test/` 資料夾。儲存 `results/roc_test.png` 和 `results/test_results.mat`。

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

## 工作分配

| 成員 | 角色 | 負責檔案 |
|---|---|---|
| **M1** | 專案負責人 / 資料管線 / 整合 | `main_train.m`、`main_test.m`、`data/`、`pipelines/` |
| **M2** | PCA 專責 | `dim_reduction/pca_train.m`、`dim_reduction/pca_project.m` |
| **M3** | LDA 專責 | `dim_reduction/lda_train.m`、`dim_reduction/lda_project.m` |
| **M4** | 貝氏分類器 | `classifiers/bayes_train.m`、`classifiers/bayes_predict.m` |
| **M5** | MLP 分類器 | `classifiers/mlp_train.m`、`classifiers/mlp_predict.m` |
| **M6** | 評估框架 | `evaluation/cross_validate.m`、`evaluation/compute_roc.m`、`evaluation/plot_roc.m`、`evaluation/compute_auc.m`、`evaluation/compute_accuracy.m` |
| **M7** | 實驗與分析 | 執行實驗、調整超參數、撰寫報告第三章 |
| **M8** | 報告與簡報負責人 | 投影片、中正大學論文格式報告、現場展示協調 |

---

## 成員 2–6 的實作規則

> **程式語言：MATLAB**（R2016b 或更新版本）

每個骨架檔案都已將完整演算法寫成注解形式的虛擬碼。
完成方式：**取消注解對應的程式碼區塊，並刪除 `error('TODO...')` 那一行。**

### 所有成員須遵守的共同規則

1. **實作完成後必須移除 `error('TODO...')` 守衛** — 這行程式碼是阻止 `main_train` 執行的原因。
2. **分類器必須設定 `model.param_count`**（整數）— 流程膠水函式會讀取此欄位來回報參數量。
3. **`bayes_predict` 與 `mlp_predict` 的 `scores` 輸出必須為 P(y=1) ∈ [0,1]**，不能只輸出硬標籤 0/1 — `cross_validate` 與 `compute_roc` 需要實數分數才能建立 ROC 曲線。

---

### 成員 2 — PCA（`dim_reduction/`）

負責檔案：`pca_train.m`、`pca_project.m`

- 取消各檔案中已提供的虛擬碼注解。
- 在 `pca_train` 中使用**緊湊技巧**：對 `X*X'`（200×200）進行特徵分解，而非對 `X'*X`（4096×4096），因為 D >> N。
- `pca_project` 只需兩行：減去 `mean_vec`，再乘以 `W`。

---

### 成員 3 — LDA（`dim_reduction/`）

負責檔案：`lda_train.m`、`lda_project.m`

- 取消各檔案中已提供的虛擬碼注解。
- 二元分類 → LDA 恰好產生 **1** 個判別方向（K−1 = 1）。
- **不可跳過 S_W 的正則化那一行**（`S_W + 1e-6 * trace(S_W)/D * eye(D)`）— 沒有它，S_W 會是奇異矩陣，`eig` 將會失敗。
- `lda_project` 只需一行：`X * W`。

---

### 成員 4 — 貝氏分類器（`classifiers/`）

負責檔案：`bayes_train.m`、`bayes_predict.m`

- 取消兩個檔案中已提供的虛擬碼注解。
- 在 `bayes_train` 中，回傳前需設定 `model.param_count = K * (D + D^2) + K`。
- 在 `bayes_predict` 中，`scores` 輸出必須是透過對數後驗 Softmax 計算出的**類別 1 的後驗機率**（注解中已說明數值穩定的位移指數技巧）。

---

### 成員 5 — MLP（`classifiers/`）

負責檔案：`mlp_train.m`、`mlp_predict.m`

- 完整的反向傳播方程式已在 `mlp_train.m` 的注解中提供 — 直接實作即可。
- `mlp_train.m` 底部的 `sigmoid_fn` 與 `sigmoid_grad_fn` 已完成，可直接呼叫。
- 回傳模型前需設定 `model.param_count = D*H + H + H + 1`。
- `mlp_predict` 只需 4 行前向傳播，再以 0.5 作為閾值。

---

### 成員 6 — 評估框架（`evaluation/`）

負責檔案：`compute_accuracy.m`、`compute_auc.m`、`compute_roc.m`、`cross_validate.m`、`plot_roc.m`

建議依以下順序實作（後者依賴前者）：

1. **`compute_accuracy`** — 一行：`mean(y_pred(:) == y_true(:))`
2. **`compute_auc`** — 兩行：依 FPR 排序，再執行 `trapz(fpr, tpr)`
3. **`compute_roc`** — 閾值掃描迴圈（已提供虛擬碼）
4. **`cross_validate`** — 協調所有元件；對 k 個 fold 的指標取平均，並將 ROC 曲線插值至共同 FPR 網格（已提供虛擬碼）
5. **`plot_roc`** — 選用輔助函式；`main_train.m` 已有內嵌繪圖，但實作此函式可使程式更模組化

---

## 模組介面規範

所有流程函式必須遵循以下簽名：

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
- `model.param_count` — 由流程膠水函式讀取以回報參數量

---

## 專案時程

| 里程碑 | 日期 |
|---|---|
| 組員名單繳交 | 9 月 29 日 |
| 口頭報告 + 現場展示 | **12 月 15 日** |
| 書面報告繳交 | **12 月 22 日** |

---

## 評分標準

| 項目 | 權重 |
|---|---|
| 口頭報告與現場展示 | 60% |
| 書面報告 | 30% |
| 測試集 AUC | 10% |

---

## 聯絡資訊

助教信箱：cvrlab308@gmail.com
