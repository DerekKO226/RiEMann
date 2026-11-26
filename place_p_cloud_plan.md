# RiEMann Place 點雲資料製作計劃（Isaac Sim）

> 建立日期：2026-11-20
> 目標：在有 Isaac Sim 的系統上，重建與 `data/mug/pick` 相同的場景（桌面、mug、rack、Franka Panda），產生 **Mug on Rack 的 Place 資料** `data/mug/place/*.npz`。格式要能直接給 `train_seg.py`、`train_mani.py` 和 `infer.py` 使用。
> 搭配工具：`scripts/data_gen/validate_npz.py`，只依賴 numpy，用來檢查產出的 npz 格式是否正確。

---

## 0. 怎麼使用這份文件

1. 先看 **§1 摘要** 和 **§2 現有資料分析**，了解為什麼要這樣設計。
2. 開工前逐項回答 **§3 需要確認的項目**。每項都附了建議預設值，不確定就先用預設值。
3. 依照 **§5 到 §11** 的 Phase 順序執行，每個 Phase 都有驗收條件。
4. 每次產出 npz 後都執行：
   ```bash
   python scripts/data_gen/validate_npz.py data/mug/place/*.npz
   ```
   出現 FAIL 時 exit code 為 1，必須修正。WARN 需要自己判斷。

---

## 1. 摘要

- 現有的 `demo.npz` **不是**隨機 SE(3) 增強。它是「手臂接近目標的 20 個時間步」，每一幀的點雲和標籤都表示在**當下的末端執行器（EE）座標系**中。
- `data/mug/pick` 很可能是**實拍資料**，桌面換成合成平面：6144 個合成桌面點加 2048 個物體點。Place 資料要照這個配方做。
- **Place 的標籤建議用「rack 掛勾上的放置框架」**，不要直接用 TCP 目標，因為 TCP 目標懸在空中，模型很難預測。EE 動作再用夾取時記錄的相對位姿換算。這也是論文模擬實驗的做法。
- **Rack 必須隨機擺放。** 範例中 rack 在每個場景幾乎都固定在 (0.7, 0)，照抄的話 Place 學不到泛化。
- 建議先用同一條流程**在 Isaac Sim 重做 pick**，驗證流程正確，再做 place。

---

## 2. 現有資料分析（從 npz 實測）

### 2.1 檔案格式

| 檔案 | 內容 | 座標系 | 標籤 |
|---|---|---|---|
| `demo.npz` | `xyz`/`rgb`: `float64 [10 軌跡, 20 幀, 8192, 3]`；`seg_center`: `[10,20,3]`；`axes`: `[10,20,9]` | 每幀為當下 EE 座標系 | 有 |
| `training/newpose/instance/distracting/all.npz` | `xyz`/`rgb`: `float64 [8192, 3]`，各一個場景 | 機器人 base 座標系，桌面在 z≈0 | **沒有** |

- `rgb` 是 0–1 的浮點數。
- `axes` 是 R 的 x、y、z 三個軸依序攤平：`R = axes.reshape(3,3).T`（見 `train_mani.py`），也就是 `axes = R.T.reshape(9)`。
- `SE3Demo` 只取每條軌跡的前 20 幀（`[:, 0:20]`）。

### 2.2 demo 的 20 幀是 EE 座標系下的軌跡

- 同一條軌跡內，幀與幀之間是精確的剛體變換：點的順序和 RGB 完全相同，Kabsch 擬合誤差約 1e-9。
- 標籤用同一個變換一起轉，代表目標在世界座標中是固定的。
- 目標距離 `|t|` 從約 0.35 m 單調降到約 0.17 m，旋轉角也逐步收斂。這和論文 A.5.2「把所有點雲轉換到 EE 座標系」一致。
- 這也表示原始資料**每條軌跡只拍一次**，每一幀只是用不同的 EE 位姿去轉換同一份點雲。

### 2.3 Mug 場景的實測規格（10 條 demo 的中位數）

| 項目 | 數值 |
|---|---|
| 工作區（eval 檔，base 座標） | x∈[0.2, 0.8]，y∈[-0.3, 0.3]，z 最高約 0.35 m |
| 桌面點 | 剛好 **6144 點**，均勻分布在上述 x、y 範圍與 z∈[-0.01, 0.01]，RGB 固定為 `[0.1176, 0.4392, 0.4078]`，是合成的 |
| 物體點 | 剛好 **2048 點**，其中 rack 約 1290、mug 約 695 |
| Rack | 高 **0.346 m**，底座寬約 0.18–0.20 m，掛勾點分布在 z 0.14–0.34 m，顏色約 `[0.62, 0.53, 0.37]`（木色） |
| Rack 位置 | 5 個 eval 場景都在 xy ≈ **(0.69–0.71, −0.01–0.03)**，是固定的 |
| Mug | 高 **0.108 m**，含把手的水平寬度 0.12–0.17 m（杯身直徑約 8–9 cm），顏色約 `[0.80, 0.63, 0.59]`（粉色系） |
| Mug 與 rack 的距離 | 0.19–0.31 m |
| 抓取點（pick 標籤） | 離桌面約 0.085 m（杯口下方約 2.4 cm），離杯身軸約 4.7 cm，代表夾在杯口邊緣；**夾爪 z 軸朝下**；離最近物體點 2–9 mm |
| 觀測涵蓋 | mug 和 rack 的 8 個方位角區間都有點，代表多台相機 360° 融合 |

### 2.4 已知的資料問題

- **`data/mug/pick/demo.npz` 第 7 條軌跡的標籤是壞的**：每一幀都是 `seg_center=0`、`axes=單位矩陣`。驗證器會回報 FAIL，訓練 pick 前請排除。
- `data/plane/pick` 每幀約有 6000 個非桌面點，超過 `downsample_table` 的 2500 點門檻，所以**桌面點會被全部丟掉**。這不影響 mug，但說明了點數配比的重要性。

### 2.5 模型對標籤的限制（`networks/equinet.py`）

- 位置是點座標的 softmax 加權平均，所以只能落在點雲的凸包內，實務上要貼近某些點。
- 姿態是預測位置 **`feature_point_radius`（2 cm）內**各點特徵的平均。半徑內沒有點時，會退回用整體加權平均。
- **結論：標籤位置必須在場景點的 2 cm 內。**

### 2.6 論文中的相關設定（arXiv 2403.19460，附錄 A.4–A.6）

- 模擬實驗用 Sapien、6 台 RGB-D、8192 點、r1=0.2 m、r2=0.02 m。
- 真實世界用 4 台 RealSense D435i、裁成 1.5 m 立方體，Mug on Rack 用 r1=0.16 m。
- 模擬中 place 的動作是 `T = T_place · T_object⁻¹`。真實世界直接預測 EE 目標。
- 論文會先讓機器人移到 pre-grasp/pre-place 位姿，以排除運動規劃器的影響；無法規劃的案例會手動剔除。

---

## 3. 需要確認的項目（開工前）

| # | 項目 | 建議預設值 | 影響 |
|---|---|---|---|
| C1 | Isaac Sim 版本（4.5 或 5.x）和**機器架構**（x86_64 或 aarch64） | 用最新穩定版 | 4.x 的 API 是 `omni.isaac.*`，5.x 改成 `isaacsim.*`。`pixi.toml` **只支援 linux-64**，aarch64 要另外處理訓練環境 |
| C2 | GPU 型號和數量 | — | config 寫死 `cuda:1`，單卡要改 `cuda:0`。Isaac Sim 需要 RTX 等級 GPU |
| C3 | **Pick 是否也在 Isaac Sim 重做** | **是**（§8） | 能驗證流程；否則 pick 是實拍、place 是模擬，兩者分屬不同資料域 |
| C4 | **Place 標籤定義**：A 用 TCP 目標，B 用 rack 掛勾的放置框架 | **B，兩者都存** | 見 §4 |
| C5 | Rack 來源：自己用 primitive 建，或取得實物 CAD | **自建** | 自建才能精確知道掛勾位姿，碰撞形狀也乾淨 |
| C6 | Rack 有幾支掛勾、目標是哪一支 | 2–3 支，目標固定為**最高的一支**（同論文） | 標籤必須唯一；多支都可以掛會造成標籤模糊 |
| C7 | Mug 模型來源和授權 | Objaverse/ShapeNet 中有把手的杯子，縮放到高 0.108 m | NI 設定需要第二個杯子（論文 NI 用高約 0.19 m 的杯子） |
| C8 | 是否要 sim2real | 先**不做** | 要做的話，相機位置要對齊實機，並加深度雜訊和邊緣缺失 |
| C9 | 運動規劃器 | RMPflow（內建），失敗率高再換 cuRobo | 決定軌跡品質和收集速度 |
| C10 | TCP 偏移與 EE 慣例 | `panda_hand` 沿 z 軸 +0.1034 m，z 軸 = 接近方向 | 要和現有 pick 資料一致，否則 pick 和 place 的 EE 定義會不同 |
| C11 | 裁切的 z 上限 | **0.6 m** | Place 運輸中舉起的杯子會超過原本的 0.35 m |
| C12 | 評估集數量 | 每種設定 1 個相容檔加 20 個 seed | 論文用 20 個 seed |
| C13 | 是否做閉環評估（實際放開看有沒有掛住） | 是 | 只看 geodesic 誤差，無法反映實際成功率 |
| C14 | 資料版控 | demo.npz 約 79 MB，用 Git LFS 或不進版控 | `origin` 指向上游 `HeegerGao/RiEMann`，要 push 的話需要自己的 remote |

---

## 4. 核心設計：Place 標籤

### 4.1 座標定義

- `T_ee_k`：第 k 幀時 EE（TCP）在世界座標的位姿。
- `T_peg`：目標掛勾上「放置框架」的世界位姿。
  - 原點：把手掛上去時，把手內側和掛勾的接觸點，位於掛勾的點上。
  - 軸向：z 沿掛勾方向，x 取水平，其餘依右手定則。
- `T_H`：杯子上「把手框架」的世界位姿，定義成掛好時 `T_H = T_peg`。
- `G = T_ee⁻¹ · T_H`：夾取後，把手框架相對 EE 的位姿。夾住之後就固定不變，直接從 Isaac Sim 讀取真值。

### 4.2 標籤 B（建議做法，寫入 `seg_center` 和 `axes`）

```
T_label_world = T_peg
T_label_k     = inv(T_ee_k) @ T_peg          # 第 k 幀的標籤（EE 座標）
seg_center[k] = T_label_k[:3, 3]
axes[k]       = T_label_k[:3, :3].T.reshape(9)
```

執行時這樣換算 EE 目標：`T_ee_target = T_peg_pred · inv(G)`。

**優點：**
- 標籤落在 rack 的點上，符合 §2.5 的 2 cm 限制。
- 杯子怎麼被夾住都不影響標籤，夾取差異由 `G` 吸收。

### 4.3 標籤 A（對照組，額外存成其他 key）

```
T_ee_target   = T_peg @ inv(G)
ee_target_pos[k]  = (inv(T_ee_k) @ T_ee_target)[:3, 3]
ee_target_axes[k] = (inv(T_ee_k) @ T_ee_target)[:3, :3].T.reshape(9)
grasp_in_ee       = G                         # [traj, 4, 4]
```

`SE3Demo` 只讀 `xyz/rgb/seg_center/axes`，多加的 key 不影響訓練。之後如果想試 A，只要把這兩個 key 改名成 `seg_center`/`axes`，不必重新收集。

---

## 5. Phase 0：環境遷移與軟體

1. **複製專案時排除 macOS 的 `._*` 檔：**
   ```bash
   rsync -a --exclude '._*' itri_riemann/ <新機器>:/path/itri_riemann/
   ```
   目前 `.git/objects/pack/._pack-*.idx` 已經會讓 git 報錯（`index file ... is too small`），請在新機器刪除：
   ```bash
   find .git -name '._*' -delete
   ```
2. **兩個 Python 環境分開：**
   - Isaac Sim 自帶 Python（3.10/3.11），負責**場景、模擬、擷取、輸出 npz**，只依賴 numpy。
   - pixi 環境（Python 3.8、torch 2.0.1 cu118、DGL、PyG），負責**訓練和推論**。
   - 不要把 DGL、PyG 裝進 Isaac Sim 的 Python，反之亦然。
3. 訓練環境：`pixi install`，再執行 `pixi run check`，確認 GPU 看得到。架構不是 linux-64 的話，依 C1 另外處理。
4. 依 C2 把 config 的 `device` 改成正確的 GPU 編號。
5. **驗收：** `pixi run check` 列出 GPU；Isaac Sim 能開啟 Franka 範例場景。

---

## 6. Phase 1：場景建置（Isaac Sim）

### 6.1 座標與機器人

- 世界座標 = 機器人 base 座標，**桌面表面在 z=0**，也就是 base 和桌面同高，這和 eval 檔一致。
- Franka Panda 放在原點，工作區是 x∈[0.2, 0.8]、y∈[-0.3, 0.3]。
- TCP 與 EE 慣例依 C10。

### 6.2 物體

| 物體 | 規格 | 注意事項 |
|---|---|---|
| 桌面 | 比工作區大即可，材質不重要（後處理會換掉） | 加 semantic label `table` |
| Rack（自建） | 立柱高 0.346 m，底座約 0.18 m，2–3 支約 45° 向上的掛勾，掛勾 z 介於 0.14 和 0.34 m，直徑 ≥ 1 cm | 每支掛勾加一個 Xform 當 `T_peg`；label `rack`；設為 kinematic 或固定在桌上 |
| Mug | 高 0.108 m，杯身直徑約 8–9 cm，有把手 | **碰撞形狀要保留把手的洞**：用 SDF mesh collision 或 convex decomposition。預設的 convex hull 會把洞填滿，杯子**掛不上去**。label `mug`；在模型上定義把手框架 `T_H` |
| 干擾物（DO 用） | 2–4 個 YCB 之類的小物件 | label `distractor` |

**顏色限制：** 物體 RGB 和 `[0.1176, 0.4392, 0.4078]` 的距離要大於 0.1，否則 `downsample_table` 會把它當成桌面。建議沿用範例色：mug 用粉色，rack 用木色。

### 6.3 相機

- 4 台 RGB-D（和實拍一樣；論文模擬用 6 台），距離桌面中心約 1 m，仰角 35–50°，間隔約 90°，要能 360° 涵蓋。
- Replicator annotator：`rgb`、`distance_to_image_plane`（或直接用 `pointcloud`）、`semantic_segmentation`。
- **要確認的事：** `pointcloud` annotator 輸出的是世界座標還是相機座標，請依你的 Isaac Sim 版本查文件，並用已知位置的物體實測。
- 只用於模擬時不加雜訊。依 C8 決定是否加。

### 6.4 擺放隨機化

| 設定 | Mug（pick 的起始位置） | Rack（place 目標） |
|---|---|---|
| T（training） | 左下象限，只繞 z 軸 ±90° | 右下象限，只繞 z 軸 ±90° |
| NP（newpose） | 任意象限，pick 時可以任意 SO(3)；place 時要能夾 | 任意象限，任意 yaw |
| NI（instance） | 換一個杯子 | 同 T |
| DO（distracting） | 同 T，另加干擾物 | 同 T |
| ALL | NI + NP + DO | NP |

- 兩物體的距離保持在約 0.19–0.31 m。
- 不能互相碰撞，也要在 Franka 可達範圍內。

**驗收：** 單幀點雲處理完（§7）後，用 `validate_npz.py` 檢查；尺寸要和 §2.3 一致；用 Open3D 目視確認。

---

## 7. Phase 2：點雲擷取與後處理

### 7.1 每條軌跡只拍一次，再用 EE 位姿轉換

這是原始資料的做法（§2.2）。

**Pick：**
- 場景是靜態的。在軌跡起點拍一次，得到世界座標點雲 `P_w`。
- 每一幀：`xyz_k = inv(T_ee_k) · P_w`。

**Place：**
- 夾起、舉高後，在運輸起點拍一次。
- 把點分成兩部分：
  - 靜態部分（桌面和 rack）：每幀用 `inv(T_ee_k)` 轉換。
  - 手上的杯子：會跟著 EE 移動，所以在 EE 座標中固定，每幀都用 `inv(T_ee_0) · P_mug_w`。
- 想要更擬真（每幀遮擋不同）時，可以改成每幀重新擷取。但桌面點仍要用同一組、同樣順序，否則驗證器的一致性檢查會被跳過。

### 7.2 處理步驟（世界座標）

1. 合併 4 台相機的點，用 semantic segmentation 去掉**機器人的所有 link**。Place 時**保留手上的杯子**。
2. 裁切到 x∈[0.2, 0.8]、y∈[-0.3, 0.3]、z∈[-0.03, 0.6]（z 上限依 C11）。
3. 刪除真實桌面的點，加入合成桌面：
   ```python
   rng = np.random.default_rng(seed)
   table = np.stack([rng.uniform(0.2, 0.8, 6144),
                     rng.uniform(-0.3, 0.3, 6144),
                     rng.uniform(-0.01, 0.01, 6144)], 1)
   table_rgb = np.tile([0.1176, 0.4392, 0.4078], (6144, 1))
   ```
4. 物體點降到 **2048 點**。建議用 FPS（最遠點取樣），確保細掛勾也取得到點；隨機取樣也可以。
5. 合併成 `[table(6144); objects(2048)] = 8192` 點，`rgb` 為 0–1 的 float64。**桌面點放在前面，每幀順序都一樣。**

### 7.3 轉換到 EE 座標系並寫入 npz

```python
def to_ee(P_w, T_ee):                      # P_w: [N,3], T_ee: [4,4] world pose of EE/TCP
    R, t = T_ee[:3, :3], T_ee[:3, 3]
    return (P_w - t) @ R                    # = R^T (p - t)

def label_in_ee(T_label_w, T_ee):
    T = np.linalg.inv(T_ee) @ T_label_w
    return T[:3, 3], T[:3, :3].T.reshape(9) # seg_center, axes（R 的三個軸依序攤平）

# 驗證慣例：R == axes.reshape(3,3).T
```

輸出 `data/mug/place/demo.npz`：

```python
np.savez("data/mug/place/demo.npz",
         xyz=xyz,               # [10, 20, 8192, 3] float64
         rgb=rgb,               # [10, 20, 8192, 3] float64, 0–1
         seg_center=seg_center, # [10, 20, 3]   標籤 B
         axes=axes,             # [10, 20, 9]   標籤 B
         ee_target_pos=...,     # [10, 20, 3]   標籤 A（選用）
         ee_target_axes=...,    # [10, 20, 9]   標籤 A（選用）
         grasp_in_ee=G)         # [10, 4, 4]    （選用）
```

另外保留一份 **raw 資料**（世界座標點雲、所有 `T_ee_k`、`T_peg`、`T_mug`、`G`、隨機 seed），存到 `data/mug/place/raw/`。之後要換標籤定義或重新取樣時，就不必重跑模擬。

---

## 8. Phase 3（建議）：先在 Isaac Sim 重現 pick

1. 用 §6、§7 的流程產生 `data/mug_sim/pick/demo.npz`，共 10 條軌跡。
   - 標籤：杯口邊緣的抓取 TCP 位姿，夾爪 z 軸朝下。
   - 軌跡：從 pre-grasp 往下接近，取 20 幀。
2. 新增 `config/mug_sim/pick/config.json`（從 `config/mug/pick` 複製），執行 `pixi run train mug_sim pick`。
3. **驗收：**
   - 驗證器全部 OK。
   - 和原始 mug pick 統計比較，抓取高度約 0.085 m、`|t|` 要遞減。
   - 訓練後測試集的 pos loss、ori loss 收斂到和原始資料相近的量級。
   - 原始資料要先排除第 7 條軌跡再比較。
4. 這一步也會產出 place 需要的「夾取」流程和 `G`。

---

## 9. Phase 4：Place 示範收集（10 條成功軌跡）

每條軌跡的流程：

1. 依 §6.4 的 T 設定隨機擺放 mug 和 rack，記錄 seed。
2. **夾取：** 規劃到杯口的抓取位姿，閉合夾爪。為了穩定，夾到後可以用 **fixed joint 把杯子固定在夾爪上**；這裡只需要觀測和標籤，不需要真實摩擦。記錄 `G = inv(T_ee) @ T_H`。
3. 舉高到運輸起點，例如往上 0.15 m。
4. **計算目標：** `T_ee_target = T_peg @ inv(G)`。pre-place 位姿取 `T_ee_target` 沿掛勾方向往後退約 0.10 m。
5. **規劃運輸軌跡：** 從運輸起點到 pre-place，平均取 20 幀的 `T_ee_k`。如果軌跡太短，前幾幀可以是靜止的，原始資料前幾幀也幾乎不動。
6. 在運輸起點擷取點雲（§7.1），計算 20 幀的 `xyz`、`seg_center`、`axes`。
7. **成功判定：**
   - 執行 pre-place → `T_ee_target` → 解除 fixed joint、張開夾爪 → 夾爪後退。
   - 模擬約 2 秒後，杯子仍掛在掛勾上，就算成功。
   - 判定方式：把手框架和 `T_peg` 的距離 < 2 cm，且杯子沒有掉到桌上。
   - 失敗的軌跡丟掉，換 seed 重來。
8. 收滿 10 條成功軌跡，輸出 `demo.npz` 加 raw 資料。

**驗收（`validate_npz.py data/mug/place/demo.npz`）：**
- 0 FAIL。
- 每條軌跡的 `|t|` 遞減。
- **標籤離最近點 ≤ 2 cm**。出現 WARN 時，要增加掛勾附近的點，或調整 `T_peg` 原點。
- EE 座標一致性 OK。
- 用 Open3D 看 1–2 幀，把 `seg_center` 和 `axes` 畫成座標軸，確認落在目標掛勾上。

---

## 10. Phase 5：評估集

- 對 T、NP、NI、DO、ALL 五種設定，各產生：
  - `data/mug/place/{training,newpose,instance,distracting,all}.npz`：單一場景，`xyz`/`rgb` 形狀 `[8192,3]`，**世界座標**。這和現有檔案以及 `infer.py` 相容。
  - **加入真值** `seg_center [3]`、`axes [9]`（世界座標，標籤 B），另外選擇性存 `grasp_in_ee`。`infer.py` 只讀 `xyz/rgb`，不會受影響。
  - `data/mug/place/eval/{setting}/seed_XX.npz`：共 20 個 seed，格式相同。
- Place 的評估場景是「夾著杯子、在運輸起點」的那一刻：手上有杯子，並去掉機器人的點。
- **驗收：** 每個檔案都通過驗證器，GT 離最近點 ≤ 2 cm。

---

## 11. Phase 6：專案程式修改與訓練

### 11.1 新增 `config/mug/place/config.json`

從 pick 複製，修改的地方以註解標示（實際 JSON 不能有註解，請刪掉）：

```jsonc
{
  "seg":  { ... 同 pick ..., "device": "cuda:0" },          // 依 C2 修改
  "mani": { ... 同 pick ...,
            "device": "cuda:0",
            "exp_name": "mug",
            "distance_threshold": 0.2,                      // 論文模擬 r1=0.2（pick 實拍用 0.16）
            "feature_point_radius": 0.02 }
}
```

訓練腳本的 `--pick_or_place` 已經接受 `place`：

```bash
pixi run train mug place      # train_seg → train_mani，輸出 experiments/mug/place/
```

### 11.2 修正 `scripts/testing/infer.py` 的已知問題（不修會跑不起來）

1. `output_pos, output_direction, _, _, _ = policy_mani(...)` 改成 `output_pos, output_direction = policy_mani(...)`，因為 `forward` 只回傳 2 個值。
2. 在 `np.savez` 前加上 `os.makedirs(wd, exist_ok=True)`。
3. 建立 `SE3ManiNet` 時補上 `feature_point_radius=cfg_mani.feature_point_radius`。

### 11.3 選做

- 把 `downsample_table` 的 `reference_rgb` 和 `total_points` 改成可在 config 設定，不再寫死。
- 新增一支 `scripts/testing/eval_place.py`：
  - 讀 `eval/{setting}/seed_*.npz`，計算預測值和 GT 的位置誤差（m）和旋轉誤差（°）。可以參考 `utils/loss_utils.double_geodesic_distance_between_poses`。
  - 用 `T_ee_target = T_peg_pred @ inv(G)` 換算 EE 目標，回到 Isaac Sim 做閉環測試（C13）。
- 訓練 pick 時排除原始 mug demo 的第 7 條軌跡。

### 11.4 驗收

- 訓練時 test pos loss 和 ori loss 都下降，並存出 `maninet.pth`。
- 每種設定都計算 geodesic 誤差。論文模擬的 RiEMann 在 Mug on Rack 的 T/NI/NP/DO/ALL 上約為 0.053–0.069，Plane on Shelf 約為 0.10–0.12（Table 2），可以拿來參考。
- 在 Isaac Sim 閉環測試 T 和 NP 設定的成功率。

---

## 12. 驗證工具：`scripts/data_gen/validate_npz.py`

```bash
python scripts/data_gen/validate_npz.py data/mug/place/*.npz      # Isaac Sim Python 或 pixi 都可以執行
```

| 檢查項目 | 等級 |
|---|---|
| key 和形狀（demo 為 `[traj,T,N,3]`，eval 為 `[N,3]`）、幀數是否 ≥ 20 | FAIL |
| NaN/Inf、rgb 不在 [0,1] | FAIL |
| `axes` 不是合法旋轉矩陣（正交且行列式為 1） | FAIL |
| 標籤退化（每幀都是 0 加單位矩陣，即原始 mug 第 7 條的狀況） | FAIL |
| 桌面點不是剛體運動，或標籤沒有跟著場景變換（EE 轉換寫錯） | FAIL |
| 沒有桌面色的點 | FAIL |
| 點數不是 8192、非桌面點 > 2500、標籤離最近點 > 2 cm、軌跡少於 5 條、eval 檔沒有 GT | WARN |

已用現有資料測試：`data/mug/pick/demo.npz` 正確抓到第 7 條 FAIL，其他 9 條的 EE 一致性都 OK；`data/plane/pick/demo.npz` 對每條軌跡回報「非桌面點 > 2500」的 WARN。

---

## 13. 風險與注意事項

1. **杯子掛不上去：** 碰撞形狀把把手的洞填滿了。用 SDF 或 convex decomposition（§6.2）。
2. **標籤離點太遠：** 掛勾太細或取樣太稀，姿態的 2 cm 取樣就會落空。用 FPS 取樣、掛勾直徑 ≥ 1 cm，並用驗證器檢查。
3. **物體被當成桌面：** 物體顏色接近 `[0.1176, 0.4392, 0.4078]`（§6.2）。
4. **非桌面點超過 2500：** `downsample_table` 會把桌面點全部丟掉，plane 資料就是這樣。物體點請固定為 2048。
5. **EE 座標慣例不一致：** TCP 偏移或軸向和 pick 不同，會讓 pick 和 place 無法串接（C10）。
6. **Rack 固定不動：** Place 會退化成記住一個常數位置，NP 設定會失敗（§6.4）。
7. **多支掛勾都能掛：** 標籤不唯一，要固定目標掛勾（C6）。
8. **Isaac Sim API 隨版本改名：** 4.x 和 5.x 的模組名稱不同，程式碼請以實際安裝的版本為準（C1）。
9. **軌跡長度：** 少於 20 幀時，`SE3Demo` 不會報錯，但形狀會不一致。請固定輸出 20 幀，驗證器會檢查。
10. **`._*` 檔：** 會污染 git 和 glob，例如 `*.npz` 也會匹配 `._*.npz`（§5）。

---

## 14. 執行順序總覽

```
Phase 0 環境遷移 ─► Phase 1 場景建置 ─► Phase 2 點雲流程 ─► Phase 3 pick 重現（建議）
                                                                  │
                                    Phase 6 訓練與評估 ◄─ Phase 5 評估集 ◄─ Phase 4 place 示範
```

| 階段 | 產出 | 驗收條件 |
|---|---|---|
| P0 | 可用的 Isaac Sim 和 pixi 環境 | `pixi run check` 通過 |
| P1 | 場景 USD 加隨機化腳本 | 尺寸符合 §2.3 |
| P2 | 擷取與轉換腳本 | 單幀通過驗證器 |
| P3 | `data/mug_sim/pick/demo.npz` 加模型 | 驗證器通過；loss 和原始資料相近 |
| P4 | `data/mug/place/demo.npz` 加 raw 資料 | 0 FAIL；標籤 ≤ 2 cm；10 條都是成功軌跡 |
| P5 | 5 種 eval 檔，各 20 個 seed，含 GT | 驗證器通過 |
| P6 | `experiments/mug/place/*.pth` 和評估結果 | geodesic 誤差和閉環成功率 |
---

## 15. 執行紀錄（2026-11-20）

### 15.1 §3 的實際決定

| # | 決定 |
|---|---|
| C1 | Isaac Sim 5.1（pip 安裝，`isaac-python`，Python 3.11），x86_64；API 用 `isaacsim.*` |
| C2 | 單張 RTX 4090 → 所有 config 改 `cuda:0` |
| C3 | 是，`data/mug_sim/pick` |
| C4 | B 寫入 `seg_center/axes`，A 另存 `ee_target_pos/axes`，並存 `grasp_in_ee` |
| C5 | 自建 primitive rack（`geometry.RACK`） |
| C6 | 2 支 45° 掛勾（z=0.25 朝 rack −x、z=0.16 朝 +x），目標為最高的一支；`T_peg` 原點在掛勾軸上、距立柱軸 `s=0.055` m |
| C7 | 程序化建模（無授權問題）：`mug_a` 高 0.108、粉色；NI 用 `mug_b` 高 0.150、黃色 |
| C8 | 不做 sim2real、不加雜訊 |
| C9 | 不用 RMPflow：Lula IK 取起點，之後用 DLS 沿路徑追蹤（保證連續），Lula 碰撞球檢查碰撞 |
| C10 | TCP = `panda_hand` +0.1034 m（Lula 的 `right_gripper` 是 +0.100 m 且繞 z 轉 180°，不採用） |
| C11 | 裁切 z ≤ 0.6 |
| C12 | 每種設定 20 個 seed，`{setting}.npz` = `seed_00` |
| C13 | 是，`closed_loop_place.py` |
| C14 | data 不進版控（`raw/` 與 `logs/` 已加入 `.gitignore`），由使用者決定是否用 LFS |

### 15.2 各 Phase 狀態

| 階段 | 狀態 | 結果 |
|---|---|---|
| P0 | 完成 | 刪除 120 個 `._*`（含 `.git` 內），`git fsck` 正常；pixi 環境可見 4090；config 改 `cuda:0` |
| P1 | 完成 | `scripts/data_gen/geometry.py`（numpy）、`isaac_scene.py`；6 台 RGB-D（論文模擬也是 6 台；4 台時目標掛勾常被手臂擋住） |
| P2 | 完成 | 反投影實測：rack 立柱中心誤差 < 0.3 mm；渲染色校正後 mug (0.78,0.65,0.61)、rack (0.63,0.54,0.38) |
| P3 | 完成（未訓練） | `data/mug_sim/pick`：驗證器 106 檔 0 FAIL 0 WARN；抓取高度 0.084 m（原始 0.084）、`|t|` 0.33→0.17 m（原始 0.34→0.15） |
| P4 | 完成 | `data/mug/place/demo.npz`：10 條軌跡、0 FAIL 0 WARN、標籤離最近點 0.6 cm、EE 一致性 OK |
| P5 | 完成 | 5 設定 × 20 seed，全部通過驗證器；GT 閉環 100/100 掛住 |
| P6 | 程式完成；訓練改在其他系統進行 | config、`infer.py` 修正、`eval_place.py`、`closed_loop_place.py` 已就緒；兩組 `demo.npz` 已用 `SE3Demo` + config 的 aug 在 CPU 上載入確認（各 200 筆、去桌後 2500 點） |

### 15.3 與原計劃不同之處

1. **擷取同步**：`world.step(render=True)` 的 annotator 資料落後約 4 幀，會拿到前一個場景的點雲。改用 `rep.orchestrator.step`，並在 `gen_mug_sim.check_capture` 檢查點雲是否貼合已知幾何，不符就中止。
2. **手臂只做運動學**（C9）：沒有用關節驅動執行軌跡，也沒有模擬夾取摩擦。掛杯測試用另一個不可見的動態杯子，直接放在釋放位姿上模擬 3 秒。
3. **成功判定**：計劃寫「把手框架離 `T_peg` < 2 cm」，實作改成「仍掛在目標掛勾上且已靜止」。杯子會沿 45° 掛勾滑到立柱，位移中位數 1.0 cm、最大 2.4 cm，110 次中有 64% 小於 2 cm。
4. **運輸路徑**：用二次 Bézier 曲線，最後一段沿掛勾軸進入；直線路徑常讓杯子擦過掛勾。
5. **Pick NP 只隨機 yaw**：沒有做任意 SO(3)（例如杯子側躺）。
6. **Pick 閉環未做**：只檢查 IK 和碰撞，沒有實際夾取。

### 15.4 待辦（訓練在其他系統）

訓練機需要的檔案（`raw/` 是除錯用，不需要）：

```
config/mug/place/  config/mug_sim/pick/
data/mug/place/{demo,training,newpose,instance,distracting,all}.npz  data/mug/place/eval/
data/mug_sim/pick/{demo,training,newpose,instance,distracting,all}.npz  data/mug_sim/pick/eval/
scripts/testing/infer.py  scripts/testing/eval_place.py   （加上原本的 networks/、utils/、scripts/training/）
```

```bash
# 訓練機（config 的 device 是 cuda:0，視情況修改）
python scripts/training/train_seg.py  --exp_name mug --pick_or_place place
python scripts/training/train_mani.py --exp_name mug --pick_or_place place
python scripts/testing/eval_place.py -exp_name mug -pick_or_place place   # -> experiments/mug/place/eval_predictions.npz
# 把 eval_predictions.npz 帶回有 Isaac Sim 的這台機器
isaac-python scripts/data_gen/closed_loop_place.py --exp mug              # -> experiments/mug/place/closed_loop.json
```

## 16. Plane on Shelf place 資料（2026-11-25 追加）

- 指令：`isaac-python scripts/data_gen/gen_plane_sim.py --out data/plane/place`。流程與 mug place 相同，`PlaneShelfScene` 繼承 `MugRackScene`。
- **飛機**（`geometry.PLANES`）：用程式建模，有機身膠囊、主翼、水平尾翼、垂直尾翼。尺寸和顏色照 `data/plane/pick` 實測。
  - T/NP/DO 用 `plane_a`：藍色，長 0.34 m、翼展 0.32 m、高 8.4 cm。
  - NI/ALL 用 `plane_b`：綠色，翼展 0.26 m。
  - 渲染色經 albedo 迭代校正：`plane_a` 為 (0.24,0.43,0.66)，目標 (0.18,0.43,0.65)，紅色通道受環境光限制降不下去；`plane_b` 為 (0.33,0.67,0.47)。
- **夾取**：夾在機翼後緣再往後 3 cm 的機身上，由上往下夾，手指橫向夾住機身，沿機身方向有 ±8 mm 隨機。
- **架子**（`geometry.SHELF`）：
  - 箱子 0.20×0.40×0.10 m，頂面有兩根支撐條，各長 6 cm、高 3.5 cm、內側間距 7 cm。機翼架在支撐條上，機身落在兩條中間。
  - −y 側有背板，用來消除 yaw 180° 的對稱。
- **標籤**：
  - B = `T_place`：箱頂上兩支撐條之間的點，x 為機頭方向，z 朝上。這裡是看得到的表面，因為 ManiNet 只從預測位置 2 cm 內的點彙整方向特徵。
  - A = `T_place @ inv(G)`。
  - 釋放時機翼在支撐條上方 3 mm。
- **成功判定**（`rest_test`）：放開 2 秒後，誤差在 1.2 cm / 8° 內並已靜止。敏感度測試顯示物理上相當寬容：yaw 25°、沿機身偏 3 cm、反向都還能停住，只有偏 6 cm 會傾倒。所以實際由容許誤差決定成功與否。
- **結果**：
  - 驗證器 106 檔 0 FAIL 0 WARN。
  - 110 次放置全部成功：位置誤差 0.3 cm，就是釋放高度；角度誤差 0°。
  - 手上飛機約 521 點固定在 EE 座標系。網路輸入約 82% 是物件點。
  - 嘗試次數：T 21、NP 45、NI 22、DO 24、ALL 27，各取 20 個。
- **Config**：`config/plane/place/config.json`，同 mug place，`distance_threshold` 0.2。
- **尚未做**：plane 的閉環評估腳本。`closed_loop_place.py` 目前只支援 mug。
