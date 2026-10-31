# DBSCAN — Gom cụm theo mật độ

## Purpose

Phát hiện các cụm dữ liệu có **hình dạng hình học bất kỳ** (hình trăng khuyết, hình sợi uốn lượn, vòng tròn đồng tâm lồng nhau) và **tự động nhận diện, cô lập các điểm nhiễu (Noise / Outliers)** mà hoàn toàn không cần chỉ định trước số lượng cụm $`K`$. Đây là thuật toán tiêu chuẩn và là đại diện xuất sắc nhất của trường phái Gom cụm dựa trên mật độ (Density-based Clustering). [1][2]

**Ví dụ dẫn nhập:** Trong phân tích dữ liệu định vị GPS để phát hiện các điểm đen tai nạn hoặc tụ điểm ùn tắc giao thông trong đô thị:
* K-Means sẽ chia bản đồ thành các hình tròn cố định, vô tình cắt đôi một tuyến đường cao tốc uốn lượn dài 20 km và ép các phương tiện chạy rải rác ở ngoại ô vào các cụm nội đô.
* DBSCAN tự động "lan truyền" dọc theo tuyến đường có mật độ xe đông đúc để nhận diện nguyên vẹn toàn bộ cung đường kẹt xe có hình dạng bất kỳ, đồng thời tự động đánh dấu các xe đơn lẻ ở vùng ngoại vi là **Nhiễu (Noise)**.

Ký hiệu: tập dữ liệu $`X = \lbrace x_1, \dots, x_n \rbrace \subset \mathbb{R}^p`$. Hai siêu tham số hình học là bán kính lân cận $`\varepsilon > 0`$ và số láng giềng tối thiểu $`\text{MinPts} \in \mathbb{N}^{+}`$. Mô hình tìm phân hoạch các cụm mật độ và tập hợp các điểm nhiễu $`\text{Noise} \subset X`$. [1]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần kỹ thuật đồ thị K-Distance bên dưới giải thích cách chọn siêu tham số; có thể quay lại sau khi hiểu quy tắc phân loại 3 loại điểm.

### Bản chất Hình học: Cụm như một Đa tạp Mật độ Liên tục

DBSCAN định nghĩa: **Một cụm là một vùng không gian liên tục có mật độ điểm cao, được bao quanh và ngăn cách với các cụm khác bởi các vùng không gian có mật độ điểm thấp (vùng thưa thớt)**.

```text
       CỤM HÌNH DẠNG BẤT KỲ                    CỤM ĐỒNG TÂM LỒNG NHAU
      (Non-convex / Curved)                      (Concentric Rings)
    . . . . . . . . .                          * * * * * * * * *
  .                   .                      *                   *
.       CỤM 1           .                  *     . . . . . . .     *
  .                   .                    *   .               .   *
    . . . . . . . . .                      *   .     CỤM 1     .   *
              ▲                            *   .               .   *
      Vùng thưa (Noise)                    *     . . . . . . .     *
              ▼                            *     CỤM 2             *
          * * * * * * * *                    *                   *
        *                 *                    * * * * * * * * *
      *      CỤM 2          *
        *                 *             DBSCAN tách hoàn hảo 2 vòng tròn!
          * * * * * * * *               (K-Means cắt ngang như nan hoa)
```

Khác với K-Means (vốn chia không gian bằng các siêu phẳng tuyến tính Voronoi), ranh giới của cụm trong DBSCAN là **vỏ bọc đa diện phi tuyến linh hoạt** uốn lượn theo hình thù tự nhiên của dữ liệu.

---

## Input

- Ma trận đặc trưng số $`X \in \mathbb{R}^{n \times p}`$ đã được chuẩn hóa qua `StandardScaler` hoặc `RobustScaler`.
- Bán kính vùng lân cận `eps` ($`\varepsilon \in \mathbb{R}^{+}`$): Khoảng cách tối đa để hai điểm được coi là láng giềng của nhau.
- Số lượng mẫu tối thiểu `min_samples` ($`\text{MinPts} \in \mathbb{N}^{+}`$): Số lượng điểm tối thiểu bắt buộc phải có trong vùng bán kính $`\varepsilon`$ để hình thành một vùng đậm đặc.
- Hàm khoảng cách `metric`: Mặc định là `euclidean`, có thể tùy biến thành `manhattan`, `cosine` hoặc `haversine` (cho tọa độ kinh độ/vĩ độ GPS).

### Tác động của Hai Siêu tham số $`\varepsilon`$ và $`\text{MinPts}`$
* Nếu chọn $`\varepsilon`$ quá nhỏ: Hầu hết dữ liệu không đủ điều kiện tạo cụm và bị coi là Nhiễu (Noise); các cụm lớn bị xé nát thành nhiều cụm vụn vặt.
* Nếu chọn $`\varepsilon`$ quá lớn: Các cụm tách biệt bị "dính chùm" vào nhau thành một siêu cụm khổng lồ bao trùm cả tập dữ liệu.
* Nếu chọn $`\text{MinPts}`$ quá nhỏ (ví dụ $`\text{MinPts} \le 2`$): Thuật toán thoái hóa thành Single Linkage của Hierarchical clustering và rất dễ bị hiện tượng dây xích (Chaining effect).
* Nếu chọn $`\text{MinPts}`$ quá lớn: Chỉ các vùng cực kỳ đậm đặc mới tạo thành cụm, phần lớn dữ liệu bị coi là ngoại lai.

---

## Output

1. **Vector nhãn phân cụm (Cluster Labels):** $`\hat{y} = (\hat{y}_1, \dots, \hat{y}_n) \in \lbrace -1, 0, 1, \dots, C-1 \rbrace^n`$:
   * Các giá trị $`\ge 0`$: Chỉ số định danh cụm mà quan sát được gán vào.
   * Giá trị $`-1`$: Đánh dấu quan sát đó là **Điểm nhiễu (Noise / Outlier)**.
2. **Chỉ số các điểm lõi (Core Sample Indices):** Mảng chỉ số các dòng trong $`X`$ thỏa mãn điều kiện làm điểm lõi: `core_sample_indices_`.
3. **Số lượng cụm tự động phát hiện được ($`C`$):** Thuật toán tự xác định số lượng cụm dựa trên cấu trúc mật độ thực tế.

---

## Evaluation metrics

### 1. Tỷ lệ Điểm Nhiễu (Noise Ratio)
Tỷ lệ phần trăm các điểm bị đánh dấu là ngoại lai:

$$
\text{Noise Ratio} = \frac{N_{\text{noise}}}{n} = \frac{\sum_{i=1}^n \mathbb{I}(\hat{y}_i = -1)}{n}
$$

* Nếu tỷ lệ này $> 30\%$, chứng tỏ tham số $`\varepsilon`$ đang quá nhỏ hoặc $`\text{MinPts}`$ đang quá lớn so với mật độ thực tế của dữ liệu.

### 2. Silhouette Score Hiệu chỉnh (Filtered Silhouette)
Khi tính điểm Silhouette cho DBSCAN, bắt buộc phải **loại bỏ các điểm nhiễu nhãn $`-1`$** ra khỏi phép tính (vì điểm nhiễu không thuộc về bất kỳ cụm nào và sẽ kéo điểm Silhouette xuống mức âm rất thấp):

$$
S_{\text{DBSCAN}} = \text{Silhouette}(X_{\hat{y} \ne -1}, \hat{y}_{\hat{y} \ne -1})
$$

---

## How it works

DBSCAN vận hành qua việc phân loại 3 loại điểm và lan truyền mở rộng cụm theo cơ chế BFS:

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             BA LOẠI ĐIỂM TRONG DBSCAN                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Core Point (Điểm lõi):    | Có ít nhất MinPts điểm nằm trong bán kính eps     │
│ 2. Border Point (Điểm biên): | Không đủ MinPts nhưng nằm trong bán kính của Core │
│ 3. Noise Point (Điểm nhiễu): | Không phải Core và không chạm bất kỳ Core nào     │
+----------------------------------------------------------------------------------+
```

```text
THUẬT TOÁN DBSCAN:
1. Đánh dấu tất cả các điểm trong X là CHƯA_THĂM (Unvisited).
2. Duyệt qua từng điểm p trong X:
   a. Nếu p đã thăm -> Bỏ qua.
   b. Đánh dấu p là ĐÃ_THĂM.
   c. Tìm tập láng giềng N_eps(p).
   d. Nếu |N_eps(p)| < MinPts:
      - Đánh dấu tạm thời p là NOISE (nhãn -1).
   e. Ngược lại (|N_eps(p)| >= MinPts):
      - p là Core Point -> Khởi tạo một CỤM MỚI C.
      - Gán p vào cụm C.
      - Cho toàn bộ N_eps(p) vào Hàng đợi mở rộng (Queue).
      - DUYỆT HÀNG ĐỢI:
        * Lấy điểm q ra khỏi hàng đợi.
        * Nếu q chưa thăm: Đánh dấu đã thăm, tìm N_eps(q).
        * Nếu |N_eps(q)| >= MinPts: Thêm toàn bộ N_eps(q) vào hàng đợi.
        * Nếu q chưa thuộc cụm nào: Gán q vào cụm C (nếu trước đó là Noise -> chuyển thành Border Point).
3. KẾT THÚC khi mọi điểm đều đã được xử lý.
```

### Độ Phức tạp Thời gian và Tăng tốc Không gian
* **Tính toán ngây thơ (Brute-force):** Với mỗi điểm phải quét qua $`n-1`$ điểm còn lại để tìm láng giềng $`\to \mathcal{O}(n^2)`$.
* **Tăng tốc với Cấu trúc Dữ liệu Không gian (Spatial Index):**
  * Với dữ liệu số chiều nhỏ ($`p < 10`$): Sử dụng cây **KD-Tree** giúp truy vấn láng giềng giảm xuống mức $`\mathcal{O}(\log n)`$, tổng độ phức tạp thuật toán là $`\mathcal{O}(n \log n)`$.
  * Với dữ liệu số chiều trung bình ($`p < 20`$): Sử dụng **Ball-Tree**.
  * Khi số chiều quá cao ($`p > 50`$), các cấu trúc cây bị suy thoái do Lời nguyền số chiều, độ phức tạp quay trở về $`\mathcal{O}(n^2)`$.

---

## Data used

Bảng đối chiếu đặc điểm thực nghiệm giữa dữ liệu bài tập và dữ liệu thực tế khi chạy DBSCAN:

| Khía cạnh | Dữ liệu Bài tập / Slide (`make_moons`) | Dữ liệu Thực tế (Production / IoT) |
|---|---|---|
| **Mật độ cụm** | Mật độ phân bố đồng đều hoàn hảo trên các cụm. | Các cụm có mật độ lệch nhau rất lớn (vùng trung tâm dày, vùng rìa thưa thớt). |
| **Lựa chọn Epsilon ($`\varepsilon`$)**| Điểm gập trên đồ thị K-Distance rất rõ ràng. | Đường cong K-Distance trơn tru, rất khó tìm một ngưỡng $`\varepsilon`$ tối ưu duy nhất. |
| **Xử lý Điểm nhiễu** | Chỉ có vài điểm nhiễu cô lập rải rác. | Dữ liệu thực tế có thể chứa 20% - 40% là nhiễu, cần phân biệt giữa nhiễu thật và cụm thưa. |
| **Bộ nhớ & Thời gian** | Chạy xong trong vài phần trăm giây với $`n = 1.000`$. | Dữ liệu hàng triệu dòng GPS gây nghẽn bộ nhớ; phải dùng Mini-Batch hoặc HDBSCAN. |

---

## Design

### Kỹ thuật Dò tìm Epsilon Tối ưu qua Đồ thị K-Distance
Quy trình chuẩn mực để xác định tham số `eps` trước khi chạy DBSCAN:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. Chuẩn hóa dữ liệu bắt buộc
X_scaled = StandardScaler().fit_transform(X)

# 2. Quy tắc ngón tay cái: MinPts = 2 * p (số chiều)
p = X_scaled.shape[1]
min_samples = 2 * p

# 3. Tính khoảng cách tới láng giềng thứ (MinPts - 1)
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Sắp xếp khoảng cách tăng dần
sorted_distances = np.sort(distances[:, min_samples - 1])

# 4. Vẽ đồ thị K-Distance để tìm điểm gập (Knee/Elbow Point)
plt.figure(figsize=(8, 4))
plt.plot(sorted_distances)
plt.title("Đồ thị K-Distance Plot để chọn Epsilon")
plt.xlabel("Chỉ số mẫu dữ liệu (đã sắp xếp)")
plt.ylabel(f"Khoảng cách tới láng giềng thứ {min_samples}")
plt.grid(True)
plt.axhline(y=0.45, color='r', linestyle='--', label='Eps đề xuất = 0.45')
plt.legend()
plt.show()

# 5. Huấn luyện DBSCAN với siêu tham số đã xác định
dbscan = DBSCAN(eps=0.45, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = np.sum(labels == -1)
print(f"Số cụm phát hiện: {n_clusters_} | Số điểm nhiễu: {n_noise_} ({n_noise_/len(labels)*100:.1f}%)")
```

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **DBSCAN** | Gom cụm mật độ không gian | Thuật toán gom cụm dựa trên mật độ lân cận, tự động phát hiện nhiễu và cụm phi tuyến. |
| **Core Point** | Điểm lõi | Điểm có tối thiểu $`\text{MinPts}`$ mẫu dữ liệu nằm trong bán kính lân cận $`\varepsilon`$. |
| **Border Point** | Điểm biên | Điểm không đủ điều kiện làm điểm lõi nhưng nằm trong vùng lân cận của điểm lõi. |
| **Noise Point** | Điểm nhiễu | Điểm không thuộc bất kỳ vùng mật độ nào, được gán nhãn định danh là $`-1`$. |
| **Density-Reachable** | Khả đạt theo mật độ | Chuỗi các bước nhảy liên tiếp giữa các điểm lõi láng giềng để nối tới một điểm đích. |
| **K-Distance Plot** | Đồ thị khoảng cách k | Công cụ kỹ thuật vẽ khoảng cách láng giềng thứ $`k`$ để tìm điểm gập xác định bán kính $`\varepsilon`$. |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning (Trang 11-13: Density-based Clustering & DBSCAN Algorithm), TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD.
- [3] Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). *DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN*. ACM TODS.
- [4] [scikit-learn User Guide: DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan).

