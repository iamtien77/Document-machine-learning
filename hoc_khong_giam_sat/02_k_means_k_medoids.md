# Gom cụm dựa trên tâm: K-Means, K-Means++ và K-Medoids

## Purpose

Gom cụm dựa trên tâm (Centroid-based Clustering) là nhóm thuật toán phân hoạch (Partitioning methods) kinh điển và phổ biến nhất trong học máy. Ý tưởng trung tâm là: Mỗi cụm được đặc trưng bởi một điểm đại diện duy nhất gọi là **Tâm cụm (Centroid)** hoặc **Medoid**. Không gian dữ liệu sau đó được phân vùng thành các ô Voronoi (Voronoi Tessellation), nơi mỗi điểm dữ liệu thuộc về cụm có tâm gần nó nhất theo khoảng cách Euclid.

**Ví dụ dẫn nhập:** Một chuỗi siêu thị muốn mở 5 kho hàng phân phối trung tâm ($`K = 5`$) để tối ưu chi phí vận chuyển đến 1.000 cửa hàng tiện lợi trên toàn quốc. Thuật toán K-Means sẽ tìm ra tọa độ của 5 kho hàng sao cho tổng bình phương khoảng cách vận chuyển từ mỗi cửa hàng đến kho hàng gần nhất là nhỏ nhất có thể.

---

### Cơ sở Toán học của Thuật toán K-Means

Cho tập dữ liệu gồm $`n`$ quan sát $`X = \lbrace x_1, x_2, \dots, x_n \rbrace`$ với $`x_i \in \mathbb{R}^p`$. Mục tiêu là phân hoạch $X$ thành $`K`$ cụm không chồng lặp $`C = \lbrace C_1, C_2, \dots, C_K \rbrace`$.

#### Hàm Mục tiêu Tối ưu: Tổng Bình phương Sai số Nội cụm (Inertia / WCSS)
K-Means tìm kiếm các tâm cụm $`\mu_1, \dots, \mu_K`$ nhằm làm cực tiểu hóa hàm **Within-Cluster Sum of Squares (WCSS)**:

$$
J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2
$$

Bài toán tối ưu hàm $`J`$ này là bài toán NP-hard. Do đó, thuật toán **Lloyd (1982)** sử dụng phương pháp tối ưu xấp xỉ cục bộ lặp hai bước luân phiên:

```text
[Khởi tạo K tâm cụm mu_1, ..., mu_K]
               │
               ▼
┌──────────────────────────────────────────────┐
│  BƯỚC 1: GÁN CỤM (Assignment Step)           │
│  Gán mỗi điểm x_i vào tâm mu_k gần nhất      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  BƯỚC 2: CẬP NHẬT TÂM (Update Step)          │
│  Tính lại mu_k bằng trọng tâm trung bình mới │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
 [Chưa hội tụ]                     [Hội tụ]
 (Tâm cụm còn dịch chuyển)        (Dừng thuật toán)
```

1. **Bước 1 — Gán cụm (Assignment Step):** Giữ cố định các tâm $`\mu_k`$, gán mỗi điểm dữ liệu $`x_i`$ vào cụm có tâm gần nhất:

$$
c_i = \mathop{\mathrm{arg\,min}}_{k \in \lbrace 1, \dots, K \rbrace} \|x_i - \mu_k\|_2^2
$$

2. **Bước 2 — Cập nhật tâm (Update Step):** Giữ cố định việc phân cụm, cập nhật lại tọa độ của từng tâm $`\mu_k`$ bằng trung bình cộng số học của toàn bộ các điểm thuộc cụm đó:

$$
\mu_k = \frac{1}{\lvert C_k \rvert} \sum_{x_i \in C_k} x_i
$$

* **Chứng minh tính hội tụ:** Ở mỗi bước (gán cụm hoặc tính lại trung bình), giá trị của hàm mục tiêu $`J`$ luôn **giảm đơn điệu hoặc giữ nguyên**. Vì số lượng cách phân hoạch $`n`$ điểm thành $`K`$ cụm là hữu hạn, thuật toán chắc chắn sẽ dừng lại sau một số hữu hạn bước lặp (thường từ 10 đến 50 vòng lặp).

---

### Khởi tạo Thông minh: Đột phá của Thuật toán K-Means++

Khởi tạo ngẫu nhiên truyền thống (Random Initialization) có nhược điểm chí mạng: nếu chọn nhầm các tâm ban đầu nằm quá sát nhau, thuật toán rất dễ bị kẹt ở **nghiệm cực tiểu cục bộ rất xấu (Poor Local Minima)**.

Năm 2007, David Arthur và Sergei Vassilvitskii công bố thuật toán **K-Means++**, trở thành tiêu chuẩn mặc định trong thư viện Scikit-learn:
1. Chọn ngẫu nhiên tâm đầu tiên $`\mu_1`$ từ tập dữ liệu theo phân phối đều.
2. Với mỗi điểm $`x_i`$ còn lại, tính khoảng cách $`D(x_i)`$ tới tâm cụm đã chọn gần nó nhất:

$$
D(x_i) = \min_{j} \|x_i - \mu_j\|_2
$$

3. Chọn tâm kế tiếp $`\mu_k`$ từ các điểm dữ liệu với xác suất tỷ lệ thuận với bình phương khoảng cách $`D(x_i)^2`$:

$$
P(x_i) = \frac{D(x_i)^2}{\sum_{x' \in X} D(x')^2}
$$

4. Lặp lại bước 2 và 3 cho đến khi đủ $`K`$ tâm cụm.

* **Bảo đảm toán học:** K-Means++ đảm bảo kỳ vọng sai số của hàm mục tiêu luôn tiệm cận nghiệm tối ưu toàn cục theo cấp số $`\mathcal{O}(\log K)`$, giúp thuật toán hội tụ nhanh hơn gấp nhiều lần và triệt tiêu hầu hết các lần chạy thất bại.

---

### K-Medoids (PAM - Partitioning Around Medoids): Giải pháp Kháng Ngoại lai

#### Hạn chế của K-Means khi có Ngoại lai (Outliers)
Vì tâm cụm $`\mu_k`$ của K-Means là trung bình cộng số học (Mean), nó cực kỳ nhạy cảm với các giá trị ngoại lai xa. Chỉ cần một điểm dữ liệu dị biệt cũng có thể kéo toàn bộ tâm cụm lệch khỏi trung tâm tự nhiên của đám mây dữ liệu.

#### Cơ chế của K-Medoids
K-Medoids giải quyết triệt để vấn đề này bằng cách:
* **Tâm cụm bắt buộc phải là một điểm dữ liệu THỰC TẾ** (gọi là **Medoid**), không phải là trung bình ảo.
* Tối thiểu hóa tổng khoảng cách tuyệt đối bất kỳ ($`L_1`$, Cosine, Manhattan) thay vì bắt buộc dùng bình phương Euclid:

$$
J_{\text{medoid}} = \sum_{k=1}^K \sum_{x_i \in C_k} D(x_i, m_k)
$$

```text
+-----------------------+----------------------------------+------------------------------------+
| Tiêu chí so sánh      | K-Means                          | K-Medoids (PAM)                    |
+-----------------------+----------------------------------+------------------------------------+
| 1. Điểm đại diện tâm  | Tọa độ trung bình ảo (Mean)      | Điểm dữ liệu thực tế (Medoid)      |
| 2. Độ nhạy ngoại lai  | Cực kỳ nhạy cảm (Dễ bị kéo lệch) | Rất bền vững (Kháng outlier mạnh)  |
| 3. Độ đo khoảng cách  | Bắt buộc dùng Euclid (L2)        | Tùy chọn tự do (Manhattan, Cosine) |
| 4. Độ phức tạp tính   | O(n * K * p * iter) (Rất nhanh)  | O(K * (n - K)^2) (Chậm trên tập lớn)|
+-----------------------+----------------------------------+------------------------------------+
```

---

### Mini-Batch K-Means cho Dữ liệu Lớn (Big Data)

Khi tập dữ liệu lên đến hàng triệu dòng ($`n > 10^6`$), K-Means truyền thống chạy rất chậm vì mỗi vòng lặp phải quét toàn bộ dữ liệu.
* **Mini-Batch K-Means:** Tại mỗi bước lặp, thuật toán chỉ lấy mẫu ngẫu nhiên một batch nhỏ (ví dụ: $`b = 1024`$ mẫu) để cập nhật vị trí tâm cụm theo cơ chế trung bình trượt (Online convex combination).
* **Kết quả:** Tăng tốc độ huấn luyện lên từ 3 đến 10 lần, tiết kiệm bộ nhớ RAM nhưng chất lượng phân cụm chỉ suy giảm không đáng kể ($< 2\%$ so với K-Means chuẩn).

---

### Phương pháp Chọn Số Cụm Tối ưu ($`K`$)

Vì học không giám sát không có nhãn, việc chọn số lượng cụm $`K`$ là bài toán tối quan trọng.

#### 1. Phương pháp Điểm khuỷu tay (The Elbow Method)
* Vẽ đồ thị giá trị Inertia ($`J`$) theo các giá trị $`K`$ tăng dần (từ 1 đến 10).
* Khi $`K`$ tăng, Inertia luôn giảm. Điểm tối ưu nằm tại vị trí **khuỷu tay (Elbow point)** — nơi tốc độ giảm của hàm mất mát bắt đầu chậm lại rõ rệt (hiệu suất biên giảm dần).

#### 2. Phân tích Hệ số Silhouette (Silhouette Analysis)
* Với mỗi điểm $`x_i`$, tính $`a(x_i)`$ là khoảng cách trung bình đến các điểm khác trong cùng cụm, và $`b(x_i)`$ là khoảng cách trung bình đến các điểm thuộc cụm gần nhất tiếp theo.

$$
s(x_i) = \frac{b(x_i) - a(x_i)}{\max\lbrace a(x_i), b(x_i) \rbrace} \in [-1, 1]
$$

* Chọn $`K`$ sao cho điểm **Silhouette Score trung bình cao nhất** và biểu đồ độ rộng của các cụm phân bố đều nhau, không có cụm nào bị điểm âm.

---

### Các Trường hợp Thất bại Kinh điển của K-Means (Failure Modes)

K-Means ngầm đưa ra 3 giả định hình học rất khắt khe: **1) Cụm có dạng hình cầu đối xứng (Spherical); 2) Các cụm có kích thước phương sai tương đương nhau; 3) Mật độ các cụm đồng đều**.

```text
TRƯỜNG HỢP 1: CỤM HÌNH DẠNG BẤT KỲ      TRƯỜNG HỢP 2: CỤM ĐỒNG TÂM LỒNG NHAU
      (Hình trăng khuyết)                      (Concentric Rings)
    . . . . .                                    * * * * * *
  .           .                                *             *
    . . . . .                                 *   . . . . .   *
            * * * * *                         *  .         .  *
          *           *                       *   . . . . .   *
            * * * * *                          *             *
  K-Means cắt đôi theo đường thẳng!              * * * * * *
  (Thất bại hoàn toàn -> Dùng DBSCAN)         K-Means cắt dọc theo nan hoa!
```

---

### Mã nguồn Thực hành với Scikit-learn

```python
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Giả lập và tiền xử lý dữ liệu
X_scaled = StandardScaler().fit_transform(X)

# 2. Huấn luyện K-Means với thuật toán khởi tạo k-means++
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    n_init=10,        # Chạy 10 lần khởi tạo độc lập để lấy nghiệm tốt nhất
    max_iter=300,
    random_state=42
)
cluster_labels = kmeans.fit_predict(X_scaled)

# 3. Đánh giá chất lượng hình học
inertia_val = kmeans.inertia_
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"Inertia (WCSS): {inertia_val:.2f} | Silhouette Score: {sil_score:.4f}")

# 4. Trích xuất tâm cụm
centroids = kmeans.cluster_centers_
```
