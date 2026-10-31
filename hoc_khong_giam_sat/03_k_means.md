# K-Means và K-Means++

## Purpose

Phân hoạch tập dữ liệu thành $`K`$ cụm không chồng lặp dựa trên khoảng cách hình học tới các điểm đại diện tâm cụm (Centroids). Đây là thuật toán gom cụm kinh điển, phổ biến nhất và là **baseline gom cụm đầu tiên** cần thử nghiệm nhờ tốc độ tính toán cực nhanh, tính trực quan và khả năng mở rộng tốt trên tập dữ liệu lớn. [1][2]

**Ví dụ dẫn nhập:** Một công ty viễn thông muốn đặt 10 trạm thu phát sóng di động ($`K = 10`$) để phục vụ 50.000 thuê bao trên địa bàn một thành phố. Thuật toán K-Means giúp xác định tọa độ đặt 10 trạm phát sao cho tổng khoảng cách di chuyển tín hiệu từ mỗi thuê bao đến trạm gần nhất là ngắn nhất, tối thiểu hóa độ trễ kết nối.

Ký hiệu: với $`n`$ mẫu và $`p`$ feature số, $`X \in \mathbb{R}^{n \times p}`$. Số cụm $`K`$ được người dùng xác định trước. Mô hình ước lượng tọa độ của $`K`$ tâm cụm $`\mu_1, \dots, \mu_K \in \mathbb{R}^p`$ và gán nhãn mỗi điểm vào cụm có tâm gần nhất. Mục tiêu là tối thiểu hóa tổng bình phương khoảng cách sai số nội cụm. [1][4]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần chứng minh hội tụ giải tích bên dưới giải thích cơ sở toán học; có thể quay lại sau khi hiểu cách dùng thuật toán trên Scikit-learn.

### Bản chất Hình học: Biểu đồ Phân vùng Voronoi (Voronoi Tessellation)

Về mặt hình học không gian, khi $`K`$ tâm cụm $`\lbrace \mu_1, \dots, \mu_K \rbrace`$ được cố định, K-Means chia toàn bộ không gian liên tục $`\mathbb{R}^p`$ thành $`K`$ đa diện lồi gọi là **Các ô Voronoi (Voronoi Cells)**.

Mỗi ô Voronoi $`V(\mu_k)`$ là tập hợp tất cả các điểm trong không gian có khoảng cách tới tâm $`\mu_k`$ nhỏ hơn hoặc bằng khoảng cách tới bất kỳ tâm nào khác:

$$
V(\mu_k) = \left\lbrace x \in \mathbb{R}^p \mid \|x - \mu_k\|_2 \le \|x - \mu_j\|_2, \quad \forall j \ne k \right\rbrace
$$

* **Ý nghĩa hình học:** Ranh giới quyết định (Decision Boundary) giữa hai cụm bất kỳ luôn là một **siêu phẳng tuyến tính vuông góc (Hyperplane Bisector)** với đoạn thẳng nối hai tâm cụm đó.
* Do đó, K-Means **chỉ có khả năng phân tách các cụm lồi (Convex Clusters)**; nó bất lực trước các cụm có ranh giới uốn lượn phi tuyến hoặc lồng ghép vào nhau.

---

## Input

- Ma trận đặc trưng số $`X \in \mathbb{R}^{n \times p}`$ đã được chuẩn hóa thang đo qua `StandardScaler` hoặc `RobustScaler`.
- Số lượng cụm mong muốn $`K \in \mathbb{N}^+`$ ($`1 < K \ll n`$).
- Số lần khởi tạo độc lập `n_init` (mặc định là 10 trong Scikit-learn).
- Ngưỡng dung sai dừng hội tụ `tol` (thường là $`10^{-4}`$) và số vòng lặp tối đa `max_iter` (thường là 300).

### Giả định Hình học & Giới hạn của Không gian Đầu vào

K-Means ngầm đưa ra 3 giả định toán học rất khắt khe về cấu trúc dữ liệu:
1. **Giả định hình cầu đối xứng (Spherical / Isotropic assumption):** Các cụm phải có dạng hình cầu trong không gian đa chiều, nghĩa là phương sai theo mọi hướng tọa độ xấp xỉ bằng nhau.
2. **Kích thước và phương sai tương đương (Equal Variance):** Các cụm có bán kính và độ phân tán tương đương nhau. Nếu một cụm rất to và một cụm rất bé nằm cạnh nhau, K-Means sẽ cắt cụm to ra để chia bớt cho cụm bé.
3. **Mật độ đồng đều (Uniform Density):** Mật độ điểm bên trong các cụm tương đương nhau.

---

## Output

1. **Nhãn phân cụm (Cluster Labels):** Vector nhãn cứng $`\hat{y} = (\hat{y}_1, \dots, \hat{y}_n) \in \lbrace 0, 1, \dots, K-1 \rbrace^n`$.
2. **Tọa độ các tâm cụm (Centroid Coordinates):** Ma trận $`M = [\mu_1, \dots, \mu_K]^\top \in \mathbb{R}^{K \times p}`$.
3. **Giá trị quán tính (Inertia / WCSS):** Giá trị tổng bình phương khoảng cách đạt được tại điểm hội tụ.

### Ý nghĩa Nghiệp vụ của Tâm Cụm (Centroid as Prototype)
Tọa độ của tâm cụm $`\mu_k`$ đóng vai trò là **mẫu chân dung đại diện (Archetype / Prototype)** cho toàn bộ các cá thể trong cụm đó:
* Bằng cách phân tích tọa độ của $`\mu_k`$ trên từng trục đặc trưng, nhà phân tích có thể đọc ra đặc điểm hành vi của nhóm (ví dụ: $`\mu_1`$ có `Tuổi = 24`, `Chi tiêu = 8.5 triệu` $`\to`$ Cụm thanh niên chi tiêu mua sắm cao).

---

## Evaluation metrics

Chất lượng nghiệm phân cụm của K-Means được đo lường qua:

### 1. Quán tính nội cụm (Inertia / Within-Cluster Sum of Squares)
Là giá trị hàm mục tiêu được tối thiểu hóa trực tiếp:

$$
J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2
$$

* *Lưu ý:* Giá trị Inertia luôn giảm đơn điệu khi $`K`$ tăng (khi $`K = n`$, $`J = 0`$). Do đó không thể dùng Inertia đơn lẻ để chọn $`K`$.

### 2. Phương pháp Điểm khuỷu tay (The Elbow Method)
Vẽ đồ thị của $`J`$ theo các giá trị $`K \in [2, 10]`$. Tìm điểm gập "khuỷu tay" — nơi đường cong bắt đầu bẻ ngang và tốc độ giảm của $`J`$ chậm lại rõ rệt.

### 3. Hệ số Silhouette trung bình (Mean Silhouette Score)
Đo lường mức độ tự tin của phép gán nhãn:

$$
s(x_i) = \frac{b(x_i) - a(x_i)}{\max\lbrace a(x_i), b(x_i) \rbrace} \in [-1, 1]
$$

* Giá trị $`> 0.5`$: Cấu trúc phân cụm rất mạnh và rõ nét.
* Giá trị $`\approx 0`$: Các cụm bị chồng lấn ranh giới mạnh.
* Giá trị $`< 0`$: Điểm dữ liệu bị gán nhầm vào sai cụm.

---

## How it works

Thuật toán chuẩn của K-Means (thuật toán Lloyd, 1982) vận hành qua cơ chế tối ưu lặp hai bước (EM-style Coordinate Descent):

```text
[BƯỚC 0: Khởi tạo K tâm cụm bằng K-Means++]
                      │
                      ▼
┌──────────────────────────────────────────────┐
│  BƯỚC 1: GÁN CỤM (Assignment Step)           │
│  c_i = argmin_k ||x_i - mu_k||^2             │
└─────────────────────┬────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────┐
│  BƯỚC 2: CẬP NHẬT TÂM (Update Step)          │
│  mu_k = (1 / |C_k|) * sum_{x in C_k} x       │
└─────────────────────┬────────────────────────┘
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
||Delta mu|| >= tol            ||Delta mu|| < tol
(Tiếp tục lặp bước 1, 2)       (Hội tụ -> Dừng thuật toán)
```

1. **Khởi tạo K-Means++ (Arthur & Vassilvitskii, 2007):**
   * Tâm đầu tiên $`\mu_1`$ được chọn ngẫu nhiên đều từ dữ liệu.
   * Các tâm tiếp theo $`\mu_k`$ được chọn với xác suất tỷ lệ thuận với bình phương khoảng cách tới tâm đã chọn gần nhất:

$$
P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}
$$

   * Giúp các tâm phân tán đều khắp không gian, triệt tiêu nguy cơ rơi vào nghiệm cực tiểu cục bộ xấu.

2. **Bước 1 — Gán cụm (Assignment):** Cố định vị trí các tâm $`\mu_k`$, phân phối mỗi điểm $`x_i`$ vào cụm có tâm gần nhất theo khoảng cách Euclid.
3. **Bước 2 — Cập nhật tâm (Update):** Cố định phép gán cụm, tính lại vị trí tâm $`\mu_k`$ bằng trung bình cộng số học của toàn bộ các điểm thuộc cụm $`C_k`$.
4. **Kiểm tra hội tụ:** Lặp lại bước 1 và 2 cho đến khi tâm không còn dịch chuyển đáng kể ($`\|\Delta \mu\| < \text{tol}`$) hoặc chạm giới hạn `max_iter`.

### Chứng minh Tính Hội tụ Đơn điệu của K-Means
Ở Bước 1, với mỗi điểm $`x_i`$, việc gán vào tâm gần nhất chắc chắn làm $`\|x_i - \mu_{c_i}\|^2 \le \|x_i - \mu_{\text{cũ}}\|^2`$, do đó hàm $`J`$ giảm hoặc giữ nguyên.
Ở Bước 2, bài toán tìm tâm làm nhỏ nhất tổng bình phương sai số của một tập điểm:

$$
\frac{\partial}{\partial \mu_k} \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2 = -2 \sum_{x_i \in C_k} (x_i - \mu_k) = 0 \iff \mu_k = \frac{1}{\lvert C_k \rvert} \sum_{x_i \in C_k} x_i
$$

Đạo hàm bằng 0 tại chính giá trị trung bình số học. Do đó Bước 2 cũng làm hàm $`J`$ giảm đơn điệu. Vì số lượng cách phân hoạch là hữu hạn ($`K^n`$), thuật toán **chắc chắn luôn hội tụ**.

---

## Data used

Bảng đối chiếu đặc điểm phân cụm trên dữ liệu bài tập và dữ liệu thực tế:

| Khía cạnh thực nghiệm | Dữ liệu Bài tập / Slide (`make_blobs`) | Dữ liệu Triển khai Thực tế (Production) |
|---|---|---|
| **Hình dạng cụm** | Các đám mây hình cầu tròn xoe, đối xứng hoàn hảo. | Các cụm hình elip méo mó, hình dải dài hoặc uốn lượn phi tuyến. |
| **Khoảng cách giữa các cụm** | Tách rời nhau rõ rệt, khoảng cách ngoại cụm rất lớn. | Chồng lấn ranh giới dày đặc, không có khoảng trống ranh giới rõ ràng. |
| **Độ nhạy ngoại lai** | Không có ngoại lai; tâm cụm cố định ổn định. | Ngoại lai cực đoan kéo lệch tâm cụm hàng chục đơn vị đo. |
| **Xác định $`K`$** | Đồ thị Elbow gập một góc 90 độ dứt khoát tại $`K = 3`$ hoặc $`4`$. | Đồ thị Elbow thoai thoải hình vòng cung, rất khó chọn một điểm duy nhất. |

---

## Design

### Pipeline Huấn luyện Scikit-learn Chuẩn mực

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Xây dựng Pipeline chuẩn: Scale kháng ngoại lai -> K-Means++
kmeans_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('kmeans', KMeans(
        n_clusters=4,
        init='k-means++',
        n_init=10,        # 10 lần khởi tạo độc lập
        max_iter=300,
        tol=1e-4,
        random_state=42
    ))
])

# Huấn luyện và trích xuất nhãn
cluster_labels = kmeans_pipeline.fit_predict(X)

# Trích xuất mô hình và đánh giá
model = kmeans_pipeline.named_steps['kmeans']
X_scaled = kmeans_pipeline.named_steps['scaler'].transform(X)

print(f"Inertia (WCSS): {model.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
print(f"Số vòng lặp hội tụ: {model.n_iter_}")
```

### Các Trường hợp Thất bại Kinh điển (Failure Modes)
1. **Cụm có kích thước phương sai lệch nhau:** K-Means sẽ cắt đôi cụm lớn để gộp một nửa vào cụm nhỏ hơn bên cạnh nhằm cân bằng WCSS.
2. **Cụm phi tuyến tính (Non-convex clusters):** Ví dụ dữ liệu hình trăng khuyết (`make_moons`) hoặc hai vòng tròn đồng tâm (`make_circles`). K-Means sẽ cắt ngang qua các dải dữ liệu thay vì bám theo đường cong $`\to`$ Bắt buộc phải chuyển sang **DBSCAN** hoặc **Spectral Clustering**.
3. **Dữ liệu có nhiều Outliers:** Bắt buộc phải loại bỏ ngoại lai trước hoặc chuyển sang dùng **K-Medoids (PAM)**.

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **K-Means** | Gom cụm K-Means | Thuật toán phân hoạch lặp dựa trên việc tối thiểu hóa bình phương khoảng cách tới tâm cụm. |
| **K-Means++** | Khởi tạo K-Means++ | Thuật toán chọn tâm ban đầu phân tán theo xác suất bình phương khoảng cách để chống nghiệm xấu. |
| **Voronoi Cell** | Ô Voronoi | Vùng không gian lồi bao quanh một tâm cụm, nơi mọi điểm đều gần tâm đó nhất. |
| **Lloyd's Algorithm** | Thuật toán Lloyd | Quy trình lặp hai bước luân phiên: Gán nhãn cụm $`\to`$ Cập nhật tọa độ tâm. |
| **Inertia / WCSS** | Quán tính nội cụm | Tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm tương ứng. |
| **Elbow Method** | Phương pháp khuỷu tay | Kỹ thuật tìm số lượng cụm $`K`$ tối ưu tại điểm uốn gập của đường cong Inertia. |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning (Trang 7-10: Centroid-based Clustering & K-means Algorithm), TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Lloyd, S. (1982). *Least squares quantization in PCM*. IEEE Transactions on Information Theory.
- [3] Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*. SODA.
- [4] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (Chapter 14.3.6: K-means Clustering). Springer.
- [5] [scikit-learn User Guide: K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means).
