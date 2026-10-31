# Trong train: tối ưu và chọn mô hình

## Purpose

Trong quá trình huấn luyện mô hình học không giám sát, kỹ sư phải giải quyết đồng thời hai nhiệm vụ: **1) Tối ưu hóa hàm mục tiêu nội tại** (như cực tiểu hóa WCSS trong K-Means, cực đại hóa Log-Likelihood trong GMM, hoặc cực tiểu hóa Reconstruction Error trong PCA); và **2) Lựa chọn siêu tham số tối ưu** (như số cụm $`K`$, bán kính $`\varepsilon`$, số thành phần $`d`$) mà hoàn toàn không có nhãn kiểm chứng từ bên ngoài. Mục tiêu là tìm ra nghiệm hội tụ ổn định, có ý nghĩa thống kê và bền vững trước các nhiễu loạn ngẫu nhiên của dữ liệu. [1][2]

**Ví dụ dẫn nhập:** Khi chạy K-Means với 5 cụm ($`K = 5`$), mỗi lần chạy với một random seed khác nhau có thể cho ra các vị trí tâm cụm khác biệt và điểm Inertia chênh lệch nhau. Quá trình huấn luyện chuẩn mực đòi hỏi phải chạy nhiều lần khởi tạo độc lập (`n_init = 10`), so sánh giá trị hàm mất mát để chọn nghiệm tốt nhất, đồng thời quét dải $`K \in [2, 10]`$ để tìm ra cấu trúc tự nhiên nhất.

Ký hiệu: ma trận huấn luyện $`X \in \mathbb{R}^{n \times p}`$. Không gian tham số mô hình cần ước lượng là $`\Theta`$ (ví dụ: tọa độ các tâm cụm $`\mu = \lbrace \mu_1, \dots, \mu_K \rbrace`$, hoặc ma trận trọng số chiếu $`W \in \mathbb{R}^{p \times d}`$). Cần tìm $`\theta^{\ast} \in \Theta`$ làm cực trị hóa hàm mục tiêu $`\mathcal{J}(\theta; X)`$. [2]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần định lý bất khả thi Kleinberg bên dưới giải thích giới hạn lý thuyết; có thể quay lại sau khi hiểu quy trình quét siêu tham số.

### Định lý Bất khả thi của Kleinberg (Axiomatic Clustering Invariance - 2002)

Trong bài báo nổi tiếng *"An Impossibility Theorem for Clustering"* (NIPS 2002), Jon Kleinberg đã chứng minh rằng: **Không tồn tại bất kỳ một thuật toán gom cụm nào có thể thỏa mãn đồng thời cả 3 tiên đề trực giác sau đây**:

1. **Tính bất biến thang đo (Scale-Invariance):** Nếu ta nhân khoảng cách giữa mọi cặp điểm với một hằng số dương $`c > 0`$ (phóng to hoặc thu nhỏ đồng dạng không gian), kết quả phân cụm không được phép thay đổi.
2. **Tính phong phú (Richness):** Thuật toán phải có khả năng sinh ra mọi phân hoạch cụm khả dĩ trên tập dữ liệu nếu khoảng cách được điều chỉnh phù hợp.
3. **Tính nhất quán (Consistency):** Nếu ta thu hẹp khoảng cách giữa các điểm trong cùng một cụm và kéo giãn khoảng cách giữa các điểm khác cụm, kết quả phân cụm bắt buộc phải giữ nguyên, không được thay đổi.

* **Hệ quả thực tiễn:** Mọi thuật toán gom cụm (K-Means, DBSCAN, Hierarchical) đều là một sự đánh đổi triết học. Không có thuật toán nào là "hoàn hảo cho mọi dữ liệu". Người kỹ sư phải hiểu rõ thuật toán đang hy sinh tiên đề nào để áp dụng đúng ngữ cảnh.

---

## Input

- Ma trận đặc trưng đã qua tiền xử lý và chuẩn hóa: $`X_{\text{scaled}} \in \mathbb{R}^{n \times p}`$.
- Không gian siêu tham số cần tìm kiếm (Hyperparameter Grid):
  * Với K-Means / K-Medoids: dải số cụm $`K \in [2, 15]`$, phương pháp khởi tạo (`k-means++` vs `random`).
  * Với DBSCAN / OPTICS: dải bán kính $`\varepsilon \in [0.1, 2.0]`$, số láng giềng tối thiểu $`\text{MinPts} \in [3, 20]`$.
  * Với GMM: loại ma trận hiệp phương sai (`covariance_type`: `full`, `tied`, `diag`, `spherical`).
  * Với PCA: số thành phần chính $`d`$ hoặc ngưỡng phương sai giải thích tối thiểu ($`95\%`$).
- Tiêu chí dừng thuật toán: số vòng lặp tối đa (`max_iter`) và ngưỡng dung sai hội tụ (`tol`).

---

## Output

- Bộ tham số tối ưu $`\theta^{\ast}`$ của mô hình sau khi hội tụ:
  * Tọa độ các tâm cụm: $`\mu_1^{\ast}, \dots, \mu_K^{\ast} \in \mathbb{R}^p`$.
  * Hoặc ma trận hiệp phương sai từng cụm: $`\Sigma_1^{\ast}, \dots, \Sigma_K^{\ast} \in \mathbb{R}^{p \times p}`$.
  * Hoặc các vector riêng trực giao: $`w_1, \dots, w_d \in \mathbb{R}^p`$.
- Vector nhãn cụm được gán trên tập train: $`\hat{y} \in \lbrace 0, 1, \dots, K-1 \rbrace^n`$.
- Lịch sử hội tụ: đồ thị hàm mất mát theo từng vòng lặp, giá trị hàm mục tiêu tại điểm dừng.
- Báo cáo phân tích siêu tham số: đồ thị Elbow, biểu đồ hệ số Silhouette theo từng giá trị $`K`$.

---

## Evaluation metrics

Trong giai đoạn huấn luyện, thuật toán dựa vào các hàm mục tiêu toán học để cập nhật tham số và đo lường độ hội tụ:

### 1. Hàm Mất mát Nội cụm (Within-Cluster Sum of Squares - WCSS)
Được tối ưu trong K-Means bằng thuật toán Coordinate Descent (Lloyd):

$$
\mathcal{J}_{\text{KMeans}}(\mu) = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2
$$

* Thuật toán dừng khi mức giảm của $`\mathcal{J}`$ giữa hai vòng lặp nhỏ hơn ngưỡng dung sai $`\text{tol} \le 10^{-4}`$.

### 2. Hàm Khả năng Hậu nghiệm (Log-Likelihood)
Được tối ưu trong Gaussian Mixture Models bằng thuật toán EM (Expectation-Maximization):

$$
\ln p(X \mid \pi, \mu, \Sigma) = \sum_{i=1}^n \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)
$$

### 3. Sai số Tái thiết (Reconstruction Error)
Được tối ưu trong giảm chiều PCA thông qua phép phân rã trị riêng (SVD):

$$
\mathcal{J}_{\text{PCA}}(W) = \|X - X W W^\top\|_F^2 = \sum_{i=1}^n \|x_i - \hat{x}_i\|_2^2
$$

---

## How it works

Quy trình tối ưu hóa và lựa chọn mô hình chuẩn mực trong giai đoạn huấn luyện:

```text
[Dữ liệu X_scaled] ──> [1. Quét dải siêu tham số] ──> [2. Đa khởi tạo (n_init)]
                                                               │
[5. Chọn mô hình tối ưu] <── [4. Thẩm định độ ổn định] <── [3. Kiểm tra hội tụ]
```

1. **Đa khởi tạo độc lập (Multi-start Optimization):**
   * Vì các hàm mất mát phân cụm là hàm phi lồi (Non-convex), thuật toán rất dễ bị mắc kẹt tại các nghiệm cực tiểu cục bộ tồi (Local Minima).
   * Luôn thiết lập $`\text{n\_init} \ge 10`$ để chạy thuật toán từ 10 bộ khởi tạo ngẫu nhiên khác nhau và chọn mô hình đạt hàm mục tiêu tốt nhất.
2. **Kiểm tra độ hội tụ (Convergence Check):**
   * Theo dõi sự dịch chuyển của các tâm cụm: $`\Delta \mu = \sum_k \|\mu_k^{(t)} - \mu_k^{(t-1)}\|`$. Nếu $`\Delta \mu < \text{tol}`$, dừng thuật toán.
3. **Quét dải siêu tham số (Hyperparameter Search):**
   * Chạy mô hình trên dải giá trị $`K \in [2, 10]`$.
   * Tính toán song song hai thước đo: Điểm khuỷu tay (Inertia) và Điểm Silhouette trung bình.
4. **Kiểm tra Độ ổn định Cụm (Cluster Stability via Bootstrap):**
   * Một cấu trúc cụm chỉ thực sự có ý nghĩa nếu nó bền vững trước các biến động dữ liệu.
   * Rút mẫu ngẫu nhiên có hoàn lại (Bootstrap 80% dữ liệu) 50 lần, huấn luyện mô hình và đo chỉ số tương đồng Jaccard giữa các phân hoạch. Nếu độ ổn định thấp, chứng tỏ số cụm $`K`$ đó là do ngẫu nhiên áp đặt chứ không phải cấu trúc thực.

---

## Data used

Bảng so sánh hành vi huấn luyện trên dữ liệu mẫu đối chiếu với môi trường sản xuất thực tế:

| Tiêu chí huấn luyện | Bài tập / Toy Dataset | Dữ liệu Huấn luyện Thực tế (Production) |
|---|---|---|
| **Thời gian hội tụ** | Vài mili-giây, hội tụ sau 5 đến 15 vòng lặp. | Vài phút đến nhiều giờ; cần dùng Mini-Batch hoặc thuật toán song song. |
| **Nghiệm cục bộ (Local Minima)**| Không gian lồi đơn giản, đa khởi tạo đều về cùng một nghiệm. | Tồn tại hàng trăm nghiệm cục bộ; bắt buộc dùng `k-means++` và `n_init >= 20`. |
| **Lựa chọn siêu tham số** | Điểm Elbow và Silhouette thể hiện cực kỳ rõ ràng, dứt khoát. | Điểm Elbow mờ nhạt (đường cong thoai thoải); phải kết hợp nhiều tiêu chí nghiệp vụ. |
| **Bộ nhớ tính toán** | Dữ liệu nằm gọn trong RAM (`< 10 MB`). | Dữ liệu hàng gigabyte; ma trận khoảng cách $`\mathcal{O}(n^2)`$ làm tràn bộ nhớ (Out-Of-Memory). |

---

## Design

### Vòng lặp Tìm kiếm Siêu tham số & Đánh giá Độ ổn định
Đoạn mã chuẩn mực thực hiện quét tìm số cụm tối ưu và ghi nhận các chỉ số đánh giá nội tại:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Dải giá trị K cần khảo sát
k_range = range(2, 9)
results = []

for k in k_range:
    # Huấn luyện đa khởi tạo
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    # Đo lường các chỉ số
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    results.append({'k': k, 'inertia': inertia, 'silhouette': sil, 'davies_bouldin': db})
    print(f"K={k:2d} | Inertia: {inertia:8.2f} | Silhouette: {sil:6.4f} | DB Index: {db:6.4f}")

# Chọn K có Silhouette cao nhất và Davies-Bouldin nhỏ nhất
best_k = max(results, key=lambda x: x['silhouette'])['k']
print(f"\n=> Số cụm khuyến nghị: K = {best_k}")
```

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **Local Minima** | Cực tiểu cục bộ | Điểm dừng của thuật toán tối ưu lặp nhưng chưa phải là nghiệm tốt nhất toàn cục. |
| **Multi-start (n_init)** | Đa khởi tạo | Kỹ thuật chạy thuật toán nhiều lần từ các điểm bắt đầu ngẫu nhiên khác nhau để tránh nghiệm xấu. |
| **Convergence Tolerance** | Dung sai hội tụ | Ngưỡng thay đổi tối thiểu của tham số hoặc hàm loss để quyết định dừng vòng lặp (`tol`). |
| **Cluster Stability** | Độ ổn định cụm | Mức độ bất biến của cấu trúc cụm khi lấy mẫu lại (Resampling) hoặc thêm nhiễu nhẹ. |
| **Elbow Point** | Điểm khuỷu tay | Vị trí trên đồ thị hàm mất mát nơi tốc độ suy giảm bắt đầu chậm lại rõ rệt. |
| **Coordinate Descent** | Hạ tọa độ | Phương pháp tối ưu luân phiên cố định một nhóm tham số để tối ưu nhóm còn lại (thuật toán Lloyd). |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning (Trang 7-10: Centroid-based Clustering; Trang 28-29: Measure Metrics), TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Kleinberg, J. (2002). *An Impossibility Theorem for Clustering*. Advances in Neural Information Processing Systems (NIPS).
- [3] Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of careful seeding*. SODA.
- [4] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (Chapter 14.3: Cluster Analysis). Springer.
- [5] [scikit-learn User Guide: K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means).

