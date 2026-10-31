# Học không giám sát: bản đồ học tập

## Purpose

Học không giám sát (unsupervised learning) tìm cấu trúc, quy luật phân phối và các nhóm tự nhiên bên trong dữ liệu mà **hoàn toàn không có nhãn/giá trị đích** ($`y`$). Khác với học có giám sát (học ánh xạ từ $X$ sang $y$), học không giám sát chỉ nhận ma trận đặc trưng $`X \in \mathbb{R}^{n \times p}`$ để tự động phát hiện các đặc tính hình học, cụm mật độ, phân phối xác suất ẩn hoặc biểu diễn chiều thấp của dữ liệu. Ba bài toán nền tảng là **clustering** (gom cụm), **dimensionality reduction** (giảm chiều dữ liệu) và **anomaly detection** (phát hiện bất thường). Đây là bản đồ để học từ bài toán đến vận hành thực tế. [1][2]

**Ví dụ dẫn nhập:** Trong phân tích khách hàng thương mại điện tử, ta có lịch sử mua hàng, tần suất truy cập và số tiền chi tiêu của 100.000 khách hàng nhưng không có sẵn nhãn phân loại nào ("khách VIP", "khách săn sale" hay "khách rời bỏ"). Học không giám sát giúp tự động gom những người có hành vi tương đồng thành từng cụm riêng biệt, từ đó hỗ trợ doanh nghiệp thiết kế chương trình chăm sóc chuyên biệt cho từng phân khúc.

Ký hiệu: với $`n`$ mẫu và $`p`$ feature số, $`X \in \mathbb{R}^{n \times p}`$. Không có vector nhãn $`y`$. Mô hình ước lượng cấu trúc tiềm ẩn $`Z`$, các tâm cụm $`\mu_k`$ hoặc ước lượng hàm mật độ xác suất $`p(x)`$. Mục tiêu là tìm ra biểu diễn nén hữu ích hoặc phân hoạch tự nhiên phản ánh đúng bản chất của tiến trình sinh dữ liệu. [2][4]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần công thức đọc sâu bên dưới giải thích cơ sở toán học; có thể quay lại sau khi hiểu quy trình chuẩn bị dữ liệu và huấn luyện mô hình.

### Động lực nghiên cứu & Bản chất học thuật (Textbook Intuition)

Trong thực tế, hơn 90% dữ liệu thu thập được trong thế giới thực là dữ liệu thô chưa được gán nhãn. Chi phí gán nhãn thủ công (Human Annotation) cho hàng triệu bản ghi là cực kỳ tốn kém và dễ chịu thiên lệch chủ quan. Học không giám sát đảo ngược hoàn toàn trọng tâm nghiên cứu: thay vì tìm hàm xấp xỉ dự đoán, ta tìm kiếm **nguyên lý tổ chức tự nhiên của dữ liệu**.

* **Bài toán không xác định duy nhất (Ill-posed Problem):**
  Trong học có giám sát, ta có hàm mất mát đo sai lệch với nhãn thật. Trong học không giám sát, **không tồn tại một đáp án đúng duy nhất (Ground Truth)**. Cùng một tập dữ liệu khách hàng, việc chia thành 3 nhóm (theo thu nhập) hay 5 nhóm (theo sở thích) đều có thể đúng về mặt toán học. Tính hữu ích phụ thuộc vào bài toán nghiệp vụ cụ thể.

* **Giả thuyết Đa tạp (The Manifold Hypothesis):**
  Dữ liệu thực tế thường nằm trong không gian có số chiều quan sát rất lớn ($`p`$ chiều), nhưng phần lớn mật độ xác suất chỉ tập trung xung quanh một đa tạp con phi tuyến $`\mathcal{M}`$ có số chiều nội tại $`d \ll p`$:

$$
\mathcal{M} \subset \mathbb{R}^p, \quad \dim(\mathcal{M}) = d \ll p
$$

  Nhiệm vụ của các thuật toán giảm chiều (như PCA, t-SNE) là "mở phẳng" hoặc tìm ra hệ tọa độ nội tại của đa tạp này, giúp loại bỏ nhiễu và trực quan hóa dữ liệu.

---

## Input

- Bài toán nghiệp vụ cần giải quyết: phân khúc khách hàng, nén ảnh, giảm chiều để hiển thị đồ thị 2D/3D hay phát hiện gian lận thẻ tín dụng.
- Dữ liệu thô $`X`$, schema các cột, đơn vị đo lường, phân phối của từng biến, tỷ lệ missing, outlier và thông tin về thời gian thu thập.
- Ràng buộc về độ trễ suy luận, bộ nhớ tính toán, khả năng giải thích cụm cho người làm nghiệp vụ và tần suất cập nhật lại mô hình.

### Không gian đầu vào & Điều kiện biên của dữ liệu

1. **Ma trận đặc trưng (Design Matrix $`X`$):**
   * Chiều dữ liệu: $`X \in \mathbb{R}^{n \times p}`$, mỗi hàng $`x_i^\top \in \mathbb{R}^p`$ là một vector thuộc tính của một quan sát.
   * Bản chất kiểu dữ liệu:
     * *Dữ liệu số (Continuous/Discrete):* Chiều cao, số tiền, số lượng sản phẩm.
     * *Dữ liệu phân loại (Categorical):* Giới tính, danh mục mua sắm. Cần xử lý cẩn trọng vì việc One-Hot encoding có thể làm méo mó cấu trúc khoảng cách Euclid.
2. **Độ nhạy cảm với thang đo (Scale Sensitivity):**
   * Hầu hết các thuật toán học không giám sát (K-Means, DBSCAN, PCA) đều dựa trên độ đo khoảng cách hình học giữa các điểm.
   * Nếu không chuẩn hóa, đặc trưng có phương sai lớn hoặc đơn vị lớn (như thu nhập hàng chục triệu) sẽ áp đảo hoàn toàn đặc trưng có đơn vị nhỏ (như tuổi từ 20 đến 60). Chuẩn hóa thang đo qua `StandardScaler` hoặc `RobustScaler` là **bắt buộc**.

---

## Output

Tùy vào nhóm thuật toán, đầu ra của học không giám sát có các dạng sau:

1. **Trong bài toán Gom cụm (Clustering):**
   * Nhãn phân cụm cứng (Hard Clustering): $`\hat{y} \in \lbrace 0, 1, \dots, K-1 \rbrace^n`$, gán mỗi quan sát vào duy nhất một cụm. Điểm nhiễu có thể được gán nhãn $`-1`$ (như trong DBSCAN).
   * Ma trận xác suất mềm (Soft Clustering): $`\Gamma \in [0, 1]^{n \times K}`$, trong đó $`\gamma_{ik} = P(z_i = k \mid x_i)`$ biểu thị xác suất mẫu $`i`$ thuộc cụm $`k`$ (như trong GMM).
   * Vector tâm cụm (Centroids / Medoids): $`\mu_1, \dots, \mu_K \in \mathbb{R}^p`$.
2. **Trong bài toán Giảm chiều (Dimensionality Reduction):**
   * Ma trận tọa độ mới trên không gian tiềm ẩn (Latent space): $`Z \in \mathbb{R}^{n \times d}`$ với $`d < p`$.
   * Ma trận thành phần chiếu (Components / Eigenvectors): $`W \in \mathbb{R}^{p \times d}`$.
3. **Trong bài toán Phát hiện bất thường (Anomaly Detection):**
   * Nhãn phân loại: $`\hat{y}_i \in \lbrace 1 \text{ (bình thường)}, -1 \text{ (bất thường)} \rbrace`$.
   * Điểm số bất thường liên tục (Anomaly Score / Decision function): $`s(x_i) \in \mathbb{R}`$.

---

## Evaluation metrics

Vì không có nhãn thực tế ($`y`$), việc đánh giá mô hình học không giám sát không thể dùng Accuracy, F1 hay MSE. Ta phân chia thành hai nhóm thước đo:

### 1. Thước đo nội tại (Internal Metrics - Khi không có Ground Truth)
Đánh giá chất lượng dựa trên chính các thuộc tính hình học của không gian dữ liệu $X$:
* **Silhouette Coefficient ($`S`$):** Đo lường sự hài hòa giữa độ gắn kết nội cụm ($`a`$) và độ tách biệt ngoại cụm ($`b`$). Giá trị chạy từ $-1$ đến $+1$; giá trị càng cao chứng tỏ cụm càng rõ nét và tách biệt:

$$
S = \frac{1}{n} \sum_{i=1}^n \frac{b(x_i) - a(x_i)}{\max\lbrace a(x_i), b(x_i) \rbrace}
$$

* **Davies-Bouldin Index ($`DB`$):** Tỷ số giữa độ phân tán nội cụm và khoảng cách giữa các tâm cụm. Giá trị càng nhỏ thể hiện mô hình phân cụm càng tốt.
* **Calinski-Harabasz Index ($`CH`$):** Tỷ số phương sai giữa các cụm và phương sai bên trong cụm (Variance Ratio Criterion). Giá trị càng lớn càng tốt.
* **Tỷ lệ phương sai giải thích (Explained Variance Ratio):** Trong giảm chiều (PCA), đo lường tỷ lệ thông tin được bảo toàn trên từng thành phần chính.

### 2. Thước đo ngoại tại (External Metrics - Khi có nhãn kiểm thử benchmark)
Sử dụng trong nghiên cứu thực nghiệm khi có một tập dữ liệu nhỏ đã được gán nhãn kiểm chứng để đo mức độ tương đồng:
* **Adjusted Rand Index (ARI):** Đo lường tỷ lệ các cặp điểm được gán đúng cùng cụm hoặc khác cụm, có hiệu chỉnh mức độ ngẫu nhiên. Chạy từ $-1$ đến $+1$; $`ARI = 1`$ là hoàn toàn trùng khớp.
* **Normalized Mutual Information (NMI):** Đo lượng thông tin chung giữa hai cách phân hoạch nhãn, chuẩn hóa về đoạn $[0, 1]$.

---

## How it works

Quy trình hoạt động cốt lõi của học không giám sát được vận hành theo chu trình khám phá tri thức:

```text
Chuẩn bị dữ liệu X ──> Chọn hàm khoảng cách D(x, z) ──> Khám phá hình học/mật độ
                                                                 │
Profiling nghiệp vụ <── Đánh giá nội tại (Silhouette/Elbow) <─── Tối ưu hóa cụm
```

1. **Chuẩn bị và làm sạch dữ liệu:** Xử lý triệt để missing values và chuẩn hóa tỉ lệ của toàn bộ các thuộc tính số về cùng thang đo.
2. **Xác định độ đo khoảng cách:** Lựa chọn hàm khoảng cách phù hợp với miền bài toán (Euclid cho cụm hình cầu, Manhattan cho dữ liệu nhiều chiều/ngoại lai, Cosine cho văn bản).
3. **Huấn luyện thuật toán phân cụm hoặc giảm chiều:** Tối ưu hóa hàm mục tiêu tương ứng (như cực tiểu hóa WCSS trong K-Means, cực đại hóa phương sai chiếu trong PCA, hoặc lan truyền vùng mật độ trong DBSCAN).
4. **Xác định số lượng cụm tối ưu ($`K`$):** Dùng phương pháp Điểm khuỷu tay (Elbow method) kết hợp phân tích biểu đồ Silhouette.
5. **Đặc tả cụm (Cluster Profiling):** Phân tích giá trị trung bình/trung vị của từng thuộc tính trên mỗi cụm để rút ra kết luận nghiệp vụ có ý nghĩa thực tế.

---

## Data used

Bảng dưới đây chỉ ra sự khác biệt bản chất giữa dữ liệu thực hành trong bài tập và dữ liệu thực tế:

| Khía cạnh | Dữ liệu mẫu / Bài tập (Toy Datasets) | Dữ liệu triển khai thực tế (Production Data) |
|---|---|---|
| **Cấu trúc cụm** | Các cụm nhân tạo tách biệt rõ ràng (Blob, Make-Moons, Make-Circles). | Các cụm chồng lấn mạnh (Overlapping), ranh giới mờ nhạt, mật độ không đồng đều. |
| **Số lượng chiều ($`p`$)** | Nhỏ ($`p = 2`$ hoặc $`4`$, như Iris), rất dễ vẽ biểu đồ trực quan hóa. | Lớn ($`p`$ từ vài chục đến hàng ngàn), chịu ảnh hưởng nặng nề của Lời nguyền số chiều. |
| **Ngoại lai & Nhiễu** | Dữ liệu được làm sạch hoàn hảo, gần như 0% nhiễu. | Chứa nhiều điểm ngoại lai, lỗi cảm biến, tài khoản ảo; dễ làm méo mó các tâm cụm. |
| **Giá trị khuyết (Missing)**| Dữ liệu đầy đủ 100%. | Tỷ lệ missing cao, cần chiến lược điền khuyết (Imputation) trước khi đo khoảng cách. |
| **Xác thực kết quả** | Có sẵn nhãn nhị phân/đa lớp để tính ARI, NMI đối chiếu. | Hoàn toàn không có nhãn; phải dựa vào đánh giá nội tại và thẩm định của chuyên gia nghiệp vụ. |

---

## Design

### Pipeline Scikit-learn Chuẩn mực
Trong học không giám sát, việc đóng gói toàn bộ quy trình từ tiền xử lý đến mô hình vào `Pipeline` là nguyên tắc bắt buộc:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Xây dựng pipeline xử lý
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('clusterer', KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42))
])

# Fit và dự đoán nhãn cụm
cluster_labels = pipeline.fit_predict(X)

# Đánh giá nội tại
sil_score = silhouette_score(pipeline.named_steps['scaler'].transform(X), cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")
```

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **Unsupervised Learning** | Học không giám sát | Học cấu trúc, phân phối và quan hệ nội tại chỉ từ ma trận $`X`$ không có nhãn $`y`$. |
| **Clustering** | Gom cụm / Phân cụm | Nhóm các quan sát tương đồng lại với nhau sao cho độ tương đồng nội cụm cao nhất. |
| **Centroid** | Tâm cụm | Vector trung bình trọng số đại diện cho vị trí trung tâm của một cụm. |
| **Density-based** | Dựa trên mật độ | Nhóm các vùng không gian có mật độ điểm cao, cách nhau bởi vùng thưa thớt. |
| **Dimensionality Reduction** | Giảm chiều dữ liệu | Chiếu dữ liệu từ không gian $`p`$ chiều lớn xuống không gian $`d`$ chiều nhỏ hơn ($`d \ll p`$). |
| **Inertia / WCSS** | Tổng bình phương nội cụm | Thước đo độ nén chặt của các cụm xung quanh tâm trong K-Means. |
| **Silhouette Score** | Hệ số dáng dấp | Đo lường độ gắn kết nội cụm so với độ tách biệt với cụm láng giềng gần nhất. |
| **Cluster Profiling** | Đặc tả cụm | Diễn giải ý nghĩa kinh doanh của từng cụm dựa trên phân phối đặc trưng. |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning, TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (Chapter 14: Unsupervised Learning). Springer.
- [3] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning* (Chapter 9: Mixture Models and EM; Chapter 12: Continuous Latent Variables). Springer.
- [4] [scikit-learn User Guide: Clustering](https://scikit-learn.org/stable/modules/clustering.html).
- [5] [scikit-learn User Guide: Decomposing signals in components](https://scikit-learn.org/stable/modules/decomposition.html).
- [6] [scikit-learn User Guide: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html).
