# Học không giám sát: Bản đồ học tập và Nguyên lý nền tảng

## Purpose

Học không giám sát (Unsupervised Learning) là nhánh học máy khám phá cấu trúc ẩn, quy luật phân phối và các nhóm tự nhiên bên trong dữ liệu mà **hoàn toàn không cần nhãn đích** ($`y`$). Khác với học có giám sát (nơi thuật toán học cách ánh xạ từ $X$ sang $y$ đã biết), học không giám sát chỉ tiếp nhận ma trận đặc trưng $X \in \mathbb{R}^{n \times p}$ và tự động tìm kiếm các đặc tính hình học hoặc thống kê nội tại của không gian dữ liệu.

**Ví dụ dẫn nhập:** Trong phân tích khách hàng thương mại điện tử, ta có lịch sử mua hàng, tần suất truy cập và số tiền chi tiêu của 100.000 người dùng, nhưng không có sẵn nhãn phân loại nào ("khách VIP", "khách săn sale" hay "khách rời bỏ"). Học không giám sát giúp tự động gom những người có hành vi tương đồng thành từng nhóm riêng biệt, từ đó hỗ trợ doanh nghiệp thiết kế chiến dịch tiếp thị chuyên biệt cho từng phân khúc.

**Cách đọc tài liệu này:** 
1. Nắm vững bản chất và ba bài toán lớn (Gom cụm, Giảm chiều, Phát hiện bất thường).
2. Hiểu rõ thách thức cốt lõi: bài toán không xác định duy nhất (Ill-posed problem) và không có nhãn kiểm chứng.
3. Đi tuần tự theo quy trình từ Tiền xử lý dữ liệu $\to$ Lựa chọn thuật toán $\to$ Đánh giá thước đo nội tại $\to$ Triển khai thực tế.

---

### Động lực nghiên cứu & Bản chất học thuật (Textbook Intuition)

Trong thực tế công nghiệp và nghiên cứu khoa học, **dữ liệu không gán nhãn chiếm hơn 90% tổng lượng dữ liệu thu thập được**. Chi phí gán nhãn thủ công (Human Annotation) cho hàng triệu hình ảnh, văn bản y khoa hay bản ghi giao dịch ngân hàng là cực kỳ đắt đỏ, tốn thời gian và dễ chịu thiên lệch chủ quan của con người. Do đó, học không giám sát đóng vai trò nền tảng trong việc khai phá tri thức thô.

```text
+-----------------------------------------------------------------------------------+
|                        KHÔNG GIAN HỌC MÁY (MACHINE LEARNING)                      |
+-----------------------------------------------------------------------------------+
|  1. HỌC CÓ GIÁM SÁT (Supervised)      | Dữ liệu (X, y) -> Tối ưu hàm mất mát L(y, f(X)) |
|  2. HỌC KHÔNG GIÁM SÁT (Unsupervised) | Chỉ có X       -> Tối ưu cấu trúc nội tại       |
|  3. HỌC BÁN GIÁM SÁT (Semi-supervised)| Ít (X, y) + Nhiều X -> Tận dụng biểu diễn       |
|  4. HỌC TĂNG CƯỜNG (Reinforcement)   | Môi trường + Phần thưởng (Reward signal)         |
+-----------------------------------------------------------------------------------+
```

#### Thách thức toán học cốt lõi: Bài toán không xác định duy nhất (Ill-posed Problem)
Trong học có giám sát, ta có hàm mục tiêu rõ ràng để đối chiếu (ví dụ: tối thiểu hóa Cross-Entropy hoặc Mean Squared Error). Trong học không giám sát, **không tồn tại một đáp án đúng duy nhất (Ground Truth)**. 

Giả sử ta có một tập hợp các hình khối gồm hình tròn và hình vuông với hai màu đỏ và xanh:
* Một thuật toán có thể gom cụm theo **hình dạng** (Nhóm hình tròn vs Nhóm hình vuông).
* Một thuật toán khác lại gom cụm theo **màu sắc** (Nhóm màu đỏ vs Nhóm màu xanh).
Cả hai cách phân nhóm đều hoàn toàn hợp lý về mặt toán học. Sự hữu ích của kết quả phụ thuộc vào **mục tiêu nghiệp vụ** của con người chứ không thuần túy là chân lý toán học khách quan.

#### Giả thuyết Đa tạp (The Manifold Hypothesis)
Tại sao học không giám sát có thể tìm ra quy luật trong không gian hàng trăm hoặc hàng ngàn chiều?
Giả thuyết Đa tạp chỉ ra rằng: Dữ liệu trong thế giới thực tuy nằm trong không gian quan sát có số chiều rất lớn ($`p`$ chiều), nhưng phần lớn mật độ xác suất lại tập trung xung quanh một **đa tạp con (Sub-manifold)** có số chiều nội tại ($`d \ll p`$) nhỏ hơn rất nhiều.

$$
\mathcal{M} \subset \mathbb{R}^p, \quad \dim(\mathcal{M}) = d \ll p
$$

* Nhiệm vụ của các thuật toán giảm chiều (PCA, t-SNE) là "mở phẳng" hoặc chiếu đa tạp này về không gian $`d`$ chiều.
* Nhiệm vụ của các thuật toán gom cụm (K-Means, DBSCAN) là phân vùng đa tạp thành các cụm mật độ cục bộ.

---

## Ba Trụ cột của Học không giám sát

```text
                           HỌC KHÔNG GIÁM SÁT (X)
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
1. GOM CỤM (Clustering)    2. GIẢM CHIỀU (Dim Reduction) 3. PHÁT HIỆN BẤT THƯỜNG
   • Dựa trên tâm (K-Means)   • Tuyến tính (PCA, SVD)       • Dựa trên mật độ (LOF)
   • Dựa trên mật độ (DBSCAN) • Phi tuyến (t-SNE, UMAP)     • Dựa trên cây (Isolation Forest)
   • Phân phối (GMM/EM)       • Nhân (Kernel PCA)           • Biên ranh giới (One-Class SVM)
   • Phân cấp (Hierarchical)
```

### 1. Phân cụm dữ liệu (Clustering)
Gom các quan sát $`x_1, x_2, \dots, x_n`$ thành $`K`$ nhóm rời rạc $`C_1, C_2, \dots, C_K`$ sao cho:
* **Tính đồng nhất nội cụm (Compactness / High intra-cluster similarity):** Các điểm trong cùng một cụm có khoảng cách hình học hoặc mức độ tương đồng rất cao.
* **Tính phân tách ngoại cụm (Separation / Low inter-cluster similarity):** Các điểm thuộc các cụm khác nhau có khoảng cách càng xa nhau càng tốt.

$$
C_i \cap C_j = \emptyset \quad (\forall i \ne j), \quad \bigcup_{k=1}^K C_k = \lbrace 1, 2, \dots, n \rbrace
$$

### 2. Giảm chiều dữ liệu & Trích xuất biểu diễn (Dimensionality Reduction)
Biến đổi dữ liệu từ không gian $`p`$ chiều ban đầu sang không gian $`d`$ chiều mới ($`d < p`$):

$$
f: \mathbb{R}^p \to \mathbb{R}^d \quad (d \ll p)
$$

* **Mục đích:** Khắc phục lời nguyền số chiều (Curse of Dimensionality), nén dữ liệu, giảm chi phí tính toán cho các mô hình xuôi dòng (downstream models), và trực quan hóa dữ liệu trên đồ thị 2D/3D.

### 3. Phát hiện bất thường (Anomaly / Novelty Detection)
Xác định các mẫu quan sát có cơ chế sinh dữ liệu khác biệt đáng kể so với đại đa số dữ liệu bình thường còn lại trong không gian:

$$
\text{Anomaly}(x) = \mathbb{I}\left( p(x) < \tau \right) \quad \text{hoặc} \quad \text{Score}(x) > \text{threshold}
$$

* **Ứng dụng:** Phát hiện gian lận giao dịch thẻ tín dụng, phát hiện tấn công xâm nhập mạng máy tính, cảnh báo sớm hỏng hóc cơ khí trong IoT công nghiệp.

---

## Input & Output trong Học không giám sát

### Input
* **Ma trận đặc trưng (Design Matrix):** $`X \in \mathbb{R}^{n \times p}`$, gồm $`n`$ hàng (quan sát) và $`p`$ cột (thuộc tính số hoặc phân loại đã được mã hóa).
* **Đặc tả hình học & Miền giá trị:** Khoảng cách và độ tương đồng giữa các quan sát là yếu tố quyết định. Dữ liệu bắt buộc phải được xử lý missing, chuẩn hóa tỉ lệ (Scale) trước khi đưa vào mô hình.
* **Siêu tham số hình học (Hyperparameters):** Tùy thuật toán, người dùng cần cung cấp giả định tiên nghiệm: số cụm mong muốn ($`K`$), bán kính lân cận ($`\varepsilon`$), số láng giềng tối thiểu ($\text{MinPts}$), hoặc tỷ lệ ô nhiễm dự kiến ($\text{contamination}$).

### Output
Tùy vào nhóm bài toán, đầu ra của học không giám sát gồm:
1. **Trong bài toán Gom cụm:**
   * Vector nhãn phân cụm cứng (Hard Clustering): $`\hat{y} \in \lbrace 0, 1, \dots, K-1 \rbrace^n`$.
   * Hoặc ma trận xác suất phân cụm mềm (Soft Clustering): $`\Gamma \in [0, 1]^{n \times K}`$, trong đó $`\gamma_{ik} = P(z_i = k \mid x_i)`$.
   * Tọa độ các điểm tâm cụm (Centroids / Prototypes): $`\mu_1, \dots, \mu_K \in \mathbb{R}^p`$.
2. **Trong bài toán Giảm chiều:**
   * Ma trận tọa độ mới trên không gian tiềm ẩn (Latent Space): $`Z \in \mathbb{R}^{n \times d}`$.
   * Ma trận trọng số phép chiếu (Loading Matrix / Eigenvectors): $`W \in \mathbb{R}^{p \times d}`$.
3. **Trong bài toán Phát hiện bất thường:**
   * Vector nhãn nhị phân: $`\hat{y}_i \in \lbrace +1 \text{ (inlier)}, -1 \text{ (outlier)} \rbrace`$.
   * Điểm số bất thường (Anomaly Score / Decision Function value): $`s(x_i) \in \mathbb{R}`$.

---

## Evaluation metrics: Đánh giá mô hình khi không có nhãn chuẩn

Đánh giá học không giám sát là một thách thức lớn. Ta chia làm hai nhóm trường hợp:

### 1. Đánh giá nội tại (Internal Evaluation - Không có Ground Truth)
Sử dụng chính các thuộc tính hình học của không gian $X$ để chấm điểm chất lượng phân nhóm:
* **Silhouette Coefficient ($`S`$):** Đo lường sự hài hòa giữa độ gắn kết nội cụm ($`a`$) và độ tách biệt ngoại cụm ($`b`$). Giá trị chạy từ $-1$ đến $+1$; càng gần $+1$ cụm càng rõ nét.
* **Davies-Bouldin Index ($`DB`$):** Tỷ lệ giữa độ phân tán của cụm với khoảng cách giữa các tâm cụm. Giá trị càng nhỏ thể hiện sự phân tách càng tốt.
* **Calinski-Harabasz Index ($`CH`$):** Tỷ số giữa phương sai giữa các cụm và phương sai nội cụm (Variance Ratio Criterion). Giá trị càng lớn càng tốt.
* **Tỷ lệ phương sai giải thích (Explained Variance Ratio):** Trong giảm chiều, đo lường phần trăm thông tin gốc được giữ lại qua $`d`$ thành phần chính đầu tiên.

### 2. Đánh giá ngoại tại (External Evaluation - Khi có nhãn kiểm định benchmark)
Trong môi trường nghiên cứu hoặc khi có một tập dữ liệu nhỏ đã được gán nhãn kiểm thử (Ground Truth $y$), ta có thể đo độ tương đồng giữa nhãn phân cụm và nhãn thực tế:
* **Adjusted Rand Index (ARI):** Đo lường tỷ lệ các cặp điểm được gán cùng cụm hoặc khác cụm, hiệu chỉnh theo mức độ ngẫu nhiên. Chạy từ $-1$ đến $+1$; $`ARI = 1`$ là khớp hoàn hảo.
* **Normalized Mutual Information (NMI):** Đo lường lượng thông tin chung (Mutual Information) giữa hai phân hoạch nhãn, chuẩn hóa về đoạn $[0, 1]$.

---

## Quy trình chuẩn một dự án Học không giám sát (End-to-End Workflow)

```text
[1. Hiểu bài toán] ──> [2. Thu thập & Làm sạch X] ──> [3. Chuẩn hóa tỉ lệ (Scale)]
                                                              │
[6. Triển khai & Theo dõi] <── [5. Đánh giá & Profiling] <── [4. Huấn luyện Mô hình]
```

1. **Định hình mục tiêu:** Phân khúc để làm gì? Cần tìm bao nhiêu nhóm? Nếu giảm chiều thì mục tiêu là trực quan hóa (2D/3D) hay làm đầu vào cho mô hình phân loại?
2. **Tiền xử lý đặc thù:** Xử lý triệt để missing values (vì đa số thuật toán gom cụm không tự xử lý được NaN). Mã hóa biến phân loại cẩn thận vì phép đo khoảng cách trên biến One-Hot có hình học khác biến số.
3. **Chuẩn hóa bắt buộc (Feature Scaling):** Các thuật toán tính khoảng cách (Euclidean trong K-Means, DBSCAN, PCA) cực kỳ nhạy cảm với thang đo. Thuộc tính có đơn vị lớn (như thu nhập hàng chục triệu) sẽ áp đảo hoàn toàn thuộc tính có đơn vị nhỏ (như tuổi 20-60). Phải đưa về cùng thang đo qua `StandardScaler` hoặc `RobustScaler`.
4. **Huấn luyện & Dò tìm siêu tham số:** Thử nghiệm dải tham số (quét $`K`$ qua phương pháp Elbow và Silhouette; quét $\varepsilon$ qua biểu đồ k-distance).
5. **Đặc tả cụm (Cluster Profiling & Interpretation):** Tính toán giá trị trung bình, trung vị của từng thuộc tính trên mỗi cụm để đặt tên nghiệp vụ có nghĩa (ví dụ: "Cụm 0: Khách hàng trẻ thu nhập cao", "Cụm 1: Khách hàng truyền thống chi tiêu thấp").
6. **Kiểm tra độ bền vững (Cluster Stability):** Chia nhỏ dữ liệu hoặc thêm nhiễu nhẹ xem cấu trúc cụm có bị vỡ hay thay đổi đột ngột không.

---

## Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt tương đương | Định nghĩa & Trực giác |
|---|---|---|
| **Unsupervised Learning** | Học không giám sát | Học quy luật và cấu trúc chỉ từ ma trận đặc trưng $X$, không có nhãn $y$. |
| **Clustering** | Gom cụm / Phân cụm | Phân chia các điểm dữ liệu thành các nhóm dựa trên mức độ tương đồng. |
| **Centroid** | Tâm cụm | Vector trung bình trọng số đại diện cho vị trí trung tâm của một cụm. |
| **Density-based** | Dựa trên mật độ | Nhóm các vùng không gian có mật độ điểm cao, cách nhau bởi vùng mật độ thấp. |
| **Dimensionality Reduction** | Giảm chiều dữ liệu | Chiếu dữ liệu từ không gian $p$ chiều lớn xuống không gian $d$ chiều nhỏ hơn. |
| **Manifold** | Đa tạp | Một không gian tô-pô cục bộ trông giống không gian Euclid (ví dụ: mặt cầu, dải ruy-băng Swiss roll). |
| **Curse of Dimensionality** | Lời nguyền số chiều | Hiện tượng không gian dữ liệu trở nên cực kỳ thưa thớt và khoảng cách mất tính phân biệt khi số chiều tăng cao. |
| **Internal Validation** | Đánh giá nội tại | Đo lường chất lượng mô hình dựa trên chính thuộc tính hình học của tập $X$ không nhãn. |
| **External Validation** | Đánh giá ngoại tại | So sánh kết quả phân cụm với nhãn thực tế có sẵn (dùng trong thử nghiệm benchmark). |

---

## Nguồn trích dẫn & Tài liệu đọc tiếp

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning, TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (Chapter 14: Unsupervised Learning). Springer.
- [3] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning* (Chapter 9: Mixture Models and EM; Chapter 12: Continuous Latent Variables). Springer.
- [4] [scikit-learn User Guide: Clustering](https://scikit-learn.org/stable/modules/clustering.html).
- [5] [scikit-learn User Guide: Decomposing signals in components](https://scikit-learn.org/stable/modules/decomposition.html).
- [6] [scikit-learn User Guide: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html).
