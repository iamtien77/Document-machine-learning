# Chương 3 — Học không giám sát (Unsupervised Learning)

> Bản ghi chú tiếng Việt dựa trên [ML_Chapter3_new.pdf](<../Tài liệu học tập/ML_Chapter3_new.pdf>) (32 trang), bài giảng của TS. Nguyễn Thị Khánh Tiên, Trường Đại học Giao thông Vận tải TP.HCM. Số trang trong ngoặc đơn là số trang slide tương ứng trong PDF. Nội dung được hệ thống hóa, chuẩn hóa công thức toán học và bổ sung các phân tích học thuật chuyên sâu.

---

## 1. Khái niệm và Mục tiêu của Học không giám sát (tr. 2–5)

Học không giám sát (Unsupervised Learning) là phân nhánh của học máy trong đó thuật toán tiếp nhận tập dữ liệu $`X = [x_1, x_2, \dots, x_n]^\top`$ **hoàn toàn không có nhãn đích chuẩn $`y`$ (Unlabeled Data)**.

Nhiệm vụ của mô hình không phải là dự đoán một giá trị đã biết, mà là tự động phát hiện các cấu trúc ẩn (Hidden Structures), quy luật phân bố (Underlying Distributions) hoặc các nhóm tương đồng tự nhiên có sẵn trong không gian dữ liệu.

| Tiêu chí | Học có giám sát (Supervised) | Học không giám sát (Unsupervised) |
|---|---|---|
| **Dữ liệu đầu vào** | Cặp đặc trưng và nhãn $`(x_i, y_i)`$ | Chỉ có vector đặc trưng $`x_i`$ |
| **Mục tiêu học** | Xấp xỉ hàm quy luật $`f: X \to y`$ | Khám phá cấu trúc, mật độ, giảm chiều |
| **Đánh giá kết quả** | Dễ dàng qua sai số với nhãn thật (Accuracy, RMSE) | Khó khăn, chủ yếu dựa trên hình học nội tại (Silhouette) |
| **Ứng dụng chính** | Phân loại thư rác, dự đoán giá nhà, nhận diện ảnh | Phân khúc khách hàng, phát hiện gian lận, nén dữ liệu |

### Các Lĩnh vực Ứng dụng Tiêu biểu (tr. 3–5)
1. **Phân khúc Khách hàng (Customer Segmentation):** Gom nhóm người tiêu dùng dựa trên thói quen mua sắm, giá trị giỏ hàng và tần suất giao dịch để phục vụ chiến lược tiếp thị mục tiêu.
2. **Nén Dữ liệu và Giảm Chiều (Dimensionality Reduction):** Giảm dung lượng ảnh, tăng tốc độ xử lý của các mô hình học máy thông qua phép chiếu bảo toàn phương sai.
3. **Phát hiện Bất thường (Anomaly Detection):** Nhận diện các giao dịch tài chính gian lận hoặc sự cố rò rỉ mạng máy tính dựa trên các điểm nằm ngoài các cụm bình thường.
4. **Phân tích Mạng Xã hội và Tin sinh học:** Nhận diện các cộng đồng người dùng có chung sở thích; gom nhóm biểu hiện gen tế bào học.

---

## 2. Thuật toán Gom cụm K-Means (tr. 6–12)

K-Means là thuật toán gom cụm phân hoạch (Partitioning Clustering) phổ biến nhất, nhằm phân chia $`n`$ điểm dữ liệu thành $`K`$ cụm rời rạc $`C = \lbrace C_1, C_2, \dots, C_K \rbrace`$.

### Hàm Mục tiêu Toán học (WCSS / Inertia - tr. 7)
Thuật toán tìm cách tối thiểu hóa tổng bình phương khoảng cách từ mỗi điểm dữ liệu tới tâm của cụm mà nó được gán vào:

$$
J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

Trong đó $`\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x`$ là tọa độ trọng tâm (Centroid) của cụm thứ $`k`$.

### Quy trình 4 Bước của Thuật toán K-Means (tr. 8–9)
1. **Khởi tạo (Initialization):** Chọn ngẫu nhiên $`K`$ điểm dữ liệu ban đầu làm $`K`$ tâm cụm $`\mu_1, \dots, \mu_K`$.
2. **Gán cụm (Assignment Step):** Gán mỗi điểm dữ liệu $`x_i`$ vào cụm có tâm gần nhất theo khoảng cách Euclid:

$$
k^{\ast}(i) = \arg\min_{k \in \lbrace 1, \dots, K \rbrace} \|x_i - \mu_k\|^2
$$

3. **Cập nhật tâm (Update Step):** Tính toán lại tọa độ trọng tâm của từng cụm dựa trên trung bình cộng các điểm mới được gán:

$$
\mu_k^{\text{new}} = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
$$

4. **Kiểm tra Hội tụ (Convergence Check):** Lặp lại Bước 2 và 3 cho đến khi tọa độ các tâm cụm không còn dịch chuyển đáng kể hoặc đạt số vòng lặp tối đa.

### Ưu điểm và Giới hạn của K-Means (tr. 10–12)
* **Ưu điểm:** Đơn giản, trực quan, tốc độ thực thi rất nhanh với độ phức tạp tuyến tính $`O(n \cdot K \cdot p \cdot \text{iter})`$.
* **Nhược điểm lớn:**
  * Bắt buộc phải cung cấp trước số lượng cụm $`K`$.
  * Rất nhạy cảm với cách khởi tạo tâm cụm ban đầu (có thể rơi vào cực tiểu cục bộ; khắc phục bằng thuật toán `k-means++`).
  * Chỉ hiệu quả với các cụm có dạng hình cầu đối xứng; thất bại hoàn toàn với cụm có hình dạng phi tuyến, uốn lượn.
  * Cực kỳ nhạy cảm với các điểm ngoại lai (Outliers kéo lệch tâm cụm).

---

## 3. Gom cụm Phân cấp (Hierarchical Clustering - tr. 13–18)

Gom cụm phân cấp xây dựng một cây phả hệ các cụm lồng nhau mà không yêu cầu phải cố định trước số lượng cụm $`K`$.

### Hai Chiến lược Tiếp cận (tr. 13)
1. **Hợp nhất từ dưới lên (Agglomerative / Bottom-Up - Phổ biến nhất):** Ban đầu mỗi điểm là một cụm riêng biệt ($`n`$ cụm). Tại mỗi bước, thuật toán tìm và gộp hai cụm gần nhau nhất lại với nhau cho đến khi tất cả quy về 1 cụm duy nhất.
2. **Phân chia từ trên xuống (Divisive / Top-Down):** Ban đầu toàn bộ dữ liệu là 1 cụm khổng lồ. Thuật toán liên tục chia tách thành các cụm con nhỏ hơn.

### Cây Phả hệ Biểu diễn (Dendrogram - tr. 14–15)
* Kết quả gom cụm được hiển thị bằng biểu đồ hình cây **Dendrogram**. Trục tung thể hiện khoảng cách liên kết (khoảng cách giữa hai cụm khi được gộp).
* Người dùng có thể cắt cây bằng một đường ngang tùy ý: chiều cao cắt càng thấp thì thu được càng nhiều cụm vi mô; chiều cao cắt càng cao thì thu được số cụm ít hơn với mức độ bao quát rộng hơn.

### Các Tiêu chuẩn Đo Khoảng cách Giữa Hai Cụm (Linkage Criteria - tr. 16–18)
Cho hai cụm $`A`$ và $`B`$:
1. **Single Linkage (Khoảng cách ngắn nhất):** $`D(A, B) = \min_{x \in A, y \in B} \|x - y\|`$ (Dễ tìm cụm phi tuyến nhưng dễ bị hiệu ứng dây chuyền - Chaining Effect).
2. **Complete Linkage (Khoảng cách xa nhất):** $`D(A, B) = \max_{x \in A, y \in B} \|x - y\|`$ (Tạo các cụm nhỏ, chặt chẽ, tránh xâu chuỗi).
3. **Average Linkage (Khoảng cách trung bình):** $`D(A, B) = \frac{1}{|A||B|} \sum_{x \in A} \sum_{y \in B} \|x - y\|`$ (Cân bằng, ít nhạy cảm với nhiễu).
4. **Ward's Linkage (Tối thiểu hóa phương sai):** Tối thiểu hóa mức gia tăng tổng phương sai nội cụm khi hợp nhất $`A`$ và $`B`$ (Tiêu chuẩn tối ưu nhất cho cụm dạng cầu).

---

## 4. Gom cụm Dựa trên Mật độ (DBSCAN - tr. 19–25)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) định nghĩa: *Một cụm là một vùng không gian liên tục có mật độ điểm cao, được phân tách bởi các vùng có mật độ điểm thấp*.

### Hai Siêu tham số Nền tảng (tr. 20)
1. **$`\varepsilon`$ (Epsilon):** Bán kính lân cận hình cầu xung quanh mỗi điểm dữ liệu.
2. **$`\text{MinPts}`$ (Minimum Samples):** Số lượng điểm tối thiểu bắt buộc phải có bên trong bán kính $`\varepsilon`$.

### Phân loại 3 Loại Điểm trong Không gian (tr. 21–22)
* **Điểm lõi (Core Point):** Điểm $`p`$ có ít nhất $`\text{MinPts}`$ điểm nằm trong bán kính $`\varepsilon`$: $`|N_\varepsilon(p)| \ge \text{MinPts}`$.
* **Điểm biên (Border Point):** Không đủ $`\text{MinPts}`$ điểm lân cận nhưng nằm trong bán kính $`\varepsilon`$ của ít nhất một Điểm lõi khác.
* **Điểm nhiễu (Noise Point):** Không phải điểm lõi cũng không chạm vào điểm lõi nào (DBSCAN gán nhãn là $`-1`$).

### Khái niệm Khả đạt theo Mật độ (Density Reachability - tr. 23–24)
* **Trực tiếp khả đạt (Directly Reachable):** Điểm $`q`$ nằm trong bán kính $`\varepsilon`$ của điểm lõi $`p`$.
* **Khả đạt theo mật độ (Density-Reachable):** Tồn tại chuỗi điểm trung gian $`p_1, \dots, p_m`$ sao cho mỗi điểm kế tiếp trực tiếp khả đạt từ điểm lõi trước đó.
* **Kết nối theo mật độ (Density-Connected):** Hai điểm $`p`$ và $`q`$ cùng khả đạt từ một điểm lõi chung $`o`$.

### Ưu điểm Nổi bật của DBSCAN (tr. 25)
* Không cần cung cấp trước số lượng cụm $`K`$.
* Nhận diện hoàn hảo các cụm có hình dạng hình học phức tạp bất kỳ (hình chữ U, trăng khuyết, xoắn ốc).
* Tự động phát hiện và loại bỏ các điểm nhiễu (Outliers) mà không để chúng làm hỏng cấu trúc cụm.

---

## 5. Các Thước đo Khoảng cách trong Học không giám sát (tr. 26–29)

Chất lượng của mọi thuật toán gom cụm phụ thuộc hoàn toàn vào hàm đo khoảng cách (Distance Metric) giữa hai vector $`x, y \in \mathbb{R}^p`$:

1. **Khoảng cách Euclid (Chuẩn $`L_2`$ - tr. 26):**

$$
d_{\text{Euclidean}}(x, y) = \sqrt{\sum_{j=1}^p (x_j - y_j)^2}
$$

2. **Khoảng cách Manhattan (Chuẩn $`L_1`$ / City Block - tr. 27):**

$$
d_{\text{Manhattan}}(x, y) = \sum_{j=1}^p |x_j - y_j|
$$

3. **Khoảng cách Minkowski (Dạng tổng quát chuẩn $`L_q`$ - tr. 28):**

$$
d_{\text{Minkowski}}(x, y) = \left( \sum_{j=1}^p |x_j - y_j|^q \right)^{1/q}
$$

   * Với $`q=1`$: tương đương Manhattan; với $`q=2`$: tương đương Euclid.
4. **Khoảng cách Cosine (Cosine Similarity - tr. 29):**
   Đo góc lệch giữa hai vector thay vì độ dài, rất hiệu quả cho văn bản (TF-IDF):

$$
\text{CosineSim}(x, y) = \frac{x^\top y}{\|x\|_2 \|y\|_2} \implies d_{\text{Cosine}} = 1 - \text{CosineSim}(x, y)
$$

* **Lưu ý Bắt buộc:** Khi dùng khoảng cách Euclid, các biến có thang đo lớn sẽ áp đảo hoàn toàn các biến có thang đo nhỏ. Bắt buộc phải chuẩn hóa Z-score (`StandardScaler`) trước khi tính khoảng cách.

---

## 6. Đánh giá Chất lượng Mô hình Gom cụm (tr. 30–32)

### Phương pháp Khuỷu tay (Elbow Method - tr. 30)
* Vẽ đồ thị hàm mục tiêu WCSS (Inertia) theo các giá trị $`K \in [1, 10]`$.
* Điểm gập cong rõ rệt nhất nơi WCSS giảm chậm lại chính là số cụm tối ưu $`K^{\ast}`$.

### Hệ số Silhouette (Silhouette Coefficient - tr. 31–32)
Đo lường độ gắn kết nội cụm $`a(i)`$ và độ phân tách ngoại cụm $`b(i)`$:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, +1]
$$

* $`s(i) \to +1`$: Điểm được gán cụm rất chính xác.
* $`s(i) \approx 0`$: Điểm nằm ở ranh giới giữa hai cụm.
* $`s(i) < 0`$: Điểm bị gán nhầm cụm.

---

## Nguồn Tham Khảo và Đối Chiếu

- Slide bài giảng gốc: TS. Nguyễn Thị Khánh Tiên, *Chương 3: Học không giám sát*, Trường ĐH Giao thông Vận tải TP.HCM.
- MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Proc. 5th Berkeley Symp. Math. Statist. Prob., 1, 281-297.
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD-96, 226-231.
- Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. J. Comput. Appl. Math., 20, 53-65.

