# Dữ liệu và Tiền xử lý trong Học không giám sát

## Purpose

Trong học có giám sát, nếu một thuộc tính bị lệch thang đo hoặc chứa ngoại lai, một số mô hình mạnh (như Random Forest, Gradient Boosting) vẫn có thể tự thích ứng hoặc điều chỉnh ngưỡng phân tách. Tuy nhiên, trong **Học không giám sát**, chất lượng dữ liệu và khâu tiền xử lý mang tính chất **quyết định sống còn**: hầu hết các thuật toán (K-Means, DBSCAN, Hierarchical, PCA) đều hoạt động trực tiếp dựa trên các phép đo khoảng cách hình học giữa các vector mẫu trong không gian $`\mathbb{R}^p`$.

**Ví dụ dẫn nhập:** Giả sử bạn gom cụm khách hàng dựa trên hai đặc trưng: `Tuổi` (từ 18 đến 70) và `Thu nhập hàng năm` (từ 50.000.000 VNĐ đến 2.000.000.000 VNĐ). 
* Nếu không chuẩn hóa thang đo, chênh lệch thu nhập giữa hai người (hàng chục triệu) sẽ hoàn toàn "đè bẹp" chênh lệch về tuổi tác (vài chục năm) khi tính khoảng cách Euclid. 
* Hệ quả là thuật toán gom cụm thực chất chỉ đang phân nhóm theo thu nhập, và biến `Tuổi` hoàn toàn bị vô hiệu hóa!

---

### Các Thước đo Khoảng cách và Mức độ Tương đồng (Distance & Similarity Metrics)

Khoảng cách toán học $`D(x, z)`$ giữa hai điểm dữ liệu $`x, z \in \mathbb{R}^p`$ là thước đo nền tảng xác định hai quan sát có "thuộc về cùng một nhóm" hay không.

```text
+------------------------------------------------------------------------------------------+
|                             CÁC HỌ KHOẢNG CÁCH TOÁN HỌC PHỔ BIẾN                         |
+------------------------------------------------------------------------------------------+
|  1. Minkowski (Tổng quát) | D(x, z) = ( sum |x_j - z_j|^q )^(1/q)                        |
|     • Euclidean (q = 2)   | Đường chim bay hình học chuẩn                                |
|     • Manhattan (q = 1)   | Khoảng cách đi theo lưới ô cờ đô thị (ít nhạy ngoại lai hơn) |
|     • Chebyshev (q = inf) | Khoảng cách cực đại theo một chiều bất kỳ                    |
|  2. Cosine Distance       | D_cos = 1 - (x.z) / (||x|| * ||z||) (tập trung vào góc xoay) |
|  3. Mahalanobis Distance  | Khử tương quan giữa các chiều thông qua ma trận hiệp phương sai |
|  4. Hamming / Jaccard     | Đo khoảng cách trên chuỗi bit nhị phân hoặc tập hợp rời rạc  |
+------------------------------------------------------------------------------------------+
```

#### 1. Khoảng cách Minkowski ($`L_q`$ Norm)
Là dạng tổng quát của các khoảng cách không gian Metric chuẩn:

$$
D_{\text{Minkowski}}(x, z) = \left( \sum_{j=1}^p \lvert x_j - z_j \rvert^q \right)^{1/q}
$$

* **Khoảng cách Euclid ($`q = 2`$ - $`L_2`$ Norm):** Là độ dài đường thẳng nối giữa hai điểm trong không gian thực tế:

$$
D_{\text{Euclidean}}(x, z) = \|x - z\|_2 = \sqrt{\sum_{j=1}^p (x_j - z_j)^2}
$$

  *Ưu điểm:* Trực quan, phù hợp với các cụm có hình dạng hình cầu đối xứng (Spherical clusters).
  *Nhược điểm:* Cực kỳ nhạy cảm với các điểm ngoại lai xa (Outliers) vì sai số bị bình phương lên.

* **Khoảng cách Manhattan ($`q = 1`$ - $`L_1`$ Norm / City Block):**

$$
D_{\text{Manhattan}}(x, z) = \|x - z\|_1 = \sum_{j=1}^p \lvert x_j - z_j \rvert
$$

  *Ưu điểm:* Bền vững hơn trước các điểm ngoại lai so với Euclid; hoạt động ổn định hơn khi số chiều dữ liệu $`p`$ tăng cao.

#### 2. Khoảng cách Cosine (Cosine Distance)
Đo lường góc lệch giữa hai vector mà **không phụ thuộc vào độ dài (Magnitude)** của chúng:

$$
\text{Cosine Similarity}(x, z) = \frac{x^\top z}{\|x\|_2 \|z\|_2} = \frac{\sum_{j=1}^p x_j z_j}{\sqrt{\sum_{j=1}^p x_j^2} \sqrt{\sum_{j=1}^p z_j^2}}
$$

$$
D_{\text{Cosine}}(x, z) = 1 - \text{Cosine Similarity}(x, z)
$$

* **Khi nào nên dùng:** Xử lý văn bản (Text Mining, TF-IDF vectors), hệ thống gợi ý (Recommender Systems) hoặc dữ liệu thưa có số chiều cao. Ví dụ: Hai bài báo cùng viết về chủ đề "Kinh tế" thì vector từ vựng sẽ chỉ về cùng một hướng trong không gian, dù một bài dài 500 từ và bài kia dài 5.000 từ.

#### 3. Khoảng cách Mahalanobis (Khử tương quan thuộc tính)
Khoảng cách Euclid ngầm giả định các thuộc tính hoàn toàn độc lập và có phương sai bằng nhau. Nếu các thuộc tính có tương quan mạnh (ví dụ: `Chiều cao` và `Cân nặng`), khoảng cách Euclid sẽ phóng đại khoảng cách gấp đôi. Khoảng cách Mahalanobis giải quyết vấn đề này bằng cách chuẩn hóa qua ma trận hiệp phương sai $`\Sigma`$:

$$
D_{\text{Mahalanobis}}(x, z) = \sqrt{(x - z)^\top \Sigma^{-1} (x - z)}
$$

* **Trực giác:** Phép tính này thực chất là biến đổi tọa độ để "xoay" và "co giãn" không gian elip dữ liệu thành hình cầu chuẩn trước khi đo khoảng cách.

---

### Lời nguyền số chiều (The Curse of Dimensionality)

Một sai lầm phổ biến là thu thập càng nhiều đặc trưng ($`p`$) càng tốt. Trong không gian nhiều chiều ($`p > 50`$ hoặc hàng ngàn chiều), hình học không gian hoạt động hoàn toàn trái ngược với trực giác 2D/3D của con người.

#### Hiện tượng tập trung khoảng cách (Distance Concentration Effect)
Khi số chiều $`p \to \infty`$, khoảng cách giữa điểm gần nhất và điểm xa nhất đến một mốc bất kỳ có xu hướng hội tụ về cùng một giá trị:

$$
\lim_{p \to \infty} \frac{D_{\max} - D_{\min}}{D_{\min}} = 0
$$

* **Hậu quả đối với Gom cụm:** Trong không gian quá nhiều chiều, **mọi điểm dữ liệu đều cách nhau những khoảng gần như bằng nhau**! Khái niệm "láng giềng gần nhất" hoặc "mật độ cục bộ" trong K-Means và DBSCAN bị suy thoái hoàn toàn.
* **Biện pháp xử lý:** Bắt buộc phải áp dụng kỹ thuật Giảm chiều dữ liệu (PCA, TruncatedSVD) hoặc Lựa chọn đặc trưng (Feature Selection) để lọc bớt nhiễu trước khi chạy thuật toán gom cụm.

---

### Các Kỹ thuật Chuẩn hóa Đặc trưng (Feature Scaling Strategies)

```text
+-------------------+-----------------------------------+------------------------------------------+
| Phương pháp       | Công thức chuyển đổi              | Khi nào nên sử dụng                      |
+-------------------+-----------------------------------+------------------------------------------+
| 1. StandardScaler | z = (x - mu) / sigma              | Dữ liệu gần phân phối chuẩn, ít outlier  |
| 2. MinMaxScaler   | x_norm = (x - min) / (max - min)  | Cần đưa về khoảng [0, 1] cố định        |
| 3. RobustScaler   | x_rob = (x - Q2) / (Q3 - Q1)      | Dữ liệu có nhiều điểm ngoại lai (Khuyên) |
| 4. PowerTransform | Box-Cox / Yeo-Johnson             | Dữ liệu lệch phải mạnh (Skewed data)    |
+-------------------+-----------------------------------+------------------------------------------+
```

#### 1. Chuẩn hóa Z-Score (`StandardScaler`)
Đưa từng đặc trưng về phân phối có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

* *Hạn chế:* Điểm ngoại lai cực trị sẽ kéo lệch trung bình $`\mu_j`$ và thổi phồng độ lệch chuẩn $`\sigma_j`$, làm cho phần lớn dữ liệu bình thường bị ép lại quá sát nhau.

#### 2. Chuẩn hóa Kháng Ngoại lai (`RobustScaler` - Khuyến nghị cho bài toán gom cụm thực tế)
Thay vì dùng trung bình và độ lệch chuẩn, phương pháp này dùng **Trung vị ($`Q_2`$)** và **Khoảng tứ phân vị ($`\text{IQR} = Q_3 - Q_1`$)**:

$$
x_{\text{robust}} = \frac{x - Q_2}{Q_3 - Q_1}
$$

* *Ưu điểm:* Trung vị và IQR không bị thao túng bởi 1% các điểm ngoại lai xa. Các cụm tự nhiên được giữ nguyên độ phân tán trung thực.

---

### Xử lý Dữ liệu Hỗn hợp (Mixed-Type Data: Số & Phân loại)

Đa số bài toán kinh doanh đều có cả dữ liệu số (`Tuổi`, `Số dư`) và dữ liệu danh mục (`Giới tính`, `Thành phố`, `Nghề nghiệp`).
1. **Rủi ro khi One-Hot Encoding:** Biến định danh sau khi One-Hot sẽ tạo ra các vector nhị phân thưa ($`\lbrace 0, 1 \rbrace`$). Khoảng cách Euclid giữa hai điểm khác nhau trên một thuộc tính One-Hot luôn cố định là $`\sqrt{1^2 + 1^2} = \sqrt{2}`$, làm méo mó cấu trúc hình học liên tục của các biến số.
2. **Khoảng cách Gower (Gower's Distance):** Tính toán độ tương đồng riêng cho từng cột (dùng Manhattan cho biến số, dùng Dice/Jaccard cho biến phân loại) rồi lấy trung bình có trọng số.
3. **Thuật toán K-Prototypes:** Thay thế K-Means truyền thống, tính toán khoảng cách Euclid cho các cột số kết hợp khoảng cách Hamming cho các cột phân loại.

---

### Checklist Tiền xử lý Chuẩn mực trong Dự án Thực tế

* [ ] **Audit Missing Values:** Đã loại bỏ các cột có tỷ lệ thiếu $> 50\%$; các cột còn lại đã được điền khuyết bằng Trung vị (`SimpleImputer(strategy='median')`) hoặc mô hình láng giềng (`KNNImputer`).
* [ ] **Kiểm tra phân phối lệch (Skewness):** Nếu đặc trưng có đuôi dài (như Doanh thu, Lượt tương tác), đã áp dụng phép biến đổi logarit $`\ln(1 + x)`$ hoặc Yeo-Johnson để kéo phân phối về dạng đối xứng.
* [ ] **Feature Scaling bắt buộc:** Toàn bộ ma trận số đã qua `RobustScaler()` hoặc `StandardScaler()`.
* [ ] **Kiểm tra đa cộng tuyến:** Loại bỏ các biến có tương quan tuyệt đối ($`r > 0.95`$) vì chúng làm tăng trọng số ảo của một khía cạnh trong phép đo khoảng cách.
* [ ] **Lưu Pipeline:** Toàn bộ các bước biến đổi phải được đóng gói trong `sklearn.pipeline.Pipeline` để áp dụng nhất quán cho bất kỳ mẫu dữ liệu mới nào suy luận sau này.
