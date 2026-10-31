# Gom cụm phân cấp (Hierarchical Clustering)

## Purpose

Xây dựng một cấu trúc phân cấp lồng nhau dạng cây (gọi là **Dendrogram**) giữa các mẫu dữ liệu, cho phép người dùng quan sát mối quan hệ tương đồng ở mọi mức độ chi tiết từ vi mô (từng cặp cá thể) đến vĩ mô (toàn bộ quần thể) mà **hoàn toàn không cần chỉ định trước số lượng cụm $`K`$**. Đây là thuật toán tiêu chuẩn trong phân loại sinh học (Phylogenetic Tree), phân tích cấu trúc gene và phân nhóm khách hàng đa cấp độ. [1][2]

**Ví dụ dẫn nhập:** Trong phân loại học sinh học (Taxonomy), ta có 100 loài sinh vật.
* K-Means chỉ có thể ép chúng thành một số nhóm phẳng cố định (ví dụ $`K = 5`$).
* Gom cụm phân cấp cho phép xây dựng cây phả hệ sinh học hoàn chỉnh: Hai loài linh trưởng được gộp vào cùng một Chi, các Chi gộp vào cùng một Họ, các Họ gộp vào cùng một Bộ, và cuối cùng hội tụ về cùng một Giới Động vật. Người nghiên cứu có thể chọn cắt cây ở bất kỳ tầng nào để lấy số lượng nhóm mong muốn.

Ký hiệu: tập dữ liệu $`X = \lbrace x_1, \dots, x_n \rbrace`$. Thuật toán xây dựng ma trận liên kết $`Z \in \mathbb{R}^{(n-1) \times 4}`$ mô tả $`n-1`$ bước gộp cụm tuần tự, trong đó mỗi bước xác định hai cụm được gộp, khoảng cách giữa chúng và số lượng phần tử của cụm mới. [1]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần công thức đệ quy Lance-Williams bên dưới giải thích cách tối ưu tính toán; có thể quay lại sau khi hiểu cách vẽ biểu đồ Dendrogram qua thư viện SciPy.

### Bản chất Hình học: Biểu đồ Cây Dendrogram & Phân hoạch Cắt Ngưỡng

Khác với các thuật toán phân hoạch phẳng (Partitioning methods), gom cụm phân cấp sinh ra một **họ các phân hoạch lồng nhau (Nested Partitions)**:

```text
BIỂU ĐỒ CÂY DENDROGRAM (Cắt ngang để lấy số cụm)
Khoảng cách
   ▲
   │                 ┌───────────────┐               (Ngưỡng cắt 1: K = 2 cụm)
   │ ────────────────┼───────────────┼────────────── Ngưỡng cắt cao
   │                 │               │
   │           ┌─────┴─────┐         │
   │ ──────────┼───────────┼─────────┼────────────── Ngưỡng cắt thấp (K = 3 cụm)
   │           │           │         │
   │        ┌──┴──┐        │      ┌──┴──┐
   │        │     │        │      │     │
   │        ●     ●        ●      ●     ●
   └────────x1────x2───────x3─────x4────x5─────────> Các mẫu quan sát
```

* Chiều cao của thanh nối ngang đại diện cho **khoảng cách không tương đồng (Dissimilarity Distance)** giữa hai cụm tại thời điểm chúng được sáp nhập.
* Người dùng chỉ cần "kẻ một đường nằm ngang" cắt qua cây tại một ngưỡng khoảng cách mong muốn: Số lượng đường nhánh thẳng đứng bị cắt ngang chính là số lượng cụm $`K`$ được tạo ra.

---

## Input

- Ma trận đặc trưng $`X \in \mathbb{R}^{n \times p}`$ đã được chuẩn hóa hoặc ma trận khoảng cách $`D \in \mathbb{R}^{n \times n}`$.
- Số lượng cụm mong muốn `n_clusters` HOẶC ngưỡng khoảng cách cắt cây `distance_threshold`.
- Tiêu chí liên kết (Linkage Criterion): `ward`, `complete`, `average`, hoặc `single`.
- Hàm khoảng cách cơ sở: `euclidean`, `manhattan`, `cosine` (Lưu ý: Tiêu chí `ward` bắt buộc dùng khoảng cách Euclid).

### Giới hạn Bộ nhớ và Độ phức tạp Không gian $\mathcal{O}(n^2)$
Gom cụm phân cấp yêu cầu lưu trữ toàn bộ **Ma trận khoảng cách từng đôi một (Pairwise Distance Matrix)** có kích thước $`n \times n`$:
* Với $`n = 1.000`$: Ma trận tốn khoảng $`8\text{ MB}`$ RAM (rất nhẹ).
* Với $`n = 50.000`$: Ma trận tốn $`50.000^2 \times 8 \text{ bytes} \approx 20\text{ GB}`$ RAM!
* Do đó, gom cụm phân cấp truyền thống **chỉ phù hợp với tập dữ liệu vừa và nhỏ ($`n \le 10.000`$)**. Với dữ liệu lớn hơn, bắt buộc phải dùng thuật toán cây nén như **BIRCH**.

---

## Output

1. **Ma trận liên kết Dendrogram (Linkage Matrix $`Z`$):** Ma trận kích thước $`(n - 1) \times 4`$ lưu vết quá trình sáp nhập:
   * Cột 1 và 2: Chỉ số của hai cụm được gộp ở bước này.
   * Cột 3: Khoảng cách giữa hai cụm tại thời điểm gộp.
   * Cột 4: Số lượng điểm dữ liệu nằm trong cụm mới tạo thành.
2. **Nhãn phân cụm (Cluster Labels):** Vector nhãn $`\hat{y} \in \lbrace 0, 1, \dots, K-1 \rbrace^n`$ sau khi áp dụng ngưỡng cắt.
3. **Cấu trúc cây phân cấp (Tree Hierarchy):** Cho phép truy vấn cây con (Subtree) và quan hệ họ hàng giữa các mẫu.

---

## Evaluation metrics

### 1. Hệ số Tương quan Cophenetic (Cophenetic Correlation Coefficient - $`CPCC`$)
Đo lường mức độ trung thực của biểu đồ Dendrogram so với ma trận khoảng cách gốc ban đầu:

$$
CPCC = \frac{\sum_{i < j} (D_{ij} - \bar{D})(C_{ij} - \bar{C})}{\sqrt{\sum_{i < j} (D_{ij} - \bar{D})^2 \sum_{i < j} (C_{ij} - \bar{C})^2}}
$$

* Trong đó $`D_{ij}`$ là khoảng cách thực tế giữa hai điểm $`x_i, x_j`$, và $`C_{ij}`$ là chiều cao trên Dendrogram tại thời điểm $`x_i`$ và $`x_j`$ lần đầu tiên được gộp chung vào một cụm.
* $`CPCC \in [0, 1]`$: Càng gần $`1.0`$ chứng tỏ cây phân cấp càng phản ánh trung thực hình học không gian gốc.

### 2. Silhouette Score sau khi Cắt Cụm
Dùng để xác định ngưỡng cắt cây (tương ứng số cụm $`K`$) cho ra các cụm có độ phân tách và gắn kết tốt nhất.

---

## How it works

Gom cụm phân cấp gồm hai trường phái đối nghịch:
* **Agglomerative (Kết tụ - Bottom-up):** Bắt đầu từ $`n`$ cụm đơn lẻ (mỗi điểm là một cụm), ở mỗi bước tìm hai cụm gần nhau nhất và gộp lại cho đến khi chỉ còn 1 cụm duy nhất chứa tất cả. (Phương pháp phổ biến nhất).
* **Divisive (Phân chia - Top-down):** Bắt đầu từ 1 cụm khổng lồ chứa toàn bộ $`n`$ điểm, ở mỗi bước phân chia đệ quy cụm thành các cụm con nhỏ hơn cho đến khi mỗi điểm là một cụm.

```text
THUẬT TOÁN AGGLOMERATIVE (KẾT TỤ TỪ DƯỚI LÊN)
1. Khởi tạo n cụm: C_1 = {x_1}, C_2 = {x_2}, ..., C_n = {x_n}
2. Tính ma trận khoảng cách giữa mọi cặp cụm D(C_i, C_j)
3. LẶP (n - 1) bước:
   a. Tìm cặp cụm (A, B) có khoảng cách D(A, B) nhỏ nhất
   b. Gộp A và B thành cụm mới C_mới = A U B
   c. Cập nhật lại khoảng cách từ C_mới tới tất cả các cụm còn lại
4. KẾT THÚC khi toàn bộ dữ liệu hội tụ về 1 cụm gốc duy nhất
```

### Bốn Tiêu chuẩn Liên kết Cốt lõi (Linkage Criteria)

Cách định nghĩa khoảng cách giữa hai tập hợp điểm $`D(A, B)`$ quyết định hoàn toàn hình thái của cây phân cấp:

```text
1. Single Linkage (Min)       2. Complete Linkage (Max)     3. Average Linkage (Mean)
     A           B                 A           B                 A           B
   ●───●                           ●           ●                 ●───●
   │   │   (Min distance)          │           │                 │ X │   (Avg distance)
   ●───●                           ●───────────●                 ●───●
                                   (Max distance)
```

1. **Single Linkage (Khoảng cách cực tiểu):**

$$
D_{\text{single}}(A, B) = \min_{x \in A, z \in B} D(x, z)
$$

   * *Hạn chế:* Rất dễ bị **Hiện tượng dây xích (Chaining Effect)** — chỉ cần một chuỗi các điểm nhiễu nằm rải rác giữa hai cụm lớn cũng đủ khiến thuật toán nhập hai cụm lại với nhau thành một dải dài ngoằn ngoèo.

2. **Complete Linkage (Khoảng cách cực đại):**

$$
D_{\text{complete}}(A, B) = \max_{x \in A, z \in B} D(x, z)
$$

   * *Đặc tính:* Triệt tiêu hiện tượng dây xích; tạo ra các cụm có đường kính nhỏ, đồng đều và cực kỳ nén chặt (Compact clusters).

3. **Average Linkage (Khoảng cách trung bình / UPGMA):**

$$
D_{\text{average}}(A, B) = \frac{1}{\lvert A \rvert \lvert B \rvert} \sum_{x \in A} \sum_{z \in B} D(x, z)
$$

   * *Đặc tính:* Cân bằng hài hòa giữa Single và Complete; ít nhạy cảm với ngoại lai và phản ánh trung thực cấu trúc phương sai của dữ liệu.

4. **Ward's Linkage (Phương pháp phương sai tối thiểu - Khuyên dùng mặc định):**
   Thay vì đo khoảng cách hình học đơn thuần, Ward tính toán **mức độ gia tăng của tổng bình phương sai số nội cụm ($`\Delta \text{ESS}`$)** khi gộp cụm $`A`$ và $`B`$:

$$
\Delta \text{ESS}(A, B) = \frac{\lvert A \rvert \lvert B \rvert}{\lvert A \rvert + \lvert B \rvert} \|\mu_A - \mu_B\|_2^2
$$

   * *Đặc tính:* Tạo ra các cụm hình cầu có độ biến thiên phương sai nội cụm nhỏ nhất; cho kết quả phân cụm có chất lượng tương tự hoặc vượt trội so với K-Means nhưng không bị phụ thuộc vào khởi tạo ngẫu nhiên.

---

## Data used

Bảng đối chiếu đặc điểm phân cụm phân cấp trên dữ liệu bài tập và dữ liệu thực tế:

| Khía cạnh | Dữ liệu Bài tập / Slide môn học | Dữ liệu Thực tế (Production) |
|---|---|---|
| **Quy mô mẫu ($`n`$)** | Rất nhỏ ($`n = 10 \sim 150`$), như tập `Iris` (150 mẫu). | Rất lớn ($`n > 100.000`$); thuật toán bị tràn RAM ngay ở bước tính ma trận khoảng cách. |
| **Trực quan hóa Dendrogram** | Đọc được rõ từng nhãn tên loài trên trục hoành. | Hàng ngàn đường kẻ dày đặc như mạng nhện, không thể đọc thủ công; phải cắt nhánh (Pruning). |
| **Hiện tượng Chaining** | Hiếm gặp do dữ liệu mẫu được phân cụm nhân tạo rõ nét. | Single linkage thất bại hoàn toàn vì nhiễu làm nối liền các cụm; bắt buộc dùng Ward hoặc Average. |
| **Mục tiêu sử dụng** | Minh họa thuật toán sáp nhập từng bước trên bảng tính. | Khám phá cấu trúc danh mục sản phẩm, giải trình tự gen hoặc phân loại nhóm khách hàng sâu. |

---

## Design

### Mã nguồn Thực hành Vẽ Dendrogram và Phân cụm với SciPy & Scikit-learn

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# 1. Chuẩn hóa dữ liệu
X_scaled = StandardScaler().fit_transform(X)

# 2. Xây dựng ma trận liên kết Linkage bằng thuật toán Ward qua SciPy
Z = linkage(X_scaled, method='ward')

# 3. Đánh giá tính trung thực Cophenetic Correlation
c, coph_dists = cophenet(Z, pdist(X_scaled))
print(f"Hệ số tương quan Cophenetic: {c:.4f}")

# 4. Vẽ biểu đồ cây Dendrogram
plt.figure(figsize=(10, 5))
plt.title("Biểu đồ Phân cấp Dendrogram (Tiêu chuẩn Ward)")
plt.xlabel("Mẫu dữ liệu / Chỉ số cụm")
plt.ylabel("Khoảng cách khoảng cách gộp")
dendrogram(
    Z,
    truncate_mode='lastp',  # Chỉ hiện p cụm cuối để tránh rối mắt
    p=10,
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True
)
plt.axhline(y=15.0, color='r', linestyle='--', label='Ngưỡng cắt khoảng cách')
plt.legend()
plt.show()

# 5. Phân cụm chính thức với Scikit-learn
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
cluster_labels = agg_cluster.fit_predict(X_scaled)
```

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **Hierarchical Clustering** | Gom cụm phân cấp | Thuật toán xây dựng cây phả hệ lồng nhau giữa các mẫu dữ liệu từ vi mô đến vĩ mô. |
| **Dendrogram** | Biểu đồ cây phân cấp | Đồ thị hình cây thể hiện trình tự và khoảng cách sáp nhập của các cụm dữ liệu. |
| **Agglomerative** | Kết tụ (Bottom-up) | Phương pháp gộp dần từ từng mẫu cá thể lên một cụm duy nhất. |
| **Divisive** | Phân chia (Top-down) | Phương pháp chẻ dần từ một cụm khổng lồ xuống từng cá thể đơn lẻ. |
| **Ward's Linkage** | Liên kết Ward | Tiêu chí sáp nhập làm cực tiểu hóa mức tăng tổng phương sai sai số nội cụm ($`\text{ESS}`$). |
| **Chaining Effect** | Hiện tượng dây xích | Lỗi kinh điển của Single linkage khi các điểm nhiễu vô tình tạo cầu nối liên kết hai cụm xa nhau. |
| **Cophenetic Correlation** | Tương quan Cophenetic | Hệ số đo lường mức độ bảo toàn khoảng cách không gian gốc của biểu đồ Dendrogram. |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning (Trang 20-24: Hierarchical Clustering Methods), TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Ward, J. H., Jr. (1963). *Hierarchical Grouping to Optimize an Objective Function*. Journal of the American Statistical Association.
- [3] Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis* (Chapter 5: Agglomerative Nesting). Wiley.
- [4] [scikit-learn User Guide: Hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering).
- [5] [SciPy Cluster Hierarchy documentation](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).

