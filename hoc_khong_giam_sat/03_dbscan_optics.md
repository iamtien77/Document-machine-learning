# Gom cụm dựa trên mật độ: DBSCAN và OPTICS

## Purpose

Trong thực tế, các cụm dữ liệu tự nhiên hiếm khi có dạng hình cầu đối xứng hoàn hảo như giả định của K-Means. Chúng thường uốn lượn theo các hình dạng hình học phức tạp (như hình trăng khuyết, hình sợi DNA, cụm đồng tâm lồng nhau) và luôn bị vây quanh bởi các điểm nhiễu ngẫu nhiên. Nhóm thuật toán **Gom cụm dựa trên mật độ (Density-Based Clustering)**, đại diện tiêu biểu là **DBSCAN** và **OPTICS**, giải quyết triệt để bài toán này bằng cách định nghĩa: *Một cụm là một vùng không gian liên tục có mật độ điểm cao, được phân tách với các cụm khác bởi các vùng không gian có mật độ điểm thấp (vùng thưa thớt)*.

**Ví dụ dẫn nhập:** Trong phân tích dữ liệu định vị GPS để phát hiện các tụ điểm giao thông ùn tắc trong đô thị:
* K-Means sẽ ép buộc cả bản đồ thành $`K`$ hình tròn đều nhau, khiến các cung đường cao tốc dài ngoằn ngoèo bị cắt xén vô lý.
* DBSCAN tự động "lan truyền" theo dọc các cung đường có mật độ xe cộ đông đúc, nhận diện chính xác các tuyến đường kẹt xe hình dạng bất kỳ, đồng thời tự động loại bỏ các tín hiệu xe chạy rải rác ở ngoại ô làm **Nhiễu (Noise)**.

---

### Cơ sở Toán học của Thuật toán DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise - Ester et al., 1996) hoạt động dựa trên hai siêu tham số hình học cốt lõi:
1. **$`\varepsilon`$ (Epsilon):** Bán kính lân cận hình cầu xung quanh một điểm.
2. **$`\text{MinPts}`$ (Minimum Samples):** Số lượng điểm tối thiểu bắt buộc phải có bên trong hình cầu bán kính $`\varepsilon`$ để vùng đó được coi là "đậm đặc".

#### 1. Phân loại Ba Loại Điểm trong Không gian
Với mỗi điểm dữ liệu $`p \in X`$, ta xác định vùng lân cận $`\varepsilon`$ của nó:

$$
N_\varepsilon(p) = \lbrace q \in X \mid \|p - q\| \le \varepsilon \rbrace
$$

Dựa vào lực lượng của tập $`N_\varepsilon(p)`$, không gian được phân loại thành 3 loại điểm riêng biệt:

```text
       Core Point (Điểm lõi)           Border Point (Điểm biên)         Noise (Điểm nhiễu)
            .   .   .                              .
          .   ●   .                              .   ●                           ●
        .   .   .   .                              .
     |N_eps(p)| >= MinPts              Nằm trong vùng của Core            |N_eps(p)| < MinPts
   (Tự khởi tạo một cụm mới)         nhưng không đủ điểm làm Core       và không chạm Core nào
```

1. **Điểm lõi (Core Point):** Điểm $`p`$ có ít nhất $`\text{MinPts}`$ điểm nằm trong bán kính $`\varepsilon`$:

$$
\lvert N_\varepsilon(p) \rvert \ge \text{MinPts}
$$

2. **Điểm biên (Border Point):** Điểm $`p`$ không thỏa mãn điều kiện làm điểm lõi ($\lvert N_\varepsilon(p) \rvert < \text{MinPts}$), nhưng lại nằm trong vùng lân cận $`\varepsilon`$ của ít nhất một điểm lõi khác.
3. **Điểm nhiễu (Noise / Outlier):** Điểm không phải là điểm lõi cũng không thuộc vùng lân cận của bất kỳ điểm lõi nào. DBSCAN gán nhãn cho các điểm này là $`-1`$.

---

#### 2. Các Khái niệm Khả đạt theo Mật độ (Density Reachability)

Để liên kết các điểm thành một cụm hoàn chỉnh, DBSCAN xây dựng chuỗi quan hệ topo toán học:

* **Trực tiếp khả đạt theo mật độ (Directly Density-Reachable):** Điểm $`q`$ trực tiếp khả đạt từ điểm $`p`$ nếu $`p`$ là điểm lõi và $`q \in N_\varepsilon(p)`$.
* **Khả đạt theo mật độ (Density-Reachable):** Điểm $`q`$ khả đạt từ điểm $`p`$ nếu tồn tại một chuỗi các điểm trung gian $`p_1, p_2, \dots, p_m`$ (với $`p_1 = p`$ và $`p_m = q`$) sao cho mỗi $`p_{i+1}`$ đều trực tiếp khả đạt từ điểm lõi $`p_i`$ (tính chất bắc cầu).
* **Kết nối theo mật độ (Density-Connected):** Hai điểm $`p`$ và $`q`$ được gọi là kết nối theo mật độ nếu cùng tồn tại một điểm lõi thứ ba $`o`$ sao cho cả $`p`$ và $`q`$ đều khả đạt theo mật độ từ $`o`$.

$$
\text{Cụm } C \text{ trong DBSCAN} = \lbrace q \in X \mid p \in C \text{ và } q \text{ kết nối mật độ với } p \rbrace
$$

---

### Quy trình Hoạt động của Thuật toán DBSCAN

Thuật toán duyệt qua dữ liệu tương tự như duyệt đồ thị theo chiều rộng (BFS / Region Growing):

```text
[Bắt đầu duyệt qua từng điểm p trong X]
               │
               ▼
   [p đã được gán cụm hoặc kiểm tra?] ── Có ──> [Bỏ qua, xét điểm tiếp theo]
               │ Không
               ▼
   [Tìm tập lân cận N_eps(p)]
               │
       ┌───────┴───────┐
       ▼               ▼
|N_eps(p)| < MinPts   |N_eps(p)| >= MinPts
  (Gán tạm thời: Noise)    (p là Core Point -> MỞ CỤM MỚI C)
                               │
                               ▼
               [Cho toàn bộ N_eps(p) vào hàng đợi]
                               │
                               ▼
               [Lan truyền mở rộng cụm: duyệt hàng đợi,
                nếu gặp Core mới -> nhập thêm láng giềng]
```

1. Duyệt qua từng điểm $`p \in X`$. Nếu $`p`$ đã được xử lý thì bỏ qua.
2. Tìm tập lân cận $`N_\varepsilon(p)`$.
3. Nếu $`\lvert N_\varepsilon(p) \rvert < \text{MinPts}`$, đánh dấu tạm thời $`p`$ là **Nhiễu (Noise)**.
4. Nếu $`\lvert N_\varepsilon(p) \rvert \ge \text{MinPts}`$, $`p`$ là điểm lõi: Khởi tạo một cụm mới $`C`$ chứa $`p`$. Đưa toàn bộ các điểm trong $`N_\varepsilon(p)`$ vào danh sách chờ mở rộng.
5. Lặp lại việc mở rộng vùng: Với mỗi điểm $`q`$ trong danh sách chờ, nếu $`q`$ là điểm lõi, gộp toàn bộ $`N_\varepsilon(q)`$ vào danh sách. Điểm nào trước đó bị gán tạm thời là Noise nay được chuyển thành Border Point của cụm $`C`$.
6. Quá trình kết thúc khi toàn bộ dữ liệu đã được phân loại vào các cụm hoặc xác nhận là Noise.

---

### OPTICS: Giải pháp cho Dữ liệu có Mật độ Biến thiên

#### Hạn chế lớn nhất của DBSCAN
DBSCAN sử dụng một giá trị $`\varepsilon`$ duy nhất cho toàn bộ không gian. Nếu dữ liệu có **các cụm mang mật độ rất khác nhau** (ví dụ: Cụm A cực kỳ dày đặc nhưng Cụm B lại thưa thớt hơn):
* Nếu chọn $`\varepsilon`$ nhỏ: Cụm B bị coi hoàn toàn là Nhiễu (Noise).
* Nếu chọn $`\varepsilon`$ lớn: Cụm A và Cụm B bị dính chùm vào nhau thành một cụm khổng lồ!

#### Cơ chế của OPTICS (Ordering Points To Identify the Clustering Structure)
OPTICS (Ankerst et al., 1999) không cố định một giá trị $`\varepsilon`$ duy nhất, mà xây dựng một **biểu đồ sắp xếp thứ tự các điểm dữ liệu** dựa trên hai khái niệm toán học:

1. **Khoảng cách lõi (Core Distance):** Khoảng cách nhỏ nhất từ $`p`$ tới láng giềng thứ $`\text{MinPts}`$ của nó:

$$
\text{Core-dist}_\varepsilon(p) = \begin{cases} \text{UNDEFINED} & \text{nếu } \lvert N_\varepsilon(p) \rvert < \text{MinPts} \\ \text{Khoảng cách tới láng giềng thứ } \text{MinPts} & \text{ngược lại} \end{cases}
$$

2. **Khoảng cách khả đạt (Reachability Distance):** Khoảng cách khả đạt từ một điểm lõi $`o`$ tới điểm $`p`$:

$$
\text{Reachability-dist}_\varepsilon(p, o) = \begin{cases} \text{UNDEFINED} & \text{nếu } o \text{ không phải Core Point} \\ \max\lbrace \text{Core-dist}_\varepsilon(o), \|p - o\| \rbrace & \text{ngược lại} \end{cases}
$$

```text
BIỂU ĐỒ KHẢ ĐẠT CỦA OPTICS (REACHABILITY PLOT)
Reachability
Distance
   │        Đỉnh núi (Ranh giới giữa các cụm / Nhiễu)
   │           ▲                 ▲
   │    ───────│─────────────────│───────  (Ngưỡng cắt Epsilon tùy chọn)
   │     \   /   \             /   \
   │      \_/     \___________/     \__
   │    Cụm 1        Cụm 2             Cụm 3
   └─────────────────────────────────────────> Thứ tự duyệt các điểm (Order)
```

* **Trực giác:** Mỗi "thung lũng" (Valley) trên biểu đồ tương ứng với một cụm mật độ. Người dùng có thể linh hoạt chọn các ngưỡng cắt khác nhau để trích xuất cấu trúc cụm phân cấp từ nông đến sâu.

---

### Bảng So sánh Đối đầu: DBSCAN vs K-Means

```text
+-----------------------+---------------------------------------+--------------------------------------+
| Đặc tính              | K-Means                               | DBSCAN                               |
+-----------------------+---------------------------------------+--------------------------------------+
| 1. Số lượng cụm K     | Phải cung cấp trước cố định           | Tự động phát hiện dựa trên mật độ    |
| 2. Hình dạng cụm      | Chỉ tốt với hình cầu đối xứng         | Tìm được mọi hình dạng uốn lượn bất kỳ|
| 3. Xử lý ngoại lai    | Ép mọi outlier vào một cụm (Rất nhạy) | Tự động phát hiện và gán nhãn Noise  |
| 4. Siêu tham số       | Số cụm K                              | Bán kính eps và MinPts               |
| 5. Độ phức tạp        | O(n * K * p * iter) (Tuyến tính)      | O(n * log n) với kd-tree; O(n^2) tệ nhất|
| 6. Nhược điểm chính   | Không tìm được cụm phi tuyến          | Nhạy cảm khi mật độ các cụm lệch nhau |
+-----------------------+---------------------------------------+--------------------------------------+
```

---

### Hướng dẫn Kỹ thuật Dò tìm Siêu tham số cho DBSCAN

Việc chọn $`\varepsilon`$ và $`\text{MinPts}`$ không được làm ngẫu nhiên:

1. **Cách chọn $`\text{MinPts}`$:**
   * Quy tắc ngón tay cái: $`\text{MinPts} \ge p + 1`$ (với $`p`$ là số chiều của dữ liệu).
   * Trong dữ liệu thực tế có nhiều nhiễu, nên chọn $`\text{MinPts} = 2p`$. Với tập dữ liệu lớn ($`n > 10^5`$), có thể tăng lên $`\text{MinPts} = 10 \sim 50`$.
2. **Cách chọn $`\varepsilon`$ thông qua Đồ thị K-Distance (K-Distance Plot):**
   * Đặt $`k = \text{MinPts} - 1`$.
   * Với mỗi điểm trong tập dữ liệu, tính khoảng cách tới láng giềng gần thứ $`k`$ của nó (dùng `NearestNeighbors(n_neighbors=k)`).
   * Sắp xếp các khoảng cách này theo thứ tự tăng dần và vẽ đồ thị.
   * Điểm **"gập đầu gối" (Knee/Elbow Point)** — nơi khoảng cách bắt đầu tăng vọt dựng đứng — chính là giá trị $`\varepsilon`$ tối ưu!

---

### Mã nguồn Thực hành với Scikit-learn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. Chuẩn hóa dữ liệu (bắt buộc)
X_scaled = StandardScaler().fit_transform(X)

# 2. Dò tìm Epsilon qua K-Distance Plot
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sắp xếp khoảng cách tới láng giềng thứ k
k_distances = np.sort(distances[:, k - 1], axis=0)

# 3. Huấn luyện DBSCAN
dbscan = DBSCAN(eps=0.45, min_samples=k)
labels = dbscan.fit_predict(X_scaled)

# 4. Phân tích kết quả
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"Số lượng cụm phát hiện được: {n_clusters}")
print(f"Số lượng điểm nhiễu (Noise): {n_noise} ({n_noise / len(labels) * 100:.2f}%)")
```
