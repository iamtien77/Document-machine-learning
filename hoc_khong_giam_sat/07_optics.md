# OPTICS — Mở rộng mật độ biến thiên

## Purpose

Giải quyết bài toán gom cụm mật độ khi tập dữ liệu chứa **các cụm có mật độ rất khác nhau** (cụm này cực kỳ đậm đặc, cụm kia lại thưa thớt hơn). Trong khi DBSCAN bắt buộc phải cố định một bán kính $`\varepsilon`$ duy nhất cho toàn bộ không gian (khiến cụm thưa bị coi là nhiễu, hoặc các cụm dày bị dính chùm vào nhau), OPTICS (Ordering Points To Identify the Clustering Structure) vượt trội nhờ cơ chế không cố định $`\varepsilon`$: thuật toán sắp xếp tuyến tính toàn bộ các điểm dữ liệu để sinh ra **Biểu đồ khả đạt (Reachability Plot)**, cho phép người dùng quan sát và trích xuất cấu trúc phân cụm ở mọi cấp độ mật độ. [1][2]

**Ví dụ dẫn nhập:** Trong phân tích mật độ dân cư và cơ sở hạ tầng từ dữ liệu vệ tinh viễn thám:
* Khu vực trung tâm thành phố (Quận 1) có mật độ tòa nhà cực kỳ dày đặc.
* Khu vực ngoại thành nông thôn có mật độ nhà ở thưa thớt hơn gấp 10 lần nhưng vẫn tạo thành các làng xã rõ rệt.
* DBSCAN thất bại vì nếu chọn $`\varepsilon`$ theo khu vực nông thôn thì cả thành phố bị gộp thành một khối dính chùm; nếu chọn theo thành phố thì cả vùng nông thôn bị xóa sổ thành Nhiễu (Noise).
* OPTICS nhận diện hoàn hảo cả hai cấu trúc nhờ biểu đồ mật độ khả đạt đa tầng.

Ký hiệu: tập dữ liệu $`X = \lbrace x_1, \dots, x_n \rbrace`$, số láng giềng tối thiểu $`\text{MinPts} \in \mathbb{N}^+`$ và bán kính tối đa $`\varepsilon_{\max}`$. Mô hình xuất ra mảng thứ tự sắp xếp các điểm $`\text{ordering}`$ và mảng khoảng cách khả đạt tương ứng $`\text{reachability}`$. [1]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần toán học về khoảng cách lõi và khoảng cách khả đạt bên dưới giải thích cơ sở giải thuật; có thể quay lại sau khi hiểu cách đọc biểu đồ Reachability Plot.

### Bản chất Hình học: Biểu đồ Khả đạt (Reachability Plot)

OPTICS biến bài toán phân cụm không gian đa chiều $`\mathbb{R}^p`$ phức tạp thành bài toán **phân tích tín hiệu 1 chiều trực quan**:

```text
BIỂU ĐỒ KHẢ ĐẠT CỦA OPTICS (REACHABILITY PLOT)
Khoảng cách
Reachability
   ▲
   │        ĐỈNH NÚI (Điểm nhiễu / Ranh giới giữa các cụm)
   │             ▲                        ▲
   │     \      / \                      / \      /
   │      \    /   \                    /   \    /
   │       \__/     \__________________/     \__/
   │      THUNG LŨNG 1    THUNG LŨNG 2 (Sâu & Rộng)  THUNG LŨNG 3
   │      (Cụm dày đặc)       (Cụm mật độ vừa)        (Cụm nhỏ)
   └──────────────────────────────────────────────────────────> Thứ tự duyệt (Order)
```

* Mỗi **"Thung lũng" (Valley)** trên đồ thị đại diện cho một cụm mật độ tự nhiên: Đáy thung lũng càng trũng sâu chứng tỏ cụm đó có mật độ điểm càng đậm đặc.
* Các **"Đỉnh núi nhọn" (Peaks)** ngăn cách giữa các thung lũng đại diện cho các vùng thưa thớt hoặc các điểm ngoại lai (Noise).

---

## Input

- Ma trận đặc trưng số $`X \in \mathbb{R}^{n \times p}`$ đã được chuẩn hóa thang đo.
- `min_samples` ($`\text{MinPts} \in \mathbb{N}^+`$): Số mẫu tối thiểu để xác định một vùng đậm đặc (mặc định là 5).
- `max_eps` ($`\varepsilon_{\max}`$): Bán kính tối đa để tìm kiếm láng giềng (mặc định là $`\infty`$ để quét toàn bộ không gian; có thể đặt một giá trị hữu hạn để giảm chi phí tính toán).
- `cluster_method`: Phương pháp trích xuất cụm từ biểu đồ khả đạt:
  * `'xi'` ($\xi$ - trích xuất dựa trên độ dốc thung lũng, tự động tìm cụm phân cấp).
  * `'dbscan'` (trích xuất bằng một ngưỡng cắt ngang $`\varepsilon`$ cố định tương tự DBSCAN).
- `xi`: Ngưỡng độ dốc tương đối biểu thị sự thay đổi mật độ khi dùng phương pháp `cluster_method='xi'` (thường từ $`0.01`$ đến $`0.1`$).

---

## Output

1. **Thứ tự duyệt các điểm dữ liệu (`ordering_`):** Mảng thứ tự các chỉ số hàng trong $`X`$ được sắp xếp sao cho các điểm gần nhau về mật độ nằm liền kề nhau: $`\text{ordering} \in \mathbb{N}^n`$.
2. **Khoảng cách khả đạt (`reachability_`):** Mảng giá trị khoảng cách khả đạt tương ứng với từng điểm theo thứ tự duyệt: $`\text{reachability} \in \mathbb{R}^n`$.
3. **Khoảng cách lõi (`core_distances_`):** Mảng khoảng cách lõi của từng điểm: $`\text{core\_distances} \in \mathbb{R}^n`$.
4. **Nhãn phân cụm (`labels_`):** Vector nhãn $`\hat{y} \in \lbrace -1, 0, 1, \dots, C-1 \rbrace^n`$ sau khi trích xuất cụm.

---

## Evaluation metrics

### 1. Phân tích Độ sâu Thung lũng trên Biểu đồ Khả đạt
Chất lượng cấu trúc phân cụm của OPTICS được đánh giá trực quan thông qua độ tương phản của biểu đồ:
* Cụm có chất lượng cao khi đáy thung lũng bằng phẳng, nằm ở mức khoảng cách thấp và có sườn dốc đứng rõ rệt.
* Nếu biểu đồ phẳng lì không có thung lũng rõ ràng, chứng tỏ dữ liệu phân bố đều ngẫu nhiên (Uniform noise), không có cấu trúc cụm mật độ.

### 2. Silhouette Score Hiệu chỉnh (Không tính Noise)
Tương tự DBSCAN, loại bỏ các điểm nhãn $`-1`$ trước khi tính toán điểm Silhouette để đo lường độ phân tách giữa các thung lũng trích xuất được.

---

## How it works

Cốt lõi toán học của OPTICS dựa trên hai định nghĩa khoảng cách mở rộng:

### 1. Hai Khái niệm Khoảng cách Nền tảng
Cho bán kính cực đại $`\varepsilon`$ và số láng giềng $`\text{MinPts}`$:

* **Khoảng cách lõi (Core Distance):** Khoảng cách nhỏ nhất từ điểm $`p`$ tới láng giềng thứ $`\text{MinPts}`$ gần nó nhất:

$$
\text{Core-dist}_{\varepsilon, \text{MinPts}}(p) = \begin{cases} \text{UNDEFINED} & \text{nếu } \lvert N_\varepsilon(p) \rvert < \text{MinPts} \\ \text{Khoảng cách tới láng giềng thứ } \text{MinPts} & \text{ngược lại} \end{cases}
$$

* **Khoảng cách khả đạt (Reachability Distance):** Khoảng cách khả đạt từ một điểm lõi $`o`$ tới điểm $`p`$:

$$
\text{Reachability-dist}_{\varepsilon, \text{MinPts}}(p, o) = \begin{cases} \text{UNDEFINED} & \text{nếu } o \text{ không phải là Core Point} \\ \max\lbrace \text{Core-dist}(o), D(p, o) \rbrace & \text{ngược lại} \end{cases}
$$

* *Trực giác:* Nếu điểm $`p`$ nằm rất gần $`o`$ (bên trong hình cầu bán kính Core-dist), khoảng cách khả đạt bị "ghìm" lại ở mức Core-dist($o$). Nếu $`p`$ nằm xa hơn, khoảng cách khả đạt chính là khoảng cách thực tế $`D(p, o)`$.

```text
THUẬT TOÁN OPTICS:
1. Đánh dấu tất cả các điểm là CHƯA_XỬ_LÝ.
2. Khởi tạo danh sách thứ tự kết quả Order_List = [] và Hàng đợi ưu tiên Priority_Queue.
3. Với mỗi điểm chưa xử lý p trong X:
   a. Tính Core-dist(p). Đánh dấu p là ĐÃ_XỬ_LÝ, thêm p vào Order_List.
   b. Nếu p là Core Point:
      - Cập nhật Reachability-dist cho tất cả các láng giềng q trong N_eps(p).
      - Đưa/cập nhật các láng giềng vào Priority_Queue (ưu tiên điểm có Reachability nhỏ nhất).
      - TRONG KHI Priority_Queue không rỗng:
        * Lấy điểm có Reachability nhỏ nhất ra: q_min.
        * Tính Core-dist(q_min). Đánh dấu đã xử lý, thêm vào Order_List.
        * Nếu q_min là Core Point: Tiếp tục cập nhật láng giềng của q_min vào Priority_Queue.
4. KẾT THÚC: Thu được mảng thứ tự duyệt và mảng Reachability hoàn chỉnh.
```

---

## Data used

Bảng đối chiếu đặc điểm thực nghiệm giữa dữ liệu bài tập và dữ liệu thực tế khi chạy OPTICS:

| Khía cạnh | Dữ liệu Bài tập / Slide môn học | Dữ liệu Thực tế (Production) |
|---|---|---|
| **Độ đồng nhất mật độ** | Thường dùng dữ liệu mật độ đơn (chạy DBSCAN hay OPTICS cho kết quả giống hệt nhau). | Mật độ biến thiên phức tạp (ví dụ: trung tâm giao dịch ngân hàng vs chi nhánh vùng sâu). |
| **Trích xuất cụm** | Dùng phương pháp cắt ngưỡng ngang `cluster_method='dbscan'`. | Dùng phương pháp trích xuất theo độ dốc `cluster_method='xi'` để bắt các cụm phân cấp lồng nhau. |
| **Độ phức tạp tính toán** | $`n < 1.000`$, chạy trong 0.1 giây. | $`n > 50.000`$, hàng đợi ưu tiên cập nhật liên tục tốn nhiều CPU; cần giới hạn `max_eps`. |
| **Ứng dụng thực tế** | Minh họa biểu đồ thung lũng 2D. | Phát hiện gian lận đa tầng, phân cụm tín hiệu địa chấn, phân tích cụm viễn thông. |

---

## Design

### Mã nguồn Vẽ Biểu đồ Reachability Plot & Trích xuất Cụm với Scikit-learn

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import StandardScaler

# 1. Chuẩn hóa dữ liệu
X_scaled = StandardScaler().fit_transform(X)

# 2. Huấn luyện thuật toán OPTICS
optics_model = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
optics_model.fit(X_scaled)

# 3. Trích xuất nhãn theo phương pháp xi và phương pháp cắt ngang DBSCAN (eps = 0.5)
labels_xi = optics_model.labels_
labels_dbscan = cluster_optics_dbscan(
    reachability=optics_model.reachability_,
    core_distances=optics_model.core_distances_,
    ordering=optics_model.ordering_,
    eps=0.5
)

# 4. Trực quan hóa Biểu đồ Khả đạt (Reachability Plot)
space = np.arange(len(X_scaled))
reachability = optics_model.reachability_[optics_model.ordering_]

plt.figure(figsize=(10, 4))
plt.plot(space, reachability, 'b-', alpha=0.7, label='Khoảng cách khả đạt')
plt.axhline(y=0.5, color='r', linestyle='--', label='Ngưỡng cắt Epsilon = 0.5')
plt.title("Biểu đồ Khả đạt OPTICS (Reachability Plot)")
plt.xlabel("Chỉ số thứ tự sắp xếp (Ordering Index)")
plt.ylabel("Reachability Distance")
plt.legend()
plt.grid(True)
plt.show()

print(f"Số cụm phát hiện (theo xi): {len(set(labels_xi)) - (1 if -1 in labels_xi else 0)}")
print(f"Số cụm phát hiện (theo dbscan eps=0.5): {len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)}")
```

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **OPTICS** | Sắp xếp điểm nhận diện cụm | Thuật toán gom cụm mật độ biến thiên dựa trên việc sắp xếp thứ tự duyệt điểm không gian. |
| **Core Distance** | Khoảng cách lõi | Khoảng cách nhỏ nhất từ một điểm tới láng giềng thứ $`\text{MinPts}`$ của nó. |
| **Reachability Distance** | Khoảng cách khả đạt | Độ đo cực đại giữa khoảng cách lõi của điểm gốc và khoảng cách thực tế tới điểm đích. |
| **Reachability Plot** | Biểu đồ khả đạt | Đồ thị 2D biểu diễn khoảng cách khả đạt theo thứ tự duyệt; các thung lũng đại diện cho cụm. |
| **Xi Extraction ($\xi$)** | Trích xuất theo độ dốc | Kỹ thuật tự động phát hiện cụm phân cấp dựa trên tỷ lệ sườn dốc suy giảm mật độ của thung lũng. |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning (Trang 14: OPTICS Algorithm), TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Ankerst, M., Breunig, M. M., Kriegel, H. P., & Sander, J. (1999). *OPTICS: Ordering Points To Identify the Clustering Structure*. ACM SIGMOD.
- [3] Schubert, E., & Gertz, M. (2018). *Improving the Cluster Structure Extracted from OPTICS Plots*. EDBT.
- [4] [scikit-learn User Guide: OPTICS](https://scikit-learn.org/stable/modules/clustering.html#optics).

