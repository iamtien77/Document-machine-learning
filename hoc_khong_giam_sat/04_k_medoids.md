# K-Medoids (PAM — Partitioning Around Medoids)

## Purpose

Phân hoạch dữ liệu thành $`K`$ cụm dựa trên các **điểm đại diện thực tế (Medoids)** thay vì trung bình số học ảo như K-Means. Đây là thuật toán gom cụm **kháng ngoại lai (Robust Clustering)** kinh điển nhất: bằng cách tối thiểu hóa tổng khoảng cách tuyệt đối bất kỳ ($`L_1`$, Manhattan, Cosine, Gower) thay vì bình phương khoảng cách Euclid ($`L_2`$), K-Medoids hoàn toàn không bị thao túng bởi các điểm ngoại lai cực đoan (Outliers) hay các thuộc tính có đuôi phân phối dài. [1][2]

**Ví dụ dẫn nhập:** Trong phân tích dữ liệu lâm sàng y tế, mỗi bệnh nhân được mô tả qua hàng chục xét nghiệm sinh hóa phức tạp và có nhiều chỉ số dị biệt do bệnh lý hiếm.
* K-Means sẽ tạo ra một tâm cụm là trung bình số học của các chỉ số — một "bệnh nhân ảo" không hề tồn tại trong thực tế và không thể giải thích về mặt y khoa.
* K-Medoids bắt buộc tâm cụm phải là một **bệnh nhân có thật trong hồ sơ** ($`m_k \in X`$). Bác sĩ có thể dùng trực tiếp hồ sơ của bệnh nhân đại diện này làm ca bệnh tham chiếu tiêu chuẩn (Archetype Patient) cho toàn bộ nhóm bệnh nhân đó.

Ký hiệu: tập dữ liệu $`X = \lbrace x_1, \dots, x_n \rbrace`$, số cụm $`K`$. Mô hình tìm tập hợp con gồm $`K`$ điểm dữ liệu thực tế $`M = \lbrace m_1, \dots, m_K \rbrace \subset X`$ và hàm gán cụm $`c: X \to \lbrace 1, \dots, K \rbrace`$ sao cho tổng khoảng cách từ mỗi điểm tới medoid của nó là nhỏ nhất. [1]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần phân tích thuật toán PAM và CLARA bên dưới giải thích cơ chế tối ưu; có thể quay lại sau khi hiểu cách dùng thư viện `scikit-learn-extra`.

### Bản chất Toán học: Trung vị Đa chiều vs Trung bình Số học

Sự vượt trội của K-Medoids trước ngoại lai bắt nguồn từ sự khác biệt cơ bản giữa **Trung bình (Mean)** và **Trung vị (Median / Medoid)** trong thống kê học:

* **Trong không gian 1 chiều:** Cho tập giá trị $`\lbrace 1, 2, 3, 4, 100 \rbrace`$.
  * Giá trị trung bình là $`\mu = 22`$ (bị kéo lệch nghiêm trọng bởi số $`100`$).
  * Giá trị trung vị là $`m = 3`$ (hoàn toàn miễn nhiễm trước sự xuất hiện của số $`100`$).
* **Trong không gian đa chiều $`\mathbb{R}^p`$:** Khái niệm "trung vị đa chiều" rất khó định nghĩa giải tích. Do đó, K-Medoids định nghĩa **Medoid** của cụm $`C_k`$ là điểm $`m_k \in C_k`$ làm cực tiểu hóa tổng khoảng cách tới mọi điểm khác trong cùng cụm:

$$
m_k = \mathop{\mathrm{arg\,min}}_{y \in C_k} \sum_{x_i \in C_k} D(x_i, y)
$$

Điểm $`m_k`$ được bảo đảm luôn là một mẫu dữ liệu hợp lệ, nằm ở vùng mật độ trung tâm nhất của cụm.

---

## Input

- Ma trận đặc trưng số $`X \in \mathbb{R}^{n \times p}`$ hoặc **Ma trận độ không tương đồng (Dissimilarity / Distance Matrix)** $`D \in \mathbb{R}^{n \times n}`$, trong đó $`D_{ij} = D(x_i, x_j)`$.
- Số lượng cụm mong muốn $`K \in \mathbb{N}^{+}`$ ($`1 < K \ll n`$).
- Hàm khoảng cách tùy chọn: `metric='manhattan'`, `metric='cosine'`, `metric='euclidean'` hoặc ma trận khoảng cách tùy biến (như khoảng cách Gower cho dữ liệu hỗn hợp).
- Phương pháp thuật toán: `pam` (chính xác), `alternate` (nhanh tương tự Lloyd), hoặc `clara` cho tập dữ liệu lớn.

### Khả năng Chấp nhận Ma trận Khoảng cách Bất kỳ (Arbitrary Metric Input)
Khác với K-Means (bắt buộc phải có tọa độ không gian thực để tính trung bình), K-Medoids **chỉ cần biết khoảng cách giữa các cặp điểm ($`D_{ij}`$)** mà không cần biết tọa độ thực của chúng. Điều này cho phép K-Medoids phân cụm trên các cấu trúc dữ liệu phi Euclid:
* Phân cụm đồ thị mạng xã hội (dùng khoảng cách đường đi ngắn nhất - Shortest Path).
* Phân cụm chuỗi DNA/Protein (dùng khoảng cách Levenshtein hoặc Needleman-Wunsch).
* Phân cụm văn bản (dùng khoảng cách Cosine hoặc Jaccard).

---

## Output

1. **Nhãn phân cụm (Cluster Labels):** Vector nhãn $`\hat{y} = (\hat{y}_1, \dots, \hat{y}_n) \in \lbrace 0, 1, \dots, K-1 \rbrace^n`$.
2. **Chỉ số các Medoid đại diện (Medoid Indices):** Mảng chỉ số các dòng trong $`X`$ được chọn làm tâm: $`\text{medoid\_indices\_} = [i_1, i_2, \dots, i_K]`$.
3. **Tọa độ các Medoid thực tế:** $`M = [x_{i_1}, x_{i_2}, \dots, x_{i_K}]^\top \in \mathbb{R}^{K \times p}`$.
4. **Tổng chi phí khoảng cách (Total Dissimilarity / Cost):** Tổng khoảng cách từ mọi điểm tới medoid tương ứng.

---

## Evaluation metrics

Chất lượng nghiệm phân cụm của K-Medoids được đánh giá thông qua:

### 1. Tổng Chi phí Tuyệt đối (Absolute Deviation Cost)
Hàm mục tiêu trực tiếp mà K-Medoids tối thiểu hóa:

$$
\mathcal{J}_{\text{medoid}} = \sum_{k=1}^K \sum_{x_i \in C_k} D(x_i, m_k)
$$

### 2. Hệ số Silhouette trung bình (Mean Silhouette Score)
Được tính toán trực tiếp trên ma trận khoảng cách $D$:

$$
s(x_i) = \frac{b(x_i) - a(x_i)}{\max\lbrace a(x_i), b(x_i) \rbrace}
$$

* Vì K-Medoids tối ưu hóa trên cấu trúc khoảng cách tùy ý, Silhouette Score là thước đo đánh giá nội tại phù hợp và khách quan nhất.

---

## How it works

Thuật toán chuẩn mực nhất của K-Medoids là **PAM (Partitioning Around Medoids - Kaufman & Rousseeuw, 1990)**, gồm hai giai đoạn:

```text
┌────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 1: KHỞI TẠO (BUILD Step)                    │
│  Chọn tham lam K medoid ban đầu ở vị trí trung tâm     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 2: HOÁN ĐỔI (SWAP Step)                     │
│  Xét mọi cặp (m_i, x_h) giữa medoid và non-medoid:     │
│  • Tính biến thiên chi phí: Delta T_ih                 │
│  • Nếu Delta T_ih < 0 lớn nhất -> Thực hiện hoán đổi!  │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
     Có phép hoán đổi                Không còn cặp nào
      làm giảm chi phí                làm giảm chi phí
(Tiếp tục lặp vòng SWAP)             (Hội tụ -> Dừng)
```

1. **Giai đoạn BUILD (Khởi tạo):**
   * Chọn medoid đầu tiên là điểm làm nhỏ nhất tổng khoảng cách tới mọi điểm khác.
   * Lần lượt chọn các medoid tiếp theo theo cơ chế tham lam (Greedy): chọn điểm làm giảm tổng chi phí khoảng cách nhiều nhất.
2. **Giai đoạn SWAP (Tối ưu hoán đổi):**
   * Với mỗi medoid hiện tại $`m_i`$ và mỗi điểm không phải medoid $`x_h`$:
     * Thử thay thế $`m_i`$ bằng $`x_h`$.
     * Tính toán sự thay đổi tổng chi phí khoảng cách: $`\Delta T_{ih} = \mathcal{J}_{\text{mới}} - \mathcal{J}_{\text{cũ}}`$.
   * Nếu tồn tại cặp $`(m_i, x_h)`$ có $`\Delta T_{ih} < 0`$ nhỏ nhất, thực hiện hoán đổi chính thức: $`x_h`$ trở thành medoid mới thay cho $`m_i`$.
   * Lặp lại giai đoạn SWAP cho đến khi không còn bất kỳ phép hoán đổi nào làm giảm chi phí ($`\Delta T_{ih} \ge 0`$ với mọi cặp).

### Độ Phức tạp Tính toán & Các Thuật toán Cải tiến (CLARA, CLARANS)

* **Điểm yếu của PAM:** Trong mỗi vòng lặp SWAP, thuật toán phải duyệt qua $`K(n - K)`$ cặp hoán đổi, mỗi phép tính chi phí tốn $`\mathcal{O}(n - K)`$. Tổng độ phức tạp mỗi vòng lặp là $`\mathcal{O}(K (n - K)^2)`$. Khi $`n > 10.000`$, PAM chạy cực kỳ chậm và không khả thi trên thực tế.
* **CLARA (Clustering Large Applications):** Rút ngẫu nhiên một tập mẫu con nhỏ ($`n' \ll n`$, thường là $`n' = 40 + 2K`$), chạy PAM hoàn chỉnh trên tập mẫu này để tìm medoid, sau đó gán nhãn cho toàn bộ $`n`$ điểm còn lại. Lặp lại quá trình trên 5 lần lấy mẫu độc lập để chọn nghiệm tốt nhất.
* **CLARANS (Randomized Search):** Không xét toàn bộ $`K(n - K)`$ cặp hoán đổi mà chỉ lấy mẫu ngẫu nhiên một số lượng cố định các cặp hoán đổi (Randomized Neighborhood Exploration), cân bằng hoàn hảo giữa tốc độ và chất lượng cụm.

---

## Data used

Bảng đối chiếu hành vi thực nghiệm giữa dữ liệu bài tập và dữ liệu thực tế khi chạy K-Medoids:

| Đặc điểm thực nghiệm | Dữ liệu Bài tập / Kiểm thử | Dữ liệu Thực tế (Production / Lâm sàng) |
|---|---|---|
| **Sự hiện diện của Ngoại lai**| Hầu như không có ngoại lai (K-Means và K-Medoids cho kết quả y hệt nhau). | Chứa từ 1% đến 5% các điểm dị biệt cực đoan; K-Means bị kéo lệch, K-Medoids đứng vững. |
| **Loại khoảng cách** | Khoảng cách Euclid chuẩn ($`L_2`$). | Khoảng cách Manhattan ($`L_1`$), Cosine hoặc ma trận bất đối xứng phức tạp. |
| **Kích thước dữ liệu ($`n`$)**| Nhỏ ($`n < 1.000`$), chạy PAM hội tụ trong tích tắc. | Lớn ($`n > 50.000`$), PAM quá tải bộ nhớ; bắt buộc phải dùng CLARA hoặc thuật toán Alternate. |
| **Tính giải thích của tâm** | Tâm cụm chỉ là con số đại diện không quan trọng. | Tâm cụm bắt buộc phải là một hồ sơ cá thể có thật để làm ca bệnh hoặc khách hàng chuẩn mẫu. |

---

## Design

### Cài đặt Chuẩn mực với `scikit-learn-extra`

Thư viện chuẩn `scikit-learn` không tích hợp sẵn PAM, thuật toán được cung cấp qua gói mở rộng chính thức `scikit-learn-extra`:

```python
# Cài đặt: pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

# 1. Pipeline xử lý bền vững: RobustScaler -> K-Medoids với khoảng cách Manhattan
kmedoids_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('kmedoids', KMedoids(
        n_clusters=4,
        metric='manhattan',      # Dùng khoảng cách L1 kháng ngoại lai mạnh mẽ
        method='pam',            # 'pam' cho dữ liệu nhỏ; 'alternate' cho dữ liệu lớn
        init='k-medoids++',      # Khởi tạo thông minh tương tự k-means++
        max_iter=300,
        random_state=42
    ))
])

# 2. Huấn luyện
cluster_labels = kmedoids_pipeline.fit_predict(X)

# 3. Trích xuất medoids thực tế
model = kmedoids_pipeline.named_steps['kmedoids']
medoid_indices = model.medoid_indices_
actual_medoids = X[medoid_indices]

print(f"Chỉ số các hàng đại diện làm Medoids: {medoid_indices}")
print(f"Tổng chi phí sai số tuyệt đối (Inertia): {model.inertia_:.2f}")
```

### So sánh Đối đầu Toàn diện: K-Means vs K-Medoids

```text
+-----------------------+---------------------------------------+---------------------------------------+
| Tiêu chí              | K-Means                               | K-Medoids (PAM)                       |
+-----------------------+---------------------------------------+---------------------------------------+
| 1. Tâm đại diện       | Tọa độ trung bình ảo (Mean vector)    | Điểm dữ liệu thực tế (Medoid sample)  |
| 2. Hàm mục tiêu       | Cực tiểu hóa tổng bình phương L2      | Cực tiểu hóa tổng khoảng cách L1/D    |
| 3. Độ nhạy ngoại lai  | Cực kỳ nhạy cảm (1 outlier phá hỏng)  | Rất bền vững (Miễn nhiễm ngoại lai)   |
| 4. Hàm khoảng cách    | Bắt buộc dùng Euclid                  | Tùy ý: Manhattan, Cosine, Gower, v.v. |
| 5. Độ phức tạp thời   | O(n * K * p * iter) (Tuyến tính, nhanh)| O(K * (n - K)^2) (Bậc hai, chậm)      |
| 6. Khi nào nên dùng   | Dữ liệu sạch, rất lớn, cụm hình cầu   | Dữ liệu có nhiều ngoại lai, cần giải thích|
+-----------------------+---------------------------------------+---------------------------------------+
```

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **K-Medoids** | Gom cụm K-Medoids | Thuật toán phân hoạch chọn tâm là các điểm dữ liệu thực tế làm nhỏ nhất tổng khoảng cách. |
| **Medoid** | Điểm trung vị đa chiều | Điểm trong cụm có tổng khoảng cách tới mọi điểm khác trong cụm là nhỏ nhất. |
| **PAM** | Phân hoạch quanh Medoid | Thuật toán kinh điển 2 giai đoạn (BUILD & SWAP) của Kaufman & Rousseeuw. |
| **CLARA** | Gom cụm dữ liệu lớn | Biến thể của PAM dựa trên việc lấy mẫu con ngẫu nhiên nhiều lần để giảm chi phí tính toán. |
| **Dissimilarity Matrix** | Ma trận không tương đồng | Ma trận vuông $`n \times n`$ lưu khoảng cách giữa mọi cặp quan sát trong tập dữ liệu. |
| **Robust Clustering** | Gom cụm bền vững | Khả năng duy trì cấu trúc phân cụm ổn định khi dữ liệu bị ô nhiễm bởi ngoại lai. |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning (Trang 9-10: K-medoids Algorithm), TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis* (Chapter 2: Partitioning Around Medoids). John Wiley & Sons.
- [3] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (Chapter 14.3.10: K-medoids). Springer.
- [4] [scikit-learn-extra documentation: KMedoids](https://scikit-learn-extra.readthedocs.io/en/stable/modules/clustering.html#k-medoids).
- [5] Ng, R. T., & Han, J. (2002). *CLARANS: A method for clustering objects for spatial data mining*. IEEE TKDE.

