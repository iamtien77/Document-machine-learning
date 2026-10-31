# k-Nearest Neighbors (kNN)

## Purpose

Dự đoán từ các mẫu train gần nhất; baseline phi tuyến dễ hiểu cho classification hoặc regression. [1]

### Định lý Cover & Hart (1967) & Sức mạnh của Mô hình Phi tham số

k-Nearest Neighbors (kNN) là đại diện tiêu biểu nhất của trường phái **Học lười (Lazy Learning / Instance-based Learning)**. Khác với các mô hình tham số (như Hồi quy Tuyến tính hay Logistic) vốn cố gắng ép dữ liệu vào một phương trình toán học cố định trong quá trình huấn luyện, kNN không học bất kỳ một trọng số nào trong giai đoạn `fit`. Mô hình lưu trữ toàn bộ dữ liệu mẫu và chỉ thực sự tính toán khi có một truy vấn dự đoán mới xuất hiện.

Năm 1967, hai nhà khoa học Thomas Cover và Peter Hart đã công bố một trong những định lý nền tảng nhất của lý thuyết nhận dạng mẫu:
* Gọi $`R^{\ast}`$ là **Sai số Tối ưu Bayes (Bayes Error Rate)** — giới hạn sai số lý thuyết tối thiểu mà không một thuật toán nào trên thế giới có thể vượt qua khi biết trước phân phối xác suất thực tế.
* Gọi $`R_{1\text{-NN}}`$ là sai số tiệm cận của thuật toán 1-Láng giềng gần nhất ($`k=1`$) khi kích thước tập huấn luyện tiến ra vô cùng ($`n \to \infty`$).
* **Định lý Cover & Hart:**

$$
R^{\ast} \le R_{1\text{-NN}} \le 2 R^{\ast} (1 - R^{\ast}) \le 2 R^{\ast}
$$

* *Ý nghĩa khoa học sâu sắc:* Ngay cả với phiên bản đơn giản nhất ($`k=1`$), sai số của thuật toán láng giềng gần nhất không bao giờ vượt quá **hai lần sai số tối ưu Bayes**! Đây là minh chứng toán học hùng hồn cho thấy các phương pháp phi tham số cục bộ có khả năng biểu diễn vô cùng mạnh mẽ mà không cần giả định trước bất kỳ dạng phân phối nào của dữ liệu.

---

## Input

`X` số/đã vector hóa, `y` là lớp hoặc giá trị liên tục; chọn khoảng cách (thường Euclidean), số láng giềng `k` và cách trọng số theo khoảng cách.

Nếu dữ liệu có `n` mẫu train, `p` chiều, mỗi bản ghi inference phải qua đúng phép scale/vector hóa đã fit. Với biến categorical, khoảng cách Euclidean trên mã số tùy ý thường không có ý nghĩa; cần encoding/metric phù hợp với ngữ nghĩa. Chọn $`k\le n_{train}`$ và kiểm cả tỷ lệ lớp trong vùng lân cận. [1]

### Lời nguyền Số chiều (Curse of Dimensionality - Richard Bellman)

kNN dựa trên giả định căn bản: **Các điểm nằm gần nhau trong không gian đặc trưng thì có nhãn hoặc giá trị mục tiêu tương tự nhau**. Tuy nhiên, giả định này sụp đổ hoàn toàn khi số chiều $`p`$ tăng cao:

1. **Sự loãng không gian (Empty Space Phenomenon):** Thể tích của một khối cầu $`p`$-chiều tỷ lệ với $`r^p`$. Để bao phủ một tỷ lệ thể tích cố định (ví dụ 10% dữ liệu) nhằm tìm $`k`$ láng giềng, bán kính lân cận $`r`$ phải mở rộng rất nhanh theo công thức $`r = (0.1)^{1/p}`$. Khi $`p = 50`$, $`r \approx 0.95`$, nghĩa là thuật toán buộc phải quét gần như toàn bộ không gian dữ liệu để gom đủ $`k`$ mẫu! Khi đó, các mẫu được gọi là "gần nhất" thực chất lại ở rất xa nhau.
2. **Sự cô lập khoảng cách (Distance Concentration):** Khi số chiều $`p \to \infty`$, khoảng cách giữa điểm gần nhất ($`d_{\min}`$) và điểm xa nhất ($`d_{\max}`$) từ một điểm truy vấn bất kỳ hội tụ về cùng một giá trị:

$$
\lim_{p \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0
$$

   Mọi điểm trong không gian đều có khoảng cách gần như bằng nhau tới điểm truy vấn, khiến khái niệm "láng giềng gần nhất" mất hoàn toàn giá trị phân biệt.
* *Quy tắc vàng:* kNN hoạt động tốt nhất khi $`p < 20`$. Khi $`p`$ lớn, bắt buộc phải giảm chiều dữ liệu (PCA, t-SNE) hoặc chọn lọc đặc trưng trước.

---

## Output

Classification: lớp theo đa số hoặc tỉ lệ có trọng số của `k` láng giềng; regression: trung bình/giá trị có trọng số của target lân cận. Có thể trả các láng giềng và khoảng cách để kiểm tra. [1]

### Bản chất Hình học: Biểu đồ Phân vùng Voronoi (Voronoi Tessellation)

Đối với bộ phân loại 1-NN ($`k=1`$), không gian đặc trưng $`\mathbb{R}^p`$ được chia thành $`n`$ vùng đa diện lồi không chồng lặp gọi là **Các ô Voronoi (Voronoi Cells)**.
* Mỗi ô Voronoi $`V(x_i)`$ bao gồm tất cả các điểm trong không gian có khoảng cách tới mẫu $`x_i`$ nhỏ hơn khoảng cách tới bất kỳ mẫu nào khác.
* Ranh giới quyết định (Decision Boundary) giữa hai lớp khác nhau chính là hợp của các đoạn trung trực vuông góc ngăn cách giữa các ô Voronoi của các lớp đối lập. Ranh giới này có tính chất phi tuyến cục bộ phức tạp và cực kỳ linh hoạt.

```text
       +-----------------------------------+
       |       .   A         |      B      |
       |     .      \        |     /       |
       |   .         \       |    /        |
       |  .      A    \      |   /    B    |
       | .             \     |  /          |
       |----------------+----+-------------|
       |          B     |       A          |
       |                |                  |
       +-----------------------------------+
       (Ranh giới 1-NN là các đường phân cách giữa các ô Voronoi)
```

---

## Evaluation metrics

Dùng metric tương ứng bài toán (F1/PR-AUC cho classification, MAE/RMSE cho regression), thêm latency và bộ nhớ khi suy luận. So với baseline và kiểm tra hiệu quả theo nhóm mật độ dữ liệu.

---

## How it works

`fit` chủ yếu lưu dữ liệu train. Khi dự đoán, tính khoảng cách đến các mẫu train, chọn `k` mẫu gần nhất, rồi gộp nhãn/giá trị. `k` nhỏ dễ nhiễu; `k` lớn làm mượt quá mức. [1][2]

**Ví dụ tính tay:** dữ liệu train một chiều gồm `(x=1,y=A)`, `(x=2,y=A)`, `(x=5,y=B)`. Với mẫu mới $`x=4`$, khoảng cách Euclidean lần lượt là `3,2,1`; $`k=1`$ dự đoán `B`, còn $`k=3`$ đa số dự đoán `A`. Với regression và nhãn `10,12,20`, $`k=2`$ không trọng số sẽ lấy hai điểm gần nhất và dự đoán $`\frac{12+20}{2}=16`$.

**Fit/predict:** lưu vector train và nhãn → tính khoảng cách → chọn `k` láng giềng → gộp nhãn. Nếu dùng trọng số khoảng cách, cần định nghĩa cách xử lý khoảng cách bằng 0. [1]

### Khoảng cách và cách gộp

Với `p` feature, $`d_2(x,z)=\sqrt{\sum_j(x_j-z_j)^2}`$ (Euclidean), $`d_{1}(x,z)=\sum _j\lvert x_j-z_j\rvert`$ (Manhattan). Minkowski tổng quát bậc $`q\ge 1`$ là $`\left(\sum_j\lvert x_j-z_j\rvert^q\right)^{1/q}`$. Metric chỉ có ích nếu giá trị “gần” phản ánh sự giống nhau liên quan target; chuẩn hóa feature là một phần của định nghĩa khoảng cách, không chỉ là bước làm đẹp dữ liệu. [1]

Classification không trọng số chọn lớp có số láng giềng lớn nhất. Nếu dùng trọng số $`a_i\ge 0`$, điểm lớp `c` có thể viết $`S_c=\sum_{i\in N_k(x)} a_i\,\mathbf{1}[y_i=c]`$; xác suất ước lượng là $`\frac{S_c}{\sum_i a_i}`$. Regression có $`\hat{y}=\frac{\sum_i a_i y_i}{\sum_i a_i}`$; khi mọi $`a_i=1`$, đó là trung bình. Với $`a_i\approx 1/d_i`$, điểm trùng dữ liệu train ($`d_i=0`$) cần quy tắc riêng để tránh chia cho 0; kiểm hành vi của implementation được dùng. Tỷ lệ láng giềng không mặc nhiên là xác suất đã calibration. [1]

**Ví dụ gộp có trọng số:** ba láng giềng của mẫu mới có nhãn `A,A,B`, khoảng cách `1,2,0.5`. Bỏ trọng số cho kết quả `A` (2 phiếu). Dùng trọng số nghịch đảo khoảng cách, điểm $`A=1+0.5=1.5`$, điểm $`B=2`$; kết quả thành `B`. Vì vậy metric, `k` và cách gộp là một phần của thiết kế mô hình.

```text
fit: lưu X_train sau preprocessing và y_train (có thể xây chỉ mục tìm kiếm)
predict(x): tính/tìm khoảng cách → chọn k mẫu nhỏ nhất → gộp theo weights
```

### Các Cấu trúc Dữ liệu Tăng tốc Truy vấn (KD-Tree & Ball-Tree)

Nếu dùng tìm kiếm vét cạn thô (Brute-force Search), chi phí cho mỗi lần dự đoán là $`\mathcal{O}(n \cdot p)`$, khiến hệ thống bị nghẽn khi $`n`$ lớn. Để giải quyết, hai cấu trúc cây không gian được ứng dụng:

1. **KD-Tree (K-dimensional Tree - Jon Bentley, 1975):**
   * *Nguyên lý:* Phân chia không gian đệ quy bằng các siêu phẳng trực giao với các trục tọa độ. Tại mỗi tầng của cây, chọn trục có phương sai lớn nhất và chia đôi tập dữ liệu tại điểm trung vị (Median).
   * *Hiệu năng:* Giảm chi phí truy vấn xuống $`\mathcal{O}(p \cdot \log n)`$ khi số chiều $`p`$ nhỏ ($`p \le 20`$).
   * *Giới hạn:* Khi $`p > 20`$, KD-Tree phải quay lui (backtracking) qua hầu hết các nhánh, tốc độ thoái hóa về ngang bằng Brute-force.
2. **Ball-Tree:**
   * *Nguyên lý:* Phân chia không gian thành các siêu cầu lồng nhau (Nested Hyperspheres) dựa trên bất đẳng thức tam giác:

$$
|d(x, c) - d(y, c)| \le d(x, y) \le d(x, c) + d(y, c)
$$

   * *Ưu điểm:* Vượt trội hơn KD-Tree trong các không gian có cấu trúc hình học phức tạp hoặc số chiều trung bình cao.

---

## Data used

Phù hợp khi khoảng cách giữa vector mang ý nghĩa và số chiều không quá lớn; Iris là ví dụ học tập. Scale feature số bằng thống kê train để một feature có đơn vị lớn không áp đảo khoảng cách. Với văn bản vector thưa, cân nhắc cosine distance và chiều cao. [1][3]

---

## Design

- Tuning `k`, metric khoảng cách, weights bằng CV; lưu toàn bộ dữ liệu train nếu thuật toán cần khi inference.
- Ưu điểm: ít giả định hình dạng ranh giới, không cần tối ưu trọng số. Hạn chế: inference và lưu trữ tốn kém; số chiều cao làm khái niệm gần/xa kém hữu ích.
- Failure modes: dữ liệu thưa vùng lân cận, outlier, feature chưa scale, group leakage giữa train/test.

| Hyperparameter | Cách thử | Rủi ro |
|---|---|---|
| `n_neighbors=k` | CV trên train, bắt đầu từ vài giá trị nhỏ đến vừa | $`k=1`$ nhạy nhiễu; `k` rất lớn làm mất cấu trúc cục bộ |
| `metric`/`p` | Chọn theo ngữ nghĩa feature, so trên cùng split | Khoảng cách vô nghĩa trên feature mã hóa sai |
| `weights` | Đồng đều hoặc theo khoảng cách | Mẫu rất gần có thể chi phối dự đoán |
| Cách tìm láng giềng | Brute, KDTree, BallTree theo dữ liệu | Chi phí tìm kiếm thay đổi theo số chiều và phân phối |

Nếu có hòa phiếu ở classification, dự đoán phụ thuộc quy tắc tie-breaking của implementation; hãy kiểm trực tiếp thay vì giả định luôn chọn một lớp cụ thể. Với lớp hiếm, xem recall theo lớp và lân cận của các mẫu sai, không chỉ accuracy. [1]

**Chi phí:** với tìm kiếm brute force trên `n` mẫu, `p` chiều, một truy vấn khoảng $`O(np)`$ để tính khoảng cách, cộng chi phí chọn láng giềng. KDTree/BallTree có thể giúp trong một số dữ liệu chiều thấp, nhưng lợi ích giảm khi chiều cao; đo thời gian trên dữ liệu thật. [1]

Bộ nhớ lưu `X_train` khoảng $`O(np)`$ cho dữ liệu dense, cộng nhãn/chỉ mục. Nếu dataset rất lớn hoặc yêu cầu latency chặt, chi phí inference có thể là lý do chọn mô hình tham số; nếu dữ liệu ít và cấu trúc cục bộ rõ, kNN là baseline phi tuyến hữu ích.

### Phân tích Đánh đổi Bias-Variance theo Siêu tham số $`k`$

Siêu tham số $`k`$ điều khiển trực tiếp bậc tự do thực tế của mô hình:
* **Khi $`k = 1`$ (Mô hình Cực kỳ Phức tạp):**
  * Training Error bằng đúng $`0`$ tuyệt đối.
  * **Bias cực kỳ thấp**, nhưng **Variance cực kỳ cao**. Mô hình nhạy cảm tuyệt đối với mọi điểm nhiễu ngoại lai. Ranh giới quyết định gồ ghề và phân mảnh. Dẫn đến **Overfitting**.
* **Khi $`k = n`$ (Mô hình Cực kỳ Đơn giản):**
  * Mô hình luôn dự đoán lớp chiếm đa số hoặc trung bình toàn bộ tập train cho mọi điểm mới.
  * **Variance bằng đúng 0**, nhưng **Bias cực kỳ cao**. Mô hình hoàn toàn không học được bất kỳ cấu trúc cục bộ nào. Dẫn đến **Underfitting**.
* *Kinh nghiệm thực nghiệm:* Chọn $`k`$ là số lẻ để tránh hòa phiếu trong bài toán nhị phân, và tìm $`k^{\ast}`$ tối ưu bằng Cross-Validation trong khoảng $`k \in [3, 25]`$.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Neighbor / láng giềng** | Mẫu train có khoảng cách nhỏ với mẫu cần dự đoán theo metric đã chọn. | Cung cấp nhãn hoặc giá trị tham chiếu cho điểm mới | Các mẫu train gần nhất được truy xuất ở inference |
| **`k`** | Số láng giềng dùng để gộp nhãn/giá trị; là hyperparameter, không phải trọng số học bằng gradient. | Quyết định số mẫu tham gia vote hoặc trung bình | Hyperparameter cân bằng nhiễu và độ mượt, chọn trên validation |
| **Distance metric** | Quy tắc đo gần/xa như Euclidean hay Manhattan; quyết định láng giềng nào được chọn. | Quyết định thứ tự gần xa giữa các mẫu | Thành phần cốt lõi của truy xuất láng giềng ở inference |
| **Distance weighting** | Cho mẫu gần hơn trọng số lớn hơn khi vote hoặc tính trung bình. | Làm láng giềng gần ảnh hưởng mạnh hơn | Quy tắc tổng hợp dự đoán sau khi đã tìm k điểm |
| **Feature scaling** | Đổi thang feature dựa trên train để đơn vị lớn không tự động chi phối khoảng cách. | Ngăn cột có đơn vị lớn lấn át khoảng cách | Transformer fit trên train rồi áp cùng phép biến đổi khi inference |
| **Curse of dimensionality** | Khi số chiều cao, khoảng cách giữa các mẫu có thể kém phân biệt, khiến “gần nhất” ít hữu ích. | Làm khái niệm gần xa kém tin cậy khi nhiều chiều | Giới hạn cần cân nhắc khi chọn KNN hoặc giảm chiều |
| **Brute-force search** | Tính khoảng cách với các mẫu train khi dự đoán, thay vì dùng chỉ mục KDTree/BallTree. | Cho kết quả chính xác theo metric nhưng tốn phép tính khi train lớn | Chiến lược tìm láng giềng ở inference |
| **Tie / hòa phiếu** | Hai lớp có cùng điểm sau gộp; kết quả tùy quy tắc xử lý của implementation. | Có thể làm nhãn dự đoán phụ thuộc cách phá hòa | Trường hợp biên cần kiểm tra nếu bài toán có nhiều lớp |
| **Voronoi Diagram** | Phân hoạch không gian thành các ô đa diện lồi dựa trên khoảng cách tới điểm mẫu. | Thể hiện trực quan ranh giới quyết định của 1-NN. | Công cụ hình học mô tả cấu trúc ranh giới phi tham số. |
| **KD-Tree** | Cấu trúc cây nhị phân phân chia không gian theo các siêu phẳng trực giao. | Tăng tốc truy vấn tìm kiếm láng giềng trong không gian số chiều thấp. | Cấu trúc chỉ mục tối ưu hóa tốc độ inference trong Scikit-learn. |

---

## Nguồn

- [1] [scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [2] [Cover & Hart: Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964)
- [3] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
- [4] Bentley, J. L. (1975). *Multidimensional binary search trees used for associative searching*. Communications of the ACM, 18(9), 509-517.
- [5] Bellman, R. (1966). *Dynamic Programming*. Science, 153(3731), 34-37. (Khái niệm Curse of Dimensionality).
