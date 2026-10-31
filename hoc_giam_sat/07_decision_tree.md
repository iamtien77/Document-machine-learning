# Decision Tree

## Purpose

Học các quy tắc chia feature thành vùng dự đoán, dùng cho classification hoặc regression. Một cây nhỏ dễ theo dõi đường quyết định và là mốc cho các mô hình ensemble. [1]

### Lịch sử CART & Bản chất Phân hoạch Trực giao Không gian (Axis-Aligned Partitioning)

Cây Quyết định hiện đại có nguồn gốc từ công trình kinh điển năm 1984 của Leo Breiman, Jerome Friedman, Richard Olshen và Charles Stone mang tên **CART (Classification and Regression Trees)**. Song song với đó là các thuật toán ID3 (1986) và C4.5 (1993) của Ross Quinlan.

Khác với các mô hình tuyến tính vốn tạo ra một siêu phẳng xiên duy nhất chia cắt không gian, Cây Quyết định hoạt động theo triết lý **"Chia để trị" (Divide and Conquer)**:
1. **Phân hoạch Đệ quy (Recursive Binary Splitting):** Tại mỗi bước, thuật toán chọn một trục tọa độ $`x_j`$ và một điểm ngưỡng $`t`$ để cắt đôi không gian thành hai nửa siêu hộp chữ nhật trực giao với trục tọa độ.
2. **Khả năng Biểu diễn Phi tuyến:** Bằng cách lặp lại phép cắt trực giao này nhiều lần ở các tầng sâu hơn, cây quyết định có thể xấp xỉ bất kỳ ranh giới phi tuyến phức tạp nào dưới dạng một tập hợp các quy tắc logic `IF-THEN` hoàn toàn minh bạch.
3. **Tính Bất biến với Phép Biến đổi Đơn điệu (Monotonic Invariance):**
   Phép so sánh $`x_j \le t`$ chỉ phụ thuộc vào **thứ tự xếp hạng** của các giá trị chứ không phụ thuộc vào độ lớn số học tuyệt đối. Nếu ta áp dụng bất kỳ phép biến đổi đơn điệu tăng nào (như nhân với hằng số dương, hàm logarit, hay chuẩn hóa Z-score), thứ tự các mẫu không hề đổi và điểm cắt tối ưu tương ứng chỉ đơn giản là $`f(t)`$. Do đó, **Cây Quyết định hoàn toàn không yêu cầu chuẩn hóa thang đo đặc trưng (Feature Scaling)**!

---

## Input

`X` số hoặc categorical đã biểu diễn phù hợp với implementation, `y` là nhãn lớp hoặc số liên tục. Cần định nghĩa cách xử lý missing và feature categorical trước train.

Trong scikit-learn, cách tách thông thường xét một điều kiện dạng $`x_j \le t`$; biến categorical cần mã hóa phù hợp, còn hỗ trợ missing phụ thuộc estimator/criterion và phiên bản cụ thể. Split phải diễn ra trước mọi bước mã hóa hoặc điền giá trị có học thống kê. [1]

---

## Output

Cấu trúc cây gồm các điều kiện tách, dự đoán tại lá; classifier có thể trả tỉ lệ lớp tại lá làm xác suất ước lượng. [1]

Với `DecisionTreeRegressor(criterion="squared_error")`, dự đoán ở lá là trung bình `y` của các mẫu train tại lá; với `absolute_error`, giá trị lá là trung vị. Vì vậy đầu ra regression thường là hàm bậc thang theo feature và khó extrapolate ngoài vùng train. [4]

### Tại sao Cây Hồi quy Hoàn toàn Bất lực khi Ngoại suy (Extrapolation Failure)?

Một đặc tính hình học cốt lõi của Cây Hồi quy là: **Đầu ra dự đoán của nó là một hàm hằng số từng khúc (Piecewise Constant Function)**.
* Mỗi chiếc lá $`R_m`$ đại diện cho một khối siêu hộp trong không gian và dự đoán một giá trị hằng số duy nhất $`c_m = \mathrm{mean}(y \mid x \in R_m)`$.
* Khi một điểm dữ liệu mới $`x_{\text{new}}`$ nằm xa ra ngoài miền dữ liệu huấn luyện (ví dụ: dự đoán giá nhà cho căn hộ $`1{,}000\,\text{m}^2`$ trong khi dữ liệu train lớn nhất chỉ là $`300\,\text{m}^2`$), điểm này sẽ bị dẫn vào chiếc lá ở rìa ngoài cùng và trả về giá trị trung bình của chiếc lá đó.
* *Hệ quả:* Cây quyết định **không thể học được xu hướng tăng trưởng (Trend)**. Dự đoán của nó sẽ đi ngang thành một đường thẳng nằm ngang bất biến ở vùng ngoại suy!

```text
 y ^                        o  Điểm thực tế (Xu hướng tiếp tục tăng)
   |                    o
   |            o---+-------.  Cây dự đoán ĐI NGANG (Hằng số của lá ngoài cùng)
   |        +---'   |
   |    +---'       |
   +----+-----------+--------> x
       Miền Train   | Vùng Ngoại suy (Extrapolation)
```

---

## Evaluation metrics

Metric theo bài toán; so chênh train/validation để phát hiện overfit. Với xác suất từ lá, kiểm calibration và số mẫu mỗi lá nếu dùng vào quyết định. [2]

---

## How it works

Từ nút gốc, chọn feature/ngưỡng tách làm giảm độ hỗn tạp (Gini/entropy) cho classification hoặc giảm lỗi bình phương cho regression; lặp đệ quy đến điều kiện dừng. Pruning hoặc giới hạn độ sâu/số mẫu lá giúp kiểm soát độ phức tạp. [1]

**Ví dụ tính tay:** ở nút có 6 mẫu dương và 4 mẫu âm, Gini là $`1-(0.6^{2}+0.4^{2})=0.48`$. Một cách tách tạo lá trái gồm 4 dương/0 âm và lá phải 2 dương/4 âm. Gini có trọng số sau tách là $`(4/10)\times 0 + (6/10)\times [1-(2/6)^{2}-(4/6)^{2}]\approx 0.267`$; mức giảm impurity khoảng `0.213`. Cây so nhiều feature/ngưỡng và chọn tách tốt nhất trong các ứng viên hợp lệ. [1]

**Fit/predict:** tìm tách ở nút → chia mẫu → lặp ở các nút con đến điều kiện dừng; khi dự đoán đi từ gốc theo các điều kiện đến lá. Một lá có rất ít mẫu có thể cho xác suất rất nhiễu dù dự đoán class trên train đúng.

### Tiêu chí tách và cây hồi quy

Với lớp `k` có tần suất $`p_k`$ trong nút, $`Gini=1-\sum _k p_k^{2}`$ và $`Entropy=-\sum _k p_k \log _{2}(p_k)`$ (quy ước $`0\log 0=0`$). Một tách có mức cải thiện $`I(\mathrm{parent})-\frac{n_L}{n}I(\mathrm{left})-\frac{n_R}{n}I(\mathrm{right})`$. Thuật toán chọn feature/ngưỡng cho mức giảm impurity cao nhất trong các ứng viên, dưới các ràng buộc số mẫu lá và độ sâu. [1]

Với regression, thay impurity phân loại bằng sai số của target trong nút. `squared_error` giảm tổng bình phương sai lệch quanh trung bình; `absolute_error` giảm tổng trị tuyệt đối quanh trung vị; `poisson` dành cho target không âm phù hợp với bài toán count/rate theo điều kiện của estimator. [4]

**Ví dụ fit regression:** bốn mẫu có một feature lần lượt $`x=[1,2,3,4]`$, target $`y=[2,4,10,12]`$. Nút gốc dự đoán trung bình `7`, tổng bình phương sai số là `68`. Tách tại $`x\le 2.5`$ tạo lá trái $`y=[2,4]`$ (dự đoán `3`, SSE=`2`) và lá phải $`y=[10,12]`$ (dự đoán `11`, SSE=`2`); tổng SSE sau tách là `4`. Khi mẫu mới có $`x=3.2`$, cây đi sang lá phải và dự đoán `11`. Đây là ví dụ tính tay, chưa phải kết quả thử nghiệm trên dataset thực.

```text
fit(node):
    nếu đạt điều kiện dừng: lưu phân phối lớp hoặc thống kê target ở lá
    duyệt feature/ngưỡng hợp lệ và tính impurity sau tách
    chọn tách giảm impurity nhiều nhất; gọi fit cho hai nhánh
predict(x): đi từ gốc theo điều kiện đến lá; trả nhãn/xác suất/giá trị lá
```

#### So sánh Giải tích: Gini Impurity vs. Entropy

Xét bài toán phân loại nhị phân với xác suất lớp dương $`p \in [0, 1]`$:
* **Gini Impurity:** $`I_G(p) = 2p(1 - p)`$. Cực đại tại $`p = 0.5`$ với giá trị $`0.5`$.
* **Entropy:** $`I_E(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)`$. Cực đại tại $`p = 0.5`$ với giá trị $`1.0`$.
* *Nhận xét:* Hai hàm có hình dáng đồ thị chuông úp ngược gần như y hệt nhau. Entropy phạt nặng hơn một chút ở các vùng nút có độ hỗn tạp cao. Tuy nhiên, Gini không cần tính toán logarit cơ số 2 nên tốc độ thực thi nhanh hơn đáng kể. Trong thực tế, hai tiêu chí này chọn ra cùng một điểm cắt trong hơn 98% trường hợp.

### Thuật toán Tỉa cành Chi phí - Độ phức tạp (Minimal Cost-Complexity Pruning - CCP)

Để ngăn chặn một cây phát triển quá sâu và học thuộc toàn bộ nhiễu của tập train, CART sử dụng hàm mục tiêu phạt độ phức tạp (Breiman et al., 1984):

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

Trong đó:
* $`R(T)`$: Tổng sai số trên tập dữ liệu của toàn bộ các lá trong cây $`T`$.
* $`|T|`$: Tổng số lá của cây.
* $`\alpha \ge 0`$: Siêu tham số điều khiển mức độ phạt kích thước cây (`ccp_alpha` trong Scikit-learn).
* *Quy trình Weakest Link Pruning:* Khi $`\alpha`$ tăng dần từ 0 lên $`+\infty`$, tại mỗi nút con $`t`$, ta tính hệ số hiệu quả:

$$
\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}
$$

  Nút có $`\alpha_{\text{eff}}`$ nhỏ nhất sẽ bị cắt bỏ (biến thành lá) đầu tiên. Quá trình này sinh ra một dãy hữu hạn các cây con lồng nhau. Ta dùng K-Fold Cross-Validation để tìm ra giá trị $`\alpha^*`$ tối ưu nhất.

---

## Data used

Dữ liệu bảng có tương tác/quan hệ phi tuyến; UCI Adult là ví dụ để thực hành sau encode/impute. Cây không cần standardization chỉ để tính ngưỡng tách, nhưng vẫn cần phòng leakage và xử lý missing. [1][3]

---

## Design

- Tuning `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha` trên validation/CV. Cây nông là baseline dễ diễn giải.
- Ưu điểm: quy tắc rõ, học tương tác tự nhiên. Hạn chế: cây sâu dễ overfit, kết quả có thể thay đổi mạnh khi dữ liệu thay đổi nhẹ.
- Failure modes: lá quá ít mẫu, feature có nhiều mức làm tách thiếu ổn định, diễn giải feature importance theo impurity như quan hệ nhân quả.

| Hyperparameter | Vai trò | Dấu hiệu cần kiểm |
|---|---|---|
| `max_depth` | Giới hạn số tầng | Cây sâu: train tốt nhưng validation giảm |
| `min_samples_leaf` | Mức dữ liệu tối thiểu tại lá | Lá ít mẫu: dự đoán/xác suất không ổn định |
| `min_samples_split` | Điều kiện cho phép tiếp tục tách | Quá lớn: underfit; quá nhỏ: dễ tạo cây phức tạp |
| `ccp_alpha` | Pruning theo cost-complexity | Chọn bằng CV, không dùng test |

**Pre-pruning** dừng trước bằng giới hạn độ sâu/lá; **cost-complexity pruning** cắt bớt cây đã học. Chọn mức phức tạp trên validation/CV. Cây một mình có thể dễ giải thích nếu nông, nhưng một cây lớn với hàng trăm nút không còn dễ đọc; khi cần độ ổn định hơn, so với forest. [1]

**Chi phí:** cây sâu tối đa có thể có nhiều nút và dễ ghi nhớ train; một dự đoán đi qua số nút bằng độ sâu đường đi. Khi lớp hiếm, kiểm số mẫu theo lớp ở từng lá và metric của lớp hiếm. Cây đơn thường có variance cao; Random Forest giảm variance bằng gộp nhiều cây. [1]

### Điểm Yếu Chết người: Tính Bất ổn định Phương sai Cao (High Variance)

Hạn chế lý thuyết nghiêm trọng nhất của một Cây Quyết định đơn lẻ là **tính bất ổn định cực kỳ cao**:
* Cây quyết định học theo phương pháp **Tham lam (Greedy Algorithm)**: Tại nút gốc, nó chọn điểm cắt tốt nhất ở thời điểm hiện tại mà không quan tâm đến tương lai.
* Nếu tập dữ liệu huấn luyện bị thay đổi dù chỉ một vài điểm dữ liệu, điểm cắt ở nút gốc có thể bị thay đổi sang một đặc trưng khác.
* Sự thay đổi ở nút gốc này sẽ kéo theo **toàn bộ các nhánh con bên dưới bị đảo lộn hoàn toàn**, sinh ra một cấu trúc cây hoàn toàn khác biệt!
* Chính sự bất ổn định (High Variance) này là động lực thúc đẩy sự ra đời của các phương pháp Ensemble như **Bagging** và **Random Forest**.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Node / nút** | Tập mẫu đang được xét tại một vị trí trong cây. | Tập hợp các mẫu đang chờ xét tại một điểm của cây | Đơn vị mà thuật toán đánh giá phép tách trong training |
| **Split / phép tách** | Điều kiện trên feature chia mẫu ở nút thành các nhánh con. | Chia mẫu thành các nhóm theo điều kiện feature | Quy tắc được học khi fit và áp theo đường đi ở inference |
| **Impurity / độ hỗn tạp** | Mức trộn lớp hoặc sai số target trong nút; cây tìm cách giảm qua các phép tách. | Cho biết nút còn lẫn lớp hoặc biến thiên target đến mức nào | Tiêu chí chọn split tốt hơn trong training |
| **Gini / entropy** | Hai tiêu chí đo hỗn tạp cho classification; thấp khi một lớp chiếm ưu thế. | Lượng hóa độ trộn lớp theo hai công thức khác nhau | Các lựa chọn criterion khi fit cây phân loại |
| **Leaf / lá** | Nút không tách tiếp; chứa phân phối lớp hoặc thống kê target để dự đoán. | Lưu giá trị hoặc tỉ lệ lớp cho dự đoán cuối | Điểm kết thúc đường đi của mẫu mới trong inference |
| **Stopping rule** | Điều kiện dừng như độ sâu tối đa hoặc số mẫu lá tối thiểu. | Giới hạn cây tiếp tục phân nhánh | Hyperparameter kiểm soát độ phức tạp và overfit |
| **Pruning** | Cắt bớt nhánh để giảm độ phức tạp và nguy cơ overfit. | Bỏ nhánh ít giá trị để cây gọn hơn | Bước kiểm soát độ phức tạp sau hoặc trong quá trình tạo cây |
| **SSE/MSE** | Tổng/trung bình bình phương sai số dùng để đánh giá tách nút trong hồi quy bậc hai. | Đánh giá mức sai số số thực của nút | Tiêu chí chọn split cho cây hồi quy squared-error |
| **CART** | Thuật toán xây dựng Cây Phân loại và Hồi quy nhị phân kinh điển (Breiman, 1984). | Chuẩn mực thuật toán cây được triển khai trong Scikit-learn. | Nền tảng cốt lõi của họ mô hình cây hiện đại. |
| **Axis-aligned Split** | Phép cắt ranh giới luôn vuông góc góc 90 độ với các trục tọa độ. | Tạo ra các khối hộp siêu chiều dễ diễn giải thành quy tắc IF-THEN. | Bản chất hình học đặc thù của cây quyết định chuẩn. |

---

## Nguồn

- [1] [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [2] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [3] [UCI: Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [4] [scikit-learn: DecisionTreeRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [5] Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.
- [6] Quinlan, J. R. (1986). *Induction of decision trees*. Machine Learning, 1(1), 81-106.
