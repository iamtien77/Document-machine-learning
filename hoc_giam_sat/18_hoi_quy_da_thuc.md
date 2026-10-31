# Polynomial Regression — hồi quy đa thức

> PDF [Chương 2](<../Tài liệu học tập/ML_Chapter2_new.pdf>) giới thiệu Polynomial Regression ở tr. 51 và 55. Đây là **linear model trên feature đã biến đổi phi tuyến**, không phải một thuật toán tối ưu hoàn toàn tách khỏi OLS/Ridge/Lasso. [1]

## Purpose

Mô hình hóa quan hệ cong giữa feature và target số bằng cách tạo các lũy thừa/tương tác rồi fit một mô hình tuyến tính theo hệ số. Nó là bước thử hợp lý khi residual của linear regression có cấu trúc cong nhưng vẫn muốn giữ một mô hình tương đối đơn giản và dễ kiểm tra. [1]

### Định lý Xấp xỉ Weierstrass & Cầu nối Tuyến tính - Phi tuyến

Một ngộ nhận kinh điển của người học máy là nghĩ rằng: *"Muốn học các quy luật phi tuyến tính phức tạp, bắt buộc phải dùng các mô hình hộp đen như Mạng Nơ-ron hay Rừng ngẫu nhiên."*

Hồi quy Đa thức chứng minh điều ngược lại thông qua phương pháp **Mở rộng Hàm cơ sở (Basis Function Expansion)**:
* **Tuyến tính trong tham số (Linear in Parameters):** Mặc dù mối quan hệ giữa biến đầu vào $`x`$ và nhãn $`y`$ là đường cong phi tuyến, nhưng hàm mục tiêu vẫn hoàn toàn tuyến tính đối với vector trọng số $`w`$:

  ```math
  \hat{y} = w_0 + w_1 \phi_1(x) + w_2 \phi_2(x) + \dots + w_M \phi_M(x) = w^\top \phi(x)
  ```

  Do đó, toàn bộ các công cụ giải tích mạnh mẽ của Hồi quy Tuyến tính (phương trình chuẩn, tính lồi toàn cục, kiểm định t-test, giải thuật Ridge/Lasso) đều được tái sử dụng nguyên vẹn mà không cần bất kỳ thuật toán tối ưu phi tuyến phức tạp nào.
* **Định lý Xấp xỉ Weierstrass (1885):**
  Mọi hàm số liên tục $`f(x)`$ xác định trên một đoạn đóng $`[a, b]`$ đều có thể được xấp xỉ đều với độ chính xác tùy ý $`\varepsilon > 0`$ bằng một đa thức bậc $`d`$ đủ lớn:

  ```math
  \sup_{x \in [a, b]} \lvert f(x) - P_d(x) \rvert < \varepsilon
  ```

  Định lý này đặt nền móng lý thuyết cho việc nâng bậc đa thức để nắm bắt mọi cấu trúc cong tiềm ẩn trong dữ liệu.

---

## Input

$`X \in \mathbb{R}^{n\times p}`$, target liên tục `y`, bậc đa thức `d`, có/không tạo interaction, estimator cuối (OLS/Ridge/Lasso), quy tắc split và metric regression. Feature phải có sẵn tại thời điểm dự đoán; missing/categorical cần được xử lý trong pipeline trước hoặc theo thiết kế phù hợp.

### Bùng nổ Số chiều Tổ hợp (Combinatorial Feature Explosion)

Số lượng đặc trưng tạo ra sau phép biến đổi đa thức bậc $`d`$ từ $`p`$ biến ban đầu (tính cả số hạng hằng số bias) tuân theo công thức tổ hợp lặp:

```math
M = \binom{p + d}{d} = \frac{(p + d)!}{p! \, d!}
```

| Số biến gốc ($`p`$) | Bậc đa thức ($`d=1`$) | Bậc đa thức ($`d=2`$) | Bậc đa thức ($`d=3`$) | Bậc đa thức ($`d=5`$) |
|---|---|---|---|---|
| **$`p = 2`$** | 3 | 6 | 10 | 21 |
| **$`p = 5`$** | 6 | 21 | 56 | 252 |
| **$`p = 10`$** | 11 | 66 | 286 | **3,003** |
| **$`p = 50`$** | 51 | 1,326 | 23,426 | **3,478,761** |

*Cảnh báo nghiên cứu:* Khi số biến ban đầu $`p`$ chỉ ở mức vài chục, việc tăng bậc đa thức lên $`d=3`$ hoặc $`5`$ sẽ làm số cột bùng nổ lên hàng triệu, dẫn tới cạn kiệt bộ nhớ RAM và làm mô hình overfit trầm trọng.

---

## Output

Pipeline gồm `PolynomialFeatures` đã định cấu hình và estimator đã fit; dự đoán $`\hat{y}`$ là số theo đơn vị target. Với một biến và bậc 2, $`\hat{y}=b+w_{1}x+w_{2}x^{2}`$. Với hai biến và bậc 2, có thể có $`x_{1}`$, $`x_{2}`$, $`x_{1}^{2}`$, $`x_{1}x_{2}`$, $`x_{2}^{2}`$. Mô hình phi tuyến theo `x` nhưng **tuyến tính theo các hệ số `w`**. [1]

---

## Evaluation metrics

So MAE/RMSE/R² với hồi quy tuyến tính bậc 1 và DummyRegressor trên cùng split. Xem residual theo `x`, theo $`\hat{y}`$ và theo vùng ngoại suy; báo cáo cả độ dao động giữa các fold. Nếu sai số train giảm nhưng validation tăng khi `d` lớn, đó là dấu hiệu overfit. [2]

---

## How it works

Đặt ánh xạ $`\phi _d(x)`$ chứa các monomial đến bậc `d`; sau đó fit $`\hat{y}=b+w^{\top}\phi _d(x)`$ bằng OLS hoặc mô hình có regularization. Với $`p=2`$, $`d=2`$, $`\phi (x)=[x_{1},x_{2},x_{1}^{2},x_{1}x_{2},x_{2}^{2}]`$ nếu intercept được fit riêng. Objective của Ridge sau biến đổi là $`\lVert \Phi w-y\rVert_{2}^{2}+\alpha \lVert w\rVert_{2}^{2}`$; $`\Phi`$ là ma trận feature đa thức. [1]

**Ví dụ fit tay:** train có `(x,y)=(-1,1),(0,0),(1,1)`. Đường bậc 2 $`\hat{y}=x^{2}`$ khớp cả ba điểm; với $`x=2`$, nó dự đoán `4`. Bậc 1 không thể đi qua ba điểm này cùng lúc. Tuy nhiên ba điểm train không chứng minh $`\hat{y}=4`$ là đúng ngoài miền `[-1,1]`; phải kiểm validation/test phản ánh vùng sử dụng. [1]

```text
split → với mỗi fold: fit imputer, tạo PolynomialFeatures
      → scale các feature đã mở rộng nếu dùng regularization/solver cần scale
      → fit LinearRegression/Ridge trên train fold
      → đánh giá validation fold cho từng cặp (degree, alpha)
khóa pipeline → test cuối → inference: transform bằng cùng feature map rồi predict
```

Số monomial đến bậc `d` (gồm bias) là $`\binom{p+d}{d}`$ trong trường hợp đầy đủ; với `p` hoặc `d` lớn, số cột tăng nhanh. Tránh tạo feature từ toàn bộ dữ liệu rồi dùng feature selection ngoài CV; mọi bước học thống kê/feature selection phải ở trong train fold. [1][3]

### Hiện tượng Runge: Thảm họa Dao động ở Biên khi Nâng bậc Đa thức

Năm 1901, nhà toán học Carl Runge đã phát hiện ra một nghịch lý chấn động: Khi nâng bậc đa thức lên quá cao ($`d \ge 10`$) để nội suy các điểm dữ liệu cách đều nhau của hàm $`f(x) = \frac{1}{1 + 25 x^2}`$, **đa thức bắt đầu dao động dữ dội với biên độ khổng lồ ở sát hai biên mút của khoảng xác định**!

```text
       y ^            _..---.._  Đa thức bậc cao dao động dữ dội ở hai biên
         |           /         \ (Hiện tượng Runge / Overfitting)
         |     /\   /           \   /\
         |    /  \ /             \ /  \
         |  o/    o       o       o    \o   o : Điểm dữ liệu quan sát
         +--+-----+-------+-------+-----+----> x
           -1                           +1
```

*Bài học cho Kỹ sư Học máy:*
1. Không bao giờ mù quáng nâng bậc đa thức với hy vọng giảm training error về 0.
2. Tại các vùng nằm ngoài khoảng huấn luyện (Extrapolation Zone), đa thức bậc cao sẽ phân kỳ lao dốc xuống $`-\infty`$ hoặc vọt lên $`+\infty`$, tạo ra những dự đoán phi lý trong thực tế.

### Tầm quan trọng Sinh tử của Feature Scaling sau khi Mở rộng Đa thức

Giả sử đặc trưng $`x \in [1, 100]`$:
* Đặc trưng bậc 1: $`x \in [1, 100]`$.
* Đặc trưng bậc 2: $`x^2 \in [1, 10{,}000]`$.
* Đặc trưng bậc 3: $`x^3 \in [1, 1{,}000{,}000]`$.

Sự chênh lệch thang đo hàng triệu lần giữa các cột sẽ gây ra hai thảm họa:
1. **Ma trận suy biến (Ill-conditioned Gram Matrix):** Các giải thuật đại số số học bị tràn số hoặc tích lũy sai số làm tròn nghiêm trọng.
2. **Hỏng cơ chế Regularization:** Số hạng phạt L2 $`\alpha \sum w_j^2`$ sẽ chỉ tập trung co cụm hệ số của các bậc thấp mà bỏ qua bậc cao, hoặc ngược lại.
* **Chuẩn mực Pipeline bắt buộc:**

  ```math
  \text{Raw Features } X \longrightarrow \text{PolynomialFeatures} \longrightarrow \mathbf{StandardScaler} \longrightarrow \text{Ridge / Lasso}
  ```

  Việc chuẩn hóa (Scaling) bắt buộc phải nằm **sau** bước mở rộng đa thức và **bên trong** từng fold huấn luyện.

---

## Data used

Phù hợp dữ liệu số có quan hệ mượt và số feature ban đầu không quá lớn. Ví dụ giá nhà theo diện tích có thể có hiệu ứng cong, nhưng phải kiểm xem đường bậc 2 có phản ánh dữ liệu hay chỉ khớp nhiễu. Với biến categorical, one-hot rồi tạo mọi tích có thể gây số chiều rất lớn và cách diễn giải khó. Với dữ liệu chuỗi thời gian, chia theo thời gian trước khi chọn `degree`.

---

## Design

| Hyperparameter | Tác dụng | Rủi ro khi tăng |
|---|---|---|
| `degree` | Tạo quan hệ cong/tương tác bậc cao hơn | Bùng nổ feature, overfit, ngoại suy dao động |
| `interaction_only` | Chỉ tạo tích giữa feature khác nhau | Bỏ qua lũy thừa như $`x^{2}`$ nếu chúng quan trọng |
| `alpha` của Ridge/Lasso | Co/chọn hệ số trong không gian đa thức | Quá mạnh gây underfit; tune trong CV |

Nên chọn khi cần mô hình cong có kiểm soát và số feature vừa phải. Không nên tăng bậc để khớp mọi điểm train hoặc dùng nó cho ngoại suy xa mà không kiểm tra. Scale/regularization có thể cần thiết vì $`x^{2}`$, $`x^{3}`$ có thang rất khác `x`; nếu scale trước khi mở rộng, các monomial vẫn có thể khác thang, nên cân nhắc scale **sau** expansion trong từng train fold. Giữ `PolynomialFeatures`, scaler và estimator trong cùng pipeline để suy luận dùng đúng thứ tự cột. [1][3]

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Basis function $`\phi (x)`$** | Phép biến đổi đầu vào thành feature mới như $`x^{2}`$ hoặc $`x_{1}x_{2}`$. | Tạo biểu diễn mới để mô hình tuyến tính học quan hệ cong | Bước biến đổi feature trước khi fit hồi quy |
| **Degree / bậc `d`** | Tổng số mũ lớn nhất của các monomial được tạo. | Điều khiển mức phức tạp và số hạng được tạo | Hyperparameter chọn trên validation |
| **Monomial** | Một tích của các lũy thừa feature, ví dụ $`x_{1}^{2}x_{2}`$. | Là một cột cụ thể của ma trận feature đa thức | Thành phần mà hồi quy gán hệ số riêng |
| **Interaction term** | Tích giữa các feature khác nhau, như $`x_{1}x_{2}`$, mô tả tác động kết hợp. | Cho mô hình biểu diễn ảnh hưởng phối hợp của nhiều feature | Cột bổ sung trong feature expansion |
| **Feature expansion** | Tăng số cột từ feature gốc sang tập các monomial; có thể làm số chiều bùng nổ. | Tăng sức biểu diễn nhưng cũng tăng nguy cơ overfit | Bước tiền xử lý phải fit/áp nhất quán trong pipeline |
| **Linear in coefficients** | $`\hat{y}`$ vẫn là tổng $`w_j\phi _j(x)`$ nên tuyến tính theo trọng số `w`, dù cong theo `x`. | Cho phép dùng tối ưu của hồi quy tuyến tính trên feature mới | Tính chất của mô hình sau khi mở rộng feature |
| **Extrapolation** | Dự đoán ngoài miền train; đa thức bậc cao có thể tăng rất nhanh ở vùng này. | Có thể làm giá trị dự đoán tăng mạnh ngoài vùng quan sát | Giới hạn cần xét khi sử dụng mô hình cho đầu vào mới |
| **Runge's Phenomenon** | Hiện tượng dao động dữ dội ở sát hai biên khi tăng bậc đa thức nội suy. | Gây ra sai số dự đoán khổng lồ ở vùng biên phân phối. | Lời cảnh tỉnh kinh điển chống lại việc over-parameter hóa đa thức. |
| **Weierstrass Theorem** | Định lý toán học chứng minh đa thức có thể xấp xỉ mọi hàm liên tục. | Cơ sở lý thuyết của phương pháp mở rộng đa thức. | Nền tảng giải tích của xấp xỉ hàm trong Machine Learning. |

---

## Nguồn

- [1] [scikit-learn: Polynomial regression with basis functions](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
- [2] [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [3] [scikit-learn: Pipelines and common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [4] Runge, C. (1901). *Über empirische Funktionen und die Interpolation zwischen äquidistanten Ordinaten*. Zeitschrift für Mathematik und Physik, 46, 224-243.
- [5] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. (Chương 5: Basis Expansions and Regularization).
