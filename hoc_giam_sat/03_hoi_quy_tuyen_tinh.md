# Linear regression, Ridge, Lasso và Elastic Net

## Purpose

Dự đoán target liên tục bằng quan hệ tuyến tính giữa feature và target. Đây là baseline regression dễ giải thích; Ridge/Lasso/Elastic Net thêm regularization để giảm overfitting hoặc chọn feature. [1]

### Định lý Gauss-Markov & Nền tảng Ước lượng Tuyến tính Không chệch Tốt nhất (BLUE)

Trong lịch sử thống kê học, Hồi quy Tuyến tính Bình phương Tối thiểu (Ordinary Least Squares - OLS) là viên gạch nền móng đầu tiên. Theo **Định lý Gauss-Markov**, nếu mô hình hồi quy thỏa mãn 5 giả định kinh điển:
1. **Tính tuyến tính trong tham số (Linearity in Parameters):** $`y = Xw + \varepsilon`$.
2. **Kỳ vọng sai số bằng 0 (Strict Exogeneity):** $`\mathbb{E}[\varepsilon \mid X] = 0`$.
3. **Phương sai sai số thuần nhất (Homoscedasticity):** $`\mathrm{Var}(\varepsilon_i \mid X) = \sigma^2 < \infty`$.
4. **Không có hiện tượng tự tương quan giữa các sai số (No Autocorrelation):** $`\mathrm{Cov}(\varepsilon_i, \varepsilon_j \mid X) = 0`$ với mọi $`i \neq j`$.
5. **Không có hiện tượng đa cộng tuyến hoàn hảo (No Perfect Multicollinearity):** $`\mathrm{rank}(X) = p \le n`$.

Khi đó, ước lượng OLS $`\hat{w}_{\text{OLS}} = (X^\top X)^{-1} X^\top y`$ là **Ước lượng Tuyến tính Không chệch Tốt nhất (BLUE - Best Linear Unbiased Estimator)**, nghĩa là nó đạt phương sai nhỏ nhất trong tất cả các ước lượng không chệch tuyến tính.

Tuy nhiên, trong học máy hiện đại ($`p \approx n`$ hoặc $`p > n`$), việc duy trì tính "không chệch" (unbiased) thường khiến phương sai (variance) bùng nổ dữ dội. Do đó, các nhà nghiên cứu đã phát triển **Ridge (Hoerl & Kennard, 1970)**, **Lasso (Tibshirani, 1996)** và **Elastic Net (Zou & Hastie, 2005)**: chấp nhận đưa vào một lượng thiên lệch nhỏ (slight bias) để giảm thiểu mạnh mẽ phương sai, đem lại tổng sai số dự đoán (MSE) thấp hơn nhiều trên dữ liệu mới.

---

## Input

`X` có `n` hàng và `p` feature số (hoặc categorical đã encode), `y` là số thực; mỗi hàng ứng với một quan sát độc lập theo cách split đã chọn.

Nếu `y` có nhiều target, shape có thể là $`n\times m`$ với estimator hỗ trợ multi-output. Cần định nghĩa đơn vị target, khoảng giá trị hợp lý và thời điểm feature có sẵn. Với biến categorical one-hot, phải lưu mapping cột; với target lệch mạnh, cân nhắc biến đổi target nhưng phải hoàn nguyên trước khi báo cáo MAE/RMSE theo đơn vị thật. [1]

### Điều kiện Biên của Ma trận Thiết kế $`X`$

* Kích thước ma trận: $`X \in \mathbb{R}^{n \times p}`$, vector nhãn $`y \in \mathbb{R}^n`$.
* **Hiện tượng Ma trận Kỳ dị / Suy biến (Ill-conditioned Matrix):**
  Nếu hai biến bất kỳ có tương quan tuyệt đối (ví dụ: $`x_2 = 2 x_1`$), ma trận Gram $`X^\top X`$ sẽ có định thức $`\det(X^\top X) = 0`$, không thể tính toán nghịch đảo số học. Khi các biến tương quan rất cao, ma trận gần như suy biến (Condition Number lớn), khiến các thuật toán giải số (Solvers) bị mất ổn định và tạo ra các hệ số $`w`$ khổng lồ mang dấu trái ngược nhau.

---

## Output

Giá trị $`\hat{y} = b + \sum_j w_j x_j`$; hệ số `w`, intercept `b`, pipeline biến đổi feature. Hệ số chỉ có ý nghĩa trong đơn vị/thang đo của feature sau preprocessing.

### Ý nghĩa Thống kê & Giới hạn Diễn giải của Hệ số $`w_j`$

* **Định nghĩa toán học:** $`w_j = \frac{\partial \mathbb{E}[y \mid X]}{\partial x_j}`$.
* **Diễn giải Ceteris Paribus:** Khi đặc trưng $`x_j`$ tăng thêm 1 đơn vị, giá trị kỳ vọng của $`y`$ sẽ thay đổi một lượng là $`w_j`$, **với điều kiện tiên quyết là tất cả các đặc trưng còn lại trong mô hình được giữ cố định hoàn toàn không đổi**.
* **Cảnh báo nghiên cứu:** Mối liên hệ tuyến tính $`w_j`$ là mối tương quan thống kê có điều kiện, **tuyệt đối không chứng minh quan hệ nhân quả (Correlation is NOT Causation)**. Nếu trong dữ liệu thực tế tồn tại một biến ẩn (Confounding Variable) bị bỏ sót, dấu của $`w_j`$ có thể bị đảo ngược hoàn toàn (Nghịch lý Simpson).

---

## Evaluation metrics

MAE cho sai số tuyệt đối dễ diễn giải; RMSE phạt lỗi lớn mạnh hơn; R² so với dự đoán hằng theo trung bình. Xem phân phối residual theo mức dự đoán, thời gian và nhóm; so với DummyRegressor. [2]

### Các Thước đo Kiểm định Suy diễn Nâng cao (Inferential Statistics)

1. **Kiểm định Student's t-test cho từng hệ số:**

$$
t_j = \frac{\hat{w}_j}{\mathrm{SE}(\hat{w}_j)}
$$

   Trong đó sai số chuẩn (Standard Error) của hệ số được xác định bởi:

$$
\mathrm{SE}(\hat{w}_j) = \sigma \sqrt{[(X^\top X)^{-1}]_{jj}}
$$

   Nếu giá trị $`p\text{-value} > 0.05`$, biến $`x_j`$ không có ý nghĩa thống kê trong việc dự đoán $`y`$ khi đã có mặt các biến còn lại.
2. **Khoảng tin cậy cho Dự đoán (Prediction vs Confidence Intervals):**
   * *Confidence Interval:* Đo lường độ bất định khi ước lượng giá trị trung bình $`\mathbb{E}[y \mid x_0]`$.
   * *Prediction Interval:* Đo lường độ bất định khi dự đoán giá trị cụ thể của một quan sát cá thể mới $`y_0 = f(x_0) + \varepsilon`$. Khoảng dự đoán luôn rộng hơn khoảng tin cậy vì phải cộng thêm phương sai nhiễu $`\sigma^2`$.

---

## How it works

- **OLS:** tìm `w,b` làm nhỏ tổng $`(y_i - \hat{y}_i)^2`$.
- **Ridge (L2):** thêm $`\alpha \sum_j w_j^{2}`$, co hệ số và hữu ích khi feature tương quan.
- **Lasso (L1):** thêm $`\alpha \sum_j \lvert w_j\rvert`$, có thể đưa một số hệ số về 0.
- **Elastic Net:** kết hợp L1 và L2; cân bằng chọn feature với ổn định khi feature tương quan. [1]

### Objective của từng biến thể

Các biểu thức dưới đây viết tường minh intercept `b` không bị phạt, tương đương cách viết rút gọn với dữ liệu đã center phù hợp trong nhiều tài liệu. Hệ số `alpha` giữa các estimator **không so trực tiếp** vì cách chuẩn hóa loss khác nhau. Ký hiệu `1` là vector toàn số 1 dài `n`. [1]

**OLS** — Khớp bình phương sai số

$$
\min_{w,b}\;\lVert Xw+b\mathbf{1}-y\rVert_2^2
$$

**Ridge** — Co hệ số, ổn định hơn khi feature tương quan

$$
\min_{w,b}\;\lVert Xw+b\mathbf{1}-y\rVert_2^2 + \alpha\lVert w\rVert_2^2
$$

**Lasso** — Có thể đưa hệ số về đúng 0

$$
\min_{w,b}\;\frac{1}{2n}\lVert Xw+b\mathbf{1}-y\rVert_2^2 + \alpha\lVert w\rVert_1
$$

**Elastic Net** — `r=l1_ratio` cân bằng thưa và co hệ số

$$
\min_{w,b}\;\frac{1}{2n}\lVert Xw+b\mathbf{1}-y\rVert_2^2 + \alpha r\lVert w\rVert_1 + \frac{\alpha(1-r)}{2}\lVert w\rVert_2^2
$$

$`\alpha =0`$ trong Ridge tương ứng objective OLS; với Lasso/Elastic Net, không nên đặt $`\alpha =0`$ rồi mong solver có hành vi y hệt LinearRegression. Chọn $`\alpha`$ trong một dải hợp lý bằng CV và dùng scaler **trong** pipeline để penalty tác động công bằng hơn giữa feature. [1]

**Ví dụ tính tay:** với một feature và mô hình $`\hat{y} = b + wx`$, nếu $`b=2`$, $`w=3`$, $`x=4`$ thì $`\hat{y}=14`$. Nếu nhãn thật là `12`, residual $`e=y-\hat{y}=-2`$, squared error là `4`, absolute error là `2`. Train thay `b,w` để giảm tổng loss trên nhiều mẫu; ví dụ một mẫu không đủ xác định mô hình đáng tin cậy.

**Fit/predict:** OLS giải bài toán least squares bằng thuật toán đại số số học; Ridge/Lasso/Elastic Net giải objective có phạt. Sau khi fit, predict chỉ áp dụng đúng phép biến đổi feature đã fit và tính $`b+w^{\top}x`$. Không dùng công thức nghịch đảo $`(X^{\top}X)^{-1}`$ một cách mù quáng khi ma trận suy biến; implementation thực tế dùng solver ổn định hơn. [1]

**Ví dụ fit OLS một biến:** ba điểm `(1,2)`, `(2,4)`, `(3,6)` có $`\bar{x}=2`$, $`\bar{y}=4`$. Công thức $`w=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2}=\frac{4}{2}=2`$, $`b=\bar{y}-w\bar{x}=0`$; với $`x=4`$, mô hình dự đoán `8`. Đây là tính trên train; để biết có tổng quát hóa hay không vẫn phải đánh giá trên các điểm giữ riêng. Nếu mọi `x` giống nhau, mẫu số bằng 0 và không ước lượng được slope theo công thức này.

**Quy trình:** split → impute/encode/scale trong từng fold → fit OLS/Ridge/Lasso/Elastic Net → chọn `alpha`/`l1_ratio` bằng CV → xem residual và lỗi theo nhóm trên validation → khóa pipeline → test. Trong scikit-learn, Lasso/Elastic Net thường dùng coordinate descent; OLS/Ridge có các solver khác nhau theo dữ liệu. [1]

### Bản chất Hình học của OLS: Phép chiếu Trực giao (Orthogonal Projection)

Xét không gian vector $`n`$-chiều $`\mathbb{R}^n`$. Vector nhãn $`y \in \mathbb{R}^n`$ là một điểm trong không gian này.
Các cột của ma trận $`X = [x_1, x_2, \dots, x_p]`$ căng ra một không gian con tuyến tính $`p`$-chiều gọi là Không gian Cột $`\mathrm{Col}(X)`$.

* Dự đoán của mô hình là một tổ hợp tuyến tính $`\hat{y} = Xw = \sum_{j=1}^p w_j x_j`$, do đó $`\hat{y}`$ **bắt buộc phải nằm trọn vẹn bên trong $`\mathrm{Col}(X)`$**.
* Để tối thiểu hóa khoảng cách Euclid $`\|y - \hat{y}\|_2^2`$, điểm $`\hat{y}`$ tốt nhất phải là **hình chiếu trực giao của $`y`$ lên $`\mathrm{Col}(X)`$**.
* Khi đó, vector phần dư $`e = y - \hat{y}`$ vuông góc (trực giao) với mọi vector cột $`x_j`$:

$$
X^\top (y - \hat{y}) = 0 \iff X^\top (y - Xw) = 0 \iff X^\top X w = X^\top y
$$

  Từ đó suy ra phương trình chuẩn nổi tiếng: $`w = (X^\top X)^{-1} X^\top y`$.

```text
       y (Vector thực tế)
       ^
       | \
       |   \  e = y - y_hat (Trực giao vuông góc với Col(X))
       |     \
       |       v
       +--------+--------> Col(X) (Không gian cột của các biến độc lập)
       0      y_hat = X*w (Hình chiếu trực giao)
```

### Tại sao Lasso Triệt tiêu Hệ số về 0 trong khi Ridge thì Không?

Bài toán Regularization có thể viết lại dưới dạng bài toán tối ưu hóa ràng buộc Karush-Kuhn-Tucker (KKT):

$$
\min_{w} \|y - Xw\|_2^2 \quad \text{sao cho} \quad \|w\|_p \le t
$$

```text
          LASSO (L1 - Khối Kim Cương)                   RIDGE (L2 - Khối Cầu Tròn)
               w2                                           w2
               ^                                            ^
               |   .---. Đường mức MSE                      |   .---.
             / | \/     \                                   |  /     \
           /   | /\      |                              .---+---.     |
         <-----+---+-----> w1                        /      |    \    |
           \   | /  \   /                           |       |     |  /
             \ | /    `'                             \      |    / -'
               v                                      `-----+---'
                                                            v
  (Tiếp xúc tại ĐỈNH trục tọa độ -> w2 = 0)     (Tiếp xúc mượt mà -> w1, w2 đều khác 0)
```

1. **Hình học L2 (Ridge):** Vùng ràng buộc $`\|w\|_2^2 \le t`$ là một hình cầu tròn mịn. Khi các elip đường mức của hàm mất mát MSE mở rộng từ tâm nghiệm OLS, chúng sẽ tiếp xúc với hình cầu tại một điểm bất kỳ trên đường cong. Xác suất để điểm tiếp xúc rơi chính xác vào các trục tọa độ là bằng 0. Do đó, Ridge chỉ co nhỏ các hệ số lại (Shrinkage) mà không bao giờ triệt tiêu về 0.
2. **Hình học L1 (Lasso):** Vùng ràng buộc $`\|w\|_1 \le t`$ là một khối đa diện hình thoi/kim cương có **các đỉnh nhọn nằm chính xác trên các trục tọa độ**. Khi các elip đường mức MSE mở rộng, điểm tiếp xúc đầu tiên có xác suất cực cao rơi trúng vào một trong các góc nhọn này. Tại các đỉnh góc nhọn, tọa độ của một hoặc nhiều biến số bằng 0 tuyệt đối. Đây chính là cơ chế tạo ra **nghiệm thưa (Sparsity)** giúp Lasso tự động chọn lọc biến quan trọng.

### Thuật toán Coordinate Descent & Toán tử Soft-Thresholding trong Lasso

Do hàm chuẩn $`L_1`$ không khả vi tại điểm $`0`$, Lasso không thể giải trực tiếp bằng Gradient Descent thông thường mà sử dụng **Coordinate Descent (Hạ từng tọa độ)**. Tại mỗi bước lặp, thuật toán cố định toàn bộ các tham số khác và tối ưu duy nhất biến $`w_j`$ thông qua toán tử ngưỡng mềm (Soft-Thresholding Operator):

$$
w_j^* = \mathcal{S}_{\lambda}(\rho_j) = \mathrm{sign}(\rho_j) \max(0, \lvert \rho_j \rvert - \lambda)
$$

Trong đó $`\rho_j = x_j^\top (y - \sum_{k \neq j} w_k x_k)`$ là mức độ tương quan giữa biến $`x_j`$ và phần dư còn lại của mô hình. Nếu $`|\rho_j| \le \lambda`$, hệ số $`w_j`$ lập tức bị ép chặt về $`0`$.

---

## Data used

Dữ liệu bảng với target liên tục, ví dụ California Housing để thực hành regression. Cần kiểm tra missing, outlier, phân phối target và các feature tạo sau thời điểm dự đoán. Scale nên nằm trong pipeline khi dùng regularization, vì mức phạt phụ thuộc thang hệ số. [1][3]

### Tác động Phá hủy của Điểm Ngoại lai (Outliers & Leverage Points)

Hàm mất mát bình phương sai số của OLS ($`e_i^2`$) cực kỳ nhạy cảm với các điểm ngoại lai:
1. **Điểm đòn bẩy cao (High Leverage Points):** Những quan sát có giá trị đặc trưng $`x_i`$ nằm xa rời trung tâm dữ liệu.
2. **Điểm ảnh hưởng (Influential Points):** Vừa có đòn bẩy cao vừa có sai số lớn. Một điểm dị biệt duy nhất có thể bẻ cong toàn bộ đường hồi quy OLS, làm sai lệch toàn bộ các hệ số còn lại.
3. **Chẩn đoán:** Sử dụng chỉ số **Khoảng cách Cook (Cook's Distance $`D_i`$)** để phát hiện các điểm có ảnh hưởng quá mức ($`D_i > 4/n`$).

---

## Design

- Chọn OLS làm mốc; thử Ridge khi hệ số không ổn định; Lasso/Elastic Net khi muốn mô hình thưa.
- Tuning $`\alpha`$ và tỉ lệ L1/L2 bằng cross-validation chỉ trên train. Tránh diễn giải hệ số như quan hệ nhân quả; tương quan feature làm cách đọc từng hệ số khó hơn.
- Trade-off: train/inference nhanh, giải thích được nhưng quan hệ phi tuyến và tương tác phải được thiết kế thành feature. Failure modes: outlier làm lệch OLS, extrapolation ngoài vùng train, leakage, residual có cấu trúc.

**Độ phức tạp:** dự đoán một mẫu tốn xấp xỉ $`O(p)`$ phép tính sau khi feature đã sẵn sàng; chi phí fit tùy solver, `n`, `p` và sparsity. Không nên chọn chỉ từ một công thức Big-O vì tiền xử lý, số chiều và điều kiện số của `X` cũng ảnh hưởng.

**Khi nào chọn:** dữ liệu bảng có quan hệ gần tuyến tính, cần baseline và diễn giải hệ số. Khi residual có hình cong hoặc tương tác rõ, thử feature engineering, tree ensemble hoặc mô hình phi tuyến rồi so trên cùng split.

**Residual diagnostics:** vẽ $`e=y-\hat{y}`$ theo $`\hat{y}`$, thời gian và nhóm. Hình cong gợi ý thiếu tính phi tuyến/tương tác; sai số tăng theo mức $`\hat{y}`$ gợi ý phương sai thay đổi; một vài residual cực lớn cần kiểm dữ liệu và độ nhạy của squared loss. Kiểm tra ngoài miền feature train để nhận ra extrapolation. Tín hiệu này giúp đặt giả thuyết, không tự động xác định nguyên nhân. [1][2]

| Quyết định | Chọn khi | Tránh suy luận quá mức |
|---|---|---|
| OLS | Baseline, feature ít, quan hệ gần tuyến tính | Hệ số không chứng minh nhân quả |
| Ridge | Nhiều feature tương quan hoặc hệ số biến động | Không loại feature tự động |
| Lasso | Muốn mô hình thưa | Tập feature được chọn có thể không ổn định khi tương quan cao |
| Elastic Net | Muốn thưa nhưng cần xử lý nhóm feature tương quan tốt hơn | Phải tune cả `alpha` và `l1_ratio` |

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Coefficient / hệ số $`w_j`$** | Mức thay đổi của dự đoán khi feature $`x_j`$ tăng một đơn vị, với các feature khác giữ nguyên trong mô hình; không tự chứng minh quan hệ nhân quả. | Cho biết hướng và mức đổi của dự đoán theo một feature | Tham số được học khi fit; cần đọc cùng đơn vị và cách chuẩn hóa feature |
| **Intercept `b`** | Giá trị dự đoán khi các feature bằng 0 theo thang dữ liệu đã biến đổi. | Đặt mức dự đoán gốc của phương trình | Tham số được học cùng các hệ số; diễn giải chỉ khi điểm feature bằng 0 có ý nghĩa |
| **Residual $`e=y-\hat{y}`$** | Sai lệch có dấu giữa nhãn thật và dự đoán; dùng để kiểm mô hình bỏ sót cấu trúc nào. | Làm lộ sai lệch có hệ thống hoặc điểm bất thường | Dùng kiểm giả định và chẩn đoán sau khi fit trên dữ liệu phù hợp |
| **Objective / hàm mục tiêu** | Biểu thức thuật toán giảm khi fit; gồm loss dữ liệu và có thể có penalty. | Quy định mô hình phải tối ưu điều gì | Được giảm trong training; không đồng nhất với mọi metric dùng báo cáo |
| **L1 / L2 regularization** | L1 phạt tổng trị tuyệt đối hệ số và có thể tạo nghiệm thưa; L2 phạt tổng bình phương để co hệ số. | Co hệ số và hạn chế overfit; L1 có thể đưa một số hệ số về 0 | Là penalty thêm vào objective của Lasso hoặc Ridge |
| **`alpha`** | Độ mạnh của penalty trong estimator tương ứng; không so số `alpha` trực tiếp giữa Ridge và Lasso. | Điều khiển mức phạt, lớn hơn thường co hệ số mạnh hơn | Hyperparameter cần chọn bằng validation hoặc CV trong cùng mô hình |
| **Collinearity** | Các feature mang thông tin gần trùng nhau, làm hệ số OLS nhạy thay đổi dữ liệu. | Làm hệ số riêng lẻ kém ổn định dù dự đoán có thể vẫn tốt | Cần kiểm trước khi diễn giải từng hệ số và khi chọn regularization |
| **Extrapolation / ngoại suy** | Dự đoán ở vùng feature nằm ngoài miền train; sai số có thể tăng dù test gần train tốt. | Có thể tạo dự đoán xa thực tế dù khớp tốt trong miền train | Là giới hạn cần nêu khi dùng mô hình cho đầu vào mới |
| **BLUE** | Ước lượng tuyến tính không chệch tốt nhất (Best Linear Unbiased Estimator). | Đảm bảo phương sai nhỏ nhất theo định lý Gauss-Markov. | Tiêu chuẩn lý thuyết đánh giá chất lượng ước lượng OLS. |
| **Soft-Thresholding** | Phép toán co cụm phi tuyến đưa các giá trị nhỏ về 0. | Cơ chế cốt lõi tạo nghiệm thưa trong thuật toán Lasso. | Bước tính toán trong solver Coordinate Descent. |
| **Cook's Distance** | Thước đo lượng hóa ảnh hưởng của từng điểm dữ liệu lên toàn bộ vector hệ số. | Phát hiện các điểm ngoại lai có tính đòn bẩy nguy hiểm. | Công cụ chẩn đoán dữ liệu hồi quy chuyên sâu. |

---

## Nguồn

- [1] [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [2] [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [3] [scikit-learn: California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- [4] Hoerl, A. E., & Kennard, R. W. (1970). *Ridge regression: Biased estimation for nonorthogonal problems*. Technometrics, 12(1), 55-67.
- [5] Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso*. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.
- [6] Zou, H., & Hastie, T. (2005). *Regularization and variable selection via the elastic net*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2), 301-320.
