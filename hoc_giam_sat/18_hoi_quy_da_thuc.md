# Polynomial Regression — hồi quy đa thức

> PDF [Chương 2](<../Tài liệu học tập/ML_Chapter2_new.pdf>) giới thiệu Polynomial Regression ở tr. 51 và 55. Đây là **linear model trên feature đã biến đổi phi tuyến**, không phải một thuật toán tối ưu hoàn toàn tách khỏi OLS/Ridge/Lasso. [1]

## Purpose

Mô hình hóa quan hệ cong giữa feature và target số bằng cách tạo các lũy thừa/tương tác rồi fit một mô hình tuyến tính theo hệ số. Nó là bước thử hợp lý khi residual của linear regression có cấu trúc cong nhưng vẫn muốn giữ một mô hình tương đối đơn giản và dễ kiểm tra. [1]

## Input

$`X \in \mathbb{R}^{n\times p}`$, target liên tục `y`, bậc đa thức `d`, có/không tạo interaction, estimator cuối (OLS/Ridge/Lasso), quy tắc split và metric regression. Feature phải có sẵn tại thời điểm dự đoán; missing/categorical cần được xử lý trong pipeline trước hoặc theo thiết kế phù hợp.

## Output

Pipeline gồm `PolynomialFeatures` đã định cấu hình và estimator đã fit; dự đoán $`\hat{y}`$ là số theo đơn vị target. Với một biến và bậc 2, $`\hat{y}=b+w_{1}x+w_{2}x^{2}`$. Với hai biến và bậc 2, có thể có $`x_{1}`$, $`x_{2}`$, $`x_{1}^{2}`$, $`x_{1}x_{2}`$, $`x_{2}^{2}`$. Mô hình phi tuyến theo `x` nhưng **tuyến tính theo các hệ số `w`**. [1]

## Evaluation metrics

So MAE/RMSE/R² với hồi quy tuyến tính bậc 1 và DummyRegressor trên cùng split. Xem residual theo `x`, theo $`\hat{y}`$ và theo vùng ngoại suy; báo cáo cả độ dao động giữa các fold. Nếu sai số train giảm nhưng validation tăng khi `d` lớn, đó là dấu hiệu overfit. [2]

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

## Data used

Phù hợp dữ liệu số có quan hệ mượt và số feature ban đầu không quá lớn. Ví dụ giá nhà theo diện tích có thể có hiệu ứng cong, nhưng phải kiểm xem đường bậc 2 có phản ánh dữ liệu hay chỉ khớp nhiễu. Với biến categorical, one-hot rồi tạo mọi tích có thể gây số chiều rất lớn và cách diễn giải khó. Với dữ liệu chuỗi thời gian, chia theo thời gian trước khi chọn `degree`.

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

## Nguồn

- [1] [scikit-learn: Polynomial regression with basis functions](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
- [2] [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [3] [scikit-learn: Pipelines and common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
