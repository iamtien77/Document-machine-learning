# Linear regression, Ridge, Lasso và Elastic Net

## Purpose

Dự đoán target liên tục bằng quan hệ tuyến tính giữa feature và target. Đây là baseline regression dễ giải thích; Ridge/Lasso/Elastic Net thêm regularization để giảm overfitting hoặc chọn feature. [1]

## Input

`X` có `n` hàng và `p` feature số (hoặc categorical đã encode), `y` là số thực; mỗi hàng ứng với một quan sát độc lập theo cách split đã chọn.

Nếu `y` có nhiều target, shape có thể là $`n\times m`$ với estimator hỗ trợ multi-output. Cần định nghĩa đơn vị target, khoảng giá trị hợp lý và thời điểm feature có sẵn. Với biến categorical one-hot, phải lưu mapping cột; với target lệch mạnh, cân nhắc biến đổi target nhưng phải hoàn nguyên trước khi báo cáo MAE/RMSE theo đơn vị thật. [1]

## Output

Giá trị $`\hat{y} = b + \sum_j w_j x_j`$; hệ số `w`, intercept `b`, pipeline biến đổi feature. Hệ số chỉ có ý nghĩa trong đơn vị/thang đo của feature sau preprocessing.

## Evaluation metrics

MAE cho sai số tuyệt đối dễ diễn giải; RMSE phạt lỗi lớn mạnh hơn; R² so với dự đoán hằng theo trung bình. Xem phân phối residual theo mức dự đoán, thời gian và nhóm; so với DummyRegressor. [2]

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

## Data used

Dữ liệu bảng với target liên tục, ví dụ California Housing để thực hành regression. Cần kiểm tra missing, outlier, phân phối target và các feature tạo sau thời điểm dự đoán. Scale nên nằm trong pipeline khi dùng regularization, vì mức phạt phụ thuộc thang hệ số. [1][3]

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

## Nguồn

- [1] [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [2] [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [3] [scikit-learn: California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
