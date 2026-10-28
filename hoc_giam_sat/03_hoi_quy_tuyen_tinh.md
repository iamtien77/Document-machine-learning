# Linear regression, Ridge, Lasso và Elastic Net

## Purpose

Dự đoán target liên tục bằng quan hệ tuyến tính giữa feature và target. Đây là baseline regression dễ giải thích; Ridge/Lasso/Elastic Net thêm regularization để giảm overfitting hoặc chọn feature. [1]

## Input

`X` có `n` hàng và `p` feature số (hoặc categorical đã encode), `y` là số thực; mỗi hàng ứng với một quan sát độc lập theo cách split đã chọn.

## Output

Giá trị `y_hat = b + Σ w_j x_j`; hệ số `w`, intercept `b`, pipeline biến đổi feature. Hệ số chỉ có ý nghĩa trong đơn vị/thang đo của feature sau preprocessing.

## Evaluation metrics

MAE cho sai số tuyệt đối dễ diễn giải; RMSE phạt lỗi lớn mạnh hơn; R² so với dự đoán hằng theo trung bình. Xem phân phối residual theo mức dự đoán, thời gian và nhóm; so với DummyRegressor. [2]

## How it works

- **OLS:** tìm `w,b` làm nhỏ tổng `(y_i - y_hat_i)^2`.
- **Ridge (L2):** thêm `α Σ w_j²`, co hệ số và hữu ích khi feature tương quan.
- **Lasso (L1):** thêm `α Σ |w_j|`, có thể đưa một số hệ số về 0.
- **Elastic Net:** kết hợp L1 và L2; cân bằng chọn feature với ổn định khi feature tương quan. [1]

**Ví dụ tính tay:** với một feature và mô hình `y_hat = b + wx`, nếu `b=2`, `w=3`, `x=4` thì `y_hat=14`. Nếu nhãn thật là `12`, residual `e=y-y_hat=-2`, squared error là `4`, absolute error là `2`. Train thay `b,w` để giảm tổng loss trên nhiều mẫu; ví dụ một mẫu không đủ xác định mô hình đáng tin cậy.

**Fit/predict:** OLS giải bài toán least squares bằng thuật toán đại số số học; Ridge/Lasso/Elastic Net giải objective có phạt. Sau khi fit, predict chỉ áp dụng đúng phép biến đổi feature đã fit và tính `b+wᵀx`. Không dùng công thức nghịch đảo `(XᵀX)^-1` một cách mù quáng khi ma trận suy biến; implementation thực tế dùng solver ổn định hơn. [1]

## Data used

Dữ liệu bảng với target liên tục, ví dụ California Housing để thực hành regression. Cần kiểm tra missing, outlier, phân phối target và các feature tạo sau thời điểm dự đoán. Scale nên nằm trong pipeline khi dùng regularization, vì mức phạt phụ thuộc thang hệ số. [1][3]

## Design

- Chọn OLS làm mốc; thử Ridge khi hệ số không ổn định; Lasso/Elastic Net khi muốn mô hình thưa.
- Tuning `α` và tỉ lệ L1/L2 bằng cross-validation chỉ trên train. Tránh diễn giải hệ số như quan hệ nhân quả; tương quan feature làm cách đọc từng hệ số khó hơn.
- Trade-off: train/inference nhanh, giải thích được nhưng quan hệ phi tuyến và tương tác phải được thiết kế thành feature. Failure modes: outlier làm lệch OLS, extrapolation ngoài vùng train, leakage, residual có cấu trúc.

**Độ phức tạp:** dự đoán một mẫu tốn xấp xỉ `O(p)` phép tính sau khi feature đã sẵn sàng; chi phí fit tùy solver, `n`, `p` và sparsity. Không nên chọn chỉ từ một công thức Big-O vì tiền xử lý, số chiều và điều kiện số của `X` cũng ảnh hưởng.

**Khi nào chọn:** dữ liệu bảng có quan hệ gần tuyến tính, cần baseline và diễn giải hệ số. Khi residual có hình cong hoặc tương tác rõ, thử feature engineering, tree ensemble hoặc mô hình phi tuyến rồi so trên cùng split.

## Nguồn

- [1] [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [2] [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [3] [scikit-learn: California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
