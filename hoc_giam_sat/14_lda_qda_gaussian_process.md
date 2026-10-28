# Mở rộng: LDA, QDA và Gaussian Process

## Purpose

Ba mô hình này mở rộng bản đồ học giám sát. **LDA/QDA** là classifier xác suất dựa trên phân phối feature theo lớp; **Gaussian Process Regression (GPR)** là regression phi tham số cung cấp dự đoán và độ bất định theo giả định kernel. Chúng hữu ích để hiểu cách giả định xác suất ảnh hưởng đến ranh giới và độ tin cậy. [1][2]

## Input

- LDA/QDA: `X` số, `y` nhãn lớp; giả định phân phối `X|y=k` gần Gaussian đa biến. LDA giả định các lớp dùng chung covariance `Σ`, QDA ước lượng `Σₖ` riêng cho từng lớp. [1]
- GPR: `X` số, `y` liên tục, kernel và tham số nhiễu; số mẫu vừa phải do chi phí ma trận kernel. [2]

## Output

- LDA/QDA trả nhãn, xác suất theo lớp; LDA còn có thể tạo biểu diễn giảm chiều có giám sát. Ranh giới LDA tuyến tính, QDA thường bậc hai. [1]
- GPR trả giá trị trung bình dự đoán và có thể trả độ lệch chuẩn dưới mô hình đã chọn. Độ bất định này phụ thuộc kernel, giả định nhiễu và dữ liệu; không là bảo đảm tuyệt đối về sai số thực tế. [2]

## Evaluation metrics

LDA/QDA: confusion matrix, F1/recall theo lớp, log loss/calibration nếu dùng xác suất. GPR: MAE/RMSE trên test, residual theo vùng feature; nếu dùng interval, kiểm coverage thực nghiệm trên dữ liệu chưa thấy. [3]

## How it works

**LDA/QDA:** ước lượng prior `P(y=k)`, mean `μₖ` và covariance. Theo Bayes, chọn lớp có log-posterior lớn nhất. Bỏ hằng số chung, score QDA là `log P(y=k) - ½log|Σₖ| - ½(x-μₖ)ᵀΣₖ⁻¹(x-μₖ)`. Khi mọi `Σₖ` giống nhau, các hạng bậc hai chung triệt tiêu khi so lớp, tạo ranh giới tuyến tính của LDA. [1]

**Ví dụ 1D:** hai lớp có mean `0` và `2`, variance bằng nhau, prior bằng nhau. Với điểm `x=1.5`, khoảng cách tới mean lớp 2 nhỏ hơn nên LDA chọn lớp 2. Nếu variance/prior khác, ranh giới đổi; không thể chỉ nhìn khoảng cách Euclidean.

**GPR:** đặt prior Gaussian lên các giá trị hàm; kernel `k(x,x')` mô tả mức tương quan giữa hai đầu vào. Từ các cặp train, tính phân phối hậu nghiệm ở điểm mới để lấy mean và variance. Tham số kernel thường chọn qua tối ưu marginal likelihood, nhưng vẫn cần đánh giá trên dữ liệu chưa thấy. [2]

## Data used

LDA/QDA hợp khi feature số và giả định Gaussian theo lớp tương đối hợp lý; QDA cần đủ mẫu mỗi lớp để ước lượng covariance riêng. GPR phù hợp regression tập nhỏ/vừa khi muốn mô hình hóa độ bất định và quan hệ mượt theo kernel. Iris có thể dùng minh họa LDA/QDA, còn một tập regression nhỏ như Diabetes có thể dùng minh họa GPR. [1][2][4]

## Design

- LDA ít tham số covariance hơn QDA, thường ổn định hơn khi mẫu/lớp ít. QDA linh hoạt hơn nhưng dễ ước lượng covariance kém; cân nhắc shrinkage/regularization theo implementation. [1]
- GPR cần chọn kernel phù hợp và scale feature; train với kernel dense thường tăng mạnh theo `n` (ma trận `n×n`, phép phân tích ma trận có chi phí bậc ba trong cách tính chuẩn), nên không mặc định dùng cho dữ liệu rất lớn. [2]
- Failure modes: giả định Gaussian không hợp, covariance suy biến, extrapolation thiếu dữ liệu, độ bất định GPR bị hiểu là hiệu chuẩn tự động.

## Nguồn

- [1] [scikit-learn: Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
- [2] [scikit-learn: Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [3] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [scikit-learn: Datasets](https://scikit-learn.org/stable/datasets.html)
