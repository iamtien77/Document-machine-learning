# Mở rộng: LDA, QDA và Gaussian Process

## Purpose

Ba mô hình này mở rộng bản đồ học giám sát. **LDA/QDA** là classifier xác suất dựa trên phân phối feature theo lớp; **Gaussian Process Regression (GPR)** là regression phi tham số cung cấp dự đoán và độ bất định theo giả định kernel. Chúng hữu ích để hiểu cách giả định xác suất ảnh hưởng đến ranh giới và độ tin cậy. [1][2]

## Input

- LDA/QDA: `X` số, `y` nhãn lớp; giả định phân phối `X|y=k` gần Gaussian đa biến. LDA giả định các lớp dùng chung covariance `Σ`, QDA ước lượng `Σₖ` riêng cho từng lớp. [1]
- GPR: `X` số, `y` liên tục, kernel và tham số nhiễu; số mẫu vừa phải do chi phí ma trận kernel. [2]

Ký hiệu `n` mẫu và `p` feature. LDA/QDA cần đủ mẫu mỗi lớp để ước lượng mean và covariance đáng tin; QDA ước lượng một ma trận `p×p` cho **mỗi** lớp nên đòi hỏi dữ liệu nhiều hơn LDA. GPR tạo ma trận kernel `n×n`, vì thế số mẫu ảnh hưởng mạnh đến chi phí fit. [1][2]

## Output

- LDA/QDA trả nhãn, xác suất theo lớp; LDA còn có thể tạo biểu diễn giảm chiều có giám sát. Ranh giới LDA tuyến tính, QDA thường bậc hai. [1]
- GPR trả giá trị trung bình dự đoán và có thể trả độ lệch chuẩn dưới mô hình đã chọn. Độ bất định này phụ thuộc kernel, giả định nhiễu và dữ liệu; không là bảo đảm tuyệt đối về sai số thực tế. [2]

## Evaluation metrics

LDA/QDA: confusion matrix, F1/recall theo lớp, log loss/calibration nếu dùng xác suất. GPR: MAE/RMSE trên test, residual theo vùng feature; nếu dùng interval, kiểm coverage thực nghiệm trên dữ liệu chưa thấy. [3]

## How it works

**LDA/QDA:** ước lượng prior `P(y=k)`, mean `μₖ` và covariance. Theo Bayes, chọn lớp có log-posterior lớn nhất. Bỏ hằng số chung, score QDA là `log P(y=k) - ½log|Σₖ| - ½(x-μₖ)ᵀΣₖ⁻¹(x-μₖ)`. Khi mọi `Σₖ` giống nhau, các hạng bậc hai chung triệt tiêu khi so lớp, tạo ranh giới tuyến tính của LDA. [1]

**Ví dụ 1D:** hai lớp có mean `0` và `2`, variance bằng nhau, prior bằng nhau. Với điểm `x=1.5`, khoảng cách tới mean lớp 2 nhỏ hơn nên LDA chọn lớp 2. Nếu variance/prior khác, ranh giới đổi; không thể chỉ nhìn khoảng cách Euclidean.

**GPR:** đặt prior Gaussian lên các giá trị hàm; kernel `k(x,x')` mô tả mức tương quan giữa hai đầu vào. Từ các cặp train, tính phân phối hậu nghiệm ở điểm mới để lấy mean và variance. Tham số kernel thường chọn qua tối ưu marginal likelihood, nhưng vẫn cần đánh giá trên dữ liệu chưa thấy. [2]

### LDA so với QDA: fit và dự đoán

Ở mỗi lớp `k`, fit prior `π_k=n_k/n` và vector mean `μ_k`. LDA gộp thông tin các lớp để ước lượng **một** covariance chung `Σ`; QDA ước lượng covariance riêng `Σ_k`. Khi dự đoán, tính log-posterior từng lớp và chọn lớp có điểm lớn nhất. Nếu prior lệch so với dữ liệu vận hành, phân phối xác suất và ranh giới có thể lệch dù các mean/covariance đã được fit đúng trên train. [1]

| | LDA | QDA |
|---|---|---|
| Ranh giới | Tuyến tính | Bậc hai trong trường hợp tổng quát |
| Covariance | Chung cho các lớp | Riêng cho từng lớp |
| Dữ liệu cần | Ít tham số hơn | Nhiều mẫu mỗi lớp hơn để ước lượng ổn định |
| Failure mode | Underfit nếu covariance khác nhau rõ | Covariance suy biến/overfit khi lớp ít mẫu |

**Ví dụ quyết định:** nếu hai lớp 1D có prior bằng nhau, mean `0` và `2`, variance cùng bằng `1`, ranh giới LDA ở `x=1`. Điểm `x=1.5` thuộc lớp có mean `2`; khi variance của lớp thứ hai lớn hơn, QDA không còn ranh giới đơn giản tại trung điểm. Đây là ví dụ giả định, không phải điểm số thực nghiệm. [1]

### GPR: posterior mean và variance

Với prior mean bằng 0 và nhiễu Gaussian có phương sai `σ_n²`, đặt `K = [k(x_i,x_j)] + σ_n²I`, `k_*=[k(x_i,x_*)]`. Dự đoán giá trị hàm ẩn tại điểm mới có `μ_* = k_*ᵀK⁻¹y` và `v_* = k(x_*,x_*) - k_*ᵀK⁻¹k_*`. Đây là công thức cho **latent function**; phương sai của quan sát mới còn cộng thành phần nhiễu nếu mô hình hóa observation noise. Trong tính toán số, giải hệ tuyến tính bằng phân rã ma trận thay vì lập nghịch đảo tường minh. [2]

**Trực giác:** gần các điểm train theo kernel, `k_*` lớn nên mean chịu ảnh hưởng mạnh từ nhãn train và variance thường nhỏ hơn prior. Xa vùng dữ liệu, mean quay về prior mean và variance có thể tăng. Mức tăng phụ thuộc kernel/length-scale và noise; một khoảng `μ_*±1.96√v_*` chỉ có cách hiểu xác suất dưới giả định posterior phù hợp, không tự bảo đảm coverage thực tế. [2]

**Fit/predict:** chọn kernel (ví dụ RBF hoặc Matérn) và mô hình nhiễu → fit tham số kernel qua log-marginal-likelihood hoặc giữ cố định → giải hệ kernel trên train → tính mean/variance cho điểm mới → kiểm MAE/RMSE và coverage trên dữ liệu giữ riêng. Tối ưu marginal likelihood có thể gặp nghiệm cục bộ; thử nhiều khởi tạo hoặc kiểm CV khi cần. [2]

## Data used

LDA/QDA hợp khi feature số và giả định Gaussian theo lớp tương đối hợp lý; QDA cần đủ mẫu mỗi lớp để ước lượng covariance riêng. GPR phù hợp regression tập nhỏ/vừa khi muốn mô hình hóa độ bất định và quan hệ mượt theo kernel. Iris có thể dùng minh họa LDA/QDA, còn một tập regression nhỏ như Diabetes có thể dùng minh họa GPR. [1][2][4]

## Design

- LDA ít tham số covariance hơn QDA, thường ổn định hơn khi mẫu/lớp ít. QDA linh hoạt hơn nhưng dễ ước lượng covariance kém; cân nhắc shrinkage/regularization theo implementation. [1]
- GPR cần chọn kernel phù hợp và scale feature; train với kernel dense thường tăng mạnh theo `n` (ma trận `n×n`, phép phân tích ma trận có chi phí bậc ba trong cách tính chuẩn), nên không mặc định dùng cho dữ liệu rất lớn. [2]
- Failure modes: giả định Gaussian không hợp, covariance suy biến, extrapolation thiếu dữ liệu, độ bất định GPR bị hiểu là hiệu chuẩn tự động.

| Hyperparameter/thiết kế | Ý nghĩa | Kiểm bằng gì |
|---|---|---|
| LDA/QDA `priors` | Tần suất lớp trước khi nhìn feature | Log loss, calibration, lớp hiếm |
| LDA `shrinkage` / QDA regularization theo API | Ổn định ước lượng covariance | CV trên train; kiểm cảnh báo suy biến |
| GPR kernel và length-scale | Độ mượt và phạm vi tương quan | Residual/MAE, coverage, ngoại suy |
| GPR `alpha` hoặc `WhiteKernel` | Mô hình hóa/ổn định nhiễu | Chất lượng dự đoán và độ bất định |

`alpha` trong `GaussianProcessRegressor` là mức nhiễu/regularization được thêm vào đường chéo kernel; không nhầm nó với `alpha` regularization của Ridge/Lasso. GPR chuẩn với kernel dense cần lưu `O(n²)` phần tử và giải hệ thường có chi phí bậc ba theo `n`; cân nhắc phương pháp xấp xỉ hoặc mô hình khác khi dữ liệu lớn. [2]

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Prior lớp `π_k`** | Tần suất/xác suất lớp trước khi xét feature của mẫu mới. | Cho lớp phổ biến hơn lợi thế ban đầu khi bằng chứng feature tương đương | Thành phần của hàm phân biệt LDA và QDA |
| **Mean `μ_k`** | Vector trung bình feature của các mẫu thuộc lớp `k`. | Xác định tâm phân phối của mỗi lớp | Tham số ước lượng từ train cho LDA và QDA |
| **Covariance `Σ`** | Ma trận mô tả độ biến thiên từng feature và mức biến thiên cùng nhau giữa hai feature. | Mô tả độ phân tán và tương quan giữa feature | Tham số phân phối Gaussian; LDA dùng chung, QDA dùng riêng theo lớp |
| **Shrinkage** | Trộn ước lượng covariance từ dữ liệu với một cấu trúc ổn định hơn để giảm nhiễu ước lượng. | Ổn định ma trận khi ít dữ liệu hoặc nhiều feature | Kỹ thuật ước lượng covariance trong LDA |
| **Kernel `k(x,x')`** | Hàm biểu diễn mức tương quan giả định của giá trị hàm tại hai đầu vào trong GP. | Quy định độ liên hệ giữa giá trị hàm tại các điểm đầu vào | Giả định cấu trúc của Gaussian Process Regression |
| **Length-scale** | Tham số kernel mô tả khoảng cách mà đầu vào còn tương quan đáng kể theo mô hình. | Điều khiển độ mượt và phạm vi ảnh hưởng của điểm train | Hyperparameter kernel của GP |
| **Posterior mean/variance** | Trung bình và phương sai dự đoán sau khi điều kiện hóa GP trên dữ liệu train. | Cho giá trị dự đoán trung tâm và độ bất định theo mô hình | Đầu ra của GP khi suy luận tại điểm mới |
| **Coverage** | Tỷ lệ nhãn thật nằm trong khoảng dự đoán trên dữ liệu giữ riêng; dùng kiểm độ bất định thực nghiệm. | Kiểm xem khoảng dự đoán có bao phủ đúng tần suất kỳ vọng | Metric đánh giá độ bất định trên dữ liệu giữ riêng |

## Nguồn

- [1] [scikit-learn: Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
- [2] [scikit-learn: Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [3] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [scikit-learn: Datasets](https://scikit-learn.org/stable/datasets.html)
