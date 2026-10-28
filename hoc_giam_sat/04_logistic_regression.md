# Logistic regression

## Purpose

Phân loại bằng mô hình tuyến tính trên log-odds; baseline mạnh, dễ triển khai cho classification nhị phân và nhiều lớp. Tên gọi có chữ regression nhưng đây là thuật toán **classification**. [1]

## Input

`X` dạng số hoặc đã encode, `y` là nhãn lớp; cần xác định lớp dương và chi phí false positive/false negative.

Với `n` mẫu và `p` feature, `X ∈ R^(n×p)`. Binary target có hai lớp; multiclass một nhãn/mẫu có `K>2` lớp. Nếu dữ liệu lớp hiếm, lưu `support` của từng lớp và kiểm các fold có đủ lớp để fit và tính metric. [1][2]

## Output

Điểm tuyến tính `z=b+wᵀx`; nhị phân có `p(y=1|x)=1/(1+e^-z)`, nhãn được quyết định bằng threshold. Mô hình đa lớp có thể trả phân phối xác suất qua các lớp tùy cấu hình. [1]

## Evaluation metrics

Precision/recall/F1 theo lớp, confusion matrix; ROC-AUC hoặc PR-AUC cho khả năng xếp hạng; log loss/Brier và calibration khi xác suất được sử dụng; so với DummyClassifier. Accuracy chỉ phù hợp khi phân phối lớp và chi phí lỗi cho phép. [2]

## How it works

Fit trọng số bằng cách giảm log loss (negative log-likelihood), thường kèm regularization L2 hoặc L1. Điều chỉnh `C` (nghịch đảo cường độ regularization trong scikit-learn) và threshold trên validation; threshold không phải tham số học của logistic regression. [1][3]

Với nhị phân `y∈{0,1}`, loss trung bình là `-(1/n)Σ[yᵢ log(pᵢ)+(1-yᵢ)log(1-pᵢ)]`; mô hình giảm loss để tăng xác suất cho nhãn đúng. Với `z=0`, sigmoid cho `p=0.5`; `z=ln(3)` cho `p=0.75`. Nếu threshold là `0.8`, ví dụ thứ hai vẫn thuộc lớp âm. Do đó xác suất và quyết định là hai tầng khác nhau. [1][3]

**Fit/predict:** encode/scale từ train → tối ưu trọng số theo log loss + phạt → tính score `z` ở mẫu mới → chuyển thành xác suất → áp threshold đã chọn. Với nhiều lớp, công thức và solver phụ thuộc chiến lược đa lớp; cần kiểm cấu hình thư viện đang dùng. [1]

### Binary: gradient và một bước học

Với `pᵢ=σ(wᵀxᵢ+b)` và log loss chưa regularization, gradient theo `w` là `(1/n)Xᵀ(p-y)` và theo `b` là trung bình `(pᵢ-yᵢ)`. Ví dụ một mẫu `x=2`, `y=1`, khởi tạo `w=b=0` cho `p=0.5`; gradient `dw=-1`, `db=-0.5`. Một bước gradient descent với learning rate `0.1` cho `w=0.1`, `b=0.05`, score mới ở `x=2` là `0.25`, nên xác suất tăng lên khoảng `0.562`. Đây là ví dụ cơ chế tối ưu, chưa phản ánh kết quả khi train nhiều mẫu và regularization. [1]

### Multiclass: softmax

Với mỗi lớp `k`, tính logit `z_k=w_kᵀx+b_k`, rồi `p_k=exp(z_k)/Σ_j exp(z_j)`; các xác suất cộng bằng 1. Loss của một mẫu có lớp thật `c` là `-log p_c`. Để ổn định số, phần mềm thường trừ `max(z)` trước khi tính exp. Nếu logits là `[0, ln(2), ln(3)]`, xác suất tương ứng `[1/6, 2/6, 3/6]`, nên lớp thứ ba được chọn khi dùng argmax. Với multiclass, threshold và chi phí có thể cần quy tắc quyết định riêng, không máy móc áp một ngưỡng nhị phân cho mọi lớp. [1]

**Quy trình chọn mô hình:** split → fit encoder/scaler trong fold train → fit LogisticRegression → tune regularization/solver phù hợp bằng CV → so precision/recall và log loss trên validation → chọn threshold nếu cần trên dữ liệu hợp lệ → test cuối. Nếu fit calibrator, dữ liệu dùng cho calibrator phải độc lập với dữ liệu fit classifier hoặc lấy từ dự đoán out-of-fold đúng cách. [1][3][5]

## Data used

Dữ liệu bảng hoặc vector thưa của văn bản, ví dụ UCI Adult cho phân loại thu nhập hoặc Iris cho nhiều lớp. Cần encode categorical, impute missing và scale feature số khi regularization/solver nhạy thang đo. [4]

## Design

- Bắt đầu với LogisticRegression + pipeline tiền xử lý; dùng `class_weight` hoặc chiến lược lấy mẫu chỉ khi metric và phân phối lớp yêu cầu.
- Ưu điểm: nhanh, hệ số giúp giải thích hướng liên hệ, có đầu ra xác suất. Hạn chế: ranh giới quyết định tuyến tính trừ khi thêm tương tác/biến đổi.
- Failure modes: nhãn lệch, feature phụ thuộc mạnh, dữ liệu tương lai bị lộ, threshold 0.5 không khớp chi phí, xác suất cần kiểm calibration.

| Quyết định | Tác dụng | Cần kiểm |
|---|---|---|
| `C` | `C` nhỏ thường regularization mạnh hơn | Train–validation gap, hệ số, metric chính |
| L1/L2 theo solver hỗ trợ | L1 có thể tạo hệ số thưa; L2 co hệ số | API của phiên bản thư viện, ổn định feature |
| `class_weight` | Đổi trọng số lỗi theo lớp trong loss | Precision/recall và xác suất/calibration sau fit |
| Threshold | Biến score/xác suất thành nhãn | Chọn trên validation theo chi phí, không trên test |

Nếu resampling lớp hiếm, thực hiện **trong train fold** của CV; oversampling trước split làm bản sao/điểm tổng hợp rò sang validation. `class_weight` và threshold có tác dụng khác nhau: cái thứ nhất đổi quá trình fit, cái thứ hai đổi quy tắc quyết định sau fit. [3][6]

**Độ phức tạp:** tính score nhị phân cho một mẫu khoảng `O(p)`; fit lặp nhiều vòng và phụ thuộc solver/sparsity. `C` nhỏ thường regularization mạnh hơn. Tránh đọc `exp(w_j)` như tác động nhân quả; đó là thay đổi odds có điều kiện theo các feature trong mô hình, dưới các giả định dữ liệu tương ứng.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Logit / log-odds** | `z=log[p/(1-p)]`; mô hình logistic học `z` tuyến tính theo feature. | Biến xác suất thành thang số thực để mô hình hóa tuyến tính | Đầu ra tuyến tính trước sigmoid của phân loại nhị phân |
| **Sigmoid** | Hàm `1/(1+e^-z)` đổi một logit nhị phân thành giá trị trong `(0,1)`. | Đưa logit về khoảng xác suất từ 0 đến 1 | Tầng chuyển điểm sang xác suất trong mô hình nhị phân |
| **Softmax** | Chuẩn hóa nhiều logit thành xác suất các lớp có tổng bằng 1. | Biến nhiều logit thành một phân phối xác suất trên các lớp | Tầng đầu ra của phân loại nhiều lớp loại trừ nhau |
| **Log loss / cross-entropy** | Loss phạt khi mô hình gán xác suất thấp cho lớp thật, đặc biệt phạt dự đoán sai nhưng quá tự tin. | Phạt mạnh dự đoán sai nhưng tự tin | Hàm loss thường được tối ưu khi fit bộ phân loại xác suất |
| **Threshold / ngưỡng** | Quy tắc biến score/xác suất thành nhãn; được chọn sau fit theo chi phí lỗi. | Thay đổi cân bằng precision và recall mà không fit lại mô hình | Quy tắc ra quyết định sau khi có score hoặc xác suất |
| **`C`** | Tham số nghịch với cường độ regularization trong `LogisticRegression` của scikit-learn. | Điều khiển regularization; C nhỏ thường phạt mạnh hơn | Hyperparameter của scikit-learn, chọn bằng validation hoặc CV |
| **`class_weight`** | Trọng số lỗi theo lớp trong quá trình fit; khác với thay threshold sau fit. | Tăng ảnh hưởng của lớp được gán trọng số lớn trong loss | Xử lý bất cân bằng lớp ngay trong training |
| **Calibration / hiệu chuẩn** | Kiểm xem nhóm mẫu dự đoán xác suất gần `p` có tần suất dương thật gần `p` hay không. | Cho biết xác suất báo ra có đáng tin như tần suất thực không | Kiểm sau fit khi xác suất dùng để ước lượng rủi ro hoặc đặt ngưỡng |

## Nguồn

- [1] [scikit-learn: Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [2] [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [3] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [4] [UCI: Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [5] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [6] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
