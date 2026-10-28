# Logistic regression

## Purpose

Phân loại bằng mô hình tuyến tính trên log-odds; baseline mạnh, dễ triển khai cho classification nhị phân và nhiều lớp. Tên gọi có chữ regression nhưng đây là thuật toán **classification**. [1]

## Input

`X` dạng số hoặc đã encode, `y` là nhãn lớp; cần xác định lớp dương và chi phí false positive/false negative.

## Output

Điểm tuyến tính `z=b+wᵀx`; nhị phân có `p(y=1|x)=1/(1+e^-z)`, nhãn được quyết định bằng threshold. Mô hình đa lớp có thể trả phân phối xác suất qua các lớp tùy cấu hình. [1]

## Evaluation metrics

Precision/recall/F1 theo lớp, confusion matrix; ROC-AUC hoặc PR-AUC cho khả năng xếp hạng; log loss/Brier và calibration khi xác suất được sử dụng; so với DummyClassifier. Accuracy chỉ phù hợp khi phân phối lớp và chi phí lỗi cho phép. [2]

## How it works

Fit trọng số bằng cách giảm log loss (negative log-likelihood), thường kèm regularization L2 hoặc L1. Điều chỉnh `C` (nghịch đảo cường độ regularization trong scikit-learn) và threshold trên validation; threshold không phải tham số học của logistic regression. [1][3]

Với nhị phân `y∈{0,1}`, loss trung bình là `-(1/n)Σ[yᵢ log(pᵢ)+(1-yᵢ)log(1-pᵢ)]`; mô hình giảm loss để tăng xác suất cho nhãn đúng. Với `z=0`, sigmoid cho `p=0.5`; `z=ln(3)` cho `p=0.75`. Nếu threshold là `0.8`, ví dụ thứ hai vẫn thuộc lớp âm. Do đó xác suất và quyết định là hai tầng khác nhau. [1][3]

**Fit/predict:** encode/scale từ train → tối ưu trọng số theo log loss + phạt → tính score `z` ở mẫu mới → chuyển thành xác suất → áp threshold đã chọn. Với nhiều lớp, công thức và solver phụ thuộc chiến lược đa lớp; cần kiểm cấu hình thư viện đang dùng. [1]

## Data used

Dữ liệu bảng hoặc vector thưa của văn bản, ví dụ UCI Adult cho phân loại thu nhập hoặc Iris cho nhiều lớp. Cần encode categorical, impute missing và scale feature số khi regularization/solver nhạy thang đo. [4]

## Design

- Bắt đầu với LogisticRegression + pipeline tiền xử lý; dùng `class_weight` hoặc chiến lược lấy mẫu chỉ khi metric và phân phối lớp yêu cầu.
- Ưu điểm: nhanh, hệ số giúp giải thích hướng liên hệ, có đầu ra xác suất. Hạn chế: ranh giới quyết định tuyến tính trừ khi thêm tương tác/biến đổi.
- Failure modes: nhãn lệch, feature phụ thuộc mạnh, dữ liệu tương lai bị lộ, threshold 0.5 không khớp chi phí, xác suất cần kiểm calibration.

**Độ phức tạp:** tính score nhị phân cho một mẫu khoảng `O(p)`; fit lặp nhiều vòng và phụ thuộc solver/sparsity. `C` nhỏ thường regularization mạnh hơn. Tránh đọc `exp(w_j)` như tác động nhân quả; đó là thay đổi odds có điều kiện theo các feature trong mô hình, dưới các giả định dữ liệu tương ứng.

## Nguồn

- [1] [scikit-learn: Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [2] [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [3] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [4] [UCI: Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
