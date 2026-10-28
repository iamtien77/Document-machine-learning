# Chọn thuật toán theo bài toán và dữ liệu

## Purpose

Đưa ra thứ tự thử nghiệm hợp lý thay vì tìm “thuật toán tốt nhất” chung. Đây là hướng dẫn quyết định ban đầu; kết quả cuối phải được xác nhận bằng validation/test đúng quy trình. [1]

## Input

Loại target; số mẫu `n`, số feature `p`; dữ liệu bảng/văn bản/ảnh; mật độ feature; yêu cầu giải thích, latency, bộ nhớ; mức mất cân bằng lớp; metric và baseline đã xác định.

## Output

Một shortlist 2–4 mô hình, giả thuyết vì sao mỗi mô hình có thể phù hợp, các hyperparameter đáng thử và kế hoạch so sánh trên cùng split/metric.

## Evaluation metrics

Giữ một metric chính gắn chi phí sai, metric phụ phát hiện failure mode và chi phí vận hành (thời gian fit, p95 latency, kích thước artifact). Báo cáo kết quả theo nhóm và qua thời gian khi thích hợp. [2]

## How it works

1. Dùng DummyClassifier/DummyRegressor để biết mức tối thiểu.
2. Thử linear/logistic regression hoặc cây nông làm baseline có thể giải thích.
3. Dựa trên lỗi validation để tăng năng lực: quan hệ phi tuyến bảng → forest/boosting; khoảng cách có nghĩa → kNN/SVM; văn bản count → Naive Bayes hoặc linear classifier; cần độ bất định trên tập nhỏ → xem GPR.
4. So cùng split và cùng preprocessing hợp lệ; kiểm chất lượng, chi phí và lỗi theo nhóm; chọn mô hình đơn giản nhất đáp ứng yêu cầu.

| Tình huống | Thử trước | Thử tiếp nếu chưa đạt | Điểm cần kiểm |
|---|---|---|---|
| Regression bảng | Dummy → Ridge/Lasso | Tree ensemble, SVR, GPR tập nhỏ | Residual, outlier, extrapolation |
| Classification bảng | Dummy → Logistic/cây nông | Forest/boosting, SVM | Lớp hiếm, calibration, threshold |
| Văn bản Bag-of-Words | Dummy → MultinomialNB/logistic | Linear SVM, mô hình biểu diễn khác | Từ vựng fit trên train, drift chủ đề |
| Dữ liệu chiều thấp, khoảng cách rõ | Dummy → kNN | SVM RBF | Scale, chi phí inference |
| Cần quy tắc dễ đọc | Logistic/cây nông | Mô hình phức tạp nếu lợi ích đủ lớn | Độ ổn định của hệ số/quy tắc |

Bảng này là **heuristic**, không là xếp hạng hiệu năng. Nhiều thuật toán có thể phù hợp cùng một dữ liệu; chỉ đánh giá có kiểm soát mới phân biệt được. [1]

## Data used

Khảo sát `n/p`, missing, categorical, tính độc lập, thời gian, phân phối target và số mẫu mỗi nhóm. Một mô hình tốt trên Iris hoặc California Housing không chứng minh tốt trên dữ liệu nghiệp vụ khác.

## Design

- Giữ cùng tập test cuối cho mọi mô hình, nhưng **không dùng điểm test để chọn** mô hình. Dùng CV/validation để so sánh và khóa lựa chọn trước test. [3]
- Phân tích trade-off: logistic nhanh và dễ giải thích nhưng ranh giới tuyến tính; tree ensemble học tương tác nhưng khó giải thích hơn; kNN fit rẻ nhưng inference tốn; kernel SVM và GPR có thể không mở rộng tốt khi `n` lớn.
- Nếu mô hình phức tạp chỉ cải thiện rất nhỏ hoặc không ổn định giữa folds, ưu tiên baseline đơn giản hơn và xem lại dữ liệu/nhãn.

## Nguồn

- [1] [scikit-learn: Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html)
- [2] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
