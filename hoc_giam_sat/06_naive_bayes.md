# Naive Bayes

## Purpose

Phân loại nhanh bằng định lý Bayes với giả định các feature độc lập có điều kiện khi biết lớp. Thường dùng làm baseline cho văn bản và dữ liệu chiều cao. [1]

## Input

`X` và nhãn lớp `y`; chọn biến thể theo dữ liệu: GaussianNB cho feature liên tục gần dạng Gaussian theo lớp, MultinomialNB cho count/tần suất **không âm**, BernoulliNB cho feature nhị phân. [1]

## Output

Nhãn dự đoán, điểm/xác suất lớp và các thống kê như prior lớp, phân phối feature có điều kiện.

## Evaluation metrics

Macro F1 và per-class precision/recall cho lớp không cân bằng; confusion matrix; log loss/Brier nếu xác suất cần dùng cho quyết định, kèm kiểm calibration. [2]

## How it works

Ước lượng `P(y)` và `P(x_j|y)` từ train, rồi dùng `P(y|x) ∝ P(y) ∏_j P(x_j|y)`. Tính trong log-space để ổn định số. Smoothing (ví dụ `alpha` trong MultinomialNB) tránh xác suất 0 cho feature chưa thấy. [1]

**Ví dụ tính tay:** giả sử `P(spam)=0.4`, `P(“sale”|spam)=0.5`, `P(“sale”|ham)=0.1`, `P(ham)=0.6`. Với email chỉ chứa tín hiệu “sale”, điểm chưa chuẩn hóa là `0.4×0.5=0.20` cho spam và `0.6×0.1=0.06` cho ham; sau chuẩn hóa, `P(spam|sale)=0.20/(0.20+0.06)≈0.769`. Ví dụ này giả định cách biểu diễn một feature đơn giản; email thực tế có nhiều feature và smoothing.

**Fit/predict:** đếm/ước lượng prior lớp và thống kê feature theo lớp → cộng log prior và log likelihood từng feature → chọn lớp có score cao nhất. Với nhiều feature, giả định độc lập có điều kiện giúp tính tích đơn giản, nhưng thường không đúng hoàn toàn. [1]

## Data used

Văn bản Bag-of-Words/count cho MultinomialNB là ví dụ điển hình; với bảng số liên tục dùng GaussianNB khi giả định hợp lý. Không đưa feature âm vào MultinomialNB. Chỉ fit vectorizer trên train fold để tránh leakage. [1][3]

## Design

- Ưu điểm: train/inference nhanh, hoạt động được khi dữ liệu không quá lớn; giả định độc lập thường sai nhưng vẫn có thể phân loại hữu ích.
- Hạn chế: tương tác feature không được biểu diễn trực tiếp; xác suất có thể quá tự tin. Kiểm calibration nếu xác suất được dùng.
- Failure modes: chọn sai biến thể, prior lệch so với production, từ vựng fit trên test, lớp hiếm thiếu mẫu.

**Chi phí và tuning:** thống kê count thường tuyến tính theo số phần tử khác 0 trong ma trận thưa; dự đoán cũng rẻ. Tuning `alpha` trên CV. Nếu cần xác suất để ra quyết định, kiểm calibration trên dữ liệu riêng vì Naive Bayes có thể cho xác suất cực đoan. [1][2]

## Nguồn

- [1] [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [2] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
