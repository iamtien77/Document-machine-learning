# Bài thực hành xuyên suốt: phân loại với scikit-learn

## Purpose

Thực hành một pipeline học giám sát hoàn chỉnh: định nghĩa nhãn, chia dữ liệu, baseline, cross-validation, chọn hyperparameter, chọn threshold trên validation và đánh giá test. Bộ dữ liệu Breast Cancer Wisconsin chỉ dùng để học kỹ thuật; **không dùng mô hình này để chẩn đoán y khoa**. [1]

## Input

- Python với `numpy` và `scikit-learn` đã được cài trong môi trường học tập.
- Dữ liệu `load_breast_cancer()` của scikit-learn. Quy ước lại `y=1` là lớp malignant để minh họa lớp dương; feature là các phép đo số. [1]
- Seed cố định `42` để việc chia dữ liệu có thể lặp lại; metric chính của ví dụ là F1. Trong ứng dụng thật, metric phải được chọn từ chi phí lỗi trước khi xem kết quả.

## Output

Mã in kích thước từng tập, metric baseline, kết quả cross-validation tốt nhất trên **train**, threshold chọn trên **validation**, và precision/recall/F1/AP/ROC-AUC cùng confusion matrix trên **test**. Tài liệu không ghi sẵn điểm số vì chưa chạy thí nghiệm trong môi trường của người đọc.

## Evaluation metrics

- `F1` để chọn cấu hình/threshold trong ví dụ; `precision`, `recall`, `average_precision` (AP) và `roc_auc` giúp thấy mặt khác của kết quả. AP khác diện tích hình thang PR. [2]
- Confusion matrix dùng thứ tự nhãn `[0,1]`, tức hàng là nhãn thật, cột là nhãn dự đoán: `[[TN,FP],[FN,TP]]`.
- `DummyClassifier` là baseline; kết quả trên test chỉ có ý nghĩa khi so cùng split và định nghĩa lớp dương. [2]

## How it works

Lưu đoạn sau thành một file Python riêng nếu muốn chạy. Code giữ validation độc lập với tập fit classifier; `GridSearchCV` chỉ nhìn train. Nó cố ý **không refit** mô hình trên train+validation trước khi chọn threshold. [3][4]

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, original_y = load_breast_cancer(return_X_y=True)
y = (original_y == 0).astype(int)  # 1 = malignant

# Test được tách đầu tiên và không dùng để chọn bất kỳ cấu hình nào.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
)
print("Số mẫu train/validation/test:", len(y_train), len(y_val), len(y_test))

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipeline,
    param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
    scoring="f1",
    cv=cv,
    refit=True,
)
search.fit(X_train, y_train)
print("C tốt nhất và F1 CV:", search.best_params_, search.best_score_)

# Chọn threshold trên validation; không fit thêm classifier ở đây.
val_scores = search.predict_proba(X_val)[:, 1]
candidates = np.linspace(0.10, 0.90, 81)
val_f1 = [
    f1_score(y_val, (val_scores >= t).astype(int), zero_division=0)
    for t in candidates
]
threshold = float(candidates[int(np.argmax(val_f1))])
print("Threshold chọn trên validation:", threshold)

# Test cuối cùng: cùng classifier, scaler và threshold đã khóa.
test_scores = search.predict_proba(X_test)[:, 1]
test_pred = (test_scores >= threshold).astype(int)
baseline_pred = baseline.predict(X_test)
print("Baseline F1:", f1_score(y_test, baseline_pred, zero_division=0))
print("Test precision:", precision_score(y_test, test_pred, zero_division=0))
print("Test recall:", recall_score(y_test, test_pred, zero_division=0))
print("Test F1:", f1_score(y_test, test_pred, zero_division=0))
print("Test AP:", average_precision_score(y_test, test_scores))
print("Test ROC-AUC:", roc_auc_score(y_test, test_scores))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
```

**Đọc kết quả:** nếu F1 test thấp hơn nhiều so với CV/validation, kiểm lại split, leakage, kích thước tập và khác biệt phân phối trước khi đổi thuật toán. Nếu recall thấp nhưng bài toán coi FN rất đắt, xác định hàm chi phí và chọn threshold theo hàm đó trên validation thay vì F1. [2][4]

Ví dụ chọn `C` bằng F1 trong CV ở threshold mặc định của classifier, sau đó mới chọn threshold trên validation; đây là hai bước nối tiếp để học quy trình, **không phải tối ưu đồng thời** `C` và threshold. Nếu cần tối ưu chung, thiết kế nested CV hoặc dự đoán out-of-fold với tập giữ riêng và đánh giá cuối trên test để hạn chế thiên lệch lựa chọn. [4][5]

## Data used

Dataset này có 569 mẫu, 30 feature số và 2 lớp trong bản scikit-learn; lớp gốc `0` là malignant, `1` là benign. Dữ liệu nhỏ giúp chạy nhanh, nhưng không mô phỏng dữ liệu bệnh viện đang vận hành và không đủ để kết luận lâm sàng. [1]

## Design

- `Pipeline` bảo đảm `StandardScaler` được fit lại trong từng fold train của CV, giảm rủi ro leakage. [3]
- Split stratified giữ tỉ lệ lớp gần nhau trong các tập; với dữ liệu nhiều bản ghi mỗi bệnh nhân hoặc dữ liệu theo thời gian, cần chuyển sang group/time split tương ứng. [5]
- Threshold được chọn bằng validation nên điểm F1 validation có thiên lệch lựa chọn; test mới là phép kiểm cuối. Nếu muốn calibration, tạo thêm holdout độc lập hoặc dùng quy trình CV phù hợp. [4][6]
- Bài tập mở rộng: đổi metric chính sang recall có ràng buộc precision, thêm biểu đồ calibration, và so với Random Forest trên **cùng** split mà không dùng test để chọn mô hình.

## Nguồn

- [1] [scikit-learn: Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [2] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [4] [scikit-learn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [5] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [6] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
