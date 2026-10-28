# Trong train: tối ưu và chọn mô hình

## Purpose

Ước lượng tham số mô hình từ train, chọn hyperparameter dựa trên validation/cross-validation, và kiểm soát overfitting mà không dùng test để ra quyết định. [1]

## Input

Train và validation đã split đúng; pipeline tiền xử lý; objective/loss; metric chính; baseline; ngân sách tính toán; danh sách hyperparameter cần thử.

## Output

Pipeline đã fit, hyperparameter được chọn, lịch sử train/validation, cấu hình tái lập (seed, phiên bản dữ liệu/phần mềm) và một thiết kế đã khóa trước test.

## Evaluation metrics

**Loss** dùng để fit tham số: squared/absolute error cho regression, log loss hoặc hinge loss cho một số classifier. **Metric chọn mô hình** phản ánh mục tiêu thực tế và có thể khác loss; ví dụ recall khi bỏ sót ca dương tính tốn kém. Theo dõi chênh train và validation; so với baseline. [2]

Một cách viết chung: `θ* = argminθ [(1/n)Σᵢ L(yᵢ,fθ(xᵢ)) + λR(θ)]`. `L` là loss, `R` là regularization, `λ` điều khiển mức phạt. Công thức này hợp với nhiều mô hình tham số, nhưng kNN và cây học theo cơ chế khác. Metric chọn mô hình có thể không khả vi nên không nhất thiết là loss tối ưu.

## How it works

1. Chạy baseline, rồi mô hình giải thích được như linear/logistic regression hoặc cây nông.
2. Mỗi cấu hình: fit pipeline trên train hoặc trên fold train; đánh giá fold validation; lặp qua các fold để ước lượng độ ổn định. [1]
3. Tối ưu tham số theo thuật toán: closed-form/solver cho linear models, gradient descent cho logistic/MLP/boosting, chia nút tham lam cho cây, lưu mẫu cho kNN.
4. Chỉnh regularization, độ sâu cây, số lá, `k`, kernel, learning rate hoặc số epoch theo validation. Early stopping theo validation với mô hình lặp; không nhìn test. [1][3]
5. Chọn cấu hình theo metric và chi phí. Nếu cần calibration hoặc chọn threshold, giữ dữ liệu **chưa dùng để fit classifier**, hoặc dùng dự đoán out-of-fold theo quy trình CV. Nếu vừa fit calibrator vừa tune threshold, nên có tập riêng cho từng bước hoặc một quy trình cross-fitting thích hợp để tránh chọn ngưỡng trên cùng dữ liệu đã tối ưu calibrator. Chỉ refit classifier trên train+validation khi không còn dùng chính validation đó như holdout độc lập; test vẫn giữ nguyên cho đánh giá cuối. [5][6]

**Vòng lặp thực nghiệm mẫu:**

```text
for mỗi mô hình và bộ hyperparameter:
    for mỗi split train_fold/validation_fold:
        fit toàn bộ pipeline trên train_fold
        dự đoán validation_fold và tính metric chính
    ghi trung bình, độ phân tán và chi phí của các fold
chọn cấu hình theo metric đã định nghĩa và kiểm lỗi theo nhóm
khóa pipeline, threshold, cách preprocessing; đánh giá test
```

CV chỉ giảm độ phụ thuộc vào một lần chia; các fold không độc lập hoàn toàn. Không xem độ lệch chuẩn giữa fold là khoảng tin cậy chính xác của hiệu năng production. [1]

## Data used

Train dùng để học trọng số/quy tắc; validation hoặc các fold dùng để chọn cấu hình. Test chỉ dùng sau khi toàn bộ lựa chọn kết thúc. Nếu dữ liệu có group/time, cross-validation cũng phải giữ cấu trúc đó. [1]

## Design

- Ghi rõ search space và ngân sách; tìm ít tham số có ý nghĩa trước. Nested CV hữu ích khi cần ước lượng ít thiên lệch sau khi tuning trên tập nhỏ. [1]
- Kiểm tra learning curve và chênh train/validation để phân biệt underfitting với overfitting.
- Regularization hoặc giảm độ phức tạp khi overfit; cải thiện feature/nhãn hoặc tăng năng lực mô hình khi underfit.
- Failure modes: tối ưu sai metric, tuning quá nhiều trên một validation set, fit preprocessing ngoài fold, seed không được lưu.

**Chẩn đoán:** train kém và validation kém có thể là underfitting hoặc dữ liệu/nhãn lỗi; train tốt nhưng validation kém gợi ý overfitting, leakage ở train, hoặc lệch phân phối. Trước khi tăng độ phức tạp, kiểm baseline và chất lượng nhãn. Learning curve có thể cho biết liệu thêm dữ liệu hay tăng năng lực mô hình có triển vọng. [4]

## Nguồn

- [1] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [2] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: Grid search](https://scikit-learn.org/stable/modules/grid_search.html)
- [4] [scikit-learn: Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- [5] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [6] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
