# Học giám sát: bản đồ học tập

## Purpose

Học giám sát (supervised learning) học ánh xạ từ đặc trưng `X` sang nhãn/giá trị đích `y` từ các ví dụ đã gán nhãn, để dự đoán trên ví dụ mới. Hai bài toán nền tảng là **regression** (`y` liên tục, như giá nhà) và **classification** (`y` là lớp, như thư rác/không phải thư rác). Đây là bản đồ để học từ bài toán đến vận hành, không phải một thuật toán riêng lẻ. [1][2]

Ký hiệu: với `n` mẫu và `p` feature, `X ∈ R^(n×p)`, `y = (y₁,...,yₙ)`. Mô hình có tham số `θ` và dự đoán `fθ(x)`. Mục tiêu là lỗi kỳ vọng trên **dữ liệu mới** thấp, chứ không phải học thuộc các cặp `(xᵢ,yᵢ)` đã thấy. Mẫu được lấy từ phân phối nào, và phân phối đó có giống lúc triển khai hay không, là giả định cần kiểm tra. [2][4]

## Input

- Bài toán thực tế, đơn vị dự đoán (một người, giao dịch, ảnh...), thời điểm cần dự đoán và chi phí của từng loại sai.
- Tập ví dụ có `X` và `y`; mô tả nguồn, cách tạo nhãn, thời gian thu thập, nhóm liên quan và quyền sử dụng dữ liệu.
- Ràng buộc về độ trễ, khả năng giải thích, bộ nhớ và tần suất cập nhật.

## Output

- Mô hình `f(X) -> y_hat` hoặc `f(X) -> p(y|X)`; với regression là giá trị số, với classification là nhãn/điểm/xác suất.
- Báo cáo đánh giá trên dữ liệu chưa dùng để chọn mô hình, cùng giới hạn áp dụng.
- Pipeline tiền xử lý, phiên bản dữ liệu và mô hình để suy luận nhất quán.

## Evaluation metrics

Chọn **trước khi train** theo mục tiêu: regression dùng MAE/RMSE/R²; classification dùng precision, recall, F1, ROC-AUC hoặc PR-AUC; nếu quyết định theo xác suất thì thêm log loss/Brier score và calibration. So với baseline đơn giản, rồi kiểm tra theo nhóm dữ liệu quan trọng. Không có metric nào tốt cho mọi bài toán. [3]

Ví dụ: hệ thống lọc thư rác có thể ưu tiên precision của lớp “rác” để tránh chặn nhầm thư quan trọng; sàng lọc ca bệnh có thể ưu tiên recall, nhưng vẫn phải kiểm số báo động giả và khả năng can thiệp. Đây là lựa chọn dựa trên chi phí sai cụ thể, không phải quy tắc chung cho mọi ứng dụng.

## How it works

1. Định nghĩa `X`, `y`, thời điểm dự đoán và metric.
2. Thu thập, kiểm nhãn/chất lượng dữ liệu; chia train/validation/test đúng theo đơn vị độc lập, nhóm hoặc thời gian. [4]
3. Fit preprocessing và baseline chỉ trên train; chọn thuật toán, tối ưu loss trên train, điều chỉnh bằng validation hoặc cross-validation. [5]
4. Khóa thiết kế; đánh giá test một lần; phân tích lỗi, hiệu chuẩn xác suất/ngưỡng trên validation nếu cần. [4][6]
5. Triển khai, giám sát dữ liệu và chất lượng, lên lịch đánh giá lại khi phân phối thay đổi. [7]

Đọc tiếp theo thứ tự: [01 Dữ liệu trước train](01_du_lieu_truoc_train.md) → [02 Trong train](02_trong_train.md) → các họ thuật toán [03](03_hoi_quy_tuyen_tinh.md), [04](04_logistic_regression.md), [05](05_knn.md), [06](06_naive_bayes.md), [07](07_decision_tree.md), [08](08_random_forest_gradient_boosting.md), [09](09_svm.md), [10](10_mang_neuron.md) → [11 Đánh giá](11_danh_gia.md) → [12 Sau train](12_sau_train.md).

Tài liệu bổ sung: [13 Bài thực hành xuyên suốt](13_thuc_hanh_xuyen_suot.md), [14 LDA/QDA và Gaussian Process](14_lda_qda_gaussian_process.md), [15 Chọn thuật toán](15_chon_thuat_toan.md).

| Giai đoạn | Câu hỏi cần trả lời | Sản phẩm kiểm tra được |
|---|---|---|
| Trước train | Dự đoán cái gì, tại thời điểm nào, dữ liệu nào hợp lệ? | Định nghĩa target, schema, split, baseline, metric |
| Trong train | Mô hình nào học được tín hiệu mà không overfit? | Pipeline, log thí nghiệm, cấu hình tốt nhất trên validation |
| Sau train | Kết quả có lặp lại trên dữ liệu mới và còn đúng khi triển khai? | Báo cáo test, phân tích lỗi, giám sát, quy tắc retrain |

## Data used

Một hàng thường là một ví dụ; cột là feature; `y` là nhãn. Dữ liệu có thể là bảng, văn bản, ảnh hoặc chuỗi thời gian sau khi được mã hóa phù hợp. Ví dụ học tập: Iris (classification), California Housing (regression); đây là bộ mẫu minh họa, không đại diện cho dữ liệu production. [8]

Phân biệt ba dạng target: **binary** (`0/1`), **multiclass** (mỗi mẫu thuộc đúng một trong nhiều lớp), **multilabel** (một mẫu có thể mang nhiều nhãn). Với regression, một mẫu cũng có thể có nhiều target số. Quy tắc gán nhãn và đơn vị quan sát phải được ghi rõ trước khi chọn thuật toán. [8]

## Design

- Luôn thử **dummy baseline** và mô hình dễ giải thích trước, sau đó mới tăng độ phức tạp nếu kết quả chưa đủ. [3]
- Lưu `preprocessing + model + threshold` như một đơn vị suy luận. Chọn split phản ánh cách dữ liệu mới xuất hiện; tránh data leakage.
- Cân nhắc trade-off giữa chất lượng, tính giải thích, chi phí train/inference và khả năng bảo trì. Failure mode chung: nhãn sai, lệch phân phối, mất cân bằng lớp, dữ liệu trùng qua các tập.

**Lộ trình đề xuất:** nắm split/leakage → chạy dummy baseline → linear/logistic → cây/kNN/Naive Bayes → ensemble/SVM/MLP → đánh giá xác suất và triển khai. Độ phức tạp tăng khi baseline chưa đạt mục tiêu, không tăng chỉ vì thuật toán mới hơn.

## Nguồn

- [1] [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
- [2] [Stanford CS229 notes](https://cs229.stanford.edu/main_notes.pdf)
- [3] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [Google: Training, validation, test sets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [5] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [6] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [7] [Google: Monitoring ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)
- [8] [scikit-learn: Toy and real-world datasets](https://scikit-learn.org/stable/datasets.html)
