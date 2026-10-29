# Học giám sát: bản đồ học tập

## Purpose

Học giám sát (supervised learning) học ánh xạ từ đặc trưng `X` sang nhãn/giá trị đích `y` từ các ví dụ đã gán nhãn, để dự đoán trên ví dụ mới. Hai bài toán nền tảng là **regression** (`y` liên tục, như giá nhà) và **classification** (`y` là lớp, như thư rác/không phải thư rác). Đây là bản đồ để học từ bài toán đến vận hành, không phải một thuật toán riêng lẻ. [1][2]

Ký hiệu: với `n` mẫu và `p` feature, $`X \in \mathbb{R}^{n\times p}`$, $`y = (y_{1},\ldots ,y_{n})`$. Mô hình có tham số $`\theta`$ và dự đoán $`f_\theta(x)`$. Mục tiêu là lỗi kỳ vọng trên **dữ liệu mới** thấp, chứ không phải học thuộc các cặp $`(x_{i},y_{i})`$ đã thấy. Mẫu được lấy từ phân phối nào, và phân phối đó có giống lúc triển khai hay không, là giả định cần kiểm tra. [2][4]

## Input

- Bài toán thực tế, đơn vị dự đoán (một người, giao dịch, ảnh...), thời điểm cần dự đoán và chi phí của từng loại sai.
- Tập ví dụ có `X` và `y`; mô tả nguồn, cách tạo nhãn, thời gian thu thập, nhóm liên quan và quyền sử dụng dữ liệu.
- Ràng buộc về độ trễ, khả năng giải thích, bộ nhớ và tần suất cập nhật.

## Output

- Mô hình $`f(X)\to\hat{y}`$ hoặc $`f(X) \to p(y\mid X)`$; với regression là giá trị số, với classification là nhãn/điểm/xác suất.
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

Thuật toán trong PDF chương 2 có thêm [18 Polynomial Regression](18_hoi_quy_da_thuc.md) và [19 AdaBoost](19_adaboost.md). Các file [16 Training phase](16_training_phase.md) và [17 Inference phase](17_inference_phase.md) mô tả ranh giới giữa học mô hình và dùng mô hình.

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

### Key words nền tảng

Các thuật ngữ sau được dùng xuyên suốt các bài. Trong mỗi bài thuật toán, bảng **Key words** giải thích thêm ý nghĩa cụ thể của thuật ngữ trong mô hình đó. [1][2][5]

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Sample / mẫu** | Một đơn vị quan sát, thường là một hàng dữ liệu. | Cung cấp một ví dụ để học hoặc dự đoán. | Cần xác định đúng đơn vị mẫu trước khi chia train/test; các hàng cùng người hoặc cùng phiên có thể cần chia theo nhóm. |
| **Feature / đặc trưng** | Thông tin đầu vào đo được cho một mẫu, ký hiệu thường là `x`. | Cho mô hình căn cứ để dự đoán. | Được thu thập, làm sạch và biến đổi giống nhau lúc train và inference. |
| **Target / label / nhãn** | Giá trị đúng cần dự đoán, ký hiệu thường là `y`; có thể là số hoặc lớp. | Cung cấp tín hiệu để tính lỗi trong học giám sát. | Phải có quy tắc gán nhãn rõ ràng và không được vô tình đưa vào feature. |
| **Train set** | Phần dữ liệu được phép dùng để học tham số và thống kê tiền xử lý. | Tạo mô hình đã fit. | Chỉ phần train của mỗi split/fold được dùng trong bước `fit`. |
| **Validation set / CV** | Dữ liệu giữ riêng trong quá trình chọn cấu hình; CV lặp nhiều lượt chia. | Ước lượng cấu hình nào tổng quát tốt hơn. | Dùng chọn thuật toán, hyperparameter, ngưỡng và điểm dừng; dùng nhiều lần có thể gây overfit vào validation. |
| **Test set** | Tập giữ riêng đến khi quy trình chọn mô hình hoàn tất. | Cho ước lượng cuối về chất lượng trên dữ liệu chưa thấy. | Dùng đánh giá cuối, không dùng để chọn cấu hình hoặc sửa mô hình rồi vẫn báo cùng điểm test như đánh giá độc lập. |
| **Baseline** | Cách dự đoán đơn giản làm mốc, ví dụ luôn đoán lớp phổ biến. | Cho biết mô hình mới cải thiện được bao nhiêu. | Được chạy trước mô hình phức tạp để kiểm giá trị thực của việc học. |
| **Parameter / tham số** | Giá trị mô hình học từ train, như hệ số hồi quy hoặc trọng số mạng. | Quyết định hàm dự đoán đã học. | Được cập nhật trong `fit` và lưu trong artifact để inference. |
| **Hyperparameter / siêu tham số** | Thiết lập chọn trước khi fit, như `k` của KNN hoặc độ sâu cây. | Điều khiển hình dạng và cách học của mô hình. | Chọn bằng validation/CV rồi fit lại theo quy trình đã chốt. |
| **Transformer / pipeline** | Transformer biến đổi dữ liệu; pipeline ghép tuần tự biến đổi và mô hình. | Giữ các bước xử lý đầu vào nhất quán. | Fit transformer trên train trong từng fold, sau đó lưu chung với mô hình để inference. |
| **Loss / hàm mất mát** | Biểu thức lượng hóa mức sai của dự đoán khi tối ưu. | Tạo tín hiệu cập nhật tham số hoặc chọn split. | Thuộc training; không nhất thiết trùng metric dùng báo cáo cho người dùng. |
| **Metric / thước đo** | Công thức đánh giá chất lượng, như MAE, F1 hoặc ROC-AUC. | Tóm tắt một khía cạnh hiệu quả của mô hình. | Phải chọn theo mục tiêu và chi phí lỗi của bài toán; tính trên tập đánh giá phù hợp. |
| **Data leakage / rò rỉ dữ liệu** | Thông tin không hợp lệ từ tương lai, test hoặc nhãn lọt vào quá trình học/chọn mô hình. | Làm điểm đánh giá cao giả tạo. | Cần chặn bằng cách chia dữ liệu trước và fit tiền xử lý đúng trong từng fold. |
| **Overfitting / học quá khớp** | Mô hình khớp dữ liệu đã thấy nhưng kém trên dữ liệu mới. | Làm chất lượng ngoài train giảm. | Được phát hiện qua validation và kiểm soát bằng độ phức tạp, regularization hoặc thêm dữ liệu phù hợp. |
| **Inference / suy luận** | Áp mô hình đã fit vào đầu vào mới để lấy dự đoán. | Tạo score, xác suất hoặc giá trị dự đoán. | Dùng đúng schema, transformer và phiên bản artifact đã lưu. |
| **Threshold / ngưỡng** | Mức cắt biến score hoặc xác suất thành quyết định lớp. | Thay đổi tỷ lệ bỏ sót và báo động nhầm. | Chọn theo chi phí lỗi trên validation, rồi áp cố định trong inference cho đến khi cập nhật có kiểm soát. |

## Nguồn

- [1] [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
- [2] [Stanford CS229 notes](https://cs229.stanford.edu/main_notes.pdf)
- [3] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [Google: Training, validation, test sets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [5] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [6] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [7] [Google: Monitoring ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)
- [8] [scikit-learn: Toy and real-world datasets](https://scikit-learn.org/stable/datasets.html)
