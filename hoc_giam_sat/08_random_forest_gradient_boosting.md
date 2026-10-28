# Random Forest và Gradient Boosting

## Purpose

Kết hợp nhiều cây để tăng chất lượng trên dữ liệu bảng. Random Forest giảm phương sai bằng nhiều cây độc lập tương đối; Gradient Boosting xây cây nối tiếp để sửa sai của mô hình hiện tại. [1][2][3]

## Input

`X` dữ liệu bảng đã xử lý theo yêu cầu implementation; `y` lớp hoặc số; metric chính, ngân sách CPU/RAM và thời gian inference.

## Output

Random Forest: regressor lấy trung bình dự đoán các cây; `RandomForestClassifier` của scikit-learn lấy lớp có **xác suất trung bình qua các cây** cao nhất, nên có thể khác majority vote. Gradient Boosting cộng đóng góp các cây theo learning rate; classifier tạo score rồi chuyển sang xác suất theo loss/implementation. Lưu cả preprocessing và cấu hình. [1][6]

## Evaluation metrics

Classification: F1, PR-AUC/ROC-AUC, log loss/calibration tùy mục tiêu. Regression: MAE/RMSE/R². So với một cây và baseline; đánh giá latency, bộ nhớ và độ ổn định theo nhóm. [4]

## How it works

- **Random Forest:** mỗi cây học trên mẫu bootstrap; ở mỗi lần tách xem một tập con feature ngẫu nhiên; gộp kết quả nhiều cây. [1][2]
- **Gradient Boosting:** thêm cây nhỏ theo từng vòng để giảm objective/loss của ensemble, có thể dùng learning rate và early stopping để hạn chế overfit. [1][3]

**Ví dụ gộp:** ba cây regression dự đoán `10,12,17` thì forest lấy trung bình `13`. Với classifier, nếu xác suất lớp A từ ba cây là `0.9,0.4,0.4`, trung bình là `0.567`, nên lớp A thắng ở bài toán nhị phân mặc dù chỉ một cây có `p(A)>0.5`. Ví dụ này cho thấy không nên diễn đạt implementation scikit-learn đơn thuần là majority vote. [6]

**Boosting trực giác:** bắt đầu với dự đoán hằng `F₀`; ở vòng `m`, fit cây nhỏ theo hướng giảm loss hiện tại rồi cập nhật `Fₘ(x)=Fₘ₋₁(x)+ηhₘ(x)`. Với squared error, hướng giảm loss liên quan residual `y-Fₘ₋₁(x)`; với loss khác, pseudo-residual là gradient âm tương ứng. [1][3]

## Data used

Phù hợp nhiều bài toán bảng, ví dụ UCI Adult và California Housing. Feature số/categorical cần biến đổi theo estimator cụ thể. Chỉ fit imputation/encoding trong train fold; tree ensemble không tự giải quyết mọi lỗi nhãn hay lệch phân phối. [1][5]

## Design

- Random Forest: tuning `n_estimators`, độ sâu, `min_samples_leaf`, số feature xét ở mỗi split; tăng số cây thường tăng chi phí inference.
- Gradient Boosting: tuning learning rate, số vòng, số lá/độ sâu và regularization; learning rate nhỏ thường cần nhiều vòng hơn.
- Ưu điểm: bắt quan hệ phi tuyến/tương tác, thường mạnh với bảng. Hạn chế: giải thích khó hơn cây đơn, có thể tốn thời gian; feature importance không chứng minh nhân quả.
- Failure modes: overfit do cây quá sâu/boost quá lâu, train-serving skew, dùng test để early stop, calibration kém.

**Chi phí:** forest có thể fit cây song song; boosting thường xây tuần tự nên thời gian fit tăng theo số vòng. Inference phải đi qua nhiều cây, nên đo latency theo kích thước mô hình. Không kết luận boosting luôn hơn forest: kiểm trên validation cùng feature, split và ngân sách.

## Nguồn

- [1] [scikit-learn: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html)
- [2] [Breiman: Random Forests](https://doi.org/10.1023/A:1010933404324)
- [3] [Friedman: Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451)
- [4] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [5] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [6] [scikit-learn: RandomForestClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
