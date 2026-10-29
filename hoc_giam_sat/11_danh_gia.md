# Đánh giá mô hình học giám sát

## Purpose

Ước lượng khả năng tổng quát hóa trên dữ liệu chưa thấy và kiểm tra mô hình có đáp ứng mục tiêu thực tế hay không. Metric phải được chọn trước khi so mô hình. [1][2]

## Input

Mô hình và pipeline đã khóa; `X_test, y_test` chưa dùng để fit/chọn cấu hình; baseline; metric chính và chi phí sai; các nhóm cần kiểm tra (địa lý, thời gian, nguồn dữ liệu...).

## Output

Bảng so sánh baseline và mô hình; metric chính/phụ, confusion matrix hoặc residual plots, phân tích lỗi theo nhóm, khoảng bất định nếu có thể, quyết định triển khai hoặc quay lại thu thập dữ liệu.

## Evaluation metrics

| Bài toán | Metric | Dùng khi | Cẩn trọng |
|---|---|---|---|
| Classification | Accuracy | Lớp và chi phí sai tương đối cân bằng | Có thể che lỗi lớp hiếm |
| Classification | Precision / Recall / F1 | Cần kiểm soát false positive / false negative | Phụ thuộc threshold |
| Classification | ROC-AUC / PR-AUC | So khả năng xếp hạng nhiều threshold | PR-AUC hữu ích khi chú ý lớp dương hiếm; vẫn cần điểm tại ngưỡng vận hành |
| Xác suất | Log loss / Brier score, calibration plot | Dùng xác suất để quyết định | AUC tốt không bảo đảm xác suất đúng |
| Regression | MAE | Muốn lỗi tuyệt đối dễ diễn giải | Ít nhấn mạnh lỗi cực lớn |
| Regression | RMSE | Lỗi lớn tốn kém hơn | Nhạy outlier |
| Regression | R² | So với dự đoán trung bình | Có thể âm trên test; không thay thế sai số theo đơn vị thật |

Các metric trên được định nghĩa và triển khai trong tài liệu scikit-learn. Chọn **một metric chính**, rồi dùng metric phụ để kiểm failure modes. [2]

**Công thức nhị phân:** lấy lớp dương làm chuẩn. `TP`: dự đoán dương và thật dương; `FP`: dự đoán dương nhưng thật âm; `FN`: dự đoán âm nhưng thật dương; `TN`: dự đoán âm và thật âm.

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| Accuracy | $`\frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}+\mathrm{TN}}`$ | Tỉ lệ dự đoán đúng |
| Precision | $`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`$ | Trong số dự đoán dương, bao nhiêu đúng |
| Recall/TPR | $`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`$ | Trong số dương thật, tìm được bao nhiêu |
| Specificity/TNR | $`\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}`$ | Trong số âm thật, loại đúng bao nhiêu |
| F1 | $`\frac{2\,\mathrm{precision}\,\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}`$ | Trung bình điều hòa của precision và recall |

Nếu mẫu số bằng 0, metric có thể không xác định; báo cáo số mẫu và quy tắc xử lý (`zero_division` nếu dùng scikit-learn), không âm thầm coi là 0 hay 1. Với multiclass, tính theo one-vs-rest từng lớp rồi dùng **macro** (trung bình các lớp), **weighted** (trọng số theo support) hoặc **micro** (cộng TP/FP/FN trước khi tính). Macro thể hiện lớp hiếm rõ hơn; weighted có thể che lỗi lớp hiếm. [2]

**Ví dụ confusion matrix:** $`TP=30`$, $`FP=10`$, $`FN=20`$, $`TN=940`$. Accuracy=`970/1000=0.97`, precision=`30/40=0.75`, recall=`30/50=0.60`, F1≈`0.667`. Accuracy 97% vẫn bỏ sót 40% ca dương tính. Tỉ lệ dương thực tế là `50/1000=5%`; baseline luôn đoán âm đạt 95% accuracy nhưng recall bằng 0.

**Regression:** với $`e_{i}=y_{i}-\hat{y}_{i}`$, $`\mathrm{MAE}=\frac{1}{n}\sum_i\lvert e_i\rvert`$, $`\mathrm{MSE}=\frac{1}{n}\sum_i e_i^2`$, $`\mathrm{RMSE}=\sqrt{\mathrm{MSE}}`$, $`R^2=1-\frac{\sum_i e_i^2}{\sum_i(y_i-\bar{y}_{\mathrm{test}})^2}`$. Trong phép tính R² trên test, $`\bar{y}_{test}`$ là trung bình `y_true` của **tập đánh giá** theo định nghĩa metric; nó không đồng nhất với DummyRegressor đã fit trung bình train. Nếu `y_true` hằng, R² có trường hợp biên cần xem quy ước thư viện. [2]

**PR-AUC cần gọi tên rõ:** `average_precision_score` của scikit-learn (AP) tính trung bình có trọng số các precision khi recall tăng; nó **khác** diện tích nội suy hình thang dưới PR curve. Khi báo cáo, ghi rõ “AP” hay “trapezoidal PR-AUC” và cách tính. Baseline AP của dự đoán ngẫu nhiên thường liên quan prevalence lớp dương; so sánh cùng tập dữ liệu và cùng định nghĩa. [2]

## How it works

1. Trên train, dùng validation/CV để chọn mô hình, hyperparameter, calibration và threshold. Với nhóm/thời gian, dùng splitter phù hợp. [1][3][4]
2. Chạy pipeline đã khóa trên test một lần; lưu metric và số mẫu. Nếu sau khi xem test lại chỉnh mô hình, cần một test mới để có ước lượng độc lập. [1][5]
3. Kiểm confusion matrix từng lớp hoặc residual theo mức dự đoán; xem các lát cắt quan trọng và ví dụ lỗi cụ thể.
4. So với baseline cùng split; kiểm sự khác biệt giữa validation và test, chi phí lỗi, latency và chất lượng xác suất.

**Chọn metric theo chi phí:** nếu mỗi FN tốn 10 đơn vị và mỗi FP tốn 1 đơn vị, một quy tắc vận hành có thể tối thiểu hóa $`10\times FN + 1\times FP`$ trên validation, có ràng buộc năng lực xử lý báo động. Đừng chỉ tối đa F1 nếu chi phí FN và FP không cân. Sau khi chọn threshold, báo cáo chi phí/precision/recall trên test một lần. [6]

## Data used

Test phải phản ánh dữ liệu tương lai/sản xuất và không chứa bản sao hoặc group trùng với train khi mục tiêu là group mới. Với chuỗi thời gian, test thuộc giai đoạn sau train; với dữ liệu lớp hiếm, báo cáo cả số mẫu mỗi lớp. [1][5]

## Design

- Không kết luận mô hình tốt từ một điểm số duy nhất; ghi `n`, cách split, seed và metric cho từng nhóm.
- Khi dữ liệu ít, dùng CV để so cấu hình; giữ một test cuối hoặc nested CV khi cần ước lượng sau tuning. [1]
- Failure modes: test bị dùng lặp để điều chỉnh, leakage qua preprocessing, metric không gắn chi phí, test không đại diện production.

## Nguồn

- [1] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [2] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [4] [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [5] [Google: Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [6] [scikit-learn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
