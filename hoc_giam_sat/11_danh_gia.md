# Đánh giá mô hình học giám sát

## Purpose

Ước lượng khả năng tổng quát hóa trên dữ liệu chưa thấy và kiểm tra mô hình có đáp ứng mục tiêu thực tế hay không. Metric phải được chọn trước khi so mô hình. [1][2]

### Nguyên lý Đánh giá Thực chứng & Khoảng Tin cậy Thống kê (Statistical Hypothesis Testing)

Một điểm số như “Accuracy = 92.5%” cần đi cùng kích thước mẫu ($`n`$), phân phối nhãn và cách chia dữ liệu để diễn giải. Metric trên test ước lượng hiệu năng trên phân phối mà test đại diện; không bảo đảm hiệu năng trên mọi dữ liệu tương lai.

Hai công cụ xem xét độ bất định, tùy mục tiêu và thiết kế đánh giá:

1. **Kiểm định giả thuyết thống kê (Hypothesis Testing):** Chênh lệch 0.5 điểm phần trăm chưa đủ để kết luận $`A`$ tốt hơn baseline $`B`$. McNemar có thể so kết quả đúng/sai ghép cặp trên cùng các mẫu độc lập. Không áp dụng máy móc paired t-test hoặc Wilcoxon cho điểm các fold: train chồng lặp làm điểm số phụ thuộc nhau. Nếu cần kiểm định, chọn phương pháp phù hợp thiết kế, xác định mức ý nghĩa trước và báo cáo độ lớn chênh lệch; $`p<0.05`$ không chứng minh lợi ích thực tế. [10][12]
2. **Khoảng tin cậy Bootstrap (Bootstrap Confidence Intervals):** Với mô hình đã khóa và mẫu test độc lập, có thể tái lấy mẫu có hoàn lại các cặp nhãn–dự đoán. Percentile và BCa là các cách xây dựng khoảng; công thức sau chỉ là **xấp xỉ chuẩn**, cần phân phối ước lượng gần chuẩn và bias nhỏ:

$$
\widehat{\mathrm{Metric}} \pm 1.96 \times \mathrm{SE}_{\text{boot}}
$$

   Khi so hai mô hình, dùng cùng chỉ số lấy mẫu cho cả hai. Dữ liệu theo nhóm/thời gian cần cách lấy mẫu giữ cấu trúc phụ thuộc. CI trên test cố định không bao quát biến thiên do huấn luyện lại hoặc distribution shift. [11]

---

## Input

Mô hình và pipeline đã khóa; `X_test, y_test` chưa dùng để fit/chọn cấu hình; baseline; metric chính và chi phí sai; các nhóm cần kiểm tra (địa lý, thời gian, nguồn dữ liệu...).

---

## Output

Bảng so sánh baseline và mô hình; metric chính/phụ, confusion matrix hoặc residual plots, phân tích lỗi theo nhóm, khoảng bất định nếu có thể, quyết định triển khai hoặc quay lại thu thập dữ liệu.

---

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

### Phân tích Sâu: Đánh giá Đa lớp (Multi-class Averaging Nuances)

Khi mở rộng sang bài toán phân loại $`K`$ lớp, việc gộp các chỉ số thành một điểm số duy nhất đòi hỏi sự thận trọng đặc biệt:

* **Macro-Average:** Tính metric độc lập cho từng lớp rồi lấy trung bình cộng giản đơn:

$$
\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k
$$

  *Đặc tính:* Mỗi lớp có trọng số bằng nhau, kể cả lớp hiếm. Một lớp thất bại vẫn có thể bị che khi số lớp lớn, nên xem thêm F1 từng lớp.
* **Weighted-Average:** Tính metric từng lớp rồi nhân với tỷ trọng số lượng mẫu (Support $`n_k`$):

$$
\text{Weighted-F1} = \frac{1}{n} \sum_{k=1}^K n_k \cdot \text{F1}_k
$$

  *Lưu ý:* Lớp đông mẫu đóng góp nhiều hơn. Weighted-F1 có thể cao dù F1 của lớp hiếm bằng 0, nên cần báo cáo metric và support từng lớp.
* **Micro-Average:** Cộng dồn toàn bộ $`TP, FP, FN`$ của tất cả các lớp lại trước khi tính toán:

$$
\text{Micro-F1} = \text{Micro-Precision} = \text{Micro-Recall} = \text{Accuracy}
$$

  Trong bài toán đơn nhãn (mỗi mẫu thuộc đúng 1 lớp), khi tính trên toàn bộ các lớp, Micro-F1 bằng Accuracy. Đẳng thức này không áp dụng chung cho multilabel hoặc khi chỉ chọn một số lớp.

### So sánh Toàn diện: Đường cong ROC vs. Đường cong Precision-Recall (PR)

```text
       ROC CURVE (TPR vs FPR)                   PR CURVE (Precision vs Recall)
   1.0 ^         .---'                      1.0 ^------.
       |       .'                           |           \
 TPR   |     .'                       Prec  |            \
       |   .'                               |             `---.
       | .'  (Đường chéo ngẫu nhiên)        |                  \
   0.0 +--------------------->          0.0 +--------------------->
       0.0                 1.0              0.0                 1.0
                 FPR                                      Recall
```

1. **ROC Curve (Receiver Operating Characteristic):**
   * Trục tung: True Positive Rate ($`\text{TPR} = \frac{TP}{P}`$).
   * Trục hoành: False Positive Rate ($`\text{FPR} = \frac{FP}{N}`$).
   * *Đặc tính toán học cốt lõi:* TPR và FPR được tính riêng trong từng lớp. **ROC và ROC-AUC không đổi nếu chỉ tỷ lệ lớp thay đổi, còn phân phối điểm số trong mỗi lớp giữ nguyên**. Nếu dữ liệu trong từng lớp cũng thay đổi, ROC-AUC có thể đổi. [7]
2. **Precision-Recall (PR) Curve:**
   * Trục tung: $`\text{Precision} = \frac{TP}{TP + FP}`$.
   * Trục hoành: $`\text{Recall} = \frac{TP}{TP + FN}`$.
   * *Đặc tính toán học cốt lõi:* Precision phụ thuộc vào số mẫu âm bị dự đoán nhầm thành dương ($`FP`$). Khi lớp dương hiếm, một FPR nhỏ vẫn có thể tạo nhiều $`FP`$ so với $`TP`$, khiến precision thấp.
   * **Chọn theo mục tiêu:** ROC-AUC cao vẫn có thể đi kèm precision thấp khi lớp dương hiếm; nhiều $`TN`$ không tự động làm AUC cao. Nên xem thêm PR curve, AP hoặc trapezoidal PR-AUC và precision/recall tại ngưỡng vận hành; không có một metric quyết định cho mọi bài toán mất cân bằng. Với điểm ngẫu nhiên độc lập nhãn, ROC lý tưởng có AUC $`0.5`$, còn precision kỳ vọng tại ngưỡng chọn độc lập nhãn bằng prevalence $`P/(P+N)`$. AP thực nghiệm trên mẫu hữu hạn không nhất thiết bằng prevalence. [2][7][9]

### Các Thước đo Hồi quy & Chỉ số $`R^2`$ Hiệu chỉnh (Adjusted $`R^2`$)

**Regression:** với $`e_{i}=y_{i}-\hat{y}_{i}`$, $`\mathrm{MAE}=\frac{1}{n}\sum_i\lvert e_i\rvert`$, $`\mathrm{MSE}=\frac{1}{n}\sum_i e_i^2`$, $`\mathrm{RMSE}=\sqrt{\mathrm{MSE}}`$, $`R^2=1-\frac{\sum_i e_i^2}{\sum_i(y_i-\bar{y}_{\mathrm{test}})^2}`$. Trong phép tính R² trên test, $`\bar{y}_{test}`$ là trung bình `y_true` của **tập đánh giá** theo định nghĩa metric; nó không đồng nhất với DummyRegressor đã fit trung bình train. Nếu `y_true` hằng, R² có trường hợp biên cần xem quy ước thư viện. [2]

**Chỉ số $`R^2`$ Hiệu chỉnh (Adjusted $`R^2`$):**
Với **OLS không regularization, có intercept, fit trên cùng tập train có target không hằng**, khi thêm biến mà vẫn giữ các biến cũ (mô hình lồng nhau), $`R^2`$ **trên train** không giảm: mô hình mới có thể đặt hệ số biến thêm bằng 0. Điều này không bảo đảm $`R^2`$ trên validation/test tăng và không áp dụng chung cho mọi thuật toán. Adjusted $`R^2`$ điều chỉnh mức phù hợp trên train theo số tham số: [13]

$$
R_{\text{adj}}^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]
$$

Trong đó $`n`$ là số mẫu train, $`p`$ là số hệ số predictor không tính intercept; công thức dùng cho thiết kế OLS đầy đủ hạng với $`n>p+1`$. $`R_{\text{adj}}^2`$ giảm nếu mức giảm sai số không đủ bù số tham số thêm vào. Chỉ số này không thay thế đánh giá trên dữ liệu chưa thấy.

**PR-AUC cần gọi tên rõ:** `average_precision_score` của scikit-learn (AP) tính trung bình có trọng số các precision khi recall tăng; nó **khác** diện tích nội suy hình thang dưới PR curve. Khi báo cáo, ghi rõ “AP” hay “trapezoidal PR-AUC” và cách tính. Baseline AP của dự đoán ngẫu nhiên thường liên quan prevalence lớp dương; so sánh cùng tập dữ liệu và cùng định nghĩa. [2]

---

## How it works

1. Trên train, dùng validation/CV để chọn mô hình, hyperparameter, calibration và threshold. Với nhóm/thời gian, dùng splitter phù hợp. [1][3][4]
2. Chạy pipeline đã khóa trên test một lần; lưu metric và số mẫu. Nếu sau khi xem test lại chỉnh mô hình, cần một test mới để có ước lượng độc lập. [1][5]
3. Kiểm confusion matrix từng lớp hoặc residual theo mức dự đoán; xem các lát cắt quan trọng và ví dụ lỗi cụ thể.
4. So với baseline cùng split; kiểm sự khác biệt giữa validation và test, chi phí lỗi, latency và chất lượng xác suất.

**Chọn metric theo chi phí:** nếu mỗi FN tốn 10 đơn vị và mỗi FP tốn 1 đơn vị, một quy tắc vận hành có thể tối thiểu hóa $`10\times FN + 1\times FP`$ trên validation, có ràng buộc năng lực xử lý báo động. Đừng chỉ tối đa F1 nếu chi phí FN và FP không cân. Sau khi chọn threshold, báo cáo chi phí/precision/recall trên test một lần. [6]

### Thuật toán Tìm Ngưỡng Tối ưu theo Ma trận Chi phí (Optimal Threshold Search)

Giả sử mô hình xuất ra $`\hat{p}_i`$, một ước lượng của $`P(y_i=1 \mid x_i)`$. Quyết định nhị phân $`\hat{y}_i \in \{0, 1\}`$ phụ thuộc vào ngưỡng $`\tau \in [0, 1]`$:

$$
\hat{y}_i(\tau) = \mathbb{I}(\hat{p}_i \ge \tau)
$$

Tổng tổn thất nghiệp vụ được định nghĩa:

$$
\mathrm{TotalCost}(\tau) = C_{FP} \cdot \sum_{i: y_i=0} \mathbb{I}(\hat{p}_i \ge \tau) + C_{FN} \cdot \sum_{i: y_i=1} \mathbb{I}(\hat{p}_i < \tau)
$$

* **Quy trình tìm ngưỡng khoa học:**
  1. Quét các giá trị xác suất duy nhất trên **Tập Validation**, đồng thời xét phương án luôn dương và luôn âm để không bỏ sót nghiệm biên. Nếu có điểm bằng 1, phương án luôn âm cần xét riêng vì quy tắc $`\hat{p}_i\ge\tau`$ với $`\tau\le1`$ không biểu diễn được nó.
  2. Tính toán $`\mathrm{TotalCost}(\tau)`$ tại từng điểm cắt.
  3. Chọn $`\tau^* = \mathop{\mathrm{arg\,min}}_{\tau} \mathrm{TotalCost}(\tau)`$.
  4. Đóng băng $`\tau^*`$ và áp dụng cố định sang **Tập Test** để đo lường chi phí thực tế.

---

## Data used

Test phải phản ánh dữ liệu tương lai/sản xuất và không chứa bản sao hoặc group trùng với train khi mục tiêu là group mới. Với chuỗi thời gian, test thuộc giai đoạn sau train; với dữ liệu lớp hiếm, báo cáo cả số mẫu mỗi lớp. [1][5]

---

## Design

- Không kết luận mô hình tốt từ một điểm số duy nhất; ghi `n`, cách split, seed và metric cho từng nhóm.
- Khi dữ liệu ít, dùng CV để so cấu hình; giữ một test cuối hoặc nested CV khi cần ước lượng sau tuning. [1]
- Failure modes: test bị dùng lặp để điều chỉnh, leakage qua preprocessing, metric không gắn chi phí, test không đại diện production.

### Sổ tay Chẩn đoán: Hiệu chuẩn Xác suất (Probability Calibration)

Các thuật toán như SVM, Random Forest, Naive Bayes hoặc Boosting có thể xếp hạng tốt nhưng **xác suất chưa được hiệu chuẩn tốt**, tùy dữ liệu và cách huấn luyện: [14]
* *Naive Bayes:* Có thể cho xác suất quá gần 0 hoặc 1 khi giả định độc lập có điều kiện giữa các feature không phù hợp với dữ liệu.
* *Random Forest:* Khó dự đoán xác suất sát 0 hoặc 1 do cơ chế bỏ phiếu trung bình từ nhiều cây con.

**Thước đo chất lượng dự đoán xác suất (Brier Score, nhị phân):**

$$
\mathrm{BrierScore} = \frac{1}{n} \sum_{i=1}^n (\hat{p}_i - y_i)^2
$$

Brier Score có thể phân rã thành: $`\text{Reliability} - \text{Resolution} + \text{Uncertainty}`$. Điểm nhỏ hơn thể hiện sai số xác suất bình phương nhỏ hơn, nhưng **không tự chứng minh calibration tốt hơn** vì còn phụ thuộc resolution. Xem thêm calibration plot. [14]

**Phương pháp khắc phục:** fit bộ hiệu chuẩn bằng dữ liệu tách khỏi dữ liệu fit mô hình nền hoặc bằng cross-validation phù hợp; không dùng test cuối để fit hay chọn phương pháp. [14]
1. **Platt Scaling (Hồi quy Logistic đơn biến):** Phù hợp khi đường cong hiệu chuẩn có dạng sigmoid (như điểm khoảng cách của SVM).
2. **Isotonic Regression (Hồi quy đơn điệu phi tham số):** Linh hoạt hơn, không giả định dạng hàm, nhưng yêu cầu tập dữ liệu hiệu chuẩn đủ lớn để tránh overfit.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Confusion Matrix** | Bảng thống kê chéo số mẫu theo nhãn thật và nhãn dự đoán. | Cho biết từng loại lỗi như $`FP`$ và $`FN`$. | Tính metric từ nhãn tại một ngưỡng; AUC và log loss còn cần điểm số hoặc xác suất. |
| **ROC-AUC** | Diện tích dưới đường cong True Positive Rate theo False Positive Rate. | Đo khả năng xếp hạng trên nhiều ngưỡng. | Bổ sung metric tại ngưỡng vận hành. |
| **AP / trapezoidal PR-AUC** | AP cộng precision theo mức tăng recall; trapezoidal PR-AUC dùng nội suy hình thang. Hai cách tính khác nhau. | Tóm tắt quan hệ precision–recall, hữu ích khi lớp dương hiếm. | Ghi cách tính, prevalence và metric tại ngưỡng vận hành. [9] |
| **Macro vs Micro** | Macro tính trung bình các lớp; Micro gộp TP/FP/FN trước khi tính. | Macro cho mỗi lớp trọng số bằng nhau; Micro phản ánh hiệu năng gộp. | Phân loại báo cáo trong bài toán đa lớp. |
| **Calibration** | Mức độ trùng khớp giữa xác suất dự đoán và tần suất thực tế. | Kiểm tra ý nghĩa của xác suất đầu ra. | Cần kiểm tra khi quyết định dựa vào xác suất; không phải mô hình nào cũng cần hiệu chuẩn lại. |
| **Bootstrap CI** | Khoảng tin cậy qua tái lấy mẫu có hoàn lại. | Lượng hóa bất định của metric dưới giả định lấy mẫu. | Một lựa chọn khi thiết kế dữ liệu phù hợp. [11] |

---

## Nguồn

Ghi chú phạm vi tra cứu và căn cứ hiệu đính: [Ghi chú hiệu đính](ghi_chu_hieu_dinh.md).

- [1] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [2] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [4] [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [5] [Google: Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [6] [scikit-learn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [7] Fawcett, T. (2006). *An introduction to ROC analysis*. Pattern Recognition Letters, 27(8), 861-874.
- [8] Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good probabilities with supervised learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML).
- [9] [scikit-learn: Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [10] [scikit-learn: Statistical comparison of models](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html)
- [11] [SciPy: Bootstrap confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [12] [statsmodels: McNemar test](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
- [13] [Penn State STAT 501: Multiple linear regression](https://online.stat.psu.edu/stat501/Lesson05)
- [14] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
