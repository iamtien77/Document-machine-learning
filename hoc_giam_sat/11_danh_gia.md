# Đánh giá mô hình học giám sát

## Purpose

Ước lượng khả năng tổng quát hóa trên dữ liệu chưa thấy và kiểm tra mô hình có đáp ứng mục tiêu thực tế hay không. Metric phải được chọn trước khi so mô hình. [1][2]

### Nguyên lý Đánh giá Thực chứng & Khoảng Tin cậy Thống kê (Statistical Hypothesis Testing)

Trong nghiên cứu khoa học, một điểm số số học đơn lẻ (ví dụ: "Accuracy = 92.5%") là hoàn toàn vô nghĩa nếu thiếu ngữ cảnh về kích thước mẫu ($n$), phân phối nhãn và độ biến thiên thống kê. Đánh giá mô hình thực chất là một phép ước lượng tham số thống kê về **sai số tổng quát hóa thực tế** của mô hình trên toàn bộ không gian dữ liệu chưa quan sát.

Hai nguyên tắc phương pháp luận bắt buộc:
1. **Kiểm định giả thuyết thống kê (Hypothesis Testing):** Khi so sánh mô hình mới $A$ với mô hình cơ sở $B$, không thể vội vàng kết luận $A$ tốt hơn $B$ chỉ vì điểm trung bình của $A$ cao hơn 0.5%. Cần thực hiện kiểm định ý nghĩa thống kê (như *McNemar's Test* cho bài toán phân loại hoặc *Paired Student's t-test / Wilcoxon Signed-Rank Test* trên các fold) để chứng minh sự vượt trội là có ý nghĩa thống kê chứ không phải do may rủi ngẫu nhiên ($p\text{-value} < 0.05$).
2. **Khoảng tin cậy Bootstrap (Bootstrap Confidence Intervals):** Thay vì chỉ báo cáo một con số điểm test, cần báo cáo kèm khoảng tin cậy 95% thu được qua kỹ thuật tái lấy mẫu Bootstrap trên tập test:
   $$\text{Metric} = \hat{\mu} \pm 1.96 \times \operatorname{SE}_{\text{boot}}$$

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
| Accuracy | $\frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{FP}+\mathrm{FN}+\mathrm{TN}}$ | Tỉ lệ dự đoán đúng |
| Precision | $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$ | Trong số dự đoán dương, bao nhiêu đúng |
| Recall/TPR | $\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}$ | Trong số dương thật, tìm được bao nhiêu |
| Specificity/TNR | $\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}$ | Trong số âm thật, loại đúng bao nhiêu |
| F1 | $\frac{2\,\mathrm{precision}\,\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}}$ | Trung bình điều hòa của precision và recall |

Nếu mẫu số bằng 0, metric có thể không xác định; báo cáo số mẫu và quy tắc xử lý (`zero_division` nếu dùng scikit-learn), không âm thầm coi là 0 hay 1. Với multiclass, tính theo one-vs-rest từng lớp rồi dùng **macro** (trung bình các lớp), **weighted** (trọng số theo support) hoặc **micro** (cộng TP/FP/FN trước khi tính). Macro thể hiện lớp hiếm rõ hơn; weighted có thể che lỗi lớp hiếm. [2]

**Ví dụ confusion matrix:** $TP=30$, $FP=10$, $FN=20$, $TN=940$. Accuracy=`970/1000=0.97`, precision=`30/40=0.75`, recall=`30/50=0.60`, F1≈`0.667`. Accuracy 97% vẫn bỏ sót 40% ca dương tính. Tỉ lệ dương thực tế là `50/1000=5%`; baseline luôn đoán âm đạt 95% accuracy nhưng recall bằng 0.

### Phân tích Sâu: Đánh giá Đa lớp (Multi-class Averaging Nuances)

Khi mở rộng sang bài toán phân loại $K$ lớp, việc gộp các chỉ số thành một điểm số duy nhất đòi hỏi sự thận trọng đặc biệt:

* **Macro-Average:** Tính metric độc lập cho từng lớp rồi lấy trung bình cộng giản đơn:
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
  *Đặc tính:* Coi mọi lớp đều bình đẳng như nhau. Nếu mô hình đoán sai hoàn toàn ở một lớp thiểu số cực hiếm, Macro-F1 sẽ giảm thê thảm, lập tức gióng lên hồi chuông cảnh báo.
* **Weighted-Average:** Tính metric từng lớp rồi nhân với tỷ trọng số lượng mẫu (Support $n_k$):
  $$\text{Weighted-F1} = \frac{1}{n} \sum_{k=1}^K n_k \cdot \text{F1}_k$$
  *Cảnh báo nguy hiểm:* Bị chi phối áp đảo bởi lớp chiếm đa số. Nếu lớp đa số đạt F1=99% nhưng lớp thiểu số đạt F1=0%, Weighted-F1 vẫn có thể đạt mức đẹp đẽ 98%, che giấu hoàn toàn thất bại nghiêm trọng của hệ thống.
* **Micro-Average:** Cộng dồn toàn bộ $TP, FP, FN$ của tất cả các lớp lại trước khi tính toán:
  $$\text{Micro-F1} = \text{Micro-Precision} = \text{Micro-Recall} = \text{Accuracy}$$
  Trong bài toán đơn nhãn (mỗi mẫu thuộc đúng 1 lớp), Micro-F1 toán học tương đương chính xác với Accuracy tổng thể.

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
   * Trục tung: True Positive Rate ($\text{TPR} = \frac{TP}{P}$).
   * Trục hoành: False Positive Rate ($\text{FPR} = \frac{FP}{N}$).
   * *Đặc tính toán học cốt lõi:* Cả TPR và FPR đều được tính toán riêng biệt bên trong từng lớp tương ứng. Do đó, **hình dạng đường cong ROC và chỉ số ROC-AUC hoàn toàn bất biến đối với sự thay đổi của tỷ lệ lớp (Class Prevalence Invariant)**.
2. **Precision-Recall (PR) Curve:**
   * Trục tung: $\text{Precision} = \frac{TP}{TP + FP}$.
   * Trục hoành: $\text{Recall} = \frac{TP}{TP + FN}$.
   * *Đặc tính toán học cốt lõi:* Precision phụ thuộc trực tiếp vào số lượng mẫu âm tính giả định sai ($FP$). Khi lớp dương cực hiếm (ví dụ: Gian lận thẻ tín dụng 0.1%), chỉ cần một tỷ lệ nhỏ $FP$ từ lớp âm tính khổng lồ cũng đủ làm mẫu số $TP + FP$ phình to, khiến Precision sụp đổ.
   * **Quy tắc vàng của nhà nghiên cứu:** Trong các bài toán mất cân bằng nhãn trầm trọng, **ROC-AUC thường tạo ảo giác lạc quan nguy hiểm** (vẫn đạt 0.98 vì $TN$ quá lớn kéo $FPR$ xuống gần 0). Khi đó, **bắt buộc phải sử dụng PR-AUC / Average Precision (AP)** làm thước đo định đoạt. Đường cơ sở (Baseline) của ROC-AUC luôn cố định là $0.5$, trong khi Baseline của PR-AUC chính là tỷ lệ lớp dương thực tế trong dữ liệu ($P / (P + N)$).

### Các Thước đo Hồi quy & Chỉ số $R^2$ Hiệu chỉnh (Adjusted $R^2$)

**Regression:** với $e_{i}=y_{i}-\hat{y}_{i}$, $\mathrm{MAE}=\frac{1}{n}\sum_i\lvert e_i\rvert$, $\mathrm{MSE}=\frac{1}{n}\sum_i e_i^2$, $\mathrm{RMSE}=\sqrt{\mathrm{MSE}}$, $R^2=1-\frac{\sum_i e_i^2}{\sum_i(y_i-\bar{y}_{\mathrm{test}})^2}$. Trong phép tính R² trên test, $\bar{y}_{test}$ là trung bình `y_true` của **tập đánh giá** theo định nghĩa metric; nó không đồng nhất với DummyRegressor đã fit trung bình train. Nếu `y_true` hằng, R² có trường hợp biên cần xem quy ước thư viện. [2]

**Chỉ số $R^2$ Hiệu chỉnh (Adjusted $R^2$):**
Một lỗ hổng toán học kinh điển của $R^2$ thông thường là: **nó không bao giờ giảm khi ta đưa thêm bất kỳ biến ngẫu nhiên vô nghĩa nào vào mô hình**. Để giải quyết bài toán này trong nghiên cứu thực nghiệm:
$$R_{\text{adj}}^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$
Trong đó $n$ là số mẫu quan sát, $p$ là số lượng đặc trưng. $R_{\text{adj}}^2$ trực tiếp phạt số bậc tự do bị tiêu tốn. Nếu một biến mới được thêm vào không đóng góp lượng giảm phương sai đủ lớn để bù đắp chi phí tăng số chiều, $R_{\text{adj}}^2$ sẽ lập tức sụt giảm.

**PR-AUC cần gọi tên rõ:** `average_precision_score` của scikit-learn (AP) tính trung bình có trọng số các precision khi recall tăng; nó **khác** diện tích nội suy hình thang dưới PR curve. Khi báo cáo, ghi rõ “AP” hay “trapezoidal PR-AUC” và cách tính. Baseline AP của dự đoán ngẫu nhiên thường liên quan prevalence lớp dương; so sánh cùng tập dữ liệu và cùng định nghĩa. [2]

---

## How it works

1. Trên train, dùng validation/CV để chọn mô hình, hyperparameter, calibration và threshold. Với nhóm/thời gian, dùng splitter phù hợp. [1][3][4]
2. Chạy pipeline đã khóa trên test một lần; lưu metric và số mẫu. Nếu sau khi xem test lại chỉnh mô hình, cần một test mới để có ước lượng độc lập. [1][5]
3. Kiểm confusion matrix từng lớp hoặc residual theo mức dự đoán; xem các lát cắt quan trọng và ví dụ lỗi cụ thể.
4. So với baseline cùng split; kiểm sự khác biệt giữa validation và test, chi phí lỗi, latency và chất lượng xác suất.

**Chọn metric theo chi phí:** nếu mỗi FN tốn 10 đơn vị và mỗi FP tốn 1 đơn vị, một quy tắc vận hành có thể tối thiểu hóa $10\times FN + 1\times FP$ trên validation, có ràng buộc năng lực xử lý báo động. Đừng chỉ tối đa F1 nếu chi phí FN và FP không cân. Sau khi chọn threshold, báo cáo chi phí/precision/recall trên test một lần. [6]

### Thuật toán Tìm Ngưỡng Tối ưu theo Ma trận Chi phí (Optimal Threshold Search)

Giả sử mô hình xuất ra điểm xác suất $\hat{p}_i = P(y_i=1 \mid x_i)$. Quyết định nhị phân $\hat{y}_i \in \{0, 1\}$ phụ thuộc vào ngưỡng $\tau \in [0, 1]$:
$$\hat{y}_i(\tau) = \mathbb{I}(\hat{p}_i \ge \tau)$$

Tổng tổn thất nghiệp vụ được định nghĩa:
$$\operatorname{TotalCost}(\tau) = C_{FP} \cdot \sum_{i: y_i=0} \mathbb{I}(\hat{p}_i \ge \tau) + C_{FN} \cdot \sum_{i: y_i=1} \mathbb{I}(\hat{p}_i < \tau)$$

* **Quy trình tìm ngưỡng khoa học:**
  1. Quét $\tau$ qua tập hợp các giá trị xác suất duy nhất được dự đoán trên **Tập Validation**.
  2. Tính toán $\operatorname{TotalCost}(\tau)$ tại từng điểm cắt.
  3. Chọn $\tau^* = \operatorname*{arg\,min}_{\tau} \operatorname{TotalCost}(\tau)$.
  4. Đóng băng $\tau^*$ và áp dụng cố định sang **Tập Test** để đo lường chi phí thực tế.

---

## Data used

Test phải phản ánh dữ liệu tương lai/sản xuất và không chứa bản sao hoặc group trùng với train khi mục tiêu là group mới. Với chuỗi thời gian, test thuộc giai đoạn sau train; với dữ liệu lớp hiếm, báo cáo cả số mẫu mỗi lớp. [1][5]

---

## Design

- Không kết luận mô hình tốt từ một điểm số duy nhất; ghi `n`, cách split, seed và metric cho từng nhóm.
- Khi dữ liệu ít, dùng CV để so cấu hình; giữ một test cuối hoặc nested CV khi cần ước lượng sau tuning. [1]
- Failure modes: test bị dùng lặp để điều chỉnh, leakage qua preprocessing, metric không gắn chi phí, test không đại diện production.

### Sổ tay Chẩn đoán: Hiệu chuẩn Xác suất (Probability Calibration)

Nhiều thuật toán phân loại mạnh mẽ (như SVM, Random Forest, Naive Bayes hoặc Boosting) tạo ra điểm số xếp hạng rất tốt nhưng **xác suất đầu ra bị méo mó nghiêm trọng**:
* *Naive Bayes:* Có xu hướng đẩy xác suất về sát cực trị 0 hoặc 1 do giả định độc lập phi thực tế.
* *Random Forest:* Khó dự đoán xác suất sát 0 hoặc 1 do cơ chế bỏ phiếu trung bình từ nhiều cây con.

**Thước đo Hiệu chuẩn (Brier Score):**
$$\operatorname{BrierScore} = \frac{1}{n} \sum_{i=1}^n (\hat{p}_i - y_i)^2$$
Brier Score có thể phân rã thành: $\text{Reliability} - \text{Resolution} + \text{Uncertainty}$. Giá trị càng nhỏ thể hiện xác suất càng đáng tin cậy.

**Phương pháp khắc phục:**
1. **Platt Scaling (Hồi quy Logistic đơn biến):** Phù hợp khi đường cong hiệu chuẩn có dạng sigmoid (như điểm khoảng cách của SVM).
2. **Isotonic Regression (Hồi quy đơn điệu phi tham số):** Linh hoạt hơn, không giả định dạng hàm, nhưng yêu cầu tập dữ liệu hiệu chuẩn đủ lớn để tránh overfit.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Confusion Matrix** | Bảng vuông thống kê chéo số lượng mẫu theo nhãn thực tế và nhãn dự đoán. | Phơi bày chi tiết từng loại sai lầm cụ thể ($FP$ và $FN$). | Nền móng để tính toán toàn bộ các metric phân loại. |
| **ROC-AUC** | Diện tích dưới đường cong True Positive Rate theo False Positive Rate. | Đo lường năng lực xếp hạng tổng quát độc lập ngưỡng cắt. | Tiêu chuẩn so sánh mô hình khi lớp tương đối cân bằng. |
| **PR-AUC (AP)** | Diện tích dưới đường cong Precision theo Recall. | Đánh giá năng lực phát hiện mẫu hiếm mà không bị loãng bởi $TN$. | Thước đo tối thượng cho bài toán mất cân bằng nhãn trầm trọng. |
| **Macro vs Micro** | Macro tính trung bình các lớp; Micro gộp toàn bộ mẫu trước khi tính. | Macro bảo vệ lớp thiểu số; Micro phản ánh hiệu năng mẫu gộp. | Phân loại báo cáo trong bài toán đa lớp. |
| **Calibration** | Mức độ trùng khớp giữa xác suất mô hình dự đoán và tần suất thực tế. | Biến đầu ra thành giá trị định lượng rủi ro thực tế. | Bắt buộc khi dùng xác suất để ra quyết định kinh doanh hoặc y tế. |
| **Bootstrap CI** | Khoảng tin cậy thu được bằng cách tái lấy mẫu có hoàn lại trên tập test. | Lượng hóa biên độ bất định thống kê của metric. | Tiêu chuẩn khoa học bắt buộc trong các công bố học thuật. |

---

## Nguồn

- [1] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [2] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [4] [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [5] [Google: Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [6] [scikit-learn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [7] Fawcett, T. (2006). *An introduction to ROC analysis*. Pattern Recognition Letters, 27(8), 861-874.
- [8] Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good probabilities with supervised learning*. Proceedings of the 22nd International Conference on Machine Learning (ICML).
