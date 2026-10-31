# Sau train: phân tích lỗi, triển khai và giám sát

## Purpose

Chuyển mô hình đã đánh giá thành quyết định có thể dùng và kiểm soát suy giảm chất lượng khi dữ liệu thực tế thay đổi. [1]

### Chu trình Sống của Hệ thống Học máy & Hiện tượng Thoái hóa Mô hình (Model Decay)

Trong kỹ nghệ phần mềm truyền thống, khi một module code đã vượt qua bài kiểm thử unit test, nó sẽ hoạt động ổn định và tin cậy theo thời gian trừ khi có ai đó sửa đổi mã nguồn. Tuy nhiên, trong kỹ nghệ học máy (MLOps), **mô hình bắt đầu suy giảm chất lượng ngay từ giây phút đầu tiên được đưa vào triển khai sản xuất**.

Hiện tượng này được gọi là **Sự suy thoái mô hình (Model Decay / Model Drift)**. Thế giới thực liên tục biến động: hành vi người tiêu dùng thay đổi, điều kiện kinh tế vĩ mô biến thiên, cảm biến thiết bị bị hao mòn, hoặc những kẻ gian lận tìm ra cách thức mới để qua mặt hệ thống. Một mô hình tĩnh được huấn luyện trên dữ liệu quá khứ sẽ dần trở nên lỗi thời. Do đó, giai đoạn sau huấn luyện là một quy trình kỹ thuật liên tục, bao gồm:
1. **Phân tích giải phẫu sai số (Error Diagnostics & Slicing):** Hiểu rõ tại sao mô hình đưa ra dự đoán sai và mô hình đang thất bại trên nhóm đối tượng nào.
2. **Triển khai an toàn (Safe Serving Architecture):** Thiết lập ranh giới phòng vệ để hệ thống không gây thiệt hại khi nhận dữ liệu bất thường.
3. **Giám sát thống kê liên tục (Continuous Statistical Monitoring):** Phát hiện sớm các dấu hiệu dịch chuyển phân phối trước khi chúng kịp làm suy sụp chỉ số kinh doanh.

---

## Input

Pipeline và phiên bản mô hình đã khóa, kết quả test, threshold/calibrator chọn từ các tập giữ riêng hoặc dự đoán out-of-fold hợp lệ, schema inference, yêu cầu latency/chi phí và cách nhận nhãn sau triển khai.

---

## Output

Artifact suy luận (`preprocessing + model + threshold/calibration`), báo cáo lỗi và giới hạn, kế hoạch monitor, trigger kiểm tra lại/retrain, rollback và phiên bản có thể truy vết.

---

## Evaluation metrics

- **Offline:** metric test chính; calibration/log loss nếu dùng xác suất; sai số theo nhóm, giai đoạn và điều kiện dữ liệu.
- **Online:** tỉ lệ dữ liệu thiếu/sai schema, tỉ lệ nhãn/dự đoán, drift của feature, latency, lỗi hệ thống, training-serving skew; khi nhãn đến muộn thì đo lại metric thực tế. Drift là tín hiệu điều tra, không tự chứng minh chất lượng đã giảm. [1][2]

### Phân loại Ba Hình thái Dịch chuyển Phân phối (Distribution Drift Taxonomy)

Để giám sát và chẩn đoán chính xác nguyên nhân suy thoái, khoa học học máy phân biệt rạch ròi ba dạng trôi dạt dữ liệu:

```text
                  P(X, Y) = P(X) * P(Y | X)
                     _______|_______
                    |               |
              COVARIATE SHIFT    CONCEPT DRIFT
             P(X) thay đổi       P(Y|X) thay đổi
```

1. **Covariate Shift (Dịch chuyển Biến đồng thuận):**
   * *Định nghĩa toán học:* Phân phối xác suất của các đặc trưng đầu vào thay đổi $`P_{\text{train}}(X) \neq P_{\text{prod}}(X)`$, nhưng quy luật điều kiện sinh nhãn giữ nguyên $`P(Y \mid X = x)`$.
   * *Ví dụ:* Một mô hình chấm điểm tín dụng được train chủ yếu trên người trẻ tuổi ($`20-30`$ tuổi), nhưng khi ngân hàng chạy chiến dịch quảng cáo mới, khách hàng nộp hồ sơ chủ yếu là người trung niên ($`40-60`$ tuổi).
2. **Concept Drift (Trôi dạt Khái niệm):**
   * *Định nghĩa toán học:* Bản chất mối quan hệ ánh xạ giữa đầu vào và nhãn mục tiêu bị thay đổi $`P_{\text{train}}(Y \mid X) \neq P_{\text{prod}}(Y \mid X)`$, ngay cả khi phân phối đầu vào $`P(X)`$ hoàn toàn không đổi.
   * *Ví dụ:* Cùng một căn hộ $`80\,\text{m}^2`$ tại một quận, trước đại dịch có giá 3 tỷ, sau đại dịch do lạm phát giá nhảy vọt lên 5 tỷ. Hoặc kẻ lừa đảo thay đổi thủ đoạn rửa tiền để qua mặt các quy tắc cũ.
3. **Prior Probability Shift (Dịch chuyển Tiền nghiệm nhãn):**
   * *Định nghĩa toán học:* Tỷ lệ xuất hiện của các lớp mục tiêu $`P(Y)`$ thay đổi theo thời gian.
   * *Ví dụ:* Tỷ lệ mắc một bệnh truyền nhiễm tăng vọt từ $`0.1\%`$ lên $`15\%`$ khi có đợt bùng phát dịch mới.

### Các Thước đo Thống kê Giám sát Drift Tự động

Khi mô hình chạy trên production, nhãn thực tế $`y`$ thường bị trễ hàng tuần hoặc hàng tháng (ví dụ: sau 30 ngày mới biết khách hàng có vỡ nợ hay không). Do đó, **hệ thống giám sát bắt buộc phải theo dõi drift trên phân phối đầu vào $`X`$ và phân phối điểm số đầu ra $`\hat{y}`$**:

1. **Chỉ số Ổn định Quần thể (Population Stability Index - PSI):**
   Chia miền giá trị của biến thành $`B`$ khoảng (bins) dựa trên tập huấn luyện, sau đó so sánh tỷ lệ quan sát giữa tập thực tế ($`A`$) và tập kỳ vọng lúc train ($`E`$):

   ```math
   \mathrm{PSI} = \sum_{b=1}^B \left( \mathrm{Actual}_b - \mathrm{Expected}_b \right) \times \ln\left( \frac{\mathrm{Actual}_b}{\mathrm{Expected}_b} \right)
   ```

   * *Tiêu chuẩn nghiệm thu:*
     * $`\mathrm{PSI} < 0.1`$: Phân phối ổn định, không có thay đổi đáng kể.
     * $`0.1 \le \mathrm{PSI} < 0.25`$: Bắt đầu có sự dịch chuyển nhẹ, cần đưa vào danh sách theo dõi.
     * $`\mathrm{PSI} \ge 0.25`$: Dịch chuyển phân phối nghiêm trọng, kích hoạt quy trình cảnh báo và xem xét retrain.
2. **Kiểm định Kolmogorov-Smirnov hai mẫu (Two-sample KS-Test):**
   Đo khoảng cách supremum giữa hai hàm phân phối tích lũy thực nghiệm (Empirical CDF):

   ```math
   D_{\text{KS}} = \sup_x \lvert F_{\text{train}}(x) - F_{\text{prod}}(x) \rvert
   ```

   Nếu $`p\text{-value} < 0.01`$, bác bỏ giả thuyết cho rằng hai tập dữ liệu cùng xuất phát từ một phân phối.

---

## How it works

1. Phân tích ví dụ sai, confusion matrix/residual và các lát cắt; xác định lỗi từ dữ liệu, nhãn, feature hay mô hình.
2. Nếu cần xác suất đáng tin cậy, fit calibrator trên dữ liệu tách khỏi dữ liệu fit classifier hoặc từ dự đoán out-of-fold theo CV đúng cách. Chọn threshold theo chi phí/metric trên tập dành riêng cho bước này hoặc dự đoán out-of-fold phù hợp; không dùng test. Nếu vừa calibration vừa tune threshold, tách dữ liệu cho hai bước hoặc dùng cross-fitting để tránh điều chỉnh quá mức trên cùng mẫu. Nếu đã refit classifier trên train+validation, **không dùng lại chính validation đó** như một holdout độc lập để calibration. [3][4]
3. Đóng gói pipeline, kiểm tra schema và cùng một phép biến đổi cho train và inference; version dữ liệu, code, model và threshold. [5]
4. Triển khai có kiểm soát; theo dõi dữ liệu vào, đầu ra, latency, chất lượng khi nhãn thật có sẵn. So sánh với baseline hoặc mô hình đang chạy.
5. Khi cảnh báo, điều tra nguồn thay đổi; thu thập nhãn mới, đánh giá lại trên dữ liệu gần đây, rồi quyết định retrain và kiểm thử phiên bản mới.

**Ví dụ chọn threshold:** trên tập threshold riêng, với mỗi ngưỡng ứng viên `t`, dự đoán dương khi $`p\ge t`$, tính `TP/FP/FN/TN` và chi phí $`c_{FN}\times FN+c_{FP}\times FP`$. Chọn `t` có chi phí thấp nhất trong các ngưỡng thỏa ràng buộc công suất xử lý; khóa `t` rồi chỉ báo cáo test. Nếu xác suất chưa calibration, không diễn giải $`p=0.8`$ thành “80% chắc chắn” trước khi kiểm tra calibration. [3][4]

**Runbook khi có cảnh báo:** xác nhận dữ liệu đầu vào/schema → xác định nhóm/thời gian bị ảnh hưởng → kiểm có thay đổi pipeline hay nguồn nhãn → đo chất lượng trên mẫu đã có nhãn → so với baseline và phiên bản trước → chỉ retrain khi dữ liệu mới/nhãn mới giải thích được vấn đề → kiểm test mới và rollback plan. Các ngưỡng cảnh báo phải lấy từ baseline vận hành và chi phí thực tế, không tự đặt một con số chung cho mọi hệ thống. [1]

### Các Chiến lược Phát hành Mô hình An toàn (Safe Deployment Strategies)

Không bao giờ thay thế đột ngột một mô hình đang phục vụ người dùng bằng một file model mới mà không qua các bước đệm an toàn:

1. **Triển khai Bóng (Shadow Deployment / Dark Launching):**
   * Mô hình mới ($`M_{\text{new}}`$) được triển khai song song với mô hình hiện tại ($`M_{\text{current}}`$).
   * Mọi request thực tế được gửi đến cả hai mô hình, nhưng **chỉ có dự đoán của $`M_{\text{current}}`$ được trả về cho người dùng**.
   * Dự đoán của $`M_{\text{new}}`$ được ghi vào log để đo lường độ trễ (latency), mức độ tiêu tốn tài nguyên và tính ổn định trong điều kiện tải thực tế.
2. **Triển khai Chim hoàng yến (Canary Deployment):**
   * Điều hướng một tỷ lệ lưu lượng nhỏ ($`1\% \to 5\%`$) sang mô hình mới.
   * Giám sát chặt chẽ tỷ lệ lỗi và phản hồi của người dùng. Nếu các chỉ số ổn định, tăng dần tỷ lệ lên $`25\% \to 50\% \to 100\%`$.
   * Nếu có sự cố, cơ chế **Rollback tự động** sẽ lập tức điều hướng toàn bộ traffic trở về phiên bản an toàn trước đó trong vòng vài giây.
3. **Thử nghiệm A/B Trực tuyến (Online A/B Testing):**
   * Phân chia người dùng ngẫu nhiên thành hai nhánh độc lập để so sánh chỉ số kinh doanh cốt lõi (Conversion Rate, Doanh thu, Click-Through Rate) giữa mô hình mới và mô hình cũ.

---

## Data used

Tập dành cho calibration và tập dành cho threshold không dùng để fit classifier; nếu thực hiện cả hai bước, nên tách hai tập hoặc dùng dự đoán out-of-fold/cross-fitting hợp lệ. Test cuối dùng để báo cáo. Production data và nhãn đến muộn dùng để đánh giá drift/chất lượng sau triển khai. Không giả định rằng dữ liệu production có nhãn tức thời. [1][3][4]

---

## Design

- Theo dõi metric theo nhóm và thời gian cùng với số mẫu; tránh diễn giải biến động trên nhóm rất nhỏ.
- Permutation importance trên dữ liệu held-out giúp kiểm tra mức phụ thuộc dự đoán vào feature, nhưng feature tương quan có thể làm cách đọc khó hơn. [6]
- Failure modes: train-serving skew, nhãn đến trễ, thay đổi định nghĩa target, calibration hỏng khi phân phối thay đổi, retrain tự động khuếch đại lỗi nhãn.
- Điều kiện đưa vào production nên gồm cải thiện so baseline, sai số nhóm chấp nhận được, latency phù hợp và khả năng rollback.

**Theo dõi định lượng:** ghi số mẫu mỗi cửa sổ thời gian, tỉ lệ missing từng feature, tỉ lệ giá trị ngoài miền hợp lệ, phân phối score, tỉ lệ dự đoán dương, p50/p95 latency, và metric thật khi nhãn đến. Chỉ báo thay đổi feature/score là tín hiệu kiểm tra; chất lượng mô hình cần đo bằng nhãn đáng tin cậy. Giữ khả năng nối prediction với nhãn về sau bằng ID không lộ dữ liệu nhạy cảm.

### Quy trình Phân tích Sai số Thực chứng (Error Analysis Protocol)

Khi mô hình thất bại trên một tập mẫu, kỹ sư không nên vội vã điều chỉnh siêu tham số hay đổi thuật toán. Cần tiến hành kiểm toán mẫu sai (Misclassification Auditing) theo quy trình 4 bước:

1. **Trích xuất ngẫu nhiên $`100`$ mẫu dự đoán sai nghiêm trọng nhất** (ví dụ: dự đoán xác suất $`0.95`$ nhưng thực tế nhãn là $`0`$).
2. **Phân loại nguyên nhân gốc rễ (Root-Cause Tagging):**
   * Lỗi nhãn dữ liệu (Data Annotation Error): Nhãn ban đầu bị con người gán sai.
   * Thiếu đặc trưng quyết định (Missing Information): Mẫu này không thể dự đoán nếu thiếu thông tin quan trọng.
   * Điểm ngoại lai dị biệt (Extreme Outlier): Mẫu dị thường hiếm gặp.
   * Lỗi biên quyết định của mô hình (Model Boundary Error): Mô hình thực sự chưa đủ năng lực phân tách.
3. **Phân tích theo lát cắt (Slicing Analysis):** Đo lường hiệu năng riêng biệt theo từng phân khúc nhân khẩu học hoặc thiết bị (ví dụ: người dùng dùng iOS vs Android, vùng nông thôn vs thành thị) để đảm bảo mô hình không bị thiên vị bất công (Fairness & Bias Audit).

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Model Decay** | Sự suy giảm hiệu năng tự nhiên của mô hình theo thời gian. | Nhắc nhở hệ thống không thể vận hành mãi mãi nếu không cập nhật. | Động lực thiết lập hệ thống giám sát MLOps. |
| **Covariate Shift** | Phân phối $`P(X)`$ thay đổi trong khi quan hệ $`P(Y \mid X)`$ giữ nguyên. | Làm mô hình phải dự đoán trên miền dữ liệu chưa quen thuộc. | Phát hiện qua PSI hoặc kiểm định thống kê KS-test. |
| **Concept Drift** | Quan hệ $`P(Y \mid X)`$ bị thay đổi do bản chất nghiệp vụ biến động. | Khiến mô hình đưa ra quyết định sai dù đầu vào trông bình thường. | Kích hoạt bắt buộc việc thu thập nhãn mới và retrain. |
| **Population Stability Index (PSI)** | Chỉ số lượng hóa mức độ dịch chuyển phân phối của một biến số. | Đưa ra con số cụ thể để đặt ngưỡng báo động tự động. | Tiêu chuẩn giám sát số 1 trong ngành tài chính và ngân hàng. |
| **Shadow Deployment** | Chạy mô hình mới song song ngầm với mô hình cũ trên traffic thật. | Kiểm tra tải và lỗi hệ thống mà không gây rủi ro cho người dùng. | Bước đệm an toàn bắt buộc trước khi chính thức phát hành. |
| **Canary Deployment** | Chuyển dần dần một phần nhỏ lưu lượng sang phiên bản mới. | Giảm thiểu tối đa diện ảnh hưởng nếu phiên bản mới có lỗi nghiêm trọng. | Chiến lược phát hành phần mềm hiện đại. |

---

## Nguồn

- [1] [Google: Monitoring ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)
- [2] [Google: Production ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems)
- [3] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [4] [scikit-learn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [5] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [6] [scikit-learn: Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [7] Sculley, D., et al. (2015). *Hidden technical debt in machine learning systems*. Advances in Neural Information Processing Systems (NeurIPS).
- [8] Gama, J., et al. (2014). *A survey on concept drift adaptation*. ACM Computing Surveys (CSUR), 46(4), 1-37.
