# Inference phase — giai đoạn suy luận trên dữ liệu mới

> Inference bắt đầu khi đã có artifact từ [training phase](16_training_phase.md). Bài giảng [Chương 1](<../Tài liệu học tập/ML_Chapter1_new.pdf>) mô tả bước prediction/deployment (tr. 16–22); [Chương 2](<../Tài liệu học tập/ML_Chapter2_new.pdf>) minh họa `predict` trên Iris (tr. 11–16). Tài liệu này mô tả giao diện và luồng vận hành cho dữ liệu chưa biết nhãn.

## Purpose

Nhận một bản ghi hoặc batch mới, áp dụng **đúng pipeline đã fit**, rồi trả dự đoán để hỗ trợ quyết định. Inference thông thường **không gọi `fit`, không học lại scaler/encoder và không tự cập nhật trọng số**. Việc retrain là một vòng training mới có dữ liệu, kiểm định và phiên bản riêng. [1][2]

Phải phân biệt **điểm mô hình (score)**, **xác suất đã kiểm calibration** và **quyết định cuối**. Ví dụ classifier trả $`p=0.73`$; nhãn dương chỉ được chọn sau khi áp threshold đã khóa. `0.73` không nên được diễn giải là xác suất thực 73% nếu mô hình chưa được kiểm calibration. [3][4]

### Ba Tầng Biểu diễn Đầu ra trong Suy luận (Inference Output Hierarchy)

Trong lý thuyết quyết định Bayes (Bayesian Decision Theory), quá trình suy luận không chỉ đơn thuần là gán một cái nhãn chữ. Hệ thống suy luận chuẩn mực bóc tách đầu ra thành ba tầng độc lập:

```text
                  INFERENCE OUTPUT LAYERS
  [ Raw Score / Logit ]  -->  Khoảng cách đại số f(x), chưa chuẩn hóa
          |
          v
  [ Calibrated Proba ]   -->  Xác suất hậu nghiệm thực tế P(Y=1 | X=x)
          |
          v
  [ Action / Decision ]  -->  Quyết định gán nhãn sau khi đối chiếu ngưỡng tau
```

1. **Tầng 1: Điểm số thô (Raw Score / Logits $`z = f(x)`$):**
   * Trong hồi quy tuyến tính/logistic, đây là giá trị $`w^\top x + b`$.
   * Trong SVM, đây là khoảng cách có dấu từ điểm dữ liệu tới siêu phẳng phân cách.
   * *Đặc điểm:* Giá trị số học thuần túy, có thể nằm ngoài khoảng $`[0, 1]`$, chỉ phản ánh thứ tự xếp hạng tương đối.
2. **Tầng 2: Xác suất đã hiệu chuẩn (Calibrated Probability $`\hat{p} \in [0, 1]`$):**
   * Được chuyển đổi qua hàm kích hoạt (như Sigmoid / Softmax) và đã vượt qua bài kiểm tra độ tin cậy thống kê (Reliability Diagram).
   * *Ý nghĩa:* Khi mô hình dự đoán $`\hat{p} = 0.8`$ cho $`100`$ bệnh nhân, thực tế phải có đúng xấp xỉ $`80`$ bệnh nhân mang bệnh.
3. **Tầng 3: Quyết định nhị phân / Hành động thực tế ($`\hat{y} \in \{0, 1\}`$):**
   * Áp dụng ngưỡng cắt tối ưu $`\tau^*`$ đã được tính toán từ ma trận chi phí ở giai đoạn đánh giá:

$$
\hat{y} = \begin{cases} 1 & \text{nếu } \hat{p} \ge \tau^* \\ 0 & \text{nếu } \hat{p} < \tau^* \end{cases}
$$

---

## Input

| Đầu vào | Yêu cầu |
|---|---|
| Bản ghi mới `x` hoặc batch `X_new` | Feature có sẵn **tại thời điểm dự đoán**; đúng tên, đơn vị, kiểu và miền giá trị theo schema |
| Artifact đã khóa | Fitted preprocessing + estimator + calibrator/threshold nếu có; model ID/version và thứ tự nhãn |
| Hợp đồng đầu ra | Regression hay classification; cần nhãn, score, xác suất hay nhiều output; giới hạn latency/độ tin cậy |
| Metadata vận hành | Thời điểm sự kiện, thời điểm yêu cầu, phiên bản schema và ID bản ghi nếu có để tạo `prediction_id` liên kết với nhãn tương lai |

**Không cần `y` để tạo dự đoán.** Nhãn thật thường xuất hiện sau đó, đôi khi rất muộn hoặc không có; nó được dùng để đánh giá chất lượng thực tế, không phải đầu vào bắt buộc của `predict`. [5]

### Bảng Ràng buộc Hợp đồng Đầu vào (Input Contract & Validation Rules)

Để tránh sập dịch vụ phục vụ người dùng, tầng API suy luận bắt buộc phải kiểm tra (validate) dữ liệu trước khi đẩy vào pipeline:

```text
Incoming Request -> [ Schema Validation ] -> [ Range Check ] -> Pipeline.transform()
                           |                         |
                    (Báo lỗi HTTP 422)        (Kích hoạt Fallback)
```

* **Kiểm tra Schema cứng:** Số lượng cột, tên cột chính xác, thứ tự cột khớp 100% với ma trận $`X`$ lúc train.
* **Xử lý giá trị phân loại chưa từng thấy (Unseen Categorical Levels):**
  * *Ví dụ:* Lúc train chỉ có `thành_phố` $`\in`$ `{Hà Nội, TP.HCM, Đà Nẵng}`. Lúc inference xuất hiện `Cần Thơ`.
  * *Quy tắc:* Encoder phải được cấu hình `handle_unknown='ignore'` để gán toàn bộ vector one-hot của `Cần Thơ` về vector $`0`$, hoặc ánh xạ vào nhãn đại diện `__OTHER__`. Tuyệt đối không để ứng dụng quăng ngoại lệ (Exception Crash).

---

## Output

- **Regression:** $`\hat{y}`$ là số theo đơn vị target đã định; nếu có interval/độ bất định thì phải ghi cách tính và điều kiện áp dụng.
- **Classification:** nhãn, score hoặc vector xác suất theo thứ tự lớp; threshold/phiên bản threshold dùng để biến score thành quyết định. Nếu estimator chỉ có `decision_function`, không gắn nhãn “xác suất” cho score đó.
- **Metadata:** `prediction_id` duy nhất cho mỗi dự đoán, `model_version`, `schema_version`, thời điểm sự kiện, thời điểm dự đoán và trạng thái kiểm input. `prediction_id` cho phép ghép đúng nhãn đến muộn với dự đoán gốc, kể cả khi model đã đổi phiên bản. Nếu input không hợp lệ, trả lỗi có thể xử lý hoặc chuyển sang fallback đã thiết kế; không trả một dự đoán âm thầm từ dữ liệu sai.

Ví dụ với nhận diện Iris: đầu vào là bốn phép đo cùng đơn vị như lúc train; đầu ra là một trong ba tên loài cùng, nếu cần, xác suất theo **đúng thứ tự nhãn đã lưu**. Ví dụ này minh họa kỹ thuật, không phản ánh yêu cầu của một hệ production cụ thể.

### Cấu trúc Bản tin Phản hồi Chuẩn mực (Production Inference Payload)

```json
{
  "prediction_id": "pred_a8f92c10-3b4e-4f1a-b6d3",
  "model_metadata": {
    "model_name": "credit_risk_xgb",
    "model_version": "v2.1.0",
    "pipeline_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  },
  "timestamp": "2026-10-31T15:30:00Z",
  "inference_results": {
    "predicted_class": 1,
    "decision_label": "REJECT_LOAN",
    "probabilities": {
      "0": 0.18,
      "1": 0.82
    },
    "applied_threshold": 0.65
  },
  "diagnostics": {
    "latency_ms": 14.2,
    "input_imputed": true,
    "imputed_features": ["annual_income"]
  }
}
```

---

## Evaluation metrics

**Khi chưa có nhãn thật:** không thể tính accuracy/F1/MAE của các dự đoán mới. Theo dõi tỉ lệ request lỗi, thiếu feature, sai schema, giá trị ngoài miền, phân phối feature/score, tỉ lệ các lớp dự đoán, latency p50/p95/p99 và throughput. Training-serving skew hoặc drift là **tín hiệu điều tra**, không tự chứng minh model accuracy đã giảm. [5][6]

**Khi nhãn thật đến:** nối nhãn theo `prediction_id` với đúng model version và thời điểm sự kiện, rồi tính metric đã chọn trong training: classification có precision/recall/F1/AP và calibration khi cần; regression có MAE/RMSE; kiểm theo nhóm và cửa sổ thời gian. Ghi số mẫu và độ trễ nhãn, vì nhãn đến muộn/chọn lọc có thể làm metric online bị thiên lệch. [5]

| Nhóm metric | Ví dụ | Ý nghĩa |
|---|---|---|
| Hệ thống | p95 latency, error rate, throughput | Dịch vụ có trả được dự đoán đúng hạn không? |
| Dữ liệu | Missing rate, schema violations, drift | Input còn giống điều kiện đã học không? |
| Dự đoán | Phân phối score/nhãn, tỉ lệ fallback | Có thay đổi bất thường trong output không? |
| Chất lượng khi có nhãn | Recall/F1, MAE, lỗi theo nhóm | Mô hình còn đáp ứng mục tiêu thực tế không? |

Không đặt một KPI cố định cho mọi ứng dụng; ngưỡng cảnh báo cần dựa trên baseline, chi phí sai và khả năng vận hành của bài toán.

### Tại sao Bắt buộc Giám sát Độ trễ theo Phân vị (Percentile Latency: p95, p99)?

Trong các hệ thống phân tán và ứng dụng phục vụ thời gian thực, **độ trễ trung bình (Mean Latency) là một chỉ số che giấu lỗi cực kỳ nguy hiểm**:
* Giả sử hệ thống xử lý $`1000`$ requests. Có $`990`$ requests trả về siêu nhanh trong $`10\,\text{ms}`$, nhưng có $`10`$ requests bị nghẽn mạng hoặc quá tải bộ nhớ và mất tới $`5000\,\text{ms}`$ ($`5`$ giây).
* Độ trễ trung bình:

$$
\text{Mean} \approx \frac{990 \times 10 + 10 \times 5000}{1000} = 59.9\,\text{ms}
$$

  Nhìn vào con số $`59.9\,\text{ms}`$, kỹ sư tưởng rằng hệ thống đang hoạt động trơn tru.
* Nhưng xét theo phân vị:
  * $`p50 = 10\,\text{ms}`$ (50% người dùng nhận kết quả dưới 10ms).
  * $`p99 = 5000\,\text{ms}`$ (1% người dùng quan trọng nhất phải chờ đợi tới 5 giây và rất có thể đã hủy giao dịch!).
* Do đó, SLA (Service Level Agreement) của hệ thống suy luận luôn bắt buộc phải cam kết trên **$`p95`$ và $`p99`$**.

---

## How it works

1. **Nạp artifact tin cậy** từ phiên bản được duyệt, kiểm tương thích code/dependency và schema. Model persistence trong scikit-learn có giới hạn tương thích phiên bản; artifact pickle/joblib từ nguồn không tin cậy có rủi ro thực thi mã khi load. [7]
2. **Nhận và kiểm raw input:** đủ feature bắt buộc, đúng kiểu/đơn vị, giá trị trong miền hợp lệ; xác định cách xử lý missing và categorical mới từ thiết kế training.
3. **Áp dụng fitted pipeline:** gọi `transform` của các bước đã fit; dùng `predict(X)` để lấy nhãn/giá trị, hoặc `decision_function(X)`/`predict_proba(X)` để lấy điểm/xác suất nếu estimator hỗ trợ. `Pipeline` tự áp dụng transformer trước phương thức dự đoán. Không gọi `fit_transform` trên request mới. Phương thức scikit-learn `score(X, y)` thuộc bước **đánh giá khi có nhãn**, không phải cách lấy score dự đoán. [2]
4. **Tạo quyết định:** nếu dùng ngưỡng tùy chỉnh, lấy `decision_function` hoặc `predict_proba` trước rồi áp threshold đã khóa; không áp threshold lên nhãn đã trả từ `predict`. Chỉ áp calibrator riêng nếu artifact chưa tích hợp calibration. `CalibratedClassifierCV.predict_proba` đã trả xác suất được hiệu chuẩn, nên không áp calibrator lần hai. Giữ đúng thứ tự lớp và đơn vị output; với regression, đảo biến đổi target nếu training có transform target. [3][4][8]
5. **Trả và ghi nhận:** trả `prediction_id` cùng output và model/schema version; log tối thiểu đủ để truy vết phiên bản, thời điểm sự kiện, lỗi schema, latency và nối với nhãn sau này, theo quy định dữ liệu của dự án.
6. **Theo dõi:** đo metric không cần nhãn ngay; khi nhãn đến, đo chất lượng thật. Nếu cảnh báo, kiểm schema/nguồn dữ liệu, feature engineering, model age và nhãn trước khi quyết định retrain. [5][6]

```text
New raw X
  → validate schema/time availability
  → fitted preprocessing.transform(X)
  → regression: predict(X) → giá trị
    classification: predict(X) → nhãn mặc định
                hoặc decision_function(X) / predict_proba(X)
                     → calibration riêng nếu chưa tích hợp → fixed threshold → nhãn
  → prediction + prediction_id + model/schema version
  → monitor; join ground truth by prediction_id later for quality metrics
```

**Hai kiểu vận hành:** online inference trả từng request khi phát sinh; batch inference chạy một tập bản ghi theo lịch. Cả hai phải giữ cùng hợp đồng feature và logic biến đổi, nhưng ngân sách latency/throughput khác nhau. [6]

### So sánh Chuyên sâu: Online Serving vs. Batch Inference

| Tiêu chí | Suy luận Trực tuyến (Online / Real-time Serving) | Suy luận Theo lô (Batch / Offline Inference) |
|---|---|---|
| **Phương thức kích hoạt** | Nhận yêu cầu đồng bộ qua REST API, gRPC hoặc WebSocket. | Chạy tự động theo lịch định kỳ (Cron Job, Airflow, Spark ETL). |
| **Ngân sách Độ trễ** | Cực kỳ khắt khe: từ vài mili-giây đến dưới 100ms. | Linh hoạt: tính bằng phút hoặc hàng giờ. |
| **Băng thông (Throughput)** | Thấp đến trung bình trên từng request đơn lẻ ($`1-100`$ items). | Khổng lồ: hàng triệu đến hàng tỷ bản ghi mỗi đợt chạy. |
| **Tối ưu hóa phần cứng** | Ưu tiên CPU đa luồng, tối ưu độ trễ bộ nhớ đệm (Cache locality). | Ưu tiên GPU/TPU xử lý ma trận song song quy mô lớn. |
| **Chiến lược Thất bại** | Bắt buộc phải có Fallback tức thì để trả kết quả cho người dùng. | Retry job tự động hoặc ghi nhận bản ghi lỗi vào Dead Letter Queue. |

---

## Data used

Inference dùng **dữ liệu mới chưa biết nhãn**, có thể khác phân phối train. Feature phải là thông tin đã sẵn tại thời điểm yêu cầu; không lấy field sẽ chỉ xuất hiện sau kết quả. Dữ liệu production có thể được lưu làm ứng viên cho **training phase kế tiếp** sau khi có nhãn đáng tin và đã kiểm quyền sử dụng, nhưng bản ghi mới không tự làm thay đổi model đang phục vụ. [5]

Ví dụ dự đoán giá nhà: diện tích, vị trí, số phòng có thể là input; giá giao dịch sau khi bán là nhãn tương lai. Không tính `price_per_square_meter` từ chính giá giao dịch rồi đưa vào inference, vì đó là target leakage. Các thống kê normalization vẫn là giá trị đã fit trên train, không phải tính lại từ batch mới.

---

## Design

- **Một hợp đồng train–serve:** giữ schema, đơn vị đo, thứ tự cột, encoding và phiên bản preprocessing nhất quán. Kiểm cả raw input và transformed feature để phát hiện training-serving skew. [5]
- **Thiết kế khi lỗi:** xác định rõ khi nào từ chối request, khi nào dùng fallback/baseline và khi nào cần con người kiểm tra. Không âm thầm thay missing bằng giá trị tùy ý ngoài pipeline đã đánh giá.
- **Khả năng truy vết:** mỗi dự đoán gắn `prediction_id`, thời điểm sự kiện và model/schema/threshold version; có thể rollback khi phát hiện lỗi. Không log dữ liệu nhạy cảm vượt nhu cầu kiểm tra.
- **Bảo trì:** theo dõi model age, phân phối đầu vào/đầu ra, latency và metric thật khi nhãn đến; kiểm lỗi theo nhóm. Drift chỉ mở cuộc điều tra, còn retrain cần đánh giá lại có kiểm soát. [5][6]
- **Failure modes:** feature lúc serving khác lúc train, nạp sai phiên bản artifact, đảo thứ tự lớp trong `predict_proba`, coi score SVM là xác suất, threshold cũ khi chi phí thay đổi, nhãn online đến trễ hoặc bị chọn lọc.

### 5 Nguyên nhân Gây ra Lỗi Lệch Huấn luyện - Phục vụ (Training-Serving Skew Playbook)

Training-Serving Skew là nguyên nhân hàng đầu khiến các dự án Machine Learning thất bại cay đắng khi bước ra đời thực:

1. **Khác biệt về môi trường triển khai mã (Codebase Divergence):**
   * *Triệu chứng:* Mã tiền xử lý lúc train viết bằng Python (Pandas/Scikit-learn), nhưng khi triển khai lên hệ thống backend lại được viết lại bằng Java/C++ hoặc SQL để tối ưu tốc độ.
   * *Hậu quả:* Chỉ cần một sự khác biệt nhỏ về phép làm tròn số học hoặc quy tắc bẻ chuỗi cũng làm thay đổi hoàn toàn vector đặc trưng.
   * *Giải pháp:* Dùng chuẩn đóng gói thống nhất như ONNX, Triton Inference Server hoặc xuất chung một pipeline nhị phân.
2. **Feature Lookahead (Dữ liệu tương lai vô tình xuất hiện lúc train):**
   * Lúc train có một cột dữ liệu được tính toán dựa trên toàn bộ lịch sử 30 ngày. Nhưng lúc serving trong thời gian thực, bảng cơ sở dữ liệu chưa kịp đồng bộ (bị trễ 2 tiếng), dẫn đến giá trị đưa vào model bị sai lệch.
3. **Cách điền khuyết không đồng nhất:**
   * Lúc train dùng `SimpleImputer(strategy='median')`. Lúc serving khi gặp trường hợp thiếu dữ liệu, lập trình viên backend tự ý gán `feature = 0` hoặc `-1`.
4. **Tráo đổi thứ tự cột đặc trưng (Feature Ordering Permutation):**
   * Một số mô hình (như XGBoost hoặc TensorFlow) nhận đầu vào là mảng số NumPy không có tên cột. Nếu thứ tự cột lúc gọi inference bị tráo đổi vị trí giữa `tuổi` và `thu_nhập`, mô hình vẫn chạy bình thường mà không báo lỗi, nhưng tạo ra những dự đoán hoàn toàn vô nghĩa!
5. **Lệch Phân phối do Hiệu ứng Phản hồi (Feedback Loops):**
   * Chính các dự đoán của mô hình làm thay đổi hành vi của người dùng trong tương lai, làm biến đổi phân phối dữ liệu mới thu thập được.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Inference / suy luận** | Dùng mô hình đã fit để dự đoán cho đầu vào mới, không tự học tham số. | Tạo dự đoán cho mẫu mới từ tham số đã học | Giai đoạn vận hành sau training |
| **Serving** | Hệ thống nhận yêu cầu và trả dự đoán theo hợp đồng đầu vào/đầu ra. | Đưa mô hình vào luồng xử lý request thực tế | Lớp giao tiếp, kiểm đầu vào và trả kết quả ở môi trường vận hành |
| **Schema** | Quy tắc về tên, kiểu, đơn vị, miền giá trị và sự hiện diện của feature. | Ngăn dữ liệu sai kiểu hoặc thiếu cột đi vào mô hình | Hợp đồng dữ liệu cần kiểm trước khi dự đoán |
| **Score vs probability** | Score là điểm của mô hình; probability cần nằm trong `[0,1]` và còn phải kiểm calibration nếu dùng như rủi ro thực. | Tránh hiểu nhầm điểm xếp hạng thành rủi ro định lượng | Phân biệt đầu ra mô hình trước khi đặt ngưỡng hoặc hiển thị |
| **Threshold** | Ngưỡng đổi score/xác suất thành quyết định lớp. | Chuyển đầu ra liên tục thành quyết định theo chi phí lỗi | Quy tắc ra quyết định tại inference |
| **`prediction_id`** | ID của một dự đoán, dùng nối nhãn đến muộn với đúng model version. | Cho phép nối dự đoán, phiên bản và nhãn đến muộn | Khóa theo dõi để đánh giá chất lượng vận hành |
| **Drift / skew** | Drift là thay đổi phân phối theo thời gian; training-serving skew là khác biệt giữa dữ liệu/feature lúc train và lúc phục vụ. | Phát hiện dữ liệu thực tế đổi khác dữ liệu huấn luyện | Tín hiệu giám sát sau triển khai, cần điều tra nguyên nhân |
| **Fallback / rollback** | Fallback xử lý request khi mô hình không thể dự đoán; rollback đưa hệ thống về phiên bản trước khi bản mới có lỗi. | Duy trì xử lý an toàn khi lỗi và khôi phục bản ổn định | Cơ chế vận hành khi request hoặc phiên bản mới gặp sự cố |
| **p95 / p99 Latency** | Phân vị 95% và 99% của thời gian phản hồi hệ thống. | Đo lường độ trễ thực tế ở nhóm người dùng chịu thiệt thòi nhất. | Thước đo bắt buộc trong cam kết chất lượng dịch vụ (SLA). |
| **Human-in-the-loop** | Cơ chế chuyển các trường hợp dự đoán có độ tự tin thấp sang chuyên gia con người thẩm định. | Đảm bảo an toàn tối đa cho các quyết định sinh tử (y tế, pháp lý). | Rào chắn kiểm soát rủi ro trong AI ứng dụng. |

---

## Nguồn

- [1] [PDF Chương 2: Supervised Learning](<../Tài liệu học tập/ML_Chapter2_new.pdf>)
- [2] [scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- [3] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [4] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [5] [Google: Monitoring production ML](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)
- [6] [Google: Productionization](https://developers.google.com/machine-learning/managing-ml-projects/production)
- [7] [scikit-learn: Model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [8] [scikit-learn: CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [9] Google Cloud Architecture Center. *MLOps: Continuous delivery and automation pipelines in machine learning*.
