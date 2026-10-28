# Inference phase — giai đoạn suy luận trên dữ liệu mới

> Inference bắt đầu khi đã có artifact từ [training phase](16_training_phase.md). Bài giảng [Chương 1](<../Tài liệu học tập/ML_Chapter1_new.pdf>) mô tả bước prediction/deployment (tr. 16–22); [Chương 2](<../Tài liệu học tập/ML_Chapter2_new.pdf>) minh họa `predict` trên Iris (tr. 11–16). Tài liệu này mô tả giao diện và luồng vận hành cho dữ liệu chưa biết nhãn.

## Purpose

Nhận một bản ghi hoặc batch mới, áp dụng **đúng pipeline đã fit**, rồi trả dự đoán để hỗ trợ quyết định. Inference thông thường **không gọi `fit`, không học lại scaler/encoder và không tự cập nhật trọng số**. Việc retrain là một vòng training mới có dữ liệu, kiểm định và phiên bản riêng. [1][2]

Phải phân biệt **điểm mô hình (score)**, **xác suất đã kiểm calibration** và **quyết định cuối**. Ví dụ classifier trả `p=0.73`; nhãn dương chỉ được chọn sau khi áp threshold đã khóa. `0.73` không nên được diễn giải là xác suất thực 73% nếu mô hình chưa được kiểm calibration. [3][4]

## Input

| Đầu vào | Yêu cầu |
|---|---|
| Bản ghi mới `x` hoặc batch `X_new` | Feature có sẵn **tại thời điểm dự đoán**; đúng tên, đơn vị, kiểu và miền giá trị theo schema |
| Artifact đã khóa | Fitted preprocessing + estimator + calibrator/threshold nếu có; model ID/version và thứ tự nhãn |
| Hợp đồng đầu ra | Regression hay classification; cần nhãn, score, xác suất hay nhiều output; giới hạn latency/độ tin cậy |
| Metadata vận hành | Thời điểm sự kiện, thời điểm yêu cầu, phiên bản schema và ID bản ghi nếu có để tạo `prediction_id` liên kết với nhãn tương lai |

**Không cần `y` để tạo dự đoán.** Nhãn thật thường xuất hiện sau đó, đôi khi rất muộn hoặc không có; nó được dùng để đánh giá chất lượng thực tế, không phải đầu vào bắt buộc của `predict`. [5]

## Output

- **Regression:** `ŷ` là số theo đơn vị target đã định; nếu có interval/độ bất định thì phải ghi cách tính và điều kiện áp dụng.
- **Classification:** nhãn, score hoặc vector xác suất theo thứ tự lớp; threshold/phiên bản threshold dùng để biến score thành quyết định. Nếu estimator chỉ có `decision_function`, không gắn nhãn “xác suất” cho score đó.
- **Metadata:** `prediction_id` duy nhất cho mỗi dự đoán, `model_version`, `schema_version`, thời điểm sự kiện, thời điểm dự đoán và trạng thái kiểm input. `prediction_id` cho phép ghép đúng nhãn đến muộn với dự đoán gốc, kể cả khi model đã đổi phiên bản. Nếu input không hợp lệ, trả lỗi có thể xử lý hoặc chuyển sang fallback đã thiết kế; không trả một dự đoán âm thầm từ dữ liệu sai.

Ví dụ với nhận diện Iris: đầu vào là bốn phép đo cùng đơn vị như lúc train; đầu ra là một trong ba tên loài cùng, nếu cần, xác suất theo **đúng thứ tự nhãn đã lưu**. Ví dụ này minh họa kỹ thuật, không phản ánh yêu cầu của một hệ production cụ thể.

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

## Data used

Inference dùng **dữ liệu mới chưa biết nhãn**, có thể khác phân phối train. Feature phải là thông tin đã sẵn tại thời điểm yêu cầu; không lấy field sẽ chỉ xuất hiện sau kết quả. Dữ liệu production có thể được lưu làm ứng viên cho **training phase kế tiếp** sau khi có nhãn đáng tin và đã kiểm quyền sử dụng, nhưng bản ghi mới không tự làm thay đổi model đang phục vụ. [5]

Ví dụ dự đoán giá nhà: diện tích, vị trí, số phòng có thể là input; giá giao dịch sau khi bán là nhãn tương lai. Không tính `price_per_square_meter` từ chính giá giao dịch rồi đưa vào inference, vì đó là target leakage. Các thống kê normalization vẫn là giá trị đã fit trên train, không phải tính lại từ batch mới.

## Design

- **Một hợp đồng train–serve:** giữ schema, đơn vị đo, thứ tự cột, encoding và phiên bản preprocessing nhất quán. Kiểm cả raw input và transformed feature để phát hiện training-serving skew. [5]
- **Thiết kế khi lỗi:** xác định rõ khi nào từ chối request, khi nào dùng fallback/baseline và khi nào cần con người kiểm tra. Không âm thầm thay missing bằng giá trị tùy ý ngoài pipeline đã đánh giá.
- **Khả năng truy vết:** mỗi dự đoán gắn `prediction_id`, thời điểm sự kiện và model/schema/threshold version; có thể rollback khi phát hiện lỗi. Không log dữ liệu nhạy cảm vượt nhu cầu kiểm tra.
- **Bảo trì:** theo dõi model age, phân phối đầu vào/đầu ra, latency và metric thật khi nhãn đến; kiểm lỗi theo nhóm. Drift chỉ mở cuộc điều tra, còn retrain cần đánh giá lại có kiểm soát. [5][6]
- **Failure modes:** feature lúc serving khác lúc train, nạp sai phiên bản artifact, đảo thứ tự lớp trong `predict_proba`, coi score SVM là xác suất, threshold cũ khi chi phí thay đổi, nhãn online đến trễ hoặc bị chọn lọc.

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

### Nguồn

- [1] [PDF Chương 2: Supervised Learning](<../Tài liệu học tập/ML_Chapter2_new.pdf>)
- [2] [scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- [3] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [4] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [5] [Google: Monitoring production ML](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)
- [6] [Google: Productionization](https://developers.google.com/machine-learning/managing-ml-projects/production)
- [7] [scikit-learn: Model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [8] [scikit-learn: CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
