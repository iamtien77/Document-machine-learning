# Sau train: phân tích lỗi, triển khai và giám sát

## Purpose

Chuyển mô hình đã đánh giá thành quyết định có thể dùng và kiểm soát suy giảm chất lượng khi dữ liệu thực tế thay đổi. [1]

## Input

Pipeline và phiên bản mô hình đã khóa, kết quả test, threshold/calibrator chọn từ các tập giữ riêng hoặc dự đoán out-of-fold hợp lệ, schema inference, yêu cầu latency/chi phí và cách nhận nhãn sau triển khai.

## Output

Artifact suy luận (`preprocessing + model + threshold/calibration`), báo cáo lỗi và giới hạn, kế hoạch monitor, trigger kiểm tra lại/retrain, rollback và phiên bản có thể truy vết.

## Evaluation metrics

- **Offline:** metric test chính; calibration/log loss nếu dùng xác suất; sai số theo nhóm, giai đoạn và điều kiện dữ liệu.
- **Online:** tỉ lệ dữ liệu thiếu/sai schema, tỉ lệ nhãn/dự đoán, drift của feature, latency, lỗi hệ thống, training-serving skew; khi nhãn đến muộn thì đo lại metric thực tế. Drift là tín hiệu điều tra, không tự chứng minh chất lượng đã giảm. [1][2]

## How it works

1. Phân tích ví dụ sai, confusion matrix/residual và các lát cắt; xác định lỗi từ dữ liệu, nhãn, feature hay mô hình.
2. Nếu cần xác suất đáng tin cậy, fit calibrator trên dữ liệu tách khỏi dữ liệu fit classifier hoặc từ dự đoán out-of-fold theo CV đúng cách. Chọn threshold theo chi phí/metric trên tập dành riêng cho bước này hoặc dự đoán out-of-fold phù hợp; không dùng test. Nếu vừa calibration vừa tune threshold, tách dữ liệu cho hai bước hoặc dùng cross-fitting để tránh điều chỉnh quá mức trên cùng mẫu. Nếu đã refit classifier trên train+validation, **không dùng lại chính validation đó** như một holdout độc lập để calibration. [3][4]
3. Đóng gói pipeline, kiểm tra schema và cùng một phép biến đổi cho train và inference; version dữ liệu, code, model và threshold. [5]
4. Triển khai có kiểm soát; theo dõi dữ liệu vào, đầu ra, latency, chất lượng khi nhãn thật có sẵn. So sánh với baseline hoặc mô hình đang chạy.
5. Khi cảnh báo, điều tra nguồn thay đổi; thu thập nhãn mới, đánh giá lại trên dữ liệu gần đây, rồi quyết định retrain và kiểm thử phiên bản mới.

**Ví dụ chọn threshold:** trên tập threshold riêng, với mỗi ngưỡng ứng viên `t`, dự đoán dương khi `p≥t`, tính `TP/FP/FN/TN` và chi phí `c_FN×FN+c_FP×FP`. Chọn `t` có chi phí thấp nhất trong các ngưỡng thỏa ràng buộc công suất xử lý; khóa `t` rồi chỉ báo cáo test. Nếu xác suất chưa calibration, không diễn giải `p=0.8` thành “80% chắc chắn” trước khi kiểm tra calibration. [3][4]

**Runbook khi có cảnh báo:** xác nhận dữ liệu đầu vào/schema → xác định nhóm/thời gian bị ảnh hưởng → kiểm có thay đổi pipeline hay nguồn nhãn → đo chất lượng trên mẫu đã có nhãn → so với baseline và phiên bản trước → chỉ retrain khi dữ liệu mới/nhãn mới giải thích được vấn đề → kiểm test mới và rollback plan. Các ngưỡng cảnh báo phải lấy từ baseline vận hành và chi phí thực tế, không tự đặt một con số chung cho mọi hệ thống. [1]

## Data used

Tập dành cho calibration và tập dành cho threshold không dùng để fit classifier; nếu thực hiện cả hai bước, nên tách hai tập hoặc dùng dự đoán out-of-fold/cross-fitting hợp lệ. Test cuối dùng để báo cáo. Production data và nhãn đến muộn dùng để đánh giá drift/chất lượng sau triển khai. Không giả định rằng dữ liệu production có nhãn tức thời. [1][3][4]

## Design

- Theo dõi metric theo nhóm và thời gian cùng với số mẫu; tránh diễn giải biến động trên nhóm rất nhỏ.
- Permutation importance trên dữ liệu held-out giúp kiểm tra mức phụ thuộc dự đoán vào feature, nhưng feature tương quan có thể làm cách đọc khó hơn. [6]
- Failure modes: train-serving skew, nhãn đến trễ, thay đổi định nghĩa target, calibration hỏng khi phân phối thay đổi, retrain tự động khuếch đại lỗi nhãn.
- Điều kiện đưa vào production nên gồm cải thiện so baseline, sai số nhóm chấp nhận được, latency phù hợp và khả năng rollback.

**Theo dõi định lượng:** ghi số mẫu mỗi cửa sổ thời gian, tỉ lệ missing từng feature, tỉ lệ giá trị ngoài miền hợp lệ, phân phối score, tỉ lệ dự đoán dương, p50/p95 latency, và metric thật khi nhãn đến. Chỉ báo thay đổi feature/score là tín hiệu kiểm tra; chất lượng mô hình cần đo bằng nhãn đáng tin cậy. Giữ khả năng nối prediction với nhãn về sau bằng ID không lộ dữ liệu nhạy cảm.

## Nguồn

- [1] [Google: Monitoring ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring)
- [2] [Google: Production ML systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems)
- [3] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [4] [scikit-learn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [5] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [6] [scikit-learn: Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
