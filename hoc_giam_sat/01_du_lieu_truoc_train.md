# Dữ liệu và quy trình trước train

## Purpose

Biến yêu cầu thực tế thành tập dữ liệu đáng tin cậy và một phép thử công bằng. Chất lượng split và nhãn quyết định giá trị của toàn bộ kết quả phía sau.

## Input

- Định nghĩa đơn vị dự đoán, target `y`, thời điểm dự đoán và dữ liệu thực sự có sẵn ở thời điểm đó.
- Dữ liệu thô, schema, nguồn nhãn, mã định danh nhóm, timestamp, thông tin về missing/outlier.
- Metric chính, baseline và ngưỡng chấp nhận đã thống nhất trước train.

## Output

- Train/validation/test không chồng lặp; mô tả quy tắc split và tỷ lệ lớp/nhóm/thời gian.
- Pipeline tiền xử lý có thể `fit` trên train và `transform` trên validation/test/inference.
- Báo cáo khám phá dữ liệu: số mẫu, tỉ lệ nhãn, missing, trùng lặp, biến bất thường, nguy cơ leakage.

## Evaluation metrics

- Trước khi có mô hình: tỉ lệ nhãn thiếu/sai, tỉ lệ giá trị thiếu, tỉ lệ trùng, độ phủ lớp/nhóm/thời gian; đây là **kiểm tra dữ liệu**, không phải điểm mô hình.
- Baseline: majority class/stratified/random cho classification; mean/median cho regression, dùng cùng metric chính với mô hình. [1]
- Kiểm tra test có đủ ví dụ ở từng nhóm trọng yếu; không có tỷ lệ split cố định phù hợp mọi dữ liệu. [2]

## How it works

1. Đóng khung bài toán: classification hay regression, nhãn được biết sau bao lâu, có sai lệch chọn mẫu hoặc nhãn không.
2. Audit schema, đơn vị đo, phân phối, missing, trùng, nhãn hiếm và lỗi gán nhãn. Loại feature xuất hiện **sau** thời điểm dự đoán.
3. Chia dữ liệu **trước khi fit** imputer, scaler, encoder, selector hay target encoding. IID: split ngẫu nhiên; lớp hiếm: cân nhắc stratification; người dùng/bệnh nhân lặp lại: split theo group; dự đoán tương lai: split theo thời gian. [3][4][5]
4. Fit mọi thống kê tiền xử lý trên train hoặc trong từng fold CV; áp dụng nguyên trạng sang validation/test. Giữ phép biến đổi trong `Pipeline` để tránh leakage. [3]
5. Train dummy baseline và mô hình đơn giản, ghi lại seed, phiên bản dữ liệu, schema và quy tắc split.

**Ví dụ chia theo thời gian:** dự đoán khách hàng rời dịch vụ vào ngày 1/7 từ thông tin có đến 30/6. Nếu nhãn chỉ xác nhận sau 30 ngày, feature từ tháng 7 hoặc số liệu tổng hợp đến tháng 8 là leakage. Train trên các tháng cũ, validation trên tháng kế tiếp, test trên giai đoạn muộn hơn; mọi feature cần được dựng theo thời điểm dự đoán của từng hàng.

**Thứ tự an toàn cho một fold:** `split fold → fit imputer/encoder/scaler trên fold train → transform fold validation → fit model trên fold train → score fold validation`. Lặp lại toàn bộ chuỗi cho từng fold. Không được `fit_transform` trên toàn bộ `X` trước cross-validation. [3]

## Data used

Ví dụ bảng giá nhà: `X` gồm diện tích, vị trí, năm xây; `y` là giá giao dịch. Nếu dùng giá chốt sau ngày dự đoán hoặc thống kê toàn bộ tập trước split, mô hình đã nhìn thấy thông tin tương lai. Với bệnh nhân có nhiều lần khám, mọi lần khám của cùng bệnh nhân cần ở cùng phía split khi mục tiêu là bệnh nhân mới. [3][4]

## Design

- Chuẩn hóa số thường hữu ích cho kNN, SVM và mô hình tuyến tính có regularization; cây thường ít cần scale. Mã hóa biến phân loại dựa vào train, xác định cách xử lý giá trị chưa thấy khi inference.
- Resampling lớp hiếm, feature selection và tuning đều phải nằm **bên trong fold train** khi cross-validation. Không dùng validation/test để học thống kê hoặc chọn feature. [3]
- Split theo thời gian cần tuân thủ trật tự và cân nhắc khoảng cách giữa tập train/test nếu nhãn hoặc feature bị trễ. [5]
- Failure modes: leakage, nhãn không đại diện, trùng mẫu, class imbalance, dữ liệu thiếu theo cơ chế liên quan đến target.

**Checklist dữ liệu:**

| Kiểm tra | Câu hỏi cụ thể | Hành động khi lỗi |
|---|---|---|
| Target | Nhãn được tạo bởi ai, khi nào, có bất đồng không? | Audit mẫu nhãn, sửa quy tắc gán nhãn |
| Đơn vị quan sát | Một người/tài khoản có nhiều hàng không? | Group split nếu mục tiêu là người/tài khoản mới |
| Thời gian | Feature có sẵn tại thời điểm dự đoán không? | Loại feature tương lai, time split |
| Missing | Thiếu do ngẫu nhiên hay do quy trình nghiệp vụ? | Tạo cờ missing nếu có ý nghĩa, impute trong pipeline |
| Trùng lặp | Có cùng bản ghi ở train và test không? | Gộp/lọc theo định danh phù hợp trước khi đánh giá |
| Lệch lớp | Có đủ mẫu lớp hiếm ở từng tập không? | Stratification khi hợp lý; báo cáo số mẫu theo lớp |

**Lưu ý:** Không áp dụng một tỷ lệ 70/15/15 máy móc. Tập test phải đủ lớn cho kết luận cần đưa ra, còn cách chia phải mô phỏng dữ liệu sẽ gặp lúc dùng mô hình. [2]

## Nguồn

- [1] [scikit-learn: Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
- [2] [Google: Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [3] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [4] [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [5] [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
