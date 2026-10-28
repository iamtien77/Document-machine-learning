# Chương 1 — Nhập môn Machine Learning

> Bản ghi chú tiếng Việt dựa trên [ML_Chapter1_new.pdf](../Tài liệu học tập/ML_Chapter1_new.pdf) (23 trang), bài giảng của TS. Nguyễn Thị Khánh Tiên. Số trang dưới đây là số trang PDF. Nội dung được diễn giải lại để học và ôn tập, không phải bản chép nguyên văn slide.

## 1. Mục tiêu học tập

Sau chương này, người học cần:

1. Phân biệt Data Science, Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL) và Generative AI (GenAI).
2. Giải thích dữ liệu, thuật toán, mô hình, training và testing bằng một ví dụ cụ thể.
3. Phân biệt supervised, unsupervised và reinforcement learning theo loại tín hiệu học và đầu ra mong muốn.
4. Mô tả quy trình một dự án ML từ định nghĩa vấn đề đến giám sát sau triển khai.
5. Nhận diện nhóm công cụ Python phù hợp cho dữ liệu, trực quan hóa, ML và DL.

## 2. Bản đồ kiến thức Data Science và ML (tr. 2–8)

Các sơ đồ mở đầu đặt ML trong một hệ kỹ năng rộng hơn. Một dự án không bắt đầu bằng việc chọn mô hình; người làm cần biết đặt câu hỏi, xử lý dữ liệu, đánh giá và trình bày kết quả.

| Nhóm kiến thức | Nội dung cần học | Vai trò trong dự án |
|---|---|---|
| Toán và thống kê | Đại số tuyến tính, xác suất, thống kê, tối ưu | Biểu diễn dữ liệu, hiểu loss, độ bất định và đánh giá |
| Lập trình | Python, cấu trúc dữ liệu, thư viện tính toán | Tạo pipeline có thể chạy lại |
| Dữ liệu | Thu thập, làm sạch, khám phá, trực quan hóa | Biến dữ liệu thô thành đầu vào đáng tin cậy |
| Machine Learning | Bài toán, thuật toán, huấn luyện, cross-validation | Học quy luật từ ví dụ và dự đoán dữ liệu mới |
| Triển khai | Đóng gói mô hình, phục vụ dự đoán, giám sát | Duy trì chất lượng khi điều kiện thay đổi |

Slide về Data Science Pipeline (tr. 4) thể hiện luồng: **thu thập dữ liệu → xử lý → làm sạch → phân tích khám phá → mô hình/thuật toán → truyền đạt kết quả → ra quyết định**. Đây là vòng lặp: nếu phân tích cho thấy nhãn sai hoặc dữ liệu thiếu, phải quay lại khâu dữ liệu.

Slide lược sử (tr. 10) đặt sự phát triển của AI, ML và DL trên một dòng thời gian dài: từ ý tưởng máy có hành vi thông minh, qua các phương pháp học từ dữ liệu, đến mạng neuron và ứng dụng quy mô lớn. Điều cần rút ra là kỹ thuật mới được xây trên nhiều lớp kiến thức trước đó; mốc lịch sử không thay thế việc hiểu giả định và giới hạn của từng mô hình.

### AI, ML, DL và GenAI (tr. 5–6)

- **AI:** phạm vi rộng của hệ thống thực hiện nhiệm vụ thường cần trí tuệ, gồm cả phương pháp dựa trên luật và phương pháp học từ dữ liệu.
- **ML:** nhóm phương pháp cho phép mô hình học quy luật từ dữ liệu thay vì viết tay mọi quy tắc.
- **DL:** nhóm phương pháp ML dùng mạng neuron nhiều tầng để học biểu diễn và dự đoán.
- **GenAI:** hệ thống tạo nội dung mới như văn bản, ảnh hoặc âm thanh. Nhiều hệ GenAI hiện đại dùng DL. Sơ đồ vòng lồng nhau trên slide là hình minh họa định hướng, không phải định nghĩa hình thức cho mọi hệ thống.

Ví dụ cùng một bài toán thư rác: hệ dựa trên luật có thể chặn email chứa một từ khóa cố định; mô hình ML học từ email đã gán nhãn; mô hình DL có thể học biểu diễn văn bản phức tạp hơn. Chọn cách nào cần dựa vào dữ liệu, chi phí lỗi và yêu cầu vận hành.

### Lộ trình học (tr. 7–8)

Slide gợi ý học nền tảng toán và Python, sau đó học các loại ML, đánh giá mô hình, chủ đề nâng cao và làm dự án. Trình tự thực dụng:

1. Python, mảng/bảng dữ liệu, trực quan hóa và thống kê mô tả.
2. Định nghĩa `X` (feature), `y` (target), cách chia train/validation/test và metric.
3. Học baseline, linear/logistic regression, cây quyết định; sau đó mới mở rộng sang ensemble, SVM và DL.
4. Làm một bài toán nhỏ trọn quy trình, ghi lại nguồn dữ liệu, giả định và lỗi thường gặp.

## 3. Khái niệm cơ bản (tr. 9)

| Thuật ngữ | Ý nghĩa | Ví dụ dự đoán giá nhà |
|---|---|---|
| Data | Các quan sát dùng để học/đánh giá | Bản ghi nhà đã giao dịch |
| Feature `X` | Thông tin có sẵn lúc dự đoán | Diện tích, số phòng, vị trí |
| Target `y` | Giá trị/nhãn muốn dự đoán | Giá bán |
| Algorithm | Quy trình học tham số/quy tắc | Linear Regression |
| Model | Kết quả đã học từ dữ liệu | Các hệ số và phép tiền xử lý đã fit |
| Training | Fit mô hình trên train set | Ước lượng hệ số từ nhà đã biết giá |
| Testing | Đánh giá trên dữ liệu giữ riêng | So dự đoán với giá của nhà chưa dùng để fit/chọn mô hình |

**Phân biệt thuật toán và mô hình:** `LinearRegression` là loại thuật toán/estimator; một bộ hệ số cụ thể sau khi `fit` trên dữ liệu là mô hình đã huấn luyện. Testing phải phản ánh khả năng trên dữ liệu mới, không chỉ kiểm mức ghi nhớ dữ liệu train.

## 4. Các nhánh học máy và ứng dụng (tr. 10–15)

| Loại học | Dữ liệu/tín hiệu học | Mục tiêu | Ví dụ |
|---|---|---|---|
| Supervised learning | Cặp `(X, y)` có nhãn | Dự đoán `y` cho `X` mới | Dự đoán giá nhà, nhận diện thư rác |
| Unsupervised learning | `X` không có target cho nhiệm vụ đó | Tìm cấu trúc, nhóm, biểu diễn | Gom nhóm hành vi khách hàng |
| Reinforcement learning | Tương tác với môi trường và nhận reward | Học chính sách hành động | Điều khiển tác tử trong môi trường mô phỏng |

Ở supervised learning, **classification** dự đoán lớp rời rạc; **regression** dự đoán giá trị liên tục. Unsupervised learning không đồng nghĩa với “không có con người tham gia”: vẫn cần người thiết kế dữ liệu, mục tiêu và kiểm tra kết quả. Reinforcement learning không nhất thiết thiếu mọi tín hiệu giám sát; reward chính là tín hiệu phản hồi, chỉ khác nhãn đúng cho từng mẫu.

Sơ đồ ứng dụng (tr. 12) nêu thương mại điện tử, chăm sóc sức khỏe, tài chính, giao thông, mạng xã hội và trợ lý ảo. Cần chuyển tên lĩnh vực thành bài toán đo được: *dự đoán gì, trước thời điểm nào, ai dùng kết quả, sai loại nào tốn kém?*

Slide ưu/nhược điểm (tr. 13) nhấn mạnh tự động hóa, xử lý dữ liệu phức tạp và ứng dụng rộng; đồng thời lưu ý phụ thuộc dữ liệu, thời gian train, khó giải thích, thiên lệch và overfitting. Khả năng xử lý thời gian thực phụ thuộc mô hình và hệ thống triển khai, không phải đặc tính chung của mọi ML model.

## 5. Quy trình dự án ML (tr. 16–19)

```text
Xác định vấn đề → Thu thập dữ liệu → Chuẩn bị dữ liệu → Chọn mô hình
        → Train → Đánh giá → Triển khai → Giám sát và cập nhật
```

| Bước | Câu hỏi cần trả lời | Đầu ra nên lưu |
|---|---|---|
| Xác định vấn đề | Dự đoán cho ai, lúc nào, để quyết định gì? | Mô tả target, metric, chi phí lỗi |
| Thu thập | Dữ liệu có đại diện cho tình huống dùng thật không? | Nguồn, thời gian, quyền sử dụng |
| Chuẩn bị | Missing, trùng, nhãn sai, leakage ở đâu? | Schema, quy tắc làm sạch và split |
| Feature engineering | Feature nào có sẵn lúc dự đoán? | Danh sách feature, phép biến đổi |
| Chọn/fit mô hình | Baseline và mô hình ứng viên nào phù hợp? | Pipeline, cấu hình, seed |
| Đánh giá | Mô hình hơn baseline trên dữ liệu chưa thấy không? | Metric, lỗi theo nhóm, giới hạn |
| Triển khai/giám sát | Dữ liệu mới có đổi, mô hình có suy giảm không? | Log chất lượng, kế hoạch cập nhật |

**Quy tắc quan trọng:** chia dữ liệu trước khi học thống kê tiền xử lý. Imputer, scaler, encoder và feature selector phải `fit` trên train hoặc từng train fold; test chỉ dùng cho đánh giá cuối. Điều này làm rõ thứ tự thực hiện trong sơ đồ tổng quát của slide.

### Ví dụ giá nhà (tr. 19, 22)

1. **Input:** diện tích, số phòng, tuổi nhà, vị trí có thể biết trước khi rao bán.
2. **Target:** giá giao dịch đã biết của các nhà lịch sử.
3. **Chuẩn bị:** kiểm đơn vị diện tích, missing, mã hóa vị trí và chia dữ liệu theo mục tiêu sử dụng. Nếu dự đoán giao dịch tương lai, nên cân nhắc split theo thời gian.
4. **Baseline:** dự đoán giá trung vị của train; so với Linear Regression hoặc Random Forest.
5. **Đánh giá:** MAE/RMSE theo đơn vị tiền, kèm lỗi theo khu vực và mức giá.
6. **Triển khai:** lưu cả mô hình lẫn phép tiền xử lý, kiểm feature mới có cùng schema.

**Lưu ý về ví dụ trên slide:** `price_per_square_meter = price / area` dùng chính giá cần dự đoán. Không đưa feature này vào `X` khi dự đoán giá; đó là target leakage. Chỉ dùng biến được tính từ thông tin thực sự có trước thời điểm dự đoán.

## 6. Công cụ Python được giới thiệu (tr. 20–22)

| Nhóm | Thư viện trên slide | Công việc chính |
|---|---|---|
| Tính toán và dữ liệu bảng | NumPy, pandas, SciPy | Mảng, DataFrame, đại số/tối ưu |
| Trực quan hóa | Matplotlib, Seaborn | EDA và biểu đồ đánh giá |
| ML truyền thống | scikit-learn | Preprocessing, mô hình, CV, metric |
| DL | PyTorch, TensorFlow, Keras | Xây và train mạng neuron |
| Boosting dữ liệu bảng | XGBoost, LightGBM, CatBoost | Mô hình cây tăng cường |
| NLP và Computer Vision | Transformers, OpenCV | Mô hình ngôn ngữ, xử lý ảnh/video |

Không cần cài toàn bộ thư viện để bắt đầu. Một bài thực hành nhỏ có thể chỉ dùng Python, NumPy/pandas, Matplotlib và scikit-learn. Vai trò cụ thể của mỗi công cụ phụ thuộc bài toán và phiên bản đang dùng.

## 7. Câu hỏi tự kiểm tra

1. Thuật toán khác mô hình đã train ở điểm nào?
2. Dự đoán “khách hàng có rời dịch vụ không” là classification hay regression? Vì sao?
3. Vì sao điểm train cao không đủ để kết luận mô hình tốt?
4. Ở ví dụ giá nhà, vì sao `price_per_square_meter` tính từ giá giao dịch không được dùng làm feature?
5. Nếu dữ liệu production thay đổi sau triển khai, bước nào trong quy trình cần được kích hoạt lại?

## Nguồn trong tài liệu

- PDF gốc: [ML_Chapter1_new.pdf](../Tài liệu học tập/ML_Chapter1_new.pdf), tr. 2–22; danh mục sách tham khảo ở tr. 23.
- Nội dung về quy trình và các ví dụ được viết lại bằng tiếng Việt; các lưu ý về leakage và kiểm tra sau triển khai là phần giải thích bổ sung để áp dụng slide đúng cách.
