# Decision Tree

## Purpose

Học các quy tắc chia feature thành vùng dự đoán, dùng cho classification hoặc regression. Một cây nhỏ dễ theo dõi đường quyết định và là mốc cho các mô hình ensemble. [1]

## Input

`X` số hoặc categorical đã biểu diễn phù hợp với implementation, `y` là nhãn lớp hoặc số liên tục. Cần định nghĩa cách xử lý missing và feature categorical trước train.

## Output

Cấu trúc cây gồm các điều kiện tách, dự đoán tại lá; classifier có thể trả tỉ lệ lớp tại lá làm xác suất ước lượng. [1]

## Evaluation metrics

Metric theo bài toán; so chênh train/validation để phát hiện overfit. Với xác suất từ lá, kiểm calibration và số mẫu mỗi lá nếu dùng vào quyết định. [2]

## How it works

Từ nút gốc, chọn feature/ngưỡng tách làm giảm độ hỗn tạp (Gini/entropy) cho classification hoặc giảm lỗi bình phương cho regression; lặp đệ quy đến điều kiện dừng. Pruning hoặc giới hạn độ sâu/số mẫu lá giúp kiểm soát độ phức tạp. [1]

**Ví dụ tính tay:** ở nút có 6 mẫu dương và 4 mẫu âm, Gini là `1-(0.6²+0.4²)=0.48`. Một cách tách tạo lá trái gồm 4 dương/0 âm và lá phải 2 dương/4 âm. Gini có trọng số sau tách là `(4/10)×0 + (6/10)×[1-(2/6)²-(4/6)²]≈0.267`; mức giảm impurity khoảng `0.213`. Cây so nhiều feature/ngưỡng và chọn tách tốt nhất trong các ứng viên hợp lệ. [1]

**Fit/predict:** tìm tách ở nút → chia mẫu → lặp ở các nút con đến điều kiện dừng; khi dự đoán đi từ gốc theo các điều kiện đến lá. Một lá có rất ít mẫu có thể cho xác suất rất nhiễu dù dự đoán class trên train đúng.

## Data used

Dữ liệu bảng có tương tác/quan hệ phi tuyến; UCI Adult là ví dụ để thực hành sau encode/impute. Cây không cần standardization chỉ để tính ngưỡng tách, nhưng vẫn cần phòng leakage và xử lý missing. [1][3]

## Design

- Tuning `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha` trên validation/CV. Cây nông là baseline dễ diễn giải.
- Ưu điểm: quy tắc rõ, học tương tác tự nhiên. Hạn chế: cây sâu dễ overfit, kết quả có thể thay đổi mạnh khi dữ liệu thay đổi nhẹ.
- Failure modes: lá quá ít mẫu, feature có nhiều mức làm tách thiếu ổn định, diễn giải feature importance theo impurity như quan hệ nhân quả.

**Chi phí:** cây sâu tối đa có thể có nhiều nút và dễ ghi nhớ train; một dự đoán đi qua số nút bằng độ sâu đường đi. Khi lớp hiếm, kiểm số mẫu theo lớp ở từng lá và metric của lớp hiếm. Cây đơn thường có variance cao; Random Forest giảm variance bằng gộp nhiều cây. [1]

## Nguồn

- [1] [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [2] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [3] [UCI: Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
