# k-Nearest Neighbors (kNN)

## Purpose

Dự đoán từ các mẫu train gần nhất; baseline phi tuyến dễ hiểu cho classification hoặc regression. [1]

## Input

`X` số/đã vector hóa, `y` là lớp hoặc giá trị liên tục; chọn khoảng cách (thường Euclidean), số láng giềng `k` và cách trọng số theo khoảng cách.

## Output

Classification: lớp theo đa số hoặc tỉ lệ có trọng số của `k` láng giềng; regression: trung bình/giá trị có trọng số của target lân cận. Có thể trả các láng giềng và khoảng cách để kiểm tra. [1]

## Evaluation metrics

Dùng metric tương ứng bài toán (F1/PR-AUC cho classification, MAE/RMSE cho regression), thêm latency và bộ nhớ khi suy luận. So với baseline và kiểm tra hiệu quả theo nhóm mật độ dữ liệu.

## How it works

`fit` chủ yếu lưu dữ liệu train. Khi dự đoán, tính khoảng cách đến các mẫu train, chọn `k` mẫu gần nhất, rồi gộp nhãn/giá trị. `k` nhỏ dễ nhiễu; `k` lớn làm mượt quá mức. [1][2]

**Ví dụ tính tay:** dữ liệu train một chiều gồm `(x=1,y=A)`, `(x=2,y=A)`, `(x=5,y=B)`. Với mẫu mới `x=4`, khoảng cách Euclidean lần lượt là `3,2,1`; `k=1` dự đoán `B`, còn `k=3` đa số dự đoán `A`. Với regression và nhãn `10,12,20`, `k=2` không trọng số sẽ lấy hai điểm gần nhất và dự đoán `(12+20)/2=16`.

**Fit/predict:** lưu vector train và nhãn → tính khoảng cách → chọn `k` láng giềng → gộp nhãn. Nếu dùng trọng số khoảng cách, cần định nghĩa cách xử lý khoảng cách bằng 0. [1]

## Data used

Phù hợp khi khoảng cách giữa vector mang ý nghĩa và số chiều không quá lớn; Iris là ví dụ học tập. Scale feature số bằng thống kê train để một feature có đơn vị lớn không áp đảo khoảng cách. Với văn bản vector thưa, cân nhắc cosine distance và chiều cao. [1][3]

## Design

- Tuning `k`, metric khoảng cách, weights bằng CV; lưu toàn bộ dữ liệu train nếu thuật toán cần khi inference.
- Ưu điểm: ít giả định hình dạng ranh giới, không cần tối ưu trọng số. Hạn chế: inference và lưu trữ tốn kém; số chiều cao làm khái niệm gần/xa kém hữu ích.
- Failure modes: dữ liệu thưa vùng lân cận, outlier, feature chưa scale, group leakage giữa train/test.

**Chi phí:** với tìm kiếm brute force trên `n` mẫu, `p` chiều, một truy vấn khoảng `O(np)` để tính khoảng cách, cộng chi phí chọn láng giềng. KDTree/BallTree có thể giúp trong một số dữ liệu chiều thấp, nhưng lợi ích giảm khi chiều cao; đo thời gian trên dữ liệu thật. [1]

## Nguồn

- [1] [scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [2] [Cover & Hart: Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964)
- [3] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
