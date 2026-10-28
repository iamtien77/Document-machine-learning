# k-Nearest Neighbors (kNN)

## Purpose

Dự đoán từ các mẫu train gần nhất; baseline phi tuyến dễ hiểu cho classification hoặc regression. [1]

## Input

`X` số/đã vector hóa, `y` là lớp hoặc giá trị liên tục; chọn khoảng cách (thường Euclidean), số láng giềng `k` và cách trọng số theo khoảng cách.

Nếu dữ liệu có `n` mẫu train, `p` chiều, mỗi bản ghi inference phải qua đúng phép scale/vector hóa đã fit. Với biến categorical, khoảng cách Euclidean trên mã số tùy ý thường không có ý nghĩa; cần encoding/metric phù hợp với ngữ nghĩa. Chọn `k≤n_train` và kiểm cả tỷ lệ lớp trong vùng lân cận. [1]

## Output

Classification: lớp theo đa số hoặc tỉ lệ có trọng số của `k` láng giềng; regression: trung bình/giá trị có trọng số của target lân cận. Có thể trả các láng giềng và khoảng cách để kiểm tra. [1]

## Evaluation metrics

Dùng metric tương ứng bài toán (F1/PR-AUC cho classification, MAE/RMSE cho regression), thêm latency và bộ nhớ khi suy luận. So với baseline và kiểm tra hiệu quả theo nhóm mật độ dữ liệu.

## How it works

`fit` chủ yếu lưu dữ liệu train. Khi dự đoán, tính khoảng cách đến các mẫu train, chọn `k` mẫu gần nhất, rồi gộp nhãn/giá trị. `k` nhỏ dễ nhiễu; `k` lớn làm mượt quá mức. [1][2]

**Ví dụ tính tay:** dữ liệu train một chiều gồm `(x=1,y=A)`, `(x=2,y=A)`, `(x=5,y=B)`. Với mẫu mới `x=4`, khoảng cách Euclidean lần lượt là `3,2,1`; `k=1` dự đoán `B`, còn `k=3` đa số dự đoán `A`. Với regression và nhãn `10,12,20`, `k=2` không trọng số sẽ lấy hai điểm gần nhất và dự đoán `(12+20)/2=16`.

**Fit/predict:** lưu vector train và nhãn → tính khoảng cách → chọn `k` láng giềng → gộp nhãn. Nếu dùng trọng số khoảng cách, cần định nghĩa cách xử lý khoảng cách bằng 0. [1]

### Khoảng cách và cách gộp

Với `p` feature, `d₂(x,z)=√Σ_j(x_j-z_j)²` (Euclidean), `d₁(x,z)=Σ_j|x_j-z_j|` (Manhattan). Minkowski tổng quát bậc `q≥1` là `(Σ_j|x_j-z_j|^q)^(1/q)`. Metric chỉ có ích nếu giá trị “gần” phản ánh sự giống nhau liên quan target; chuẩn hóa feature là một phần của định nghĩa khoảng cách, không chỉ là bước làm đẹp dữ liệu. [1]

Classification không trọng số chọn lớp có số láng giềng lớn nhất. Nếu dùng trọng số `a_i≥0`, điểm lớp `c` có thể viết `S_c=Σ_{i∈N_k(x)} a_i·1[y_i=c]`; xác suất ước lượng là `S_c/Σ_i a_i`. Regression có `ŷ=Σ_i a_i y_i/Σ_i a_i`; khi mọi `a_i=1`, đó là trung bình. Với `a_i≈1/d_i`, điểm trùng dữ liệu train (`d_i=0`) cần quy tắc riêng để tránh chia cho 0; kiểm hành vi của implementation được dùng. Tỷ lệ láng giềng không mặc nhiên là xác suất đã calibration. [1]

**Ví dụ gộp có trọng số:** ba láng giềng của mẫu mới có nhãn `A,A,B`, khoảng cách `1,2,0.5`. Bỏ trọng số cho kết quả `A` (2 phiếu). Dùng trọng số nghịch đảo khoảng cách, điểm `A=1+0.5=1.5`, điểm `B=2`; kết quả thành `B`. Vì vậy metric, `k` và cách gộp là một phần của thiết kế mô hình.

```text
fit: lưu X_train sau preprocessing và y_train (có thể xây chỉ mục tìm kiếm)
predict(x): tính/tìm khoảng cách → chọn k mẫu nhỏ nhất → gộp theo weights
```

## Data used

Phù hợp khi khoảng cách giữa vector mang ý nghĩa và số chiều không quá lớn; Iris là ví dụ học tập. Scale feature số bằng thống kê train để một feature có đơn vị lớn không áp đảo khoảng cách. Với văn bản vector thưa, cân nhắc cosine distance và chiều cao. [1][3]

## Design

- Tuning `k`, metric khoảng cách, weights bằng CV; lưu toàn bộ dữ liệu train nếu thuật toán cần khi inference.
- Ưu điểm: ít giả định hình dạng ranh giới, không cần tối ưu trọng số. Hạn chế: inference và lưu trữ tốn kém; số chiều cao làm khái niệm gần/xa kém hữu ích.
- Failure modes: dữ liệu thưa vùng lân cận, outlier, feature chưa scale, group leakage giữa train/test.

| Hyperparameter | Cách thử | Rủi ro |
|---|---|---|
| `n_neighbors=k` | CV trên train, bắt đầu từ vài giá trị nhỏ đến vừa | `k=1` nhạy nhiễu; `k` rất lớn làm mất cấu trúc cục bộ |
| `metric`/`p` | Chọn theo ngữ nghĩa feature, so trên cùng split | Khoảng cách vô nghĩa trên feature mã hóa sai |
| `weights` | Đồng đều hoặc theo khoảng cách | Mẫu rất gần có thể chi phối dự đoán |
| Cách tìm láng giềng | Brute, KDTree, BallTree theo dữ liệu | Chi phí tìm kiếm thay đổi theo số chiều và phân phối |

Nếu có hòa phiếu ở classification, dự đoán phụ thuộc quy tắc tie-breaking của implementation; hãy kiểm trực tiếp thay vì giả định luôn chọn một lớp cụ thể. Với lớp hiếm, xem recall theo lớp và lân cận của các mẫu sai, không chỉ accuracy. [1]

**Chi phí:** với tìm kiếm brute force trên `n` mẫu, `p` chiều, một truy vấn khoảng `O(np)` để tính khoảng cách, cộng chi phí chọn láng giềng. KDTree/BallTree có thể giúp trong một số dữ liệu chiều thấp, nhưng lợi ích giảm khi chiều cao; đo thời gian trên dữ liệu thật. [1]

Bộ nhớ lưu `X_train` khoảng `O(np)` cho dữ liệu dense, cộng nhãn/chỉ mục. Nếu dataset rất lớn hoặc yêu cầu latency chặt, chi phí inference có thể là lý do chọn mô hình tham số; nếu dữ liệu ít và cấu trúc cục bộ rõ, kNN là baseline phi tuyến hữu ích.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Neighbor / láng giềng** | Mẫu train có khoảng cách nhỏ với mẫu cần dự đoán theo metric đã chọn. | Cung cấp nhãn hoặc giá trị tham chiếu cho điểm mới | Các mẫu train gần nhất được truy xuất ở inference |
| **`k`** | Số láng giềng dùng để gộp nhãn/giá trị; là hyperparameter, không phải trọng số học bằng gradient. | Quyết định số mẫu tham gia vote hoặc trung bình | Hyperparameter cân bằng nhiễu và độ mượt, chọn trên validation |
| **Distance metric** | Quy tắc đo gần/xa như Euclidean hay Manhattan; quyết định láng giềng nào được chọn. | Quyết định thứ tự gần xa giữa các mẫu | Thành phần cốt lõi của truy xuất láng giềng ở inference |
| **Distance weighting** | Cho mẫu gần hơn trọng số lớn hơn khi vote hoặc tính trung bình. | Làm láng giềng gần ảnh hưởng mạnh hơn | Quy tắc tổng hợp dự đoán sau khi đã tìm k điểm |
| **Feature scaling** | Đổi thang feature dựa trên train để đơn vị lớn không tự động chi phối khoảng cách. | Ngăn cột có đơn vị lớn lấn át khoảng cách | Transformer fit trên train rồi áp cùng phép biến đổi khi inference |
| **Curse of dimensionality** | Khi số chiều cao, khoảng cách giữa các mẫu có thể kém phân biệt, khiến “gần nhất” ít hữu ích. | Làm khái niệm gần xa kém tin cậy khi nhiều chiều | Giới hạn cần cân nhắc khi chọn KNN hoặc giảm chiều |
| **Brute-force search** | Tính khoảng cách với các mẫu train khi dự đoán, thay vì dùng chỉ mục KDTree/BallTree. | Cho kết quả chính xác theo metric nhưng tốn phép tính khi train lớn | Chiến lược tìm láng giềng ở inference |
| **Tie / hòa phiếu** | Hai lớp có cùng điểm sau gộp; kết quả tùy quy tắc xử lý của implementation. | Có thể làm nhãn dự đoán phụ thuộc cách phá hòa | Trường hợp biên cần kiểm tra nếu bài toán có nhiều lớp |

## Nguồn

- [1] [scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [2] [Cover & Hart: Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964)
- [3] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
