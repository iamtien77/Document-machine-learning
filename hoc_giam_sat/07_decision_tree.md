# Decision Tree

## Purpose

Học các quy tắc chia feature thành vùng dự đoán, dùng cho classification hoặc regression. Một cây nhỏ dễ theo dõi đường quyết định và là mốc cho các mô hình ensemble. [1]

## Input

`X` số hoặc categorical đã biểu diễn phù hợp với implementation, `y` là nhãn lớp hoặc số liên tục. Cần định nghĩa cách xử lý missing và feature categorical trước train.

Trong scikit-learn, cách tách thông thường xét một điều kiện dạng $`x_j \le t`$; biến categorical cần mã hóa phù hợp, còn hỗ trợ missing phụ thuộc estimator/criterion và phiên bản cụ thể. Split phải diễn ra trước mọi bước mã hóa hoặc điền giá trị có học thống kê. [1]

## Output

Cấu trúc cây gồm các điều kiện tách, dự đoán tại lá; classifier có thể trả tỉ lệ lớp tại lá làm xác suất ước lượng. [1]

Với `DecisionTreeRegressor(criterion="squared_error")`, dự đoán ở lá là trung bình `y` của các mẫu train tại lá; với `absolute_error`, giá trị lá là trung vị. Vì vậy đầu ra regression thường là hàm bậc thang theo feature và khó extrapolate ngoài vùng train. [4]

## Evaluation metrics

Metric theo bài toán; so chênh train/validation để phát hiện overfit. Với xác suất từ lá, kiểm calibration và số mẫu mỗi lá nếu dùng vào quyết định. [2]

## How it works

Từ nút gốc, chọn feature/ngưỡng tách làm giảm độ hỗn tạp (Gini/entropy) cho classification hoặc giảm lỗi bình phương cho regression; lặp đệ quy đến điều kiện dừng. Pruning hoặc giới hạn độ sâu/số mẫu lá giúp kiểm soát độ phức tạp. [1]

**Ví dụ tính tay:** ở nút có 6 mẫu dương và 4 mẫu âm, Gini là $`1-(0.6^{2}+0.4^{2})=0.48`$. Một cách tách tạo lá trái gồm 4 dương/0 âm và lá phải 2 dương/4 âm. Gini có trọng số sau tách là $`(4/10)\times 0 + (6/10)\times [1-(2/6)^{2}-(4/6)^{2}]\approx 0.267`$; mức giảm impurity khoảng `0.213`. Cây so nhiều feature/ngưỡng và chọn tách tốt nhất trong các ứng viên hợp lệ. [1]

**Fit/predict:** tìm tách ở nút → chia mẫu → lặp ở các nút con đến điều kiện dừng; khi dự đoán đi từ gốc theo các điều kiện đến lá. Một lá có rất ít mẫu có thể cho xác suất rất nhiễu dù dự đoán class trên train đúng.

### Tiêu chí tách và cây hồi quy

Với lớp `k` có tần suất $`p_k`$ trong nút, $`Gini=1-\sum _k p_k^{2}`$ và $`Entropy=-\sum _k p_k \log _{2}(p_k)`$ (quy ước $`0\log 0=0`$). Một tách có mức cải thiện $`I(\mathrm{parent})-\frac{n_L}{n}I(\mathrm{left})-\frac{n_R}{n}I(\mathrm{right})`$. Thuật toán chọn feature/ngưỡng cho mức giảm impurity cao nhất trong các ứng viên, dưới các ràng buộc số mẫu lá và độ sâu. [1]

Với regression, thay impurity phân loại bằng sai số của target trong nút. `squared_error` giảm tổng bình phương sai lệch quanh trung bình; `absolute_error` giảm tổng trị tuyệt đối quanh trung vị; `poisson` dành cho target không âm phù hợp với bài toán count/rate theo điều kiện của estimator. [4]

**Ví dụ fit regression:** bốn mẫu có một feature lần lượt $`x=[1,2,3,4]`$, target $`y=[2,4,10,12]`$. Nút gốc dự đoán trung bình `7`, tổng bình phương sai số là `68`. Tách tại $`x\le 2.5`$ tạo lá trái $`y=[2,4]`$ (dự đoán `3`, SSE=`2`) và lá phải $`y=[10,12]`$ (dự đoán `11`, SSE=`2`); tổng SSE sau tách là `4`. Khi mẫu mới có $`x=3.2`$, cây đi sang lá phải và dự đoán `11`. Đây là ví dụ tính tay, chưa phải kết quả thử nghiệm trên dataset thực.

```text
fit(node):
    nếu đạt điều kiện dừng: lưu phân phối lớp hoặc thống kê target ở lá
    duyệt feature/ngưỡng hợp lệ và tính impurity sau tách
    chọn tách giảm impurity nhiều nhất; gọi fit cho hai nhánh
predict(x): đi từ gốc theo điều kiện đến lá; trả nhãn/xác suất/giá trị lá
```

## Data used

Dữ liệu bảng có tương tác/quan hệ phi tuyến; UCI Adult là ví dụ để thực hành sau encode/impute. Cây không cần standardization chỉ để tính ngưỡng tách, nhưng vẫn cần phòng leakage và xử lý missing. [1][3]

## Design

- Tuning `max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha` trên validation/CV. Cây nông là baseline dễ diễn giải.
- Ưu điểm: quy tắc rõ, học tương tác tự nhiên. Hạn chế: cây sâu dễ overfit, kết quả có thể thay đổi mạnh khi dữ liệu thay đổi nhẹ.
- Failure modes: lá quá ít mẫu, feature có nhiều mức làm tách thiếu ổn định, diễn giải feature importance theo impurity như quan hệ nhân quả.

| Hyperparameter | Vai trò | Dấu hiệu cần kiểm |
|---|---|---|
| `max_depth` | Giới hạn số tầng | Cây sâu: train tốt nhưng validation giảm |
| `min_samples_leaf` | Mức dữ liệu tối thiểu tại lá | Lá ít mẫu: dự đoán/xác suất không ổn định |
| `min_samples_split` | Điều kiện cho phép tiếp tục tách | Quá lớn: underfit; quá nhỏ: dễ tạo cây phức tạp |
| `ccp_alpha` | Pruning theo cost-complexity | Chọn bằng CV, không dùng test |

**Pre-pruning** dừng trước bằng giới hạn độ sâu/lá; **cost-complexity pruning** cắt bớt cây đã học. Chọn mức phức tạp trên validation/CV. Cây một mình có thể dễ giải thích nếu nông, nhưng một cây lớn với hàng trăm nút không còn dễ đọc; khi cần độ ổn định hơn, so với forest. [1]

**Chi phí:** cây sâu tối đa có thể có nhiều nút và dễ ghi nhớ train; một dự đoán đi qua số nút bằng độ sâu đường đi. Khi lớp hiếm, kiểm số mẫu theo lớp ở từng lá và metric của lớp hiếm. Cây đơn thường có variance cao; Random Forest giảm variance bằng gộp nhiều cây. [1]

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Node / nút** | Tập mẫu đang được xét tại một vị trí trong cây. | Tập hợp các mẫu đang chờ xét tại một điểm của cây | Đơn vị mà thuật toán đánh giá phép tách trong training |
| **Split / phép tách** | Điều kiện trên feature chia mẫu ở nút thành các nhánh con. | Chia mẫu thành các nhóm theo điều kiện feature | Quy tắc được học khi fit và áp theo đường đi ở inference |
| **Impurity / độ hỗn tạp** | Mức trộn lớp hoặc sai số target trong nút; cây tìm cách giảm qua các phép tách. | Cho biết nút còn lẫn lớp hoặc biến thiên target đến mức nào | Tiêu chí chọn split tốt hơn trong training |
| **Gini / entropy** | Hai tiêu chí đo hỗn tạp cho classification; thấp khi một lớp chiếm ưu thế. | Lượng hóa độ trộn lớp theo hai công thức khác nhau | Các lựa chọn criterion khi fit cây phân loại |
| **Leaf / lá** | Nút không tách tiếp; chứa phân phối lớp hoặc thống kê target để dự đoán. | Lưu giá trị hoặc tỉ lệ lớp cho dự đoán cuối | Điểm kết thúc đường đi của mẫu mới trong inference |
| **Stopping rule** | Điều kiện dừng như độ sâu tối đa hoặc số mẫu lá tối thiểu. | Giới hạn cây tiếp tục phân nhánh | Hyperparameter kiểm soát độ phức tạp và overfit |
| **Pruning** | Cắt bớt nhánh để giảm độ phức tạp và nguy cơ overfit. | Bỏ nhánh ít giá trị để cây gọn hơn | Bước kiểm soát độ phức tạp sau hoặc trong quá trình tạo cây |
| **SSE/MSE** | Tổng/trung bình bình phương sai số dùng để đánh giá tách nút trong hồi quy bậc hai. | Đánh giá mức sai số số thực của nút | Tiêu chí chọn split cho cây hồi quy squared-error |

## Nguồn

- [1] [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [2] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [3] [UCI: Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [4] [scikit-learn: DecisionTreeRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
