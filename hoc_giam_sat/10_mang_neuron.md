# Mạng neuron nhiều lớp (MLP)

## Purpose

Học hàm phi tuyến bằng các tầng biến đổi có trọng số và activation. MLP là điểm vào để hiểu backpropagation; mạng sâu chuyên biệt cho ảnh/văn bản cần thiết kế khác và lượng dữ liệu phù hợp. [1][2]

## Input

`X` vector số đã scale/encode, `y` là lớp hoặc giá trị số; kiến trúc số tầng/số neuron, activation, learning rate, batch size và số epoch.

Một mạng dense có `p` feature đầu vào, tầng ẩn kích thước `h₁,h₂,...` và số đầu ra phù hợp target. Classification binary cần định nghĩa lớp dương; multiclass cần danh sách lớp và thứ tự output; regression cần đơn vị của target. Missing/categorical phải được xử lý trước theo pipeline đã fit. [1]

## Output

Trọng số các tầng; classification trả score/xác suất lớp tùy đầu ra, regression trả số liên tục; lịch sử loss và metric theo epoch nếu được ghi.

Đối với MLPClassifier, đầu ra nhiều lớp biểu diễn xác suất các lớp sau phép biến đổi đầu ra; `predict` là nhãn, `predict_proba` là xác suất ước lượng theo thứ tự lớp. MLPRegressor trả giá trị số. Xác suất classifier vẫn cần kiểm calibration trước khi dùng như xác suất rủi ro thực. [1][3]

## Evaluation metrics

Metric theo bài toán (F1/PR-AUC/log loss hoặc MAE/RMSE), chênh train/validation, calibration khi dùng xác suất, độ trễ và kích thước mô hình. [3]

## How it works

Forward pass tạo dự đoán qua các tầng; loss đo sai lệch; backpropagation tính gradient theo chain rule; optimizer cập nhật trọng số qua nhiều mini-batch/epoch. Regularization và early stopping giúp kiểm soát overfit. [1][2]

Với một neuron, `z=wᵀx+b`, `a=σ(z)`; một tầng ẩn tạo `h=φ(W₁x+b₁)` và đầu ra `y_hat=g(W₂h+b₂)`. Ví dụ `x=2`, `w=3`, `b=-1` cho `z=5`; ReLU trả `max(0,5)=5`. Backpropagation truyền gradient của loss từ đầu ra qua các tầng để cập nhật trọng số. [1][2]

**Vòng train:** khởi tạo trọng số → chia mini-batch → forward → tính loss → backward → optimizer cập nhật → đánh giá validation mỗi epoch → dừng sớm theo tiêu chí đã định. Chỉ dùng validation cho early stopping, không dùng test. Với MLPClassifier của scikit-learn, kiểm solver và cách early stopping của phiên bản đang dùng. [1]

### Chọn đầu ra và loss theo bài toán

| Bài toán | Đầu ra thường dùng | Loss huấn luyện thường dùng | Dự đoán cuối |
|---|---|---|---|
| Binary classification | Sigmoid một logit | Binary cross-entropy | Threshold trên xác suất/score |
| Multiclass một nhãn | Softmax trên các logit | Multiclass cross-entropy | Lớp có xác suất lớn nhất hoặc quy tắc chi phí |
| Regression thực | Đầu ra tuyến tính/identity | Squared error (hoặc loss khác theo estimator) | Giá trị số cùng đơn vị target |

Đây là thiết kế thường gặp, không phải mọi framework/estimator đều cho chỉnh mọi tổ hợp loss–activation. Kiểm API cụ thể trước khi tái sử dụng cấu hình. [1][2]

### Backpropagation bằng một phép tính tay

Xét một neuron tuyến tính `ŷ=wx+b`, loss `L=½(ŷ-y)²`. Với `x=2`, `y=3`, `w=1`, `b=0`, ta có `ŷ=2`, `L=0.5`. Theo chain rule: `∂L/∂w=(ŷ-y)x=-2`, `∂L/∂b=ŷ-y=-1`. Một bước gradient descent với `η=0.1` cho `w_new=1-0.1(-2)=1.2`, `b_new=0-0.1(-1)=0.1`; dự đoán mới tại `x=2` là `2.5`. Với nhiều tầng, gradient được nhân tiếp qua đạo hàm activation và ma trận trọng số; optimizer cập nhật mọi tầng. Ví dụ này chỉ minh họa cơ chế một bước, không chứng minh hội tụ. [2]

```text
khởi tạo tham số (và seed để tái lập một phần)
for epoch:
    for batch lấy từ train:
        forward → loss → backward → optimizer.step
    đo loss/metric trên train và validation
    nếu đạt điều kiện early stopping: dừng và chọn checkpoint hợp lệ
khóa preprocessing, trọng số, thứ tự lớp và threshold trước test
```

## Data used

MLP trên dữ liệu bảng: feature số đã scale, categorical đã encode; Iris là ví dụ nhỏ để học cơ chế, không chứng minh ưu thế trước tree ensemble. Với ảnh/văn bản, chọn biểu diễn và kiến trúc phù hợp loại dữ liệu. [1][4]

## Design

- Chỉ tăng độ phức tạp sau baseline tuyến tính/cây và khi validation cho thấy lợi ích. Tuning kiến trúc, regularization, learning rate bằng validation.
- Ưu điểm: biểu diễn phi tuyến linh hoạt. Hạn chế: nhạy preprocessing/hyperparameter, cần nhiều tính toán và dữ liệu hơn trong nhiều tình huống, khó giải thích trực tiếp.
- Failure modes: overfit, gradient/optimization không ổn định, seed gây biến thiên, train-serving preprocessing khác nhau.

| Tham số | Ảnh hưởng | Dấu hiệu chẩn đoán |
|---|---|---|
| Số tầng/neuron | Năng lực biểu diễn và số tham số | Train tốt–validation kém gợi ý overfit |
| `alpha`/weight decay | Phạt trọng số lớn | Quá mạnh có thể underfit |
| Learning rate | Kích thước bước cập nhật | Loss dao động/NaN nếu quá lớn; học chậm nếu quá nhỏ |
| Batch size/solver | Độ nhiễu gradient, RAM, tốc độ | So thời gian fit và đường loss |
| Early stopping/patience | Số epoch cho phép không cải thiện | Dừng quá sớm hoặc train quá lâu |

Các solver như SGD/Adam cập nhật lặp; `lbfgs` dùng cách tối ưu khác và có thể hữu ích trên tập nhỏ theo tài liệu scikit-learn. Không đánh giá một cấu hình chỉ theo loss train: so baseline, validation và độ biến thiên qua seed. [1]

**Chi phí:** số tham số tầng dense từ `d_in` sang `d_out` là `d_in×d_out+d_out`; một forward pass cần xấp xỉ số phép nhân theo kích thước các ma trận. Tăng số tầng/neurons tăng năng lực nhưng cũng tăng dữ liệu và tính toán cần thiết. Khi tập nhỏ, chạy nhiều seed để thấy biến thiên và so với baseline trên cùng split.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Neuron** | Phép tính `z=wᵀx+b` rồi áp activation để tạo đầu ra. | Biến tín hiệu vào thành tín hiệu ra qua trọng số và hàm kích hoạt | Khối tính toán cơ bản của mỗi tầng |
| **Activation** | Hàm phi tuyến như ReLU/sigmoid; nếu bỏ phi tuyến, nhiều tầng tuyến tính có thể gộp thành một phép tuyến tính. | Cho mạng biểu diễn quan hệ phi tuyến | Phép biến đổi đặt sau tổ hợp tuyến tính trong các tầng ẩn |
| **Logit** | Điểm đầu ra trước sigmoid/softmax khi phân loại. | Giữ điểm số chưa chuẩn hóa cho từng lớp | Đầu ra trước sigmoid hoặc softmax và thường đưa vào loss |
| **Forward pass** | Tính dự đoán từ input qua các tầng theo chiều tiến. | Tạo dự đoán và loss từ đầu vào hiện tại | Bước tính thuận của mỗi mini-batch khi training và của inference |
| **Backpropagation** | Tính gradient từ loss trở về các tầng trước bằng chain rule. | Truyền đạo hàm loss về các tham số | Bước tính gradient để optimizer cập nhật trọng số |
| **Gradient** | Đạo hàm chỉ hướng loss thay đổi theo tham số; optimizer dùng nó để cập nhật. | Cho biết loss đổi theo tham số ra sao | Tín hiệu tối ưu dùng trong bước cập nhật tham số |
| **Mini-batch / epoch** | Mini-batch là nhóm mẫu cho một bước cập nhật; epoch là một lượt qua tập train. | Quy định kích thước mỗi lần cập nhật và một lượt qua train | Đơn vị tổ chức vòng lặp huấn luyện |
| **Early stopping** | Dừng train theo validation để hạn chế overfit; không dùng test để quyết định epoch. | Tránh tiếp tục fit khi validation xấu đi | Quy tắc chọn checkpoint hoặc epoch kết thúc |

## Nguồn

- [1] [scikit-learn: Neural network models](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [2] [Google: Neural Networks](https://developers.google.com/machine-learning/crash-course/neural-networks)
- [3] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
