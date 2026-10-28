# Mạng neuron nhiều lớp (MLP)

## Purpose

Học hàm phi tuyến bằng các tầng biến đổi có trọng số và activation. MLP là điểm vào để hiểu backpropagation; mạng sâu chuyên biệt cho ảnh/văn bản cần thiết kế khác và lượng dữ liệu phù hợp. [1][2]

## Input

`X` vector số đã scale/encode, `y` là lớp hoặc giá trị số; kiến trúc số tầng/số neuron, activation, learning rate, batch size và số epoch.

## Output

Trọng số các tầng; classification trả score/xác suất lớp tùy đầu ra, regression trả số liên tục; lịch sử loss và metric theo epoch nếu được ghi.

## Evaluation metrics

Metric theo bài toán (F1/PR-AUC/log loss hoặc MAE/RMSE), chênh train/validation, calibration khi dùng xác suất, độ trễ và kích thước mô hình. [3]

## How it works

Forward pass tạo dự đoán qua các tầng; loss đo sai lệch; backpropagation tính gradient theo chain rule; optimizer cập nhật trọng số qua nhiều mini-batch/epoch. Regularization và early stopping giúp kiểm soát overfit. [1][2]

Với một neuron, `z=wᵀx+b`, `a=σ(z)`; một tầng ẩn tạo `h=φ(W₁x+b₁)` và đầu ra `y_hat=g(W₂h+b₂)`. Ví dụ `x=2`, `w=3`, `b=-1` cho `z=5`; ReLU trả `max(0,5)=5`. Backpropagation truyền gradient của loss từ đầu ra qua các tầng để cập nhật trọng số. [1][2]

**Vòng train:** khởi tạo trọng số → chia mini-batch → forward → tính loss → backward → optimizer cập nhật → đánh giá validation mỗi epoch → dừng sớm theo tiêu chí đã định. Chỉ dùng validation cho early stopping, không dùng test. Với MLPClassifier của scikit-learn, kiểm solver và cách early stopping của phiên bản đang dùng. [1]

## Data used

MLP trên dữ liệu bảng: feature số đã scale, categorical đã encode; Iris là ví dụ nhỏ để học cơ chế, không chứng minh ưu thế trước tree ensemble. Với ảnh/văn bản, chọn biểu diễn và kiến trúc phù hợp loại dữ liệu. [1][4]

## Design

- Chỉ tăng độ phức tạp sau baseline tuyến tính/cây và khi validation cho thấy lợi ích. Tuning kiến trúc, regularization, learning rate bằng validation.
- Ưu điểm: biểu diễn phi tuyến linh hoạt. Hạn chế: nhạy preprocessing/hyperparameter, cần nhiều tính toán và dữ liệu hơn trong nhiều tình huống, khó giải thích trực tiếp.
- Failure modes: overfit, gradient/optimization không ổn định, seed gây biến thiên, train-serving preprocessing khác nhau.

**Chi phí:** số tham số tầng dense từ `d_in` sang `d_out` là `d_in×d_out+d_out`; một forward pass cần xấp xỉ số phép nhân theo kích thước các ma trận. Tăng số tầng/neurons tăng năng lực nhưng cũng tăng dữ liệu và tính toán cần thiết. Khi tập nhỏ, chạy nhiều seed để thấy biến thiên và so với baseline trên cùng split.

## Nguồn

- [1] [scikit-learn: Neural network models](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [2] [Google: Neural Networks](https://developers.google.com/machine-learning/crash-course/neural-networks)
- [3] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
