# Support Vector Machine (SVM/SVR)

## Purpose

Tìm ranh giới phân loại có margin lớn (SVC/LinearSVC) hoặc hàm dự đoán số nằm trong vùng sai số cho phép (SVR). Kernel cho phép ranh giới phi tuyến. [1][2]

## Input

`X` vector số, `y` là lớp hoặc số; chọn kernel, `C`, và `gamma` nếu dùng RBF. Feature cần scale theo thống kê train để khoảng cách/inner product có ý nghĩa. [1]

## Output

Nhãn hoặc decision score cho classification; giá trị số cho regression; support vectors với SVC/SVR kernel. Score SVM không mặc nhiên là xác suất đã calibration. [1]

## Evaluation metrics

SVC: F1, ROC-AUC/PR-AUC từ score, confusion matrix; SVR: MAE/RMSE/R². Thêm thời gian train/inference; kiểm calibration riêng nếu cần xác suất. [3]

## How it works

SVM tối ưu margin kèm phạt lỗi trên mẫu vi phạm. `C` điều khiển trade-off giữa margin và lỗi train; kernel RBF đo độ giống phi tuyến, `gamma` điều khiển tầm ảnh hưởng của mẫu. SVR tối ưu sai số với epsilon-insensitive loss. [1][2]

Với binary linear SVM và nhãn `yᵢ∈{-1,+1}`, một objective phổ biến là `½||w||² + CΣ max(0,1-yᵢ(wᵀxᵢ+b))`. Hinge loss bằng 0 khi mẫu nằm đúng phía và ngoài margin. Dự đoán nhãn theo dấu của `wᵀx+b`; điểm có độ lớn lớn hơn nằm xa ranh giới hơn, nhưng không tự động là xác suất. [1]

**Ví dụ:** nếu một điểm có `y=+1` và score `0.2`, hinge loss là `max(0,1-0.2)=0.8`; nếu score `1.5` thì loss bằng 0. Support vectors là những điểm ảnh hưởng đến ranh giới trong nghiệm SVM. [1][2]

## Data used

Vector số đã scale; Iris hoặc các tập classification kích thước vừa dùng để minh họa. Với dữ liệu rất lớn hoặc văn bản sparse nhiều chiều, mô hình tuyến tính thường là lựa chọn thực dụng hơn kernel SVC. [1][4]

## Design

- Thử LinearSVC trước; dùng RBF SVC khi ranh giới tuyến tính chưa đủ, tuning `C`/`gamma` bằng CV.
- Ưu điểm: mạnh trong không gian chiều cao và linh hoạt nhờ kernel. Hạn chế: kernel SVM có thể tốn bộ nhớ/thời gian trên tập lớn, khó giải thích trực tiếp.
- Failure modes: chưa scale, tuning trên test, `gamma` quá lớn overfit, score bị hiểu nhầm là xác suất.

**Chi phí:** linear SVM thường hợp với dữ liệu nhiều mẫu/nhiều chiều; kernel SVC có chi phí train tăng nhanh theo số mẫu và lưu support vectors để dự đoán. Dùng thử trên tập con trước khi mở rộng, và đo chi phí thực tế thay vì giả định một Big-O cố định cho mọi solver. [1]

## Nguồn

- [1] [scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [2] [Cortes & Vapnik: Support-vector networks](https://doi.org/10.1007/BF00994018)
- [3] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
