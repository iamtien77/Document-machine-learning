# Support Vector Machine (SVM/SVR)

## Purpose

Tìm ranh giới phân loại có margin lớn (SVC/LinearSVC) hoặc hàm dự đoán số nằm trong vùng sai số cho phép (SVR). Kernel cho phép ranh giới phi tuyến. [1][2]

## Input

`X` vector số, `y` là lớp hoặc số; chọn kernel, `C`, và `gamma` nếu dùng RBF. Feature cần scale theo thống kê train để khoảng cách/inner product có ý nghĩa. [1]

Với `n` mẫu và `p` feature: `X ∈ R^(n×p)`; `y ∈ {-1,+1}` cho ví dụ SVM nhị phân hoặc `y ∈ R^n` cho SVR. Dữ liệu phân loại nhiều lớp cần chiến lược đa lớp của estimator. `epsilon` chỉ thuộc SVR, không phải biên phân loại của SVC. [1][5]

## Output

Nhãn hoặc decision score cho classification; giá trị số cho regression; support vectors với SVC/SVR kernel. Score SVM không mặc nhiên là xác suất đã calibration. [1]

`SVC.decision_function(X)` trả điểm phục vụ xếp hạng/ra quyết định; `SVC.predict_proba(X)` chỉ có khi bật hỗ trợ xác suất và sử dụng thêm bước ước lượng xác suất theo implementation. `SVR.predict(X)` trả `ŷ` cùng đơn vị của target sau khi hoàn nguyên mọi biến đổi target. [1][5]

## Evaluation metrics

SVC: F1, ROC-AUC/PR-AUC từ score, confusion matrix; SVR: MAE/RMSE/R². Thêm thời gian train/inference; kiểm calibration riêng nếu cần xác suất. [3]

## How it works

SVM tối ưu margin kèm phạt lỗi trên mẫu vi phạm. `C` điều khiển trade-off giữa margin và lỗi train; kernel RBF đo độ giống phi tuyến, `gamma` điều khiển tầm ảnh hưởng của mẫu. SVR tối ưu sai số với epsilon-insensitive loss. [1][2]

Với binary linear SVM và nhãn `yᵢ∈{-1,+1}`, một objective phổ biến là `½||w||² + CΣ max(0,1-yᵢ(wᵀxᵢ+b))`. Hinge loss bằng 0 khi mẫu nằm đúng phía và ngoài margin. Dự đoán nhãn theo dấu của `wᵀx+b`; điểm có độ lớn lớn hơn nằm xa ranh giới hơn, nhưng không tự động là xác suất. [1]

**Ví dụ:** nếu một điểm có `y=+1` và score `0.2`, hinge loss là `max(0,1-0.2)=0.8`; nếu score `1.5` thì loss bằng 0. Support vectors là những điểm ảnh hưởng đến ranh giới trong nghiệm SVM. [1][2]

### SVR: vùng sai số `epsilon`

Với hàm tuyến tính `f(x)=wᵀx+b`, một cách viết objective của ε-SVR là `½||w||² + CΣᵢ max(0, |yᵢ-f(xᵢ)|-ε)`. Lỗi có độ lớn không vượt `ε` không bị phạt bởi phần ε-insensitive; lỗi vượt ra ngoài bị phạt theo khoảng vượt. Ví dụ `y=10`, `ŷ=9.8`, `ε=0.3` cho loss dữ liệu bằng `0`; nếu `ŷ=9.4` thì loss dữ liệu bằng `0.6-0.3=0.3` trước khi nhân `C`. `C` càng lớn càng ưu tiên khớp train, trong khi `ε` lớn tạo vùng miễn phạt rộng hơn và thường ít support vectors hơn. [1][5]

**Quy trình fit/predict:** (1) split đúng group/time, fit scaler trên train fold; (2) chọn `LinearSVC`/`LinearSVR` làm mốc hoặc chọn kernel khi có lý do; (3) tune `C`, `epsilon` (SVR), `gamma` (RBF) bằng CV; (4) fit trên train, lưu scaler cùng model; (5) transform mẫu mới rồi gọi `decision_function`/`predict` cho classification hoặc `predict` cho SVR. Với kernel RBF, `K(x,z)=exp(-γ||x-z||²)`; kernel tính độ giống trong không gian đặc trưng mà không cần tạo tường minh toàn bộ feature phi tuyến. [1][5]

## Data used

Vector số đã scale; Iris hoặc các tập classification kích thước vừa dùng để minh họa. Với dữ liệu rất lớn hoặc văn bản sparse nhiều chiều, mô hình tuyến tính thường là lựa chọn thực dụng hơn kernel SVC. [1][4]

## Design

- Thử LinearSVC trước; dùng RBF SVC khi ranh giới tuyến tính chưa đủ, tuning `C`/`gamma` bằng CV.
- Ưu điểm: mạnh trong không gian chiều cao và linh hoạt nhờ kernel. Hạn chế: kernel SVM có thể tốn bộ nhớ/thời gian trên tập lớn, khó giải thích trực tiếp.
- Failure modes: chưa scale, tuning trên test, `gamma` quá lớn overfit, score bị hiểu nhầm là xác suất.

| Tham số | Tăng giá trị thường làm gì? | Kiểm chứng |
|---|---|---|
| `C` | Phạt vi phạm mạnh hơn, ranh giới/hàm có thể bám train hơn | So train–validation gap và metric chính |
| `gamma` của RBF | Ảnh hưởng mỗi mẫu cục bộ hơn, dễ fit chi tiết/nhiễu hơn | Xem validation curve; scale feature trước |
| `epsilon` của SVR | Nới vùng sai số không bị phạt | Đo MAE/RMSE và số support vectors |

Các chiều hướng trên không bảo đảm metric luôn tăng hoặc giảm: chúng tương tác với nhau, với kernel và phân phối dữ liệu. Dùng cùng split và cùng metric khi so cấu hình.

**Chi phí:** linear SVM thường hợp với dữ liệu nhiều mẫu/nhiều chiều; kernel SVC có chi phí train tăng nhanh theo số mẫu và lưu support vectors để dự đoán. Dùng thử trên tập con trước khi mở rộng, và đo chi phí thực tế thay vì giả định một Big-O cố định cho mọi solver. [1]

`SVR` dựa trên libsvm có thời gian fit hơn bậc hai theo số mẫu trong tài liệu scikit-learn; với tập lớn nên xem `LinearSVR`/`SGDRegressor` hoặc kernel approximation. Khi cần giải thích hệ số và latency thấp, mô hình tuyến tính thường là mốc phù hợp; khi cần ranh giới cong trên tập vừa, mới thử RBF và so bằng CV. [5]

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Hyperplane** | Ranh giới tuyến tính `wᵀx+b=0` trong không gian feature. | Chia không gian feature thành hai phía | Ranh giới quyết định của SVM tuyến tính |
| **Margin** | Khoảng cách an toàn quanh ranh giới; SVM cân bằng margin với lỗi train. | Tăng khoảng cách với các điểm quan trọng để cải thiện khả năng tổng quát | Mục tiêu hình học của bộ phân loại SVM |
| **Support vector** | Mẫu train quyết định hoặc ảnh hưởng đến nghiệm ranh giới/hàm SVR. | Quyết định vị trí ranh giới hoặc hàm hồi quy | Các mẫu train có ảnh hưởng trực tiếp tới nghiệm SVM |
| **Hinge loss** | `max(0,1-yf(x))`, phạt mẫu ở sai phía hoặc trong margin phân loại. | Tạo tín hiệu phạt cho điểm sai lớp hoặc nằm trong margin | Thành phần loss của SVM phân loại tuyến tính |
| **Kernel** | Hàm đo độ giống `K(x,z)` để học ranh giới phi tuyến mà không tạo tường minh mọi feature mới. | Cho phép ranh giới phi tuyến thông qua độ giống nhau | Thành phần của kernel SVM khi dữ liệu không tách tuyến tính |
| **`C`** | Mức phạt vi phạm trong objective; tương tác với kernel và tham số khác. | Cân bằng phạt lỗi train với biên và độ phức tạp | Hyperparameter của SVM, chọn qua validation |
| **`gamma`** | Tham số độ cục bộ của kernel RBF. | Điều chỉnh tầm ảnh hưởng của từng điểm với RBF | Hyperparameter kernel, thường tune cùng C |
| **`epsilon`** | Bán kính vùng sai số không bị phạt của SVR; không phải margin phân loại. | Bỏ qua lỗi nhỏ trong ống epsilon | Hyperparameter của SVR, quyết định độ rộng vùng không phạt |

## Nguồn

- [1] [scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [2] [Cortes & Vapnik: Support-vector networks](https://doi.org/10.1007/BF00994018)
- [3] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
- [5] [scikit-learn: SVR API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
