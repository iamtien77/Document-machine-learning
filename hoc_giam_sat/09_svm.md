# Support Vector Machine (SVM/SVR)

## Purpose

Tìm ranh giới phân loại có margin lớn (SVC/LinearSVC) hoặc hàm dự đoán số nằm trong vùng sai số cho phép (SVR). Kernel cho phép ranh giới phi tuyến. [1][2]

### Lý thuyết Học Thống kê Vapnik & Nguyên lý Tối thiểu hóa Rủi ro Cấu trúc (SRM)

Support Vector Machine (SVM) được phát triển bởi Vladimir Vapnik và Corinna Cortes (1995) tại phòng thí nghiệm Bell Labs. Đây được coi là một trong những thành tựu toán học rực rỡ nhất của lý thuyết học máy cổ điển.

Thay vì tiếp cận theo nguyên lý Tối thiểu hóa Rủi ro Thực nghiệm thông thường (Empirical Risk Minimization - ERM, vốn rất dễ dẫn đến Overfitting), Vapnik đã đề xuất **Nguyên lý Tối thiểu hóa Rủi ro Cấu trúc (Structural Risk Minimization - SRM)** dựa trên **Chiều VC (Vapnik-Chervonenkis Dimension)**:
* Chiều VC đo lường dung lượng hoặc độ linh hoạt của một không gian hàm. Vapnik đã chứng minh rằng: Sai số tổng quát hóa trên dữ liệu mới $`\mathcal{R}(f)`$ được chặn trên bởi sai số huấn luyện thực nghiệm $`\hat{\mathcal{R}}_n(f)`$ cộng với một số hạng tỷ lệ thuận với Chiều VC:

  ```math
  \mathcal{R}(f) \le \hat{\mathcal{R}}_n(f) + \mathcal{O}\left( \sqrt{\frac{h \ln(n/h) - \ln(\eta)}{n}} \right)
  ```

  Trong đó $`h`$ là Chiều VC, $`n`$ là số mẫu, $`1-\eta`$ là độ tin cậy thống kê.
* **Định lý then chốt của Vapnik:** Trong số tất cả các siêu phẳng phân cách có thể phân chia đúng tập dữ liệu, **siêu phẳng có độ rộng lề (Margin) lớn nhất sẽ có Chiều VC nhỏ nhất**! 
* Do đó, việc tìm kiếm Siêu phẳng Phân cách Tối ưu (Maximal Margin Hyperplane) chính là phương pháp toán học trực tiếp nhất để thu nhỏ chặn trên của sai số tổng quát hóa, đảm bảo mô hình có khả năng khái quát hóa vượt trội ngay cả trong không gian số chiều cực lớn.

---

## Input

`X` vector số, `y` là lớp hoặc số; chọn kernel, `C`, và `gamma` nếu dùng RBF. Feature cần scale theo thống kê train để khoảng cách/inner product có ý nghĩa. [1]

Với `n` mẫu và `p` feature: $`X \in \mathbb{R}^{n\times p}`$; $`y \in \{-1,+1\}`$ cho ví dụ SVM nhị phân hoặc $`y \in \mathbb{R}^n`$ cho SVR. Dữ liệu phân loại nhiều lớp cần chiến lược đa lớp của estimator. `epsilon` chỉ thuộc SVR, không phải biên phân loại của SVC. [1][5]

---

## Output

Nhãn hoặc decision score cho classification; giá trị số cho regression; support vectors với SVC/SVR kernel. Score SVM không mặc nhiên là xác suất đã calibration. [1]

`SVC.decision_function(X)` trả điểm phục vụ xếp hạng/ra quyết định; `SVC.predict_proba(X)` chỉ có khi bật hỗ trợ xác suất và sử dụng thêm bước ước lượng xác suất theo implementation. `SVR.predict(X)` trả $`\hat{y}`$ cùng đơn vị của target sau khi hoàn nguyên mọi biến đổi target. [1][5]

---

## Evaluation metrics

SVC: F1, ROC-AUC/PR-AUC từ score, confusion matrix; SVR: MAE/RMSE/R². Thêm thời gian train/inference; kiểm calibration riêng nếu cần xác suất. [3]

---

## How it works

SVM tối ưu margin kèm phạt lỗi trên mẫu vi phạm. `C` điều khiển trade-off giữa margin và lỗi train; kernel RBF đo độ giống phi tuyến, `gamma` điều khiển tầm ảnh hưởng của mẫu. SVR tối ưu sai số với epsilon-insensitive loss. [1][2]

Với binary linear SVM và nhãn $`y_{i}\in \{-1,+1\}`$, một objective phổ biến là $`\tfrac{1}{2}\lVert w\rVert^{2} + C\sum_i \max (0,1-y_{i}(w^{\top}x_{i}+b))`$. Hinge loss bằng 0 khi mẫu nằm đúng phía và ngoài margin. Dự đoán nhãn theo dấu của $`w^{\top}x+b`$; điểm có độ lớn lớn hơn nằm xa ranh giới hơn, nhưng không tự động là xác suất. [1]

**Ví dụ:** nếu một điểm có $`y=+1`$ và score `0.2`, hinge loss là $`\max(0,1-0.2)=0.8`$; nếu score `1.5` thì loss bằng 0. Support vectors là những điểm ảnh hưởng đến ranh giới trong nghiệm SVM. [1][2]

### Bài toán Tối ưu Lồi & Phương pháp Nhân tử Lagrange (Primal vs. Dual Formulation)

Xét bài toán SVM phân loại nhị phân mềm (Soft-margin SVM):
Khoảng cách hình học từ một điểm tới siêu phẳng $`w^\top x + b = 0`$ là $`\frac{|w^\top x + b|}{\|w\|_2}`$. Khoảng cách lề giữa hai lớp là $`\gamma = \frac{2}{\|w\|_2}`$.
Do đó, tối đa hóa lề $`\frac{2}{\|w\|_2}`$ tương đương với tối thiểu hóa $`\frac{1}{2}\|w\|_2^2`$.

1. **Bài toán Gốc (Primal Optimization Problem):**

   ```math
   \min_{w, b, \xi} \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \xi_i \quad \text{sao cho} \quad y_i (w^\top x_i + b) \ge 1 - \xi_i, \;\; \xi_i \ge 0
   ```

   Trong đó $`\xi_i`$ là các biến bù lỏng (Slack Variables) cho phép một số mẫu vi phạm lề.
2. **Hàm Lagrange và Điều kiện KKT (Karush-Kuhn-Tucker):**
   Thiết lập hàm Lagrange với các nhân tử $`\alpha_i \ge 0`$ và $`\mu_i \ge 0`$:

   ```math
   \mathcal{L}(w, b, \xi, \alpha, \mu) = \frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i \left[ y_i (w^\top x_i + b) - 1 + \xi_i \right] - \sum_{i=1}^n \mu_i \xi_i
   ```

   Lấy đạo hàm riêng theo các biến gốc $`w, b, \xi`$ và cho bằng $`0`$:
   * $`\frac{\partial \mathcal{L}}{\partial w} = 0 \implies w = \sum_{i=1}^n \alpha_i y_i x_i`$
   * $`\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0`$
   * $`\frac{\partial \mathcal{L}}{\partial \xi_i} = 0 \implies C - \alpha_i - \mu_i = 0 \implies 0 \le \alpha_i \le C`$
3. **Bài toán Đối ngẫu Lagrange (Dual Problem):**
   Thế các biểu thức trên ngược lại vào hàm Lagrange, ta thu được bài toán đối ngẫu chỉ còn phụ thuộc vào $`\alpha`$:

   ```math
   \max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (x_i^\top x_j) \quad \text{sao cho} \quad 0 \le \alpha_i \le C, \;\; \sum_{i=1}^n \alpha_i y_i = 0
   ```

```text
                  Ý NGHĨA KHOA HỌC THEN CHỐT CỦA BÀI TOÁN ĐỐI NGẪU
  1. Tính thưa (Sparsity) & Support Vectors:
     Theo điều kiện bù KKT: alpha_i * [y_i(w^T x_i + b) - 1 + xi_i] = 0.
     - Với các điểm nằm ngoài lề an toàn: alpha_i = 0 (Hoàn toàn không tham gia vào nghiệm w).
     - Chỉ những điểm nằm TRÊN LỀ hoặc VI PHẠM LỀ mới có alpha_i > 0!
     -> ĐÂY CHÍNH LÀ CÁC SUPPORT VECTORS (Vector Hỗ trợ). Ranh giới SVM được quyết định
        duy nhất bởi các điểm này, loại bỏ 90-99% dữ liệu thừa thãi.
  2. Sự xuất hiện của Tích vô hướng (Inner Product):
     Dữ liệu x chỉ xuất hiện duy nhất dưới dạng tích vô hướng x_i^T x_j!
```

### Mánh khóe Hạt nhân (The Kernel Trick) & Định lý Mercer

Vì bài toán đối ngẫu chỉ phụ thuộc vào tích vô hướng giữa các điểm dữ liệu $`x_i^\top x_j`$, ta có thể thực hiện một bước nhảy vọt phi thường:
* Ánh xạ dữ liệu từ không gian gốc $`x \in \mathbb{R}^p`$ lên một Không gian Đặc trưng Hilbert $`\mathcal{H}`$ có số chiều cao hơn (hoặc vô hạn chiều) thông qua hàm phi tuyến $`\phi(x)`$.
* Thay vì phải tính toán tọa độ cực kỳ phức tạp $`\phi(x)`$, ta sử dụng một **Hàm nhân (Kernel Function)**:

  ```math
  K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle
  ```

* **Định lý Mercer (1909):** Bất kỳ hàm nhân đối xứng và nửa xác định dương (Positive Semi-Definite) nào cũng tương ứng với một tích vô hướng trong một không gian Hilbert nào đó.
* **Kernel RBF (Radial Basis Function / Gaussian Kernel):**

  ```math
  K(x, z) = \exp(-\gamma \|x - z\|^2)
  ```

  Khai triển chuỗi Taylor của hàm số mũ chứng minh rằng: **Kernel RBF tương ứng với một không gian đặc trưng có số chiều VÔ HẠN ($`\infty`$-dimensions)**! Nhờ đó, SVM có thể dễ dàng phân tách bất kỳ tập dữ liệu phức tạp nào mà chi phí tính toán chỉ dừng lại ở phép tính khoảng cách Euclid giữa hai điểm.

### SVR: vùng sai số `epsilon`

Với hàm tuyến tính $`f(x)=w^{\top}x+b`$, một cách viết objective của ε-SVR là $`\tfrac{1}{2}\lVert w\rVert^{2} + C\sum _{i} \max (0, \lvert y_{i}-f(x_{i})\rvert-\epsilon )`$. Lỗi có độ lớn không vượt $`\epsilon`$ không bị phạt bởi phần ε-insensitive; lỗi vượt ra ngoài bị phạt theo khoảng vượt. Ví dụ $`y=10`$, $`\hat{y}=9.8`$, $`\epsilon =0.3`$ cho loss dữ liệu bằng `0`; nếu $`\hat{y}=9.4`$ thì loss dữ liệu bằng `0.6-0.3=0.3` trước khi nhân `C`. `C` càng lớn càng ưu tiên khớp train, trong khi $`\epsilon`$ lớn tạo vùng miễn phạt rộng hơn và thường ít support vectors hơn. [1][5]

```text
       y ^                          + epsilon (Biên trên của ống)
         |               .---'     --- Hàm dự đoán f(x)
         |             .'           - epsilon (Biên dưới của ống)
         |           .'      o  (Điểm vi phạm ngoài ống -> Có slack xi_i > 0)
         |     o   .'       
         |        /      (Điểm nằm trong ống -> Sai số = 0, alpha_i = 0)
         +-------+------------------------------> x
               Vùng ống sai số Epsilon-insensitive Tube
```

**Quy trình fit/predict:** (1) split đúng group/time, fit scaler trên train fold; (2) chọn `LinearSVC`/`LinearSVR` làm mốc hoặc chọn kernel khi có lý do; (3) tune `C`, `epsilon` (SVR), `gamma` (RBF) bằng CV; (4) fit trên train, lưu scaler cùng model; (5) transform mẫu mới rồi gọi `decision_function`/`predict` cho classification hoặc `predict` cho SVR. Với kernel RBF, $`K(x,z)=\exp (-\gamma \lVert x-z\rVert^{2})`$; kernel tính độ giống trong không gian đặc trưng mà không cần tạo tường minh toàn bộ feature phi tuyến. [1][5]

---

## Data used

Vector số đã scale; Iris hoặc các tập classification kích thước vừa dùng để minh họa. Với dữ liệu rất lớn hoặc văn bản sparse nhiều chiều, mô hình tuyến tính thường là lựa chọn thực dụng hơn kernel SVC. [1][4]

---

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
| **Hyperplane** | Ranh giới tuyến tính $`w^{\top}x+b=0`$ trong không gian feature. | Chia không gian feature thành hai phía | Ranh giới quyết định của SVM tuyến tính |
| **Margin** | Khoảng cách an toàn quanh ranh giới; SVM cân bằng margin với lỗi train. | Tăng khoảng cách với các điểm quan trọng để cải thiện khả năng tổng quát | Mục tiêu hình học của bộ phân loại SVM |
| **Support vector** | Mẫu train quyết định hoặc ảnh hưởng đến nghiệm ranh giới/hàm SVR. | Quyết định vị trí ranh giới hoặc hàm hồi quy | Các mẫu train có ảnh hưởng trực tiếp tới nghiệm SVM |
| **Hinge loss** | $`\max(0,1-yf(x))`$, phạt mẫu ở sai phía hoặc trong margin phân loại. | Tạo tín hiệu phạt cho điểm sai lớp hoặc nằm trong margin | Thành phần loss của SVM phân loại tuyến tính |
| **Kernel** | Hàm đo độ giống $`K(x,z)`$ để học ranh giới phi tuyến mà không tạo tường minh mọi feature mới. | Cho phép ranh giới phi tuyến thông qua độ giống nhau | Thành phần của kernel SVM khi dữ liệu không tách tuyến tính |
| **`C`** | Mức phạt vi phạm trong objective; tương tác với kernel và tham số khác. | Cân bằng phạt lỗi train với biên và độ phức tạp | Hyperparameter của SVM, chọn qua validation |
| **`gamma`** | Tham số độ cục bộ của kernel RBF ($`\gamma = 1/(2\sigma^2)`$). | Điều chỉnh bán kính ảnh hưởng của từng điểm hỗ trợ | Hyperparameter kernel, thường tune cùng C |
| **`epsilon`** | Bán kính vùng sai số không bị phạt của SVR; không phải margin phân loại. | Bỏ qua lỗi nhỏ trong ống epsilon | Hyperparameter của SVR, quyết định độ rộng vùng không phạt |
| **Kernel Trick** | Kỹ thuật tính tích vô hướng trong không gian chiều cao bằng hàm nhân trực tiếp. | Tránh chi phí tính toán tọa độ phi tuyến tường minh. | Đột phá toán học làm nên tên tuổi của SVM. |
| **Structural Risk Minimization** | Nguyên lý cân bằng giữa sai số thực nghiệm và độ phức tạp Chiều VC. | Đảm bảo chặn trên sai số tổng quát hóa chặt chẽ nhất. | Nền tảng triết lý của Vapnik cho SVM. |

---

## Nguồn

- [1] [scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [2] [Cortes & Vapnik: Support-vector networks](https://doi.org/10.1007/BF00994018)
- [3] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [UCI: Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
- [5] [scikit-learn: SVR API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [6] Vapnik, V. N. (1998). *Statistical Learning Theory*. John Wiley & Sons.
- [7] Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.
