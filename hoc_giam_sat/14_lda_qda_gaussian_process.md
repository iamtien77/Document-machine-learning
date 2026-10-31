# Mở rộng: LDA, QDA và Gaussian Process

## Purpose

Ba mô hình này mở rộng bản đồ học giám sát. **LDA/QDA** là classifier xác suất dựa trên phân phối feature theo lớp; **Gaussian Process Regression (GPR)** là regression phi tham số cung cấp dự đoán và độ bất định theo giả định kernel. Chúng hữu ích để hiểu cách giả định xác suất ảnh hưởng đến ranh giới và độ tin cậy. [1][2]

### Bối cảnh Học thuật: Từ Ranh giới Tuyến tính đến Không gian Hàm Vô hạn Chiều

1. **Phân tích Biệt thức Tuyến tính (LDA - Fisher, 1936):**
   * Được Ronald Fisher đề xuất ban đầu như một phương pháp giảm chiều có giám sát. Fisher đặt câu hỏi: "Tìm một hướng chiếu vector $w$ sao cho sau khi chiếu các điểm dữ liệu lên $w$, khoảng cách giữa các tâm lớp là xa nhất, đồng thời độ phân tán nội bộ từng lớp là nhỏ nhất?"
   * Sau đó, góc nhìn xác suất Bayes chứng minh rằng: Nếu dữ liệu trong mỗi lớp tuân theo phân phối Gaussian đa biến với cùng một ma trận hiệp phương sai $\Sigma$, ranh giới quyết định tối ưu theo Bayes chính là một siêu phẳng tuyến tính trùng khớp với phép chiếu của Fisher.
2. **Phân tích Biệt thức Bậc hai (QDA):**
   * Nới lỏng giả định cứng nhắc của LDA bằng cách cho phép mỗi lớp có ma trận hiệp phương sai riêng $\Sigma_k$. Khi đó ranh giới quyết định trở thành các mặt cong bậc hai (Hyper-paraboloid, Hyper-ellipsoid).
3. **Quá trình Gaussian (Gaussian Process - Rasmussen & Williams, 2006):**
   * Đột phá tư duy từ không gian tham số sang **không gian hàm (Function Space)**. Thay vì giả định $y = f_w(x)$ với số lượng tham số $w$ cố định, GPR coi giá trị của hàm số tại vô số điểm đầu vào là một tập hợp các biến ngẫu nhiên có phân phối đồng thời Gaussian (Định lý Mở rộng Kolmogorov).

---

## Input

- LDA/QDA: `X` số, `y` nhãn lớp; giả định phân phối $X\mid y=k$ gần Gaussian đa biến. LDA giả định các lớp dùng chung covariance $\Sigma$, QDA ước lượng $\Sigma_{k}$ riêng cho từng lớp. [1]
- GPR: `X` số, `y` liên tục, kernel và tham số nhiễu; số mẫu vừa phải do chi phí ma trận kernel. [2]

Ký hiệu `n` mẫu và `p` feature. LDA/QDA cần đủ mẫu mỗi lớp để ước lượng mean và covariance đáng tin; QDA ước lượng một ma trận $p\times p$ cho **mỗi** lớp nên đòi hỏi dữ liệu nhiều hơn LDA. GPR tạo ma trận kernel $n\times n$, vì thế số mẫu ảnh hưởng mạnh đến chi phí fit. [1][2]

---

## Output

- LDA/QDA trả nhãn, xác suất theo lớp; LDA còn có thể tạo biểu diễn giảm chiều có giám sát. Ranh giới LDA tuyến tính, QDA thường bậc hai. [1]
- GPR trả giá trị trung bình dự đoán và có thể trả độ lệch chuẩn dưới mô hình đã chọn. Độ bất định này phụ thuộc kernel, giả định nhiễu và dữ liệu; không là bảo đảm tuyệt đối về sai số thực tế. [2]

---

## Evaluation metrics

LDA/QDA: confusion matrix, F1/recall theo lớp, log loss/calibration nếu dùng xác suất. GPR: MAE/RMSE trên test, residual theo vùng feature; nếu dùng interval, kiểm coverage thực nghiệm trên dữ liệu chưa thấy. [3]

---

## How it works

**LDA/QDA:** ước lượng prior $P(y=k)$, mean $\mu _{k}$ và covariance. Theo Bayes, chọn lớp có log-posterior lớn nhất. Bỏ hằng số chung, score QDA là $\log P(y=k)-\frac{1}{2}\log\lvert\Sigma_k\rvert-\frac{1}{2}(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)$. Khi mọi $\Sigma_{k}$ giống nhau, các hạng bậc hai chung triệt tiêu khi so lớp, tạo ranh giới tuyến tính của LDA. [1]

**Ví dụ 1D:** hai lớp có mean `0` và `2`, variance bằng nhau, prior bằng nhau. Với điểm $x=1.5$, khoảng cách tới mean lớp 2 nhỏ hơn nên LDA chọn lớp 2. Nếu variance/prior khác, ranh giới đổi; không thể chỉ nhìn khoảng cách Euclidean.

**GPR:** đặt prior Gaussian lên các giá trị hàm; kernel $k(x,x')$ mô tả mức tương quan giữa hai đầu vào. Từ các cặp train, tính phân phối hậu nghiệm ở điểm mới để lấy mean và variance. Tham số kernel thường chọn qua tối ưu marginal likelihood, nhưng vẫn cần đánh giá trên dữ liệu chưa thấy. [2]

### LDA so với QDA: fit và dự đoán

Ở mỗi lớp `k`, fit prior $\pi _k=n_k/n$ và vector mean $\mu _k$. LDA gộp thông tin các lớp để ước lượng **một** covariance chung $\Sigma$; QDA ước lượng covariance riêng $\Sigma_k$. Khi dự đoán, tính log-posterior từng lớp và chọn lớp có điểm lớn nhất. Nếu prior lệch so với dữ liệu vận hành, phân phối xác suất và ranh giới có thể lệch dù các mean/covariance đã được fit đúng trên train. [1]

| | LDA | QDA |
|---|---|---|
| Ranh giới | Tuyến tính | Bậc hai trong trường hợp tổng quát |
| Covariance | Chung cho các lớp | Riêng cho từng lớp |
| Dữ liệu cần | Ít tham số hơn | Nhiều mẫu mỗi lớp hơn để ước lượng ổn định |
| Failure mode | Underfit nếu covariance khác nhau rõ | Covariance suy biến/overfit khi lớp ít mẫu |

**Ví dụ quyết định:** nếu hai lớp 1D có prior bằng nhau, mean `0` và `2`, variance cùng bằng `1`, ranh giới LDA ở $x=1$. Điểm $x=1.5$ thuộc lớp có mean `2`; khi variance của lớp thứ hai lớn hơn, QDA không còn ranh giới đơn giản tại trung điểm. Đây là ví dụ giả định, không phải điểm số thực nghiệm. [1]

#### Chứng minh Toán học: Bản chất Khoảng cách Mahalanobis & Triệt tiêu Hạng Bậc hai

Hàm mật độ xác suất của phân phối chuẩn nhiều chiều của lớp $k$:

$$P(X = x \mid Y = k) = \frac{1}{(2\pi)^{p/2} \lvert \Sigma_k \rvert^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) \right)$$

Đại lượng $D_M^2(x, \mu_k) = (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k)$ chính là **Khoảng cách Mahalanobis bình phương**, đo khoảng cách từ điểm $x$ tới tâm $\mu_k$ có tính đến hình dạng elip phân tán của ma trận hiệp phương sai $\Sigma_k$.

Lấy logarit tự nhiên xác suất hậu nghiệm $\ln P(Y = k \mid X = x)$:

$$\delta_k(x) = -\frac{1}{2} \ln \lvert \Sigma_k \rvert - \frac{1}{2} (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) + \ln P(Y=k)$$

Đây chính là phương trình bậc hai của **QDA**.

**Tại sao LDA lại có ranh giới tuyến tính?**
Khi $\Sigma_k = \Sigma$ với mọi lớp $k$:
Khai triển số hạng Mahalanobis:

$$(x - \mu_k)^\top \Sigma^{-1} (x - \mu_k) = x^\top \Sigma^{-1} x - 2 x^\top \Sigma^{-1} \mu_k + \mu_k^\top \Sigma^{-1} \mu_k$$

Khi so sánh ranh giới giữa hai lớp bất kỳ $k$ và $l$ bằng cách giải phương trình $\delta_k(x) = \delta_l(x)$:
Số hạng bậc hai $x^\top \Sigma^{-1} x$ xuất hiện giống hệt nhau ở cả hai vế nên **bị triệt tiêu hoàn toàn**!
Biểu thức còn lại chỉ chứa các số hạng bậc nhất theo $x$:

$$x^\top \Sigma^{-1} (\mu_k - \mu_l) - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \frac{1}{2} \mu_l^\top \Sigma^{-1} \mu_l + \ln \frac{\pi_k}{\pi_l} = 0$$

Đây chính là phương trình một siêu phẳng tuyến tính trong không gian $\mathbb{R}^p$.

### GPR: posterior mean và variance

Với prior mean bằng 0 và nhiễu Gaussian có phương sai $\sigma _n^{2}$, đặt $K = [k(x_i,x_j)] + \sigma _n^{2}I$, $k_*=[k(x_i,x_*)]$. Dự đoán giá trị hàm ẩn tại điểm mới có $\mu _* = k_*^{\top}K^{-1}y$ và $v_* = k(x_*,x_*) - k_*^{\top}K^{-1}k_*$. Đây là công thức cho **latent function**; phương sai của quan sát mới còn cộng thành phần nhiễu nếu mô hình hóa observation noise. Trong tính toán số, giải hệ tuyến tính bằng phân rã ma trận thay vì lập nghịch đảo tường minh. [2]

**Trực giác:** gần các điểm train theo kernel, $k_*$ lớn nên mean chịu ảnh hưởng mạnh từ nhãn train và variance thường nhỏ hơn prior. Xa vùng dữ liệu, mean quay về prior mean và variance có thể tăng. Mức tăng phụ thuộc kernel/length-scale và noise; một khoảng $\mu_*\pm 1.96\sqrt{v_*}$ chỉ có cách hiểu xác suất dưới giả định posterior phù hợp, không tự bảo đảm coverage thực tế. [2]

**Fit/predict:** chọn kernel (ví dụ RBF hoặc Matérn) và mô hình nhiễu → fit tham số kernel qua log-marginal-likelihood hoặc giữ cố định → giải hệ kernel trên train → tính mean/variance cho điểm mới → kiểm MAE/RMSE và coverage trên dữ liệu giữ riêng. Tối ưu marginal likelihood có thể gặp nghiệm cục bộ; thử nhiều khởi tạo hoặc kiểm CV khi cần. [2]

#### Suy diễn Giải tích: Phân phối Có điều kiện trong Gaussian Process

Theo định nghĩa của GP, phân phối đồng thời giữa các quan sát đã biết $y \in \mathbb{R}^n$ và giá trị hàm chưa biết $f_*$ tại điểm mới $x_*$ là một phân phối chuẩn đa biến:

$$\begin{bmatrix} y \\ f_* \end{bmatrix} \sim \mathcal{N} \left( \mathbf{0}, \begin{bmatrix} K(X, X) + \sigma_n^2 I & K(X, x_*) \\ K(x_*, X) & k(x_*, x_*) \end{bmatrix} \right)$$

Áp dụng quy tắc suy diễn có điều kiện của phân phối chuẩn đa biến (Gaussian Conditioning Rule), ta thu được nghiệm giải tích chính xác:

$$f_* \mid X, y, x_* \sim \mathcal{N}(\mu_*, \mathrm{Var}(f_*))$$

Trong đó:

$$\mu_* = K(x_*, X) \left[ K(X, X) + \sigma_n^2 I \right]^{-1} y$$

$$\mathrm{Var}(f_*) = k(x_*, x_*) - K(x_*, X) \left[ K(X, X) + \sigma_n^2 I \right]^{-1} K(X, x_*)$$

```text
       y, f(x)
          ^               .---. Khoảng tin cậy mở rộng ở vùng xa dữ liệu
          |              /     \
          |    o        |       |        o : Điểm dữ liệu huấn luyện
          |   / \      /    o    \      ---: Đường trung bình dự đoán mu_*
          |  o   \    /      \    \
          |       `--o        `----o
          +------------------------------------> x
                      Vùng có ít dữ liệu (Độ bất định Var(f_*) tăng vọt)
```

---

## Data used

LDA/QDA hợp khi feature số và giả định Gaussian theo lớp tương đối hợp lý; QDA cần đủ mẫu mỗi lớp để ước lượng covariance riêng. GPR phù hợp regression tập nhỏ/vừa khi muốn mô hình hóa độ bất định và quan hệ mượt theo kernel. Iris có thể dùng minh họa LDA/QDA, còn một tập regression nhỏ như Diabetes có thể dùng minh họa GPR. [1][2][4]

---

## Design

- LDA ít tham số covariance hơn QDA, thường ổn định hơn khi mẫu/lớp ít. QDA linh hoạt hơn nhưng dễ ước lượng covariance kém; cân nhắc shrinkage/regularization theo implementation. [1]
- GPR cần chọn kernel phù hợp và scale feature; train với kernel dense thường tăng mạnh theo `n` (ma trận $n\times n$, phép phân tích ma trận có chi phí bậc ba trong cách tính chuẩn), nên không mặc định dùng cho dữ liệu rất lớn. [2]
- Failure modes: giả định Gaussian không hợp, covariance suy biến, extrapolation thiếu dữ liệu, độ bất định GPR bị hiểu là hiệu chuẩn tự động.

| Hyperparameter/thiết kế | Ý nghĩa | Kiểm bằng gì |
|---|---|---|
| LDA/QDA `priors` | Tần suất lớp trước khi nhìn feature | Log loss, calibration, lớp hiếm |
| LDA `shrinkage` / QDA regularization theo API | Ổn định ước lượng covariance | CV trên train; kiểm cảnh báo suy biến |
| GPR kernel và length-scale | Độ mượt và phạm vi tương quan | Residual/MAE, coverage, ngoại suy |
| GPR `alpha` hoặc `WhiteKernel` | Mô hình hóa/ổn định nhiễu | Chất lượng dự đoán và độ bất định |

`alpha` trong `GaussianProcessRegressor` là mức nhiễu/regularization được thêm vào đường chéo kernel; không nhầm nó với `alpha` regularization của Ridge/Lasso. GPR chuẩn với kernel dense cần lưu $O(n^{2})$ phần tử và giải hệ thường có chi phí bậc ba theo `n`; cân nhắc phương pháp xấp xỉ hoặc mô hình khác khi dữ liệu lớn. [2]

### Kỹ thuật Thu hẹp Hiệp phương sai (Shrinkage Covariance - Ledoit & Wolf)

Khi số chiều $p$ lớn so với số mẫu $n$, ma trận hiệp phương sai mẫu thực nghiệm $\hat{\Sigma}$ bị nhiễu nghiêm trọng và có thể không khả nghịch. Kỹ thuật **Ledoit-Wolf Shrinkage** điều chuẩn ma trận bằng cách trộn lồi với ma trận đơn vị:

$$\Sigma_{\text{shrunk}} = (1 - \gamma) \hat{\Sigma} + \gamma \frac{\mathrm{tr}(\hat{\Sigma})}{p} I$$

Hệ số $\gamma \in [0, 1]$ được tự động ước lượng tối ưu từ dữ liệu, giúp triệt tiêu hoàn toàn nguy cơ ma trận bị suy biến trong LDA.

### Chi phí Tính toán Khổng lồ của GPR: Lời nguyền $\mathcal{O}(n^3)$

* **Bộ nhớ:** Để lưu ma trận Gram $K(X, X)$, cần $\mathcal{O}(n^2)$ bộ nhớ RAM. Với $n = 100{,}000$ mẫu, ma trận chiếm tới $\approx 80\,\text{GB}$ RAM!
* **Tính toán:** Phép phân tích Cholesky để giải hệ tuyến tính $\left[ K + \sigma_n^2 I \right]^{-1} y$ có độ phức tạp thuật toán là $\mathcal{O}(n^3)$.
* *Kết luận thực tiễn:* GPR chuẩn tắc **chỉ khả thi với $n \le 10{,}000$ mẫu**. Với dữ liệu lớn hơn, bắt buộc phải dùng các phương pháp xấp xỉ thưa (Sparse Gaussian Processes như FITC, VFE) hoặc Random Fourier Features.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Prior lớp $\pi _k$** | Tần suất/xác suất lớp trước khi xét feature của mẫu mới. | Cho lớp phổ biến hơn lợi thế ban đầu khi bằng chứng feature tương đương | Thành phần của hàm phân biệt LDA và QDA |
| **Mean $\mu _k$** | Vector trung bình feature của các mẫu thuộc lớp `k`. | Xác định tâm phân phối của mỗi lớp | Tham số ước lượng từ train cho LDA và QDA |
| **Covariance $\Sigma$** | Ma trận mô tả độ biến thiên từng feature và mức biến thiên cùng nhau giữa hai feature. | Mô tả độ phân tán và tương quan giữa feature | Tham số phân phối Gaussian; LDA dùng chung, QDA dùng riêng theo lớp |
| **Shrinkage** | Trộn ước lượng covariance từ dữ liệu với một cấu trúc ổn định hơn để giảm nhiễu ước lượng. | Ổn định ma trận khi ít dữ liệu hoặc nhiều feature | Kỹ thuật ước lượng covariance trong LDA |
| **Kernel $k(x,x')$** | Hàm biểu diễn mức tương quan giả định của giá trị hàm tại hai đầu vào trong GP. | Quy định độ liên hệ giữa giá trị hàm tại các điểm đầu vào | Giả định cấu trúc của Gaussian Process Regression |
| **Length-scale** | Tham số kernel mô tả khoảng cách mà đầu vào còn tương quan đáng kể theo mô hình. | Điều khiển độ mượt và phạm vi ảnh hưởng của điểm train | Hyperparameter kernel của GP |
| **Posterior mean/variance** | Trung bình và phương sai dự đoán sau khi điều kiện hóa GP trên dữ liệu train. | Cho giá trị dự đoán trung tâm và độ bất định theo mô hình | Đầu ra của GP khi suy luận tại điểm mới |
| **Coverage** | Tỷ lệ nhãn thật nằm trong khoảng dự đoán trên dữ liệu giữ riêng; dùng kiểm độ bất định thực nghiệm. | Kiểm xem khoảng dự đoán có bao phủ đúng tần suất kỳ vọng | Metric đánh giá độ bất định trên dữ liệu giữ riêng |
| **Mahalanobis Distance** | Khoảng cách hình học có chuẩn hóa theo ma trận hiệp phương sai. | Đo độ lệch chuẩn hóa từ một điểm tới tâm cụm phân phối. | Số hạng hạt nhân trong hàm phân biệt LDA/QDA. |
| **Rayleigh Quotient** | Tỷ số giữa phương sai ngoại lớp và phương sai nội lớp trong bài toán Fisher. | Tiêu chí tối ưu hóa để tìm vector chiếu giảm chiều tốt nhất. | Cơ sở đại số tuyến tính của thuật toán LDA gốc. |

---

## Nguồn

- [1] [scikit-learn: Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
- [2] [scikit-learn: Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [3] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [4] [scikit-learn: Datasets](https://scikit-learn.org/stable/datasets.html)
- [5] Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics, 7(2), 179-188.
- [6] Rasmussen, C. E., & Williams, C. K. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- [7] Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365-411.
