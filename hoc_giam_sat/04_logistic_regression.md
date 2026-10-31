# Logistic regression

## Purpose

Phân loại bằng mô hình tuyến tính trên log-odds; baseline mạnh, dễ triển khai cho classification nhị phân và nhiều lớp. Tên gọi có chữ regression nhưng đây là thuật toán **classification**. [1]

### Tại sao Hồi quy Tuyến tính Thất bại trong Phân loại? (Motivation for Logistic Regression)

Một trực giác tự nhiên của người mới học là thử áp dụng Hồi quy Tuyến tính OLS cho bài toán phân loại nhị phân bằng cách gán nhãn $`y \in \{0, 1\}`$ và dự đoán $`\hat{y} = w^\top x + b`$. Tuy nhiên, cách tiếp cận này sụp đổ vì ba lý do lý thuyết căn bản:

1. **Vi phạm không gian xác suất:** Phương trình đường thẳng $`w^\top x + b`$ có miền giá trị trên toàn bộ trục số thực $`(-\infty, +\infty)`$. Do đó, mô hình có thể dự đoán xác suất âm ($`\hat{y} = -0.4`$) hoặc xác suất lớn hơn 1 ($`\hat{y} = 1.7`$), hoàn toàn vô nghĩa về mặt tiên đề xác suất Kolmogorov.
2. **Độ nhạy méo mó trước các điểm ngoại lai xa (Outlier Sensitivity):** Khi xuất hiện một điểm dữ liệu dương tính ($`y=1`$) nằm rất xa về phía bên phải ranh giới (một mẫu "quá chuẩn"), hàm bình phương sai số của OLS sẽ cố gắng kéo đường hồi quy dốc lên để giảm sai số của điểm này, vô tình làm xô lệch ranh giới quyết định và gây phân loại sai các điểm ở vùng trung tâm.
3. **Vi phạm giả định phương sai đồng nhất (Heteroscedasticity):** Với biến ngẫu nhiên nhị phân Bernoulli, phương sai phụ thuộc trực tiếp vào kỳ vọng: $`\mathrm{Var}(Y \mid X) = p(x)(1 - p(x))`$. Do đó phương sai biến thiên theo từng điểm dữ liệu, làm mất tính tối ưu của OLS.

Để giải quyết triệt để vấn đề này, David Cox (1958) đã đưa ra mô hình Hồi quy Logistic thuộc họ **Mô hình Tuyến tính Khái quát (GLM - Generalized Linear Models)** bằng cách đưa vào **Hàm liên kết Logit (Logit Link Function)** ánh xạ từ xác suất $`(0, 1)`$ sang toàn bộ trục số thực $`(-\infty, +\infty)`$.

---

## Input

`X` dạng số hoặc đã encode, `y` là nhãn lớp; cần xác định lớp dương và chi phí false positive/false negative.

Với `n` mẫu và `p` feature, $`X \in \mathbb{R}^{n\times p}`$. Binary target có hai lớp; multiclass một nhãn/mẫu có $`K>2`$ lớp. Nếu dữ liệu lớp hiếm, lưu `support` của từng lớp và kiểm các fold có đủ lớp để fit và tính metric. [1][2]

---

## Output

Điểm tuyến tính $`z=b+w^{\top}x`$; nhị phân có $`p(y=1\mid x)=\frac{1}{1+e^{-z}}`$, nhãn được quyết định bằng threshold. Mô hình đa lớp có thể trả phân phối xác suất qua các lớp tùy cấu hình. [1]

### Tỷ số Cược (Odds) & Ý nghĩa Diễn giải của Trọng số $`w_j`$

* **Định nghĩa Tỷ số Cược (Odds):** Là tỷ lệ giữa xác suất biến cố xảy ra và xác suất biến cố không xảy ra:

$$
\mathrm{Odds} = \frac{p}{1 - p}
$$

  Ví dụ: nếu xác suất chiến thắng là $`p = 0.8`$, tỷ số cược là $`\mathrm{Odds} = \frac{0.8}{0.2} = 4`$ (tỷ lệ "4 ăn 1").
* **Hàm Log-Odds (Logit):**

$$
\mathrm{logit}(p) = \ln\left( \frac{p}{1 - p} \right) = w_0 + w_1 x_1 + \dots + w_p x_p
$$

  Đảo ngược hàm logit, ta thu được hàm kích hoạt **Sigmoid (Logistic Function)**:

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$

* **Ý nghĩa Diễn giải Hệ số qua Tỷ số Nguy cơ (Odds Ratio - OR):**
  Lấy mũ cơ số tự nhiên hai vế:

$$
\mathrm{Odds} = e^{w_0} \cdot e^{w_1 x_1} \dots e^{w_j x_j}
$$

  Khi biến độc lập $`x_j`$ tăng thêm 1 đơn vị (các biến khác giữ nguyên):

$$
\mathrm{OR} = \frac{\mathrm{Odds}(x_j + 1)}{\mathrm{Odds}(x_j)} = e^{w_j}
$$

  * Nếu $`w_j > 0 \implies e^{w_j} > 1`$: Làm tăng tỷ số cược xảy ra lớp dương tính.
  * Nếu $`w_j < 0 \implies e^{w_j} < 1`$: Làm giảm tỷ số cược xảy ra lớp dương tính.

---

## Evaluation metrics

Precision/recall/F1 theo lớp, confusion matrix; ROC-AUC hoặc PR-AUC cho khả năng xếp hạng; log loss/Brier và calibration khi xác suất được sử dụng; so với DummyClassifier. Accuracy chỉ phù hợp khi phân phối lớp và chi phí lỗi cho phép. [2]

---

## How it works

Fit trọng số bằng cách giảm log loss (negative log-likelihood), thường kèm regularization L2 hoặc L1. Điều chỉnh `C` (nghịch đảo cường độ regularization trong scikit-learn) và threshold trên validation; threshold không phải tham số học của logistic regression. [1][3]

Với nhị phân $`y\in \{0,1\}`$, loss trung bình là $`-\frac{1}{n}\sum_i\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]`$; mô hình giảm loss để tăng xác suất cho nhãn đúng. Với $`z=0`$, sigmoid cho $`p=0.5`$; $`z=\ln (3)`$ cho $`p=0.75`$. Nếu threshold là `0.8`, ví dụ thứ hai vẫn thuộc lớp âm. Do đó xác suất và quyết định là hai tầng khác nhau. [1][3]

**Fit/predict:** encode/scale từ train → tối ưu trọng số theo log loss + phạt → tính score `z` ở mẫu mới → chuyển thành xác suất → áp threshold đã chọn. Với nhiều lớp, công thức và solver phụ thuộc chiến lược đa lớp; cần kiểm cấu hình thư viện đang dùng. [1]

### Binary: gradient và một bước học

Với $`p_{i}=\sigma (w^{\top}x_{i}+b)`$ và log loss chưa regularization, gradient theo `w` là $`(1/n)X^{\top}(p-y)`$ và theo `b` là trung bình $`(p_{i}-y_{i})`$. Ví dụ một mẫu $`x=2`$, $`y=1`$, khởi tạo $`w=b=0`$ cho $`p=0.5`$; gradient $`dw=-1`$, $`db=-0.5`$. Một bước gradient descent với learning rate `0.1` cho $`w=0.1`$, $`b=0.05`$, score mới ở $`x=2`$ là `0.25`, nên xác suất tăng lên khoảng `0.562`. Đây là ví dụ cơ chế tối ưu, chưa phản ánh kết quả khi train nhiều mẫu và regularization. [1]

#### Chứng minh Giải tích: Vẻ Đẹp của Đạo hàm Cross-Entropy

Xét hàm mất mát của một mẫu đơn lẻ:

$$
L = - \left[ y \ln p + (1 - y) \ln(1 - p) \right]
$$

Trong đó $`p = \sigma(z)`$ và $`z = w^\top x + b`$.

1. Đạo hàm của hàm Sigmoid:

$$
\sigma'(z) = \frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z) (1 - \sigma(z)) = p(1 - p)
$$

2. Áp dụng quy tắc đạo hàm hàm hợp (Chain Rule) để tìm đạo hàm theo $`z`$:

$$
\frac{\partial L}{\partial z} = - \left[ \frac{y}{p} \cdot p(1 - p) - \frac{1 - y}{1 - p} \cdot p(1 - p) \right] = - \left[ y(1 - p) - (1 - y)p \right] = p - y
$$

3. Đạo hàm theo từng trọng số $`w_j`$:

$$
\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_j} = (p - y) x_j
$$

*Nhận xét sâu sắc:* Kết quả đạo hàm $`(p - y)x_j`$ có dạng đại số **giống hệt như đạo hàm của bài toán Hồi quy Tuyến tính OLS**! Sự sai lệch giữa xác suất dự đoán $`p`$ và nhãn thực tế $`y`$ đóng vai trò như một lực đẩy tuyến tính định hướng cập nhật trọng số.

### Multiclass: softmax

Với mỗi lớp `k`, tính logit $`z_k=w_k^{\top}x+b_k`$, rồi $`p_k=\frac{\exp(z_k)}{\sum_j\exp(z_j)}`$; các xác suất cộng bằng 1. Loss của một mẫu có lớp thật `c` là $`-\log p_c`$. Để ổn định số, phần mềm thường trừ $`\max(z)`$ trước khi tính exp. Nếu logits là $`[0,\ln(2),\ln(3)]`$, xác suất tương ứng $`[1/6, 2/6, 3/6]`$, nên lớp thứ ba được chọn khi dùng argmax. Với multiclass, threshold và chi phí có thể cần quy tắc quyết định riêng, không máy móc áp một ngưỡng nhị phân cho mọi lớp. [1]

#### Ổn định Số học trong Cài đặt Softmax (Numerical Stability Trick)

Nếu vector logit có giá trị lớn (ví dụ $`z = [1000, 1001, 1002]`$), phép tính $`e^{1000}`$ sẽ gây ra lỗi tràn số học nghiêm trọng (**Numerical Overflow** dẫn đến giá trị `inf` hoặc `NaN`).
* Định lý: Hàm Softmax có **tính chất bất biến khi tịnh tiến (Shift Invariance)**:

$$
\frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{-c} e^{z_k}}{e^{-c} \sum_j e^{z_j}} = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}}
$$

* Do đó, mọi thư viện khoa học chuẩn mực đều chọn $`c = \max_j(z_j)`$, đảm bảo rằng số mũ lớn nhất luôn bằng $`e^0 = 1`$, triệt tiêu hoàn toàn nguy cơ tràn số.

**Quy trình chọn mô hình:** split → fit encoder/scaler trong fold train → fit LogisticRegression → tune regularization/solver phù hợp bằng CV → so precision/recall và log loss trên validation → chọn threshold nếu cần trên dữ liệu hợp lệ → test cuối. Nếu fit calibrator, dữ liệu dùng cho calibrator phải độc lập với dữ liệu fit classifier hoặc lấy từ dự đoán out-of-fold đúng cách. [1][3][5]

---

## Data used

Dữ liệu bảng hoặc vector thưa của văn bản, ví dụ UCI Adult cho phân loại thu nhập hoặc Iris cho nhiều lớp. Cần encode categorical, impute missing và scale feature số khi regularization/solver nhạy thang đo. [4]

---

## Design

- Bắt đầu với LogisticRegression + pipeline tiền xử lý; dùng `class_weight` hoặc chiến lược lấy mẫu chỉ khi metric và phân phối lớp yêu cầu.
- Ưu điểm: nhanh, hệ số giúp giải thích hướng liên hệ, có đầu ra xác suất. Hạn chế: ranh giới quyết định tuyến tính trừ khi thêm tương tác/biến đổi.
- Failure modes: nhãn lệch, feature phụ thuộc mạnh, dữ liệu tương lai bị lộ, threshold 0.5 không khớp chi phí, xác suất cần kiểm calibration.

| Quyết định | Tác dụng | Cần kiểm |
|---|---|---|
| `C` | `C` nhỏ thường regularization mạnh hơn | Train–validation gap, hệ số, metric chính |
| L1/L2 theo solver hỗ trợ | L1 có thể tạo hệ số thưa; L2 co hệ số | API của phiên bản thư viện, ổn định feature |
| `class_weight` | Đổi trọng số lỗi theo lớp trong loss | Precision/recall và xác suất/calibration sau fit |
| Threshold | Biến score/xác suất thành nhãn | Chọn trên validation theo chi phí, không trên test |

Nếu resampling lớp hiếm, thực hiện **trong train fold** của CV; oversampling trước split làm bản sao/điểm tổng hợp rò sang validation. `class_weight` và threshold có tác dụng khác nhau: cái thứ nhất đổi quá trình fit, cái thứ hai đổi quy tắc quyết định sau fit. [3][6]

**Độ phức tạp:** tính score nhị phân cho một mẫu khoảng $`O(p)`$; fit lặp nhiều vòng và phụ thuộc solver/sparsity. `C` nhỏ thường regularization mạnh hơn. Tránh đọc $`\exp(w_j)`$ như tác động nhân quả; đó là thay đổi odds có điều kiện theo các feature trong mô hình, dưới các giả định dữ liệu tương ứng.

### Hiện tượng Phân tách Hoàn hảo (The Separation Problem & Infinite Weights)

Một hiện tượng toán học kỳ lạ trong Hồi quy Logistic: **Khi dữ liệu quá "đẹp" (hai lớp có thể phân tách tuyến tính 100%), mô hình không có regularization sẽ bị nổ tham số!**
* *Giải thích:* Nếu tồn tại một siêu phẳng ngăn cách hoàn toàn hai lớp, mô hình sẽ cố gắng đẩy xác suất dự đoán $`p_i`$ về đúng $`1.0`$ cho lớp dương và $`0.0`$ cho lớp âm để đưa hàm Cross-Entropy Loss về đúng bằng $`0`$.
* Muốn $`\sigma(w^\top x) \to 1`$, đại lượng $`w^\top x`$ phải tiến ra $`+\infty`$. Do đó, các trọng số $`\|w\|_2 \to \infty`$. Thuật toán Gradient Descent sẽ chạy vô tận và ma trận hiệp phương sai của các tham số bị suy biến.
* *Cách khắc phục:* Luôn luôn bật Regularization (trong Scikit-learn, tham số `penalty='l2'` và `C=1.0` được bật mặc định nhằm phạt độ lớn của $`\|w\|`$, giữ cho trọng số luôn hữu hạn và ổn định).

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Logit / log-odds** | $`z=\log [p/(1-p)]`$; mô hình logistic học `z` tuyến tính theo feature. | Biến xác suất thành thang số thực để mô hình hóa tuyến tính | Đầu ra tuyến tính trước sigmoid của phân loại nhị phân |
| **Sigmoid** | Hàm $`\frac{1}{1+e^{-z}}`$ đổi một logit nhị phân thành giá trị trong `(0,1)`. | Đưa logit về khoảng xác suất từ 0 đến 1 | Tầng chuyển điểm sang xác suất trong mô hình nhị phân |
| **Softmax** | Chuẩn hóa nhiều logit thành xác suất các lớp có tổng bằng 1. | Biến nhiều logit thành một phân phối xác suất trên các lớp | Tầng đầu ra của phân loại nhiều lớp loại trừ nhau |
| **Log loss / cross-entropy** | Loss phạt khi mô hình gán xác suất thấp cho lớp thật, đặc biệt phạt dự đoán sai nhưng quá tự tin. | Phạt mạnh dự đoán sai nhưng tự tin | Hàm loss thường được tối ưu khi fit bộ phân loại xác suất |
| **Threshold / ngưỡng** | Quy tắc biến score/xác suất thành nhãn; được chọn sau fit theo chi phí lỗi. | Thay đổi cân bằng precision và recall mà không fit lại mô hình | Quy tắc ra quyết định sau khi có score hoặc xác suất |
| **`C`** | Tham số nghịch với cường độ regularization trong `LogisticRegression` của scikit-learn ($`C = 1/\lambda`$). | Điều khiển regularization; C nhỏ thường phạt mạnh hơn | Hyperparameter của scikit-learn, chọn bằng validation hoặc CV |
| **`class_weight`** | Trọng số lỗi theo lớp trong quá trình fit; khác với thay threshold sau fit. | Tăng ảnh hưởng của lớp được gán trọng số lớn trong loss | Xử lý bất cân bằng lớp ngay trong training |
| **Calibration / hiệu chuẩn** | Kiểm xem nhóm mẫu dự đoán xác suất gần `p` có tần suất dương thật gần `p` hay không. | Cho biết xác suất báo ra có đáng tin như tần suất thực không | Kiểm sau fit khi xác suất dùng để ước lượng rủi ro hoặc đặt ngưỡng |
| **Odds Ratio (OR)** | Tỷ số giữa hai tỷ số cược khi một feature tăng thêm 1 đơn vị ($`e^{w_j}`$). | Lượng hóa mức độ tác động của từng đặc trưng lên nguy cơ phân loại. | Công cụ diễn giải chính trong nghiên cứu y học và kinh tế lượng. |
| **Separation Problem** | Hiện tượng các trọng số tiến tới vô cùng khi dữ liệu phân tách hoàn hảo. | Gây sụp đổ solver tối ưu hóa nếu không có phạt ràng buộc. | Lý do bắt buộc phải có L2/L1 penalty trong thực tế. |

---

## Nguồn

- [1] [scikit-learn: Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [2] [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [3] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [4] [UCI: Adult dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [5] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [6] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [7] Cox, D. R. (1958). *The regression analysis of binary sequences*. Journal of the Royal Statistical Society: Series B (Methodological), 20(2), 215-232.
- [8] McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models*. Chapman and Hall/CRC.
