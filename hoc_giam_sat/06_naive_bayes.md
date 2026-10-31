# Naive Bayes

## Purpose

Phân loại nhanh bằng định lý Bayes với giả định các feature độc lập có điều kiện khi biết lớp. Thường dùng làm baseline cho văn bản và dữ liệu chiều cao. [1]

### Mô hình Tạo sinh (Generative) vs. Mô hình Phân biệt (Discriminative)

Để hiểu sâu sắc bản chất của Naive Bayes, nhà nghiên cứu cần phân biệt hai trường phái lớn trong học máy có giám sát:

1. **Mô hình Phân biệt (Discriminative Models - như Logistic Regression, SVM, Cây quyết định):**
   * Trực tiếp mô hình hóa xác suất hậu nghiệm $P(Y \mid X)$ hoặc ranh giới phân cách hình học giữa các lớp.
   * *Triết lý:* "Chỉ tập trung phân biệt ranh giới giữa mèo và chó, không cần biết một con mèo toàn diện được cấu thành như thế nào."
2. **Mô hình Tạo sinh (Generative Models - như Naive Bayes, LDA, HMM):**
   * Mô hình hóa phân phối xác suất đồng thời $P(X, Y) = P(Y) \cdot P(X \mid Y)$.
   * *Triết lý:* "Học cách từng lớp tạo sinh ra dữ liệu như thế nào ($P(X \mid Y)$). Sau khi có dữ liệu mới $x$, sử dụng Định lý Bayes để lật ngược bài toán tính xác suất $P(Y \mid X)$."
   * Theo nghiên cứu kinh điển của Andrew Ng và Michael Jordan (NeurIPS 2001), mô hình tạo sinh như Naive Bayes đạt sai số tiệm cận nhanh hơn theo số lượng mẫu ($n \sim \log p$) so với Logistic Regression ($n \sim p$). Do đó, khi tập dữ liệu huấn luyện rất nhỏ hoặc số chiều từ vựng văn bản cực kỳ lớn, Naive Bayes thường vượt trội hơn các mô hình phức tạp.

---

## Input

`X` và nhãn lớp `y`; chọn biến thể theo dữ liệu: GaussianNB cho feature liên tục gần dạng Gaussian theo lớp, MultinomialNB cho count/tần suất **không âm**, BernoulliNB cho feature nhị phân. [1]

Với văn bản, `X` thường là ma trận thưa $n\times V$ (`V` là số từ); phải fit vocabulary/vectorizer **chỉ** trên train fold. Với bảng số liên tục, kiểm đơn vị và missing; không dùng MultinomialNB cho feature âm do chuẩn hóa tâm hóa. [1][3]

---

## Output

Nhãn dự đoán, điểm/xác suất lớp và các thống kê như prior lớp, phân phối feature có điều kiện.

---

## Evaluation metrics

Macro F1 và per-class precision/recall cho lớp không cân bằng; confusion matrix; log loss/Brier nếu xác suất cần dùng cho quyết định, kèm kiểm calibration. [2]

---

## How it works

Ước lượng $P(y)$ và $P(x_j\mid y)$ từ train, rồi dùng $P(y\mid x) \propto P(y) \prod_j P(x_j\mid y)$. Tính trong log-space để ổn định số. Smoothing (ví dụ `alpha` trong MultinomialNB) tránh xác suất 0 cho feature chưa thấy. [1]

**Ví dụ tính tay:** giả sử $P(\mathrm{spam})=0.4$, $P(\mathrm{sale}\mid \mathrm{spam})=0.5$, $P(\mathrm{sale}\mid \mathrm{ham})=0.1$, $P(\mathrm{ham})=0.6$. Với email chỉ chứa tín hiệu “sale”, điểm chưa chuẩn hóa là $0.4\times 0.5=0.20$ cho spam và $0.6\times 0.1=0.06$ cho ham; sau chuẩn hóa, $P(\mathrm{spam}\mid \mathrm{sale})=0.20/(0.20+0.06)\approx 0.769$. Ví dụ này giả định cách biểu diễn một feature đơn giản; email thực tế có nhiều feature và smoothing.

**Fit/predict:** đếm/ước lượng prior lớp và thống kê feature theo lớp → cộng log prior và log likelihood từng feature → chọn lớp có score cao nhất. Với nhiều feature, giả định độc lập có điều kiện giúp tính tích đơn giản, nhưng thường không đúng hoàn toàn. [1]

### Likelihood của ba biến thể

| Biến thể | Thống kê fit | Phù hợp và lưu ý |
|---|---|---|
| GaussianNB | Mean $\mu_{cj}$, variance $\sigma_{cj}^2$ | Feature liên tục; `var_smoothing` giúp ổn định khi variance rất nhỏ |
| MultinomialNB | Xác suất token $\theta_{cj}$ | Count không âm; $\alpha$ smoothing chống xác suất 0 |
| BernoulliNB | Xác suất feature có mặt $\theta_{cj}$ | Feature 0/1; **vắng mặt** cũng đóng góp vào score |

**GaussianNB — mật độ có điều kiện:**

$$
p(x_j\mid y=c)=\frac{\exp\!\left[-\frac{(x_j-\mu_{cj})^2}{2\sigma_{cj}^2}\right]}{\sqrt{2\pi\sigma_{cj}^2}}
$$

**MultinomialNB — xác suất token sau smoothing:**

$$
\theta_{cj}=\frac{N_{cj}+\alpha}{\sum_t N_{ct}+\alpha V}
$$

**BernoulliNB — likelihood của một feature:**

$$
P(x_j\mid y=c)=\theta_{cj}^{x_j}(1-\theta_{cj})^{1-x_j}
$$

Ở MultinomialNB, $N_{cj}$ là tổng số lần token `j` xuất hiện trong văn bản lớp `c`; `V` là kích thước vocabulary. Công thức trên mô tả Laplace/Lidstone smoothing khi $\alpha$ là hằng chung. Từng estimator có quy ước và tham số API riêng; không dùng một công thức Gaussian cho count hoặc Bernoulli. [1]

**Ví dụ fit MultinomialNB:** vocabulary gồm `sale`, `meeting`; trong văn bản spam train có tổng count `(3,1)`, ham `(1,3)`, hai lớp có prior `0.5`. Với $\alpha =1$, ta có $P(\mathrm{sale}\mid \mathrm{spam})=4/6$, $P(\mathrm{meeting}\mid \mathrm{spam})=2/6$; ham đảo lại. Email mới `sale sale meeting` có score chưa chuẩn hóa: spam $0.5\times (4/6)^{2}\times (2/6)=2/27$, ham $0.5\times (2/6)^{2}\times (4/6)=1/27$. Posterior sau chuẩn hóa là $P(\mathrm{spam}\mid x)=2/3$. Ví dụ giả định count đơn giản; không có kết quả benchmark.

Khi `V` lớn, nhân nhiều xác suất rất nhỏ dễ underflow; tính $\log P(c)+\sum _j \log P(x_j\mid c)$ (hoặc dạng count tương ứng) rồi so score. Muốn xuất xác suất, chuẩn hóa các log-score ổn định số. [1]

### Chứng minh Toán học: Bản chất Giả định "Ngây thơ" (Conditional Independence)

Theo Định lý Bayes chính xác tuyệt đối:

$$P(Y = c \mid x_1, x_2, \dots, x_p) = \frac{P(Y = c) \cdot P(x_1, x_2, \dots, x_p \mid Y = c)}{P(x_1, x_2, \dots, x_p)}$$

Trong thực tế, việc ước lượng phân phối đồng thời đa biến $P(x_1, \dots, x_p \mid Y = c)$ đòi hỏi số lượng tham số khổng lồ cỡ $2^p$ (nếu các biến nhị phân), dẫn tới sụp đổ do "Lời nguyền số chiều" (Curse of Dimensionality).

**Giả định Độc lập Có điều kiện (The "Naive" Assumption):**
Mô hình giả định rằng với điều kiện lớp $Y = c$ đã biết, sự xuất hiện của biến $x_j$ hoàn toàn độc lập với biến $x_k$:

$$P(x_1, x_2, \dots, x_p \mid Y = c) = \prod_{j=1}^p P(x_j \mid Y = c)$$

Khi đó, bài toán từ việc ước lượng bảng xác suất $2^p$ phần tử được rút gọn xuống chỉ còn ước lượng $2p$ tham số đơn biến! 

Chuyển sang không gian Logarit để ngăn chặn hiện tượng tràn số dưới (Underflow):

$$\hat{y} = \mathop{\mathrm{arg\,max}}_{c} \left[ \ln P(Y = c) + \sum_{j=1}^p \ln P(x_j \mid Y = c) \right]$$

### Nghịch lý Thú vị: Tại sao Naive Bayes Vẫn Hoạt động Tốt khi Giả định Bị Sai?

Trong đời thực, các đặc trưng hầu như không bao giờ độc lập (ví dụ: từ "Hong" và "Kong" trong văn bản hầu như luôn đi cùng nhau). Vậy tại sao Naive Bayes vẫn là bộ phân loại thư rác và phân loại tài liệu cực kỳ thành công?

Nghiên cứu của Pedro Domingos và Michael Pazzani (1997) đã chứng minh bằng giải tích:
* **Mục tiêu phân loại là tìm $\mathop{\mathrm{arg\,max}}$, không phải ước lượng chính xác xác suất.**
* Giả sử xác suất thực tế của lớp Đúng là $0.65$ và lớp Sai là $0.35$.
* Do giả định độc lập bị vi phạm, Naive Bayes nhân lặp các bằng chứng phụ thuộc, làm xác suất bị đẩy cực đoan: dự đoán lớp Đúng thành $0.999$ và lớp Sai thành $0.001$.
* **Kết luận:** Mặc dù giá trị xác suất bị méo mó nghiêm trọng (Probability Calibration rất kém), nhưng **thứ tự xếp hạng giữa các lớp không hề thay đổi**. Nhãn dự đoán $\mathop{\mathrm{arg\,max}}$ vẫn chính xác 100%!

---

## Data used

Văn bản Bag-of-Words/count cho MultinomialNB là ví dụ điển hình; với bảng số liên tục dùng GaussianNB khi giả định hợp lý. Không đưa feature âm vào MultinomialNB. Chỉ fit vectorizer trên train fold để tránh leakage. [1][3]

---

## Design

- Ưu điểm: train/inference nhanh, hoạt động được khi dữ liệu không quá lớn; giả định độc lập thường sai nhưng vẫn có thể phân loại hữu ích.
- Hạn chế: tương tác feature không được biểu diễn trực tiếp; xác suất có thể quá tự tin. Kiểm calibration nếu xác suất được dùng.
- Failure modes: chọn sai biến thể, prior lệch so với production, từ vựng fit trên test, lớp hiếm thiếu mẫu.

| Quyết định | Cách chọn | Kiểm tra |
|---|---|---|
| Biến thể NB | Theo kiểu feature và cơ chế sinh dữ liệu | Có giá trị âm/nhị phân/liên tục không? |
| `alpha` của Multinomial/Bernoulli | Tune bằng CV trên train | Quá nhỏ: nhạy token hiếm; quá lớn: làm phẳng mạnh |
| `var_smoothing` của Gaussian | Giữ variance số học ổn định | Feature gần hằng và calibration |
| Prior lớp | Ước lượng từ train hoặc đặt theo ngữ cảnh | Nếu prevalence production đổi, xác suất có thể lệch |

Khi feature phụ thuộc mạnh (ví dụ hai biến đo cùng một tín hiệu), NB có thể đếm bằng chứng lặp và cho xác suất quá cực đoan. Trong bài toán văn bản, NB thường là baseline rẻ; nếu cần xác suất rủi ro tin cậy, kiểm calibration riêng và so với logistic regression. [1][2]

**Chi phí và tuning:** thống kê count thường tuyến tính theo số phần tử khác 0 trong ma trận thưa; dự đoán cũng rẻ. Tuning `alpha` trên CV. Nếu cần xác suất để ra quyết định, kiểm calibration trên dữ liệu riêng vì Naive Bayes có thể cho xác suất cực đoan. [1][2]

### Bẫy Phổ biến: Đếm Trùng Bằng chứng (Double Counting Evidence)

* *Hiện tượng:* Nếu ta vô tình đưa vào 2 feature hoàn toàn đồng nghĩa (ví dụ: `chiều_cao_cm` và `chiều_cao_inch`), Naive Bayes sẽ coi đây là hai bằng chứng độc lập ủng hộ cùng một lớp và nhân đôi sức nặng của bằng chứng này vào hàm tích log-likelihood.
* *Hậu quả:* Ranh giới quyết định bị bóp méo thiên lệch về phía nhóm feature bị trùng lặp. Cần loại bỏ các biến có tương quan tuyến tính cao trước khi huấn luyện Naive Bayes.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Prior $P(y)$** | Xác suất lớp trước khi xét feature của mẫu mới. | Biểu thị mức phổ biến ban đầu của từng lớp | Thành phần đầu của công thức Bayes trước khi xét feature |
| **Likelihood $P(x\mid y)$** | Mức phù hợp của feature quan sát được nếu giả sử một lớp là đúng. | Đo mức phù hợp của dữ liệu nếu thuộc một lớp | Thành phần được ước lượng từ train để so lớp ở inference |
| **Posterior $P(y\mid x)$** | Xác suất lớp sau khi đã xét feature, tính từ prior và likelihood theo Bayes. | Kết hợp prior và likelihood sau chuẩn hóa | Đại lượng dùng chọn lớp; độ tin cậy xác suất cần kiểm riêng |
| **Conditional independence** | Giả định các feature độc lập với nhau khi đã biết lớp; là chữ “naive” trong tên mô hình. | Cho phép nhân likelihood từng feature để tính nhanh | Giả định mô hình hóa của Naive Bayes, có thể sai với feature tương quan |
| **Smoothing `alpha`** | Cộng một lượng vào count để token chưa thấy không làm likelihood bằng 0 trong mô hình count. | Tránh xác suất bằng 0 cho token ít gặp hoặc chưa thấy | Hyperparameter của các biến thể NB dựa trên count |
| **Vocabulary** | Tập token/cột văn bản học từ train; phải dùng cùng mapping khi inference. | Giữ thứ tự cột đầu vào ổn định | Được học khi vector hóa train và tái sử dụng khi inference |
| **Log-space** | Cộng log xác suất thay vì nhân nhiều số rất nhỏ, giúp tránh underflow. | Giảm nguy cơ số quá nhỏ bị làm tròn về 0 | Cách tính số học ổn định khi cộng log likelihood |
| **Calibration** | Kiểm độ tin cậy của xác suất đầu ra; phân loại đúng không đồng nghĩa xác suất đúng. | Phát hiện xác suất quá tự tin hoặc thiếu tự tin | Kiểm sau fit nếu quyết định phụ thuộc xác suất |
| **Generative Model** | Mô hình học phân phối đồng thời $P(X, Y)$ thay vì chỉ học ranh giới $P(Y \mid X)$. | Cho phép sinh mẫu mới và học hiệu quả khi dữ liệu ít. | Bản chất kiến trúc học thuật của Naive Bayes. |
| **Laplace Smoothing** | Phép cộng hằng số $\alpha=1$ vào tử số và $\alpha V$ vào mẫu số khi đếm tần suất. | Triệt tiêu hoàn toàn thảm họa xác suất bằng 0. | Kỹ thuật điều chuẩn chuẩn mực trong xử lý ngôn ngữ tự nhiên. |

---

## Nguồn

- [1] [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [2] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [4] Domingos, P., & Pazzani, M. (1997). *On the optimality of the simple Bayesian classifier under zero-one loss*. Machine Learning, 29(2), 103-130.
- [5] Ng, A. Y., & Jordan, M. I. (2001). *On discriminative vs. generative classifiers: A comparison of logistic regression and naive bayes*. Advances in Neural Information Processing Systems (NeurIPS), 14.
