# Naive Bayes

## Purpose

Phân loại nhanh bằng định lý Bayes với giả định các feature độc lập có điều kiện khi biết lớp. Thường dùng làm baseline cho văn bản và dữ liệu chiều cao. [1]

## Input

`X` và nhãn lớp `y`; chọn biến thể theo dữ liệu: GaussianNB cho feature liên tục gần dạng Gaussian theo lớp, MultinomialNB cho count/tần suất **không âm**, BernoulliNB cho feature nhị phân. [1]

Với văn bản, `X` thường là ma trận thưa $`n\times V`$ (`V` là số từ); phải fit vocabulary/vectorizer **chỉ** trên train fold. Với bảng số liên tục, kiểm đơn vị và missing; không dùng MultinomialNB cho feature âm do chuẩn hóa tâm hóa. [1][3]

## Output

Nhãn dự đoán, điểm/xác suất lớp và các thống kê như prior lớp, phân phối feature có điều kiện.

## Evaluation metrics

Macro F1 và per-class precision/recall cho lớp không cân bằng; confusion matrix; log loss/Brier nếu xác suất cần dùng cho quyết định, kèm kiểm calibration. [2]

## How it works

Ước lượng $`P(y)`$ và $`P(x_j\mid y)`$ từ train, rồi dùng $`P(y\mid x) \propto P(y) \prod_j P(x_j\mid y)`$. Tính trong log-space để ổn định số. Smoothing (ví dụ `alpha` trong MultinomialNB) tránh xác suất 0 cho feature chưa thấy. [1]

**Ví dụ tính tay:** giả sử $`P(\mathrm{spam})=0.4`$, $`P(\mathrm{sale}\mid \mathrm{spam})=0.5`$, $`P(\mathrm{sale}\mid \mathrm{ham})=0.1`$, $`P(\mathrm{ham})=0.6`$. Với email chỉ chứa tín hiệu “sale”, điểm chưa chuẩn hóa là $`0.4\times 0.5=0.20`$ cho spam và $`0.6\times 0.1=0.06`$ cho ham; sau chuẩn hóa, $`P(\mathrm{spam}\mid \mathrm{sale})=0.20/(0.20+0.06)\approx 0.769`$. Ví dụ này giả định cách biểu diễn một feature đơn giản; email thực tế có nhiều feature và smoothing.

**Fit/predict:** đếm/ước lượng prior lớp và thống kê feature theo lớp → cộng log prior và log likelihood từng feature → chọn lớp có score cao nhất. Với nhiều feature, giả định độc lập có điều kiện giúp tính tích đơn giản, nhưng thường không đúng hoàn toàn. [1]

### Likelihood của ba biến thể

| Biến thể | Thống kê fit | Phù hợp và lưu ý |
|---|---|---|
| GaussianNB | Mean $`\mu_{cj}`$, variance $`\sigma_{cj}^2`$ | Feature liên tục; `var_smoothing` giúp ổn định khi variance rất nhỏ |
| MultinomialNB | Xác suất token $`\theta_{cj}`$ | Count không âm; $`\alpha`$ smoothing chống xác suất 0 |
| BernoulliNB | Xác suất feature có mặt $`\theta_{cj}`$ | Feature 0/1; **vắng mặt** cũng đóng góp vào score |

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

Ở MultinomialNB, $`N_{cj}`$ là tổng số lần token `j` xuất hiện trong văn bản lớp `c`; `V` là kích thước vocabulary. Công thức trên mô tả Laplace/Lidstone smoothing khi $`\alpha`$ là hằng chung. Từng estimator có quy ước và tham số API riêng; không dùng một công thức Gaussian cho count hoặc Bernoulli. [1]

**Ví dụ fit MultinomialNB:** vocabulary gồm `sale`, `meeting`; trong văn bản spam train có tổng count `(3,1)`, ham `(1,3)`, hai lớp có prior `0.5`. Với $`\alpha =1`$, ta có $`P(\mathrm{sale}\mid \mathrm{spam})=4/6`$, $`P(\mathrm{meeting}\mid \mathrm{spam})=2/6`$; ham đảo lại. Email mới `sale sale meeting` có score chưa chuẩn hóa: spam $`0.5\times (4/6)^{2}\times (2/6)=2/27`$, ham $`0.5\times (2/6)^{2}\times (4/6)=1/27`$. Posterior sau chuẩn hóa là $`P(\mathrm{spam}\mid x)=2/3`$. Ví dụ giả định count đơn giản; không có kết quả benchmark.

Khi `V` lớn, nhân nhiều xác suất rất nhỏ dễ underflow; tính $`\log P(c)+\sum _j \log P(x_j\mid c)`$ (hoặc dạng count tương ứng) rồi so score. Muốn xuất xác suất, chuẩn hóa các log-score ổn định số. [1]

## Data used

Văn bản Bag-of-Words/count cho MultinomialNB là ví dụ điển hình; với bảng số liên tục dùng GaussianNB khi giả định hợp lý. Không đưa feature âm vào MultinomialNB. Chỉ fit vectorizer trên train fold để tránh leakage. [1][3]

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

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Prior $`P(y)`$** | Xác suất lớp trước khi xét feature của mẫu mới. | Biểu thị mức phổ biến ban đầu của từng lớp | Thành phần đầu của công thức Bayes trước khi xét feature |
| **Likelihood $`P(x\mid y)`$** | Mức phù hợp của feature quan sát được nếu giả sử một lớp là đúng. | Đo mức phù hợp của dữ liệu nếu thuộc một lớp | Thành phần được ước lượng từ train để so lớp ở inference |
| **Posterior $`P(y\mid x)`$** | Xác suất lớp sau khi đã xét feature, tính từ prior và likelihood theo Bayes. | Kết hợp prior và likelihood sau chuẩn hóa | Đại lượng dùng chọn lớp; độ tin cậy xác suất cần kiểm riêng |
| **Conditional independence** | Giả định các feature độc lập với nhau khi đã biết lớp; là chữ “naive” trong tên mô hình. | Cho phép nhân likelihood từng feature để tính nhanh | Giả định mô hình hóa của Naive Bayes, có thể sai với feature tương quan |
| **Smoothing `alpha`** | Cộng một lượng vào count để token chưa thấy không làm likelihood bằng 0 trong mô hình count. | Tránh xác suất bằng 0 cho token ít gặp hoặc chưa thấy | Hyperparameter của các biến thể NB dựa trên count |
| **Vocabulary** | Tập token/cột văn bản học từ train; phải dùng cùng mapping khi inference. | Giữ thứ tự cột đầu vào ổn định | Được học khi vector hóa train và tái sử dụng khi inference |
| **Log-space** | Cộng log xác suất thay vì nhân nhiều số rất nhỏ, giúp tránh underflow. | Giảm nguy cơ số quá nhỏ bị làm tròn về 0 | Cách tính số học ổn định khi cộng log likelihood |
| **Calibration** | Kiểm độ tin cậy của xác suất đầu ra; phân loại đúng không đồng nghĩa xác suất đúng. | Phát hiện xác suất quá tự tin hoặc thiếu tự tin | Kiểm sau fit nếu quyết định phụ thuộc xác suất |

## Nguồn

- [1] [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [2] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
