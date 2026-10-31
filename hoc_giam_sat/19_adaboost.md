# AdaBoost — Adaptive Boosting

> PDF [Chương 2](<../Tài liệu học tập/ML_Chapter2_new.pdf>) nhắc AdaBoost trong nhóm boosting ở tr. 57. AdaBoost khác Gradient Boosting: bản kinh điển tăng trọng số các mẫu bị phân loại sai, rồi cộng có trọng số các weak learners. [1]

## Purpose

Kết hợp nhiều mô hình yếu (thường là cây rất nông) thành classifier mạnh hơn bằng cách tập trung dần vào các mẫu khó. Đây là một họ ensemble hữu ích để hiểu boosting tuần tự; cần so với cây đơn, Random Forest và Gradient Boosting trên cùng dữ liệu. [1][2]

### Câu hỏi Mở Lịch sử của Kearns & Valiant (1988) & Đột phá AdaBoost

Trong lý thuyết học máy tính toán (Computational Learning Theory), Leslie Valiant (1984) đã xây dựng khung lý thuyết **Học Khả năng Đúng Xấp xỉ (PAC Learning - Probably Approximately Correct)**. Trong khung lý thuyết này, Michael Kearns và Leslie Valiant (1988, 1989) đã đặt ra một câu hỏi mở có tính chất nền tảng làm trăn trở toàn bộ giới khoa học máy tính:
> *"Liệu một lớp các thuật toán học yếu (Weak Learners — tức là các bộ phân loại chỉ cần đạt độ chính xác nhỉnh hơn đoán mò ngẫu nhiên $50\% + \gamma$) có thể được kết hợp để tạo ra một thuật toán học mạnh (Strong Learner — có khả năng đạt độ chính xác tùy ý $1 - \varepsilon$) hay không?"*

Năm 1990, Robert Schapire đưa ra lời giải đầu tiên, và đến năm 1995, Yoav Freund cùng Robert Schapire đã hoàn thiện giải thuật **AdaBoost (Adaptive Boosting)**. Công trình này đã chứng minh về mặt toán học rằng câu trả lời là **CÓ**. Với đóng góp lịch sử này, Freund và Schapire đã vinh dự nhận Giải thưởng Gödel danh giá vào năm 2003.

---

## Input

`X` dữ liệu có nhãn lớp `y`, một base estimator hỗ trợ trọng số mẫu theo yêu cầu implementation, số vòng tối đa `n_estimators`, `learning_rate` và metric chính. Với bài toán nhị phân trong công thức kinh điển, mã hóa $y\in \{-1,+1\}$. Dữ liệu missing/categorical phải được xử lý theo khả năng estimator được dùng.

---

## Output

Danh sách weak learners $h_{1},\ldots ,h_M$ và trọng số đóng góp của từng learner. Dự đoán nhị phân kinh điển dùng dấu của tổng $\sum _m \alpha _m h_m(x)$; implementation đa lớp như `AdaBoostClassifier` có quy tắc SAMME và có thể xuất score/xác suất theo API. Xác suất ước lượng cần được kiểm calibration nếu dùng để quyết định theo rủi ro. [1][3]

### Ranh giới Quyết định Tuyến tính Tổng hợp

Dự đoán cuối cùng cho một mẫu mới $x$ được tạo ra bằng phép bỏ phiếu có trọng số giữa toàn bộ $M$ bộ phân loại yếu:
$$H(x) = \mathrm{sign}\left( \sum_{m=1}^M \alpha_m h_m(x) \right)$$
Trong đó $\alpha_m$ là "tiếng nói" hay trọng số đóng góp của mô hình cơ sở thứ $m$. Mô hình nào có tỷ lệ lỗi thấp hơn sẽ được trao quyền quyết định lớn hơn.

---

## Evaluation metrics

Classification: precision/recall/F1 theo lớp, confusion matrix, AP/ROC-AUC nếu có score thích hợp, calibration khi dùng xác suất. So baseline và cây đơn trên validation/CV; xem chênh train–validation khi tăng số vòng. Theo dõi thời gian fit/inference vì dự đoán phải gộp nhiều weak learners. [4]

---

## How it works

**AdaBoost nhị phân kinh điển:** khởi tạo $w_i=1/n$. Ở vòng `m`, fit weak learner theo trọng số $w_i$; tính lỗi có trọng số $e_m=\frac{\sum_i w_i\,\mathbf{1}[h_m(x_i)\ne y_i]}{\sum_i w_i}$; khi $0<e_m<0.5$, đặt $\alpha_m=\frac{1}{2}\ln\left(\frac{1-e_m}{e_m}\right)$; cập nhật $w_i\leftarrow w_i \exp (-\alpha _m y_i h_m(x_i))$, rồi chuẩn hóa. Mẫu bị dự đoán sai được tăng trọng số tương đối. Nếu $e_m=0$, công thức log có biên vô hạn và thuật toán thường dừng vì đã fit hoàn hảo; nếu learner không tốt hơn ngẫu nhiên ($e_m\ge 0.5$), cần dừng hoặc xử lý theo quy tắc của implementation thay vì áp công thức máy móc. Công thức này mô tả **binary AdaBoost cổ điển**; không thay thế công thức của mọi phiên bản đa lớp/API. [1][2][3]

**Ví dụ một vòng:** bốn mẫu cùng trọng số `0.25`; weak learner sai đúng một mẫu nên $e=0.25$, $\alpha=\frac{1}{2}\ln 3\approx 0.549$. Sau cập nhật và chuẩn hóa, mẫu sai có trọng số `0.5`, ba mẫu đúng mỗi mẫu khoảng `1/6`. Vòng kế tiếp vì thế chú ý mẫu sai nhiều hơn. Ví dụ này giải thích fit, không là điểm benchmark.

```text
khởi tạo sample weights → lặp:
    fit weak learner theo weights
    tính lỗi có trọng số, xác định trọng số learner
    tăng trọng số tương đối của mẫu sai, chuẩn hóa
predict: gộp các learner theo trọng số đã học
```

AdaBoost tăng trọng số **mẫu**, còn Gradient Boosting thường fit cây theo pseudo-residual/gradient của loss. Cả hai đều tuần tự, khác Random Forest dùng bootstrap và feature randomness để gộp nhiều cây. [1][2]

### Chứng minh Toán học: AdaBoost Thực chất là Tối thiểu hóa Hàm Mất mát Mũ

Năm 2000, Jerome Friedman, Trevor Hastie và Robert Tibshirani đã làm sáng tỏ bản chất của AdaBoost dưới góc độ tối ưu hóa thống kê: **AdaBoost chính là phương pháp khớp mô hình cộng dần từng bước (Forward Stagewise Additive Modeling) để tối thiểu hóa Hàm Mất mát Mũ (Exponential Loss)**:
$$L(y, f(x)) = \exp(-y f(x))$$

1. Tại vòng $m$, mô hình hiện tại là $f_m(x) = f_{m-1}(x) + \alpha_m h_m(x)$.
2. Hàm mục tiêu cần tối thiểu hóa tại bước $m$:
   $$\mathcal{J}(\alpha, h) = \sum_{i=1}^n \exp\left( -y_i [f_{m-1}(x_i) + \alpha h(x_i)] \right) = \sum_{i=1}^n w_i^{(m)} \exp(-y_i \alpha h(x_i))$$
   Trong đó $w_i^{(m)} = \exp(-y_i f_{m-1}(x_i))$ chính là trọng số của mẫu thứ $i$ được tích lũy từ các vòng trước!
3. Phân rã tổng mất mát thành hai nhóm (nhóm phân loại đúng $y_i h(x_i) = 1$ và nhóm phân loại sai $y_i h(x_i) = -1$):
   $$\mathcal{J} = e^{-\alpha} \sum_{y_i = h(x_i)} w_i^{(m)} + e^\alpha \sum_{y_i \neq h(x_i)} w_i^{(m)} = (e^\alpha - e^{-\alpha}) \sum_{i=1}^n w_i^{(m)} \mathbb{I}(y_i \neq h(x_i)) + e^{-\alpha} \sum_{i=1}^n w_i^{(m)}$$
4. Để tìm $\alpha_m$ tối ưu, lấy đạo hàm riêng theo $\alpha$ và cho bằng $0$:
   $$\frac{\partial \mathcal{J}}{\partial \alpha} = e^\alpha \sum_{y_i \neq h} w_i^{(m)} - e^{-\alpha} \sum_{y_i = h} w_i^{(m)} = 0$$
   Đặt tỷ lệ lỗi có trọng số $e_m = \frac{\sum_{y_i \neq h} w_i^{(m)}}{\sum_{i=1}^n w_i^{(m)}}$, phương trình trên trở thành:
   $$e^\alpha e_m - e^{-\alpha} (1 - e_m) = 0 \iff e^{2\alpha} = \frac{1 - e_m}{e_m} \implies \alpha_m = \frac{1}{2} \ln \left( \frac{1 - e_m}{e_m} \right)$$
   Một phép chứng minh giải tích cực kỳ thanh lịch xác nhận rằng công thức trực giác của Freund & Schapire chính là nghiệm tối ưu chính xác của hàm mất mát hàm mũ!

### Mở rộng Đa lớp: Thuật toán SAMME (Zhu et al., 2009)

Trong bài toán đa lớp ($K > 2$), một bộ phân loại đoán mò ngẫu nhiên sẽ có tỷ lệ chính xác là $1/K$, nghĩa là tỷ lệ sai lên tới $\frac{K-1}{K}$. Nếu áp dụng công thức nhị phân cổ điển (đòi hỏi $e_m < 0.5$), thuật toán sẽ bị dừng ngay lập tức khi $K \ge 3$.
Thuật toán **SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss function)** mở rộng công thức trọng số mô hình:
$$\alpha_m = \ln \left( \frac{1 - e_m}{e_m} \right) + \ln(K - 1)$$
Nhờ số hạng $\ln(K - 1)$, thuật toán chỉ yêu cầu weak learner đạt độ chính xác lớn hơn mức ngẫu nhiên $1/K$ ($e_m < \frac{K-1}{K}$) là đã có thể gán trọng số dương $\alpha_m > 0$.

---

## Data used

Dữ liệu bảng classification với nhãn tương đối đáng tin, sau tiền xử lý hợp lệ; Iris có thể minh họa cơ chế, nhưng không đại diện production. Nếu nhãn sai hoặc outlier khó, AdaBoost có thể tiếp tục tăng chú ý vào chính các mẫu nhiễu đó. Với lớp hiếm, chọn metric theo chi phí lỗi và xem lỗi từng lớp; không kết luận accuracy chung là đủ.

---

## Design

| Quyết định | Vai trò | Cần kiểm |
|---|---|---|
| Base estimator/độ sâu cây | Năng lực của mỗi weak learner | Cây quá mạnh tăng chi phí và nguy cơ fit nhiễu |
| `n_estimators` | Số vòng tối đa | Validation curve và latency |
| `learning_rate` | Mức đóng góp của mỗi learner | Tune cùng số vòng; quá lớn có thể kém ổn định |

Chọn trên CV/train; test chỉ dùng sau khi khóa cấu hình. Cần lưu toàn bộ ensemble và preprocessing cho inference. Khi nhãn nhiễu nhiều, thử mô hình có regularization/robustness phù hợp và kiểm các mẫu bị tăng trọng số; không tự cho rằng boosting luôn tốt hơn forest. `AdaBoostClassifier` trong scikit-learn có chi tiết API thay đổi theo phiên bản, nên xem tài liệu của phiên bản đang sử dụng. [1][3]

### Gót chân Achilles của AdaBoost: Độ nhạy Cực đoan với Nhiễu Nhãn

Mặc dù có nền tảng toán học đẹp đẽ, AdaBoost có một nhược điểm chí mạng trong ứng dụng thực tế: **Cực kỳ mẫn cảm với dữ liệu bị nhiễu hoặc nhãn bị gán sai (Label Noise)**.
* *Nguyên nhân:* Do hàm mất mát hàm mũ $L(y, f) = \exp(-y f)$ tăng bùng nổ theo cấp số mũ khi một mẫu bị dự đoán sai nghiêm trọng ($y f \ll 0$).
* Nếu trong tập dữ liệu có một điểm ngoại lai hoặc một hàng bị gán nhãn ngược thực tế, AdaBoost sẽ không ngừng nhân trọng số của điểm này lên gấp bội qua từng vòng lặp.
* Đến các vòng lặp sau, toàn bộ trọng số của tập dữ liệu sẽ bị dồn hết vào một vài điểm nhiễu vô nghĩa, buộc các cây con tiếp theo phải uốn cong ranh giới để khớp bằng được những điểm rác này. Kết quả là mô hình bị Overfitting nặng nề.
* *Giải pháp công nghiệp:* Với dữ liệu thực tế có nhiều nhiễu, các kỹ sư thường chuyển sang sử dụng **Gradient Boosting với hàm Log Loss** hoặc Huber Loss (vốn chỉ tăng tuyến tính chứ không tăng theo hàm mũ khi gặp mẫu sai).

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Weak learner** | Mô hình cơ sở đơn giản, ví dụ cây một tầng; các learner được gộp thành ensemble. | Tạo dự đoán đơn giản để ensemble cải thiện qua nhiều vòng | Mô hình cơ sở được fit ở mỗi vòng AdaBoost |
| **Sample weight $w_i$** | Mức ảnh hưởng của mẫu `i` khi fit learner ở vòng hiện tại. | Làm mẫu khó có ảnh hưởng lớn hơn ở vòng tiếp theo | Trọng số truyền từ vòng trước vào learner hiện tại |
| **Weighted error $e_m$** | Tỷ lệ lỗi của learner sau khi tính theo trọng số mẫu. | Đo lỗi có xét mức quan trọng của từng mẫu | Căn cứ tính hệ số learner trong công thức AdaBoost cổ điển |
| **Learner weight $\alpha _m$** | Mức đóng góp của learner `m` vào dự đoán ensemble trong công thức cổ điển. | Quy định learner nào ảnh hưởng mạnh tới dự đoán chung | Trọng số gộp đầu ra của các learner |
| **Boosting round** | Một vòng fit learner mới và cập nhật trọng số mẫu. | Thêm một learner và cập nhật mức chú ý vào mẫu | Đơn vị lặp của quá trình huấn luyện AdaBoost |
| **Stump** | Cây quyết định chỉ có một phép tách, thường dùng làm weak learner. | Tạo quy tắc phân lớp đơn giản, dễ làm weak learner | Base estimator thường gặp của AdaBoost |
| **Learning rate** | Hệ số điều chỉnh đóng góp của learner theo API, cần tune cùng số vòng. | Co mức đóng góp mỗi vòng để kiểm soát việc học | Hyperparameter tune cùng số learner |
| **SAMME** | Biến thể AdaBoost đa lớp; công thức trọng số khác công thức nhị phân minh họa ở trên. | Mở rộng cách gộp learner cho bài toán nhiều lớp | Biến thể thuật toán cần phân biệt với công thức nhị phân |
| **PAC Learning** | Khung lý thuyết học khả năng đúng xấp xỉ của Valiant (1984). | Đặt nền móng toán học xác suất cho việc chứng minh tính khả thi của Boosting. | Nguồn gốc lý thuyết của giải thuật AdaBoost. |
| **Exponential Loss** | Hàm mất mát dạng $e^{-y f(x)}$ phạt theo cấp số nhân các mẫu dự đoán sai. | Hàm mục tiêu ngầm định được tối thiểu hóa qua từng vòng AdaBoost. | Cơ sở giải tích chứng minh sự hội tụ của thuật toán. |

---

## Nguồn

- [1] [scikit-learn: Ensembles — AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
- [2] [Freund & Schapire: A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://doi.org/10.1006/jcss.1997.1504)
- [3] [scikit-learn: AdaBoostClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [4] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [5] Friedman, J., Hastie, T., & Tibshirani, R. (2000). *Additive logistic regression: a statistical view of boosting*. The Annals of Statistics, 28(2), 337-407.
- [6] Zhu, J., Zou, H., Rosset, S., & Hastie, T. (2009). *Multi-class AdaBoost*. Statistics and Its Interface, 2(3), 349-360.
