# Trong train: tối ưu và chọn mô hình

## Purpose

Ước lượng tham số mô hình từ train, chọn hyperparameter dựa trên validation/cross-validation, và kiểm soát overfitting mà không dùng test để ra quyết định. [1]

### Không gian Giả thuyết & Kiểm soát Năng lực Biểu diễn (Hypothesis Space & Capacity Control)

Trong lý thuyết học thống kê (Statistical Learning Theory), quá trình huấn luyện mô hình thực chất là một bài toán tìm kiếm có hướng trong **Không gian Giả thuyết** $`\mathcal{H}`$ (tập hợp tất cả các hàm toán học mà kiến trúc mô hình có thể biểu diễn).

Kỹ sư học máy phải giải quyết đồng thời hai bài toán tối ưu hóa mang tính đối nghịch:
1. **Khớp tín hiệu (Signal Fitting):** Tối thiểu hóa sai số giữa dự đoán của mô hình và nhãn thực tế trên tập huấn luyện thông qua một thuật toán tối ưu hóa (như Gradient Descent hoặc giải tích ma trận).
2. **Kiểm soát Năng lực (Capacity Control):** Hạn chế "sức chứa" hoặc độ linh hoạt của $`\mathcal{H}`$ để ngăn mô hình ghi nhớ các biến động ngẫu nhiên (nhiễu).

Nguyên lý then chốt của khoa học thực nghiệm: **Tập kiểm tra ($`D_{\text{test}}`$) phải được xem như một "hộp đen niêm phong" tuyệt đối.** Mọi quyết định lựa chọn kiến trúc thuật toán, điều chỉnh siêu tham số và dừng huấn luyện sớm đều phải được thực hiện hoàn toàn trên tập phát triển ($`D_{\text{dev}}`$) thông qua kỹ thuật Kiểm định chéo (Cross-Validation).

---

## Input

Train và validation đã split đúng; pipeline tiền xử lý; objective/loss; metric chính; baseline; ngân sách tính toán; danh sách hyperparameter cần thử.

### Cấu hình Toán học của Bài toán Huấn luyện

Đầu vào kỹ thuật của giai đoạn này là một bài toán quy hoạch toán học tổng quát:

```math
\theta^* = \mathop{\mathrm{arg\,min}}_{\theta \in \Theta} \;\; \mathcal{J}(\theta; D_{\text{train}}) = \mathop{\mathrm{arg\,min}}_{\theta \in \Theta} \left[ \frac{1}{n} \sum_{i=1}^n L(y_i, f_\theta(x_i)) + \lambda \Omega(\theta) \right]
```

Trong đó:
* $`\Theta`$: Không gian tham số có thể đạt được.
* $`L(y, \hat{y})`$: Hàm mất mát khả vi (Differentiable Loss Function) đo lường sai số trên từng điểm dữ liệu.
* $`\Omega(\theta)`$: Số hạng điều chuẩn (Regularizer) phạt độ phức tạp của vector tham số (ví dụ chuẩn $`L_2`$: $`\frac{1}{2}\|\theta\|_2^2`$ hoặc chuẩn $`L_1`$: $`\|\theta\|_1`$).
* $`\lambda \ge 0`$: Siêu tham số điều khiển cường độ điều chuẩn.

---

## Output

Pipeline đã fit, hyperparameter được chọn, lịch sử train/validation, cấu hình tái lập (seed, phiên bản dữ liệu/phần mềm) và một thiết kế đã khóa trước test.

### Thành phần Hồ sơ Nghiệm thu Huấn luyện (Training Artifact Registry)

Một giai đoạn huấn luyện hoàn tất chuẩn mực khoa học phải bàn giao một gói Artifact bao gồm:
1. **Trọng số & Tham số tối ưu ($`\theta^*`$):** Ma trận hệ số, bias, các giá trị split ngưỡng của cây quyết định.
2. **Trạng thái Tiền xử lý (Fitted Transformers):** Mean/std của scaler, từ điển danh mục của encoder, giá trị thay thế của imputer được học riêng trong từng fold.
3. **Ma trận Dự đoán Ngoài Fold (Out-Of-Fold Predictions Matrix):** Toàn bộ dự đoán trên tập train khi từng phần đóng vai trò validation fold, phục vụ kiểm toán hoặc làm đầu vào cho Stacking Ensemble.
4. **Đồ thị Lịch sử Huấn luyện (Learning Trajectory):** Đường cong hội tụ của Training Loss và Validation Metric qua các bước lặp (epochs/iterations).

---

## Evaluation metrics

**Loss** dùng để fit tham số: squared/absolute error cho regression, log loss hoặc hinge loss cho một số classifier. **Metric chọn mô hình** phản ánh mục tiêu thực tế và có thể khác loss; ví dụ recall khi bỏ sót ca dương tính tốn kém. Theo dõi chênh train và validation; so với baseline. [2]

Một cách viết chung: $`\theta^* = \mathop{\mathrm{arg\,min}}_{\theta} \left[\frac{1}{n}\sum_i L(y_i,f_\theta(x_i)) + \lambda R(\theta)\right]`$. `L` là loss, `R` là regularization, $`\lambda`$ điều khiển mức phạt. Công thức này hợp với nhiều mô hình tham số, nhưng kNN và cây học theo cơ chế khác. Metric chọn mô hình có thể không khả vi nên không nhất thiết là loss tối ưu.

### Bản chất của Hàm Mất mát Thay thế (Surrogate Loss Function)

Tại sao không tối ưu trực tiếp Accuracy hay F1-Score trong lúc train?
* Xét hàm lỗi 0-1 (Zero-One Loss) đo tỷ lệ phân loại sai:

  ```math
  L_{0-1}(y, f(x)) = \mathbb{I}(y \neq \mathrm{sign}(f(x)))
  ```

* Hàm này là hàm bậc thang: đạo hàm tại mọi điểm khả vi đều bằng 0, và không liên tục tại điểm phân chia ranh giới. Do đó, các thuật toán tối ưu hóa dựa trên gradient (Gradient-based Optimization) hoàn toàn bất lực.
* **Giải pháp học thuật:** Sử dụng một **hàm mất mát thay thế lồi (Convex Surrogate Loss)** chặn trên hàm 0-1:
  * *Binary Cross-Entropy / Log Loss:* $`L_{\text{logistic}}(y, f) = \log(1 + e^{-y f})`$ (mịn, khả vi vô hạn, tiệm cận xác suất hậu nghiệm).
  * *Hinge Loss (SVM):* $`L_{\text{hinge}}(y, f) = \max(0, 1 - y f)`$ (lồi, liên tục từng khúc, tạo ra nghiệm thưa với các vector hỗ trợ).

```text
 Loss
  ^
  |      \                     /  Hinge Loss: max(0, 1 - y*f)
3 |       \                   /
  |        \                 /
2 |         \               /---- Log Loss: log(1 + exp(-y*f))
  |          \             /
1 |           \___________/
  |            |         |
0 +------------+---------+-----------------> Margin (y * f)
 -2           -1    0    1         2
  |            |_________|
  |                      |-------- Zero-One Error (Bậc thang không khả vi)
```

---

## How it works

1. Chạy baseline, rồi mô hình giải thích được như linear/logistic regression hoặc cây nông.
2. Mỗi cấu hình: fit pipeline trên train hoặc trên fold train; đánh giá fold validation; lặp qua các fold để ước lượng độ ổn định. [1]
3. Tối ưu tham số theo thuật toán: closed-form/solver cho linear models, gradient descent cho logistic/MLP/boosting, chia nút tham lam cho cây, lưu mẫu cho kNN.
4. Chỉnh regularization, độ sâu cây, số lá, `k`, kernel, learning rate hoặc số epoch theo validation. Early stopping theo validation với mô hình lặp; không nhìn test. [1][3]
5. Chọn cấu hình theo metric và chi phí. Nếu cần calibration hoặc chọn threshold, giữ dữ liệu **chưa dùng để fit classifier**, hoặc dùng dự đoán out-of-fold theo quy trình CV. Nếu vừa fit calibrator vừa tune threshold, nên có tập riêng cho từng bước hoặc một quy trình cross-fitting thích hợp để tránh chọn ngưỡng trên cùng dữ liệu đã tối ưu calibrator. Chỉ refit classifier trên train+validation khi không còn dùng chính validation đó như holdout độc lập; test vẫn giữ nguyên cho đánh giá cuối. [5][6]

**Vòng lặp thực nghiệm mẫu:**

```text
for mỗi mô hình và bộ hyperparameter:
    for mỗi split train_fold/validation_fold:
        fit toàn bộ pipeline trên train_fold
        dự đoán validation_fold và tính metric chính
    ghi trung bình, độ phân tán và chi phí của các fold
chọn cấu hình theo metric đã định nghĩa và kiểm lỗi theo nhóm
khóa pipeline, threshold, cách preprocessing; đánh giá test
```

CV chỉ giảm độ phụ thuộc vào một lần chia; các fold không độc lập hoàn toàn. Không xem độ lệch chuẩn giữa fold là khoảng tin cậy chính xác của hiệu năng production. [1]

### Các Phương pháp Tối ưu hóa Tham số (Optimization Paradigms)

Tùy thuộc vào cấu trúc đại số của hàm mục tiêu, thuật toán huấn luyện áp dụng một trong các cơ chế sau:

1. **Nghiệm Đóng Giải tích (Analytical Closed-form Solution):**
   * Tiêu biểu: Hồi quy tuyến tính OLS thông qua phương trình chuẩn (Normal Equation):

     ```math
     \theta^* = (X^\top X)^{-1} X^\top y
     ```

   * *Ưu điểm:* Nghiệm chính xác tuyệt đối, không cần chọn learning rate hay số vòng lặp.
   * *Nhược điểm:* Độ phức tạp tính toán đảo ma trận cỡ $`\mathcal{O}(p^3)`$, sụp đổ khi số chiều $`p`$ lớn hoặc ma trận bị suy biến (khi đó phải dùng Ridge regularization hoặc phân tích SVD).
2. **Quy hoạch Tối ưu Dựa trên Gradient (First-Order Gradient Descent):**
   * Cập nhật tham số theo hướng ngược vector đạo hàm riêng:

     ```math
     \theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{J}(\theta^{(t)})
     ```

   * Biến thể: *Batch GD* (tính trên toàn bộ $`n`$ mẫu), *Stochastic GD (SGD)* (tính trên từng mẫu đơn lẻ), *Mini-batch GD* (tính trên lô nhỏ kích thước $`32, 64, 128`$).
3. **Phân chia Nhánh Tham lam (Greedy Recursive Partitioning):**
   * Áp dụng cho Decision Trees: Tại mỗi nút, duyệt vét cạn toàn bộ các feature và điểm chia tiềm năng để chọn điểm chia tối đa hóa mức giảm độ tinh khiết (Information Gain hoặc Gini Reduction).
4. **Ghi nhớ Mẫu (Instance-based / Lazy Learning):**
   * Áp dụng cho KNN: Giai đoạn "train" chỉ đơn thuần là nạp toàn bộ vector dữ liệu vào cấu trúc chỉ mục không gian (như KD-Tree hoặc Ball-Tree). Mọi chi phí tính toán bị dồn toàn bộ sang giai đoạn inference.

### Bản chất của Nested Cross-Validation (Kiểm định Chéo Lồng nhau)

Khi dữ liệu có kích thước hạn chế nhưng cần thực hiện cả việc tinh chỉnh siêu tham số lẫn ước lượng sai số tổng quát hóa một cách không thiên lệch:

* **Vòng ngoài (Outer Loop - $`K_{\text{outer}}`$ folds):** Dùng duy nhất để đánh giá hiệu năng tổng quát hóa của toàn bộ quy trình.
* **Vòng trong (Inner Loop - $`K_{\text{inner}}`$ folds):** Được thực thi bên trong mỗi fold train của vòng ngoài để tìm ra bộ siêu tham số $`\theta^*`$ tốt nhất.

```text
[ Toàn bộ Dữ liệu Nghiên cứu ]
 |
 +--> Outer Fold 1: [Train + Val] -----------> Outer Test 1 (Đo lường độc lập)
 |      |
 |      +--> Inner Loop CV (Tune hyperparameter C, gamma, depth)
 |      +--> Chọn được bộ tham số tốt nhất cho Fold 1
 |
 +--> Outer Fold 2: [Train + Val] -----------> Outer Test 2 (Đo lường độc lập)
 |      |
 |      +--> Inner Loop CV (Tune hyperparameter C, gamma, depth)
 |      +--> Chọn được bộ tham số tốt nhất cho Fold 2
```

Nested CV loại bỏ hoàn toàn hiện tượng "rò rỉ lựa chọn" (*Selection Bias*), đem lại ước lượng thống kê trung thực nhất cho báo cáo khoa học.

---

## Data used

Train dùng để học trọng số/quy tắc; validation hoặc các fold dùng để chọn cấu hình. Test chỉ dùng sau khi toàn bộ lựa chọn kết thúc. Nếu dữ liệu có group/time, cross-validation cũng phải giữ cấu trúc đó. [1]

---

## Design

- Ghi rõ search space và ngân sách; tìm ít tham số có ý nghĩa trước. Nested CV hữu ích khi cần ước lượng ít thiên lệch sau khi tuning trên tập nhỏ. [1]
- Kiểm tra learning curve và chênh train/validation để phân biệt underfitting với overfitting.
- Regularization hoặc giảm độ phức tạp khi overfit; cải thiện feature/nhãn hoặc tăng năng lực mô hình khi underfit.
- Failure modes: tối ưu sai metric, tuning quá nhiều trên một validation set, fit preprocessing ngoài fold, seed không được lưu.

**Chẩn đoán:** train kém và validation kém có thể là underfitting hoặc dữ liệu/nhãn lỗi; train tốt nhưng validation kém gợi ý overfitting, leakage ở train, hoặc lệch phân phối. Trước khi tăng độ phức tạp, kiểm baseline và chất lượng nhãn. Learning curve có thể cho biết liệu thêm dữ liệu hay tăng năng lực mô hình có triển vọng. [4]

### Sổ tay Chẩn đoán Đường cong Học tập (Learning Curves Diagnostic Playbook)

Đường cong học tập biểu diễn sai số (Loss) của tập Train và tập Validation theo số lượng mẫu huấn luyện $`n`$ hoặc số epoch. Đây là công cụ chẩn đoán quan trọng nhất để định hướng can thiệp kỹ thuật:

| Triệu chứng đường cong | Nguyên nhân gốc rễ (Root Cause) | Hành động khắc phục khoa học |
|---|---|---|
| **Cả Train Loss và Val Loss đều cao, hội tụ gần nhau ở mức sai số lớn.** | **High Bias (Underfitting):** Mô hình quá nông hoặc giả định quá cứng nhắc, không hấp thụ được tín hiệu. | 1. Tăng năng lực mô hình (thêm layer, tăng độ sâu cây).<br>2. Kỹ nghệ thêm đặc trưng phi tuyến ($`x_1 x_2, x^2`$).<br>3. Giảm bớt hệ số phạt điều chuẩn $`\lambda`$. |
| **Train Loss rất thấp, nhưng Val Loss cao, khoảng cách giữa 2 đường lớn (Gap rộng).** | **High Variance (Overfitting):** Mô hình ghi nhớ cả nhiễu ngẫu nhiên của tập train. | 1. Thu thập thêm dữ liệu huấn luyện mới.<br>2. Tăng cường điều chuẩn (L1/L2 penalty, Dropout).<br>3. Giảm số lượng biến (Feature Selection).<br>4. Áp dụng Early Stopping. |
| **Val Loss dao động dữ dội, không ổn định qua các epoch.** | **Tốc độ học quá lớn hoặc Batch Size quá nhỏ.** | 1. Giảm Learning Rate hoặc dùng Learning Rate Scheduler.<br>2. Tăng kích thước Mini-batch.<br>3. Kiểm tra lại nhãn bị nhiễu/lộn xộn. |
| **Train Loss cao hơn hẳn Val Loss.** | **Kỹ thuật Regularization hoặc Dropout chỉ kích hoạt lúc train.** | Hiện tượng bình thường nếu dùng Dropout hoặc Data Augmentation lúc train (vốn bị tắt khi evaluate trên Val). |

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Convex Optimization** | Bài toán tối ưu hóa với hàm mục tiêu có duy nhất một cực tiểu toàn cục. | Đảm bảo thuật toán luôn hội tụ về nghiệm tối ưu toàn cục. | Nền tảng của Linear Regression, Ridge, Lasso, Logistic Regression, SVM. |
| **Non-Convex Optimization** | Bài toán có nhiều cực tiểu cục bộ (Local Minima) và điểm yên ngựa (Saddle Points). | Kết quả phụ thuộc vào điểm khởi tạo tham số. | Đặc trưng cốt lõi của Deep Learning và Neural Networks. |
| **Early Stopping** | Dừng huấn luyện khi sai số trên validation bắt đầu tăng liên tiếp $`k`$ bước. | Ngăn chặn mô hình tiếp tục học vào vùng overfit. | Cơ chế điều chuẩn ngầm cực kỳ hiệu quả cho mô hình lặp. |
| **Out-Of-Fold (OOF)** | Dự đoán trên các fold bị giữ lại trong suốt quá trình $`K`$-Fold CV. | Tạo ra bộ đặc trưng meta không bị thiên lệch cho Stacking. | Cầu nối giữa các mô hình cơ sở và mô hình tổng hợp (Ensemble). |
| **Grid Search** | Duyệt vét cạn qua toàn bộ lưới các giá trị siêu tham số rời rạc định trước. | Tìm được cấu hình tối ưu trong không gian lưới nhỏ. | Đảm bảo bao phủ toàn diện nhưng tốn kém tài nguyên. |
| **Bayesian Optimization** | Sử dụng mô hình xác suất (Gaussian Process/TPE) định hướng điểm thử tiếp theo. | Tìm siêu tham số tối ưu với số lần thử nghiệm ít nhất. | Chuẩn mực công nghiệp hiện đại (thư viện Optuna). |

---

## Nguồn

- [1] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [2] [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3] [scikit-learn: Grid search](https://scikit-learn.org/stable/modules/grid_search.html)
- [4] [scikit-learn: Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- [5] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [6] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [7] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Chương 5: Machine Learning Basics & Chương 8: Optimization).
- [8] Vapnik, V. N. (1998). *Statistical Learning Theory*. John Wiley & Sons.
