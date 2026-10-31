# Random Forest và Gradient Boosting

## Purpose

Kết hợp nhiều cây để tăng chất lượng trên dữ liệu bảng. Random Forest giảm phương sai bằng nhiều cây độc lập tương đối; Gradient Boosting xây cây nối tiếp để sửa sai của mô hình hiện tại. [1][2][3]

### Triết lý Học Tập Hợp (Ensemble Learning) & Định lý Bồi thẩm đoàn Condorcet

Trong khoa học chính trị và thống kê toán học, **Định lý Bồi thẩm đoàn Condorcet (1785)** khẳng định: Nếu một nhóm thẩm phán cùng bỏ phiếu độc lập về một vụ án, và mỗi cá nhân có xác suất đưa ra quyết định đúng $`p > 0.5`$ (tốt hơn đoán mò ngẫu nhiên), thì khi số lượng thẩm phán tham gia hội đồng tiến ra vô cùng, xác suất quyết định chung của tập thể đúng sẽ tiệm cận tuyệt đối về $`100\%`$.

Học Tập hợp (Ensemble Learning) đưa nguyên lý này vào khoa học máy tính: Thay vì tìm kiếm một "siêu mô hình" hoàn hảo duy nhất, ta huấn luyện một tập hợp các mô hình thành phần (Base Learners) rồi gộp kết quả lại. Hai nhánh phát triển rực rỡ nhất là:

1. **Bagging (Bootstrap Aggregating - Leo Breiman, 1996):**
   * *Chiến lược:* Xây dựng các cây quyết định sâu, có năng lực biểu diễn cao (**High Variance, Low Bias**) một cách **song song và độc lập** trên các tập con dữ liệu ngẫu nhiên.
   * *Mục tiêu tối thượng:* **Giảm Phương sai (Variance Reduction)** mà không làm tăng Thiên lệch. Đỉnh cao của nhánh này là **Random Forest (2001)**.
2. **Boosting (Jerome Friedman, 2001):**
   * *Chiến lược:* Bắt đầu bằng một mô hình rất đơn giản, thậm chí yếu (**High Bias, Low Variance**), sau đó xây dựng các cây tiếp theo một cách **tuần tự (Sequential)**, mỗi cây tập trung sửa chữa các lỗi sai của toàn bộ tổ hợp trước đó.
   * *Mục tiêu tối thượng:* **Giảm Thiên lệch (Bias Reduction)** để biến các mô hình yếu thành một mô hình cực mạnh. Đỉnh cao là **Gradient Boosting (GBM)**.

---

## Input

`X` dữ liệu bảng đã xử lý theo yêu cầu implementation; `y` lớp hoặc số; metric chính, ngân sách CPU/RAM và thời gian inference.

Phân biệt mục tiêu: forest chủ yếu giảm phương sai của cây biến động mạnh; boosting tăng năng lực bằng cách cộng các cây nhỏ theo hướng giảm loss. Hai mô hình cần cùng split và metric để so sánh, nhưng search space khác nhau. [1][2][3]

---

## Output

Random Forest: regressor lấy trung bình dự đoán các cây; `RandomForestClassifier` của scikit-learn lấy lớp có **xác suất trung bình qua các cây** cao nhất, nên có thể khác majority vote. Gradient Boosting cộng đóng góp các cây theo learning rate; classifier tạo score rồi chuyển sang xác suất theo loss/implementation. Lưu cả preprocessing và cấu hình. [1][6]

---

## Evaluation metrics

Classification: F1, PR-AUC/ROC-AUC, log loss/calibration tùy mục tiêu. Regression: MAE/RMSE/R². So với một cây và baseline; đánh giá latency, bộ nhớ và độ ổn định theo nhóm. [4]

---

## How it works

- **Random Forest:** với cấu hình bootstrap thường dùng, mỗi cây học trên mẫu rút có hoàn lại; ở mỗi lần tách xem một tập con feature ngẫu nhiên; gộp kết quả nhiều cây. Một số implementation cho `bootstrap=False`, khi đó không dùng OOB theo cách mô tả bên dưới. [1][2]
- **Gradient Boosting:** thêm cây nhỏ theo từng vòng để giảm objective/loss của ensemble, có thể dùng learning rate và early stopping để hạn chế overfit. [1][3]

**Ví dụ gộp:** ba cây regression dự đoán `10,12,17` thì forest lấy trung bình `13`. Với classifier, nếu xác suất lớp A từ ba cây là `0.9,0.4,0.4`, trung bình là `0.567`, nên lớp A thắng ở bài toán nhị phân mặc dù chỉ một cây có $`p(A)>0.5`$. Ví dụ này cho thấy không nên diễn đạt implementation scikit-learn đơn thuần là majority vote. [6]

**Boosting trực giác:** bắt đầu với dự đoán hằng $`F_{0}`$; ở vòng `m`, fit cây nhỏ theo hướng giảm loss hiện tại rồi cập nhật $`F_m(x)=F_{m-1}(x)+\eta h_m(x)`$. Với squared error, hướng giảm loss liên quan residual $`y-F_{m-1}(x)`$; với loss khác, pseudo-residual là gradient âm tương ứng. [1][3]

### Random Forest: bagging và OOB

Khi `bootstrap=True`, mỗi cây nhận một bootstrap sample rút **có hoàn lại** từ train. Với một mẫu train `i`, các cây không chứa `i` trong bootstrap của chúng có thể dự đoán mẫu đó; gộp các dự đoán này tạo **out-of-bag (OOB) estimate**. OOB hữu ích để kiểm nhanh mức lỗi và ảnh hưởng của số cây, nhưng không thay một test độc lập, đặc biệt khi dữ liệu có nhóm hoặc thời gian. [1][2][7]

```text
for b = 1..B:
    lấy bootstrap sample từ train
    fit cây, mỗi split chỉ xét một tập con feature
predict regression: trung bình dự đoán B cây
predict classification (scikit-learn): lấy lớp có xác suất trung bình cao nhất
```

Bagging làm giảm độ biến động nếu lỗi của các cây không hoàn toàn giống nhau. Random feature selection giúp giảm tương quan giữa cây; cây đơn vẫn có thể overfit, nhưng trung bình nhiều cây thường ổn định hơn. Đây là trực giác, không là bảo đảm forest luôn tốt hơn cây đơn trên mọi dữ liệu. [1][2]

#### Chứng minh Toán học: Tại sao Phải Làm Giảm Tương quan Giữa các Cây?

Xét một tập hợp $`B`$ cây quyết định ngẫu nhiên $`T_1(x), T_2(x), \dots, T_B(x)`$. Giả sử mỗi cây có phương sai dự đoán là $`\sigma^2`$ và hệ số tương quan giữa hai cây bất kỳ là $`\rho = \mathrm{Corr}(T_i, T_j)`$.
Dự đoán trung bình của rừng là $`\bar{T}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)`$.

Phương sai của mô hình tổ hợp được tính chính xác bằng công thức thống kê:

```math
\mathrm{Var}(\bar{T}) = \frac{1}{B^2} \left[ \sum_{b=1}^B \mathrm{Var}(T_b) + \sum_{i \neq j} \mathrm{Cov}(T_i, T_j) \right] = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2
```

* **Phân tích giới hạn tiệm cận:**
  Khi số lượng cây $`B \to \infty`$, số hạng $`\frac{1 - \rho}{B} \sigma^2 \to 0`$. Tuy nhiên, tổng phương sai **tiến về cận dưới cố định là $`\rho \sigma^2`$**!
  * Nếu các cây giống nhau hoàn toàn ($`\rho = 1`$): Việc gộp $`1{,}000`$ cây cũng không làm giảm được dù chỉ $`1\%`$ phương sai!
  * **Đột phá của Leo Breiman (2001):** Để đẩy phương sai xuống mức tối thiểu, **bắt buộc phải làm giảm hệ số tương quan $`\rho`$**. Breiman đưa ra kỹ thuật **Random Feature Subspace Selection**: Tại mỗi nút phân nhánh, chỉ cho phép thuật toán chọn từ một tập con ngẫu nhiên gồm $`m`$ đặc trưng:

    ```math
    m = \lfloor \sqrt{p} \rfloor \quad (\text{cho phân loại}) \quad \text{hoặc} \quad m = \lfloor p / 3 \rfloor \quad (\text{cho hồi quy})
    ```

    Nhờ đó, ngay cả khi dữ liệu có một biến chi phối cực mạnh, các cây vẫn bị ép phải khám phá các biến yếu hơn, làm các cây con trở nên đa dạng và độc lập, triệt tiêu mạnh mẽ $`\rho \sigma^2`$.

### Gradient Boosting: loss và cập nhật tuần tự

Với regression dùng squared error, bắt đầu $`F_{0}`$ bằng một giá trị hằng hợp lý, thường là trung bình target theo loss. Tại mỗi vòng, tính pseudo-residual từ gradient âm của loss tại dự đoán hiện tại, fit cây $`h_m`$ vào tín hiệu đó, rồi cập nhật với learning rate $`\eta`$. Với classification log loss, cây cập nhật score/logit; xác suất được tạo từ score theo liên kết của loss. Công thức chính xác có thể khác theo implementation/loss, nên không diễn giải boosting classification như cộng trực tiếp nhãn lớp. [1][3]

**Ví dụ squared error:** $`y=[2,4]`$, dự đoán ban đầu $`F_{0}=[3,3]`$, residual `[-1,+1]`. Giả sử cây đầu tiên dự đoán đúng residual này và $`\eta =0.5`$, mô hình mới cho `[2.5,3.5]`; lỗi bình phương từng điểm giảm từ `1` xuống `0.25`. Ví dụ nhằm giải thích một bước fit, không là benchmark.

**Early stopping:** dành một validation fold từ phần phát triển hoặc dùng CV; theo dõi loss/metric sau mỗi vòng và dừng khi không còn cải thiện theo quy tắc đã đặt. Không dùng test để chọn `n_estimators`. `subsample<1` tạo stochastic gradient boosting; OOB improvement của nó là chỉ báo khác với OOB score của forest và cần đọc theo đúng estimator. [1][8]

#### Bản chất Toán học: Hạ dốc Gradient trong Không gian Hàm (Functional Gradient Descent)

Jerome Friedman (2001) đã chứng minh rằng Gradient Boosting chính là giải thuật Gradient Descent được nâng cấp từ không gian tham số số thực $`\mathbb{R}^p`$ lên **Không gian Hàm vô hạn chiều (Function Space)**:

Hàm mục tiêu cần tối thiểu hóa:

```math
\mathcal{J}(F) = \sum_{i=1}^n L(y_i, F(x_i))
```

Coi vector dự đoán $`[F(x_1), F(x_2), \dots, F(x_n)]^\top \in \mathbb{R}^n`$ như các biến số cần tối ưu. Theo giải tích, hướng giảm nhanh nhất của hàm mất mát chính là hướng của **vector gradient âm (Negative Gradient)**.

1. **Tính toán Pseudo-Residuals tại vòng $`m`$:**

   ```math
   r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
   ```

   * *Với hàm Squared Error Loss $`L = \frac{1}{2}(y - F)^2`$:* $`r_{im} = y_i - F_{m-1}(x_i)`$ (chính là phần dư số học).
   * *Với hàm Log Loss (Logistic) $`L = - [y \ln p + (1-y) \ln(1-p)]`$:* $`r_{im} = y_i - p_i`$ (chênh lệch xác suất).
2. **Khớp cây hồi quy cơ sở:** Huấn luyện cây $`h_m(x)`$ nhận đầu vào là $`X`$ để dự đoán mục tiêu là vector gradient âm $`r_{im}`$.
3. **Cập nhật Hàm Ensemble:**

   ```math
   F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)
   ```

   Hệ số co nhỏ $`\eta \in (0, 1]`$ (Learning Rate / Shrinkage) đóng vai trò điều chuẩn, làm chậm tốc độ học để ngăn ngừa hiện tượng overfit.

---

## Data used

Phù hợp nhiều bài toán bảng, ví dụ UCI Adult và California Housing. Feature số/categorical cần biến đổi theo estimator cụ thể. Chỉ fit imputation/encoding trong train fold; tree ensemble không tự giải quyết mọi lỗi nhãn hay lệch phân phối. [1][5]

---

## Design

- Random Forest: tuning `n_estimators`, độ sâu, `min_samples_leaf`, số feature xét ở mỗi split; tăng số cây thường tăng chi phí inference.
- Gradient Boosting: tuning learning rate, số vòng, số lá/độ sâu và regularization; learning rate nhỏ thường cần nhiều vòng hơn.
- Ưu điểm: bắt quan hệ phi tuyến/tương tác, thường mạnh với bảng. Hạn chế: giải thích khó hơn cây đơn, có thể tốn thời gian; feature importance không chứng minh nhân quả.
- Failure modes: overfit do cây quá sâu/boost quá lâu, train-serving skew, dùng test để early stop, calibration kém.

| Họ mô hình | Hyperparameter chính | Tác động thường gặp |
|---|---|---|
| Forest | `n_estimators` | Nhiều cây giảm dao động nhưng tăng RAM/latency |
| Forest | `max_features` | Ít feature mỗi split tăng đa dạng cây nhưng có thể làm cây đơn yếu hơn |
| Forest | `max_depth`, `min_samples_leaf` | Kiểm soát độ chi tiết và phương sai |
| Boosting | `learning_rate`, `n_estimators` | Learning rate nhỏ thường cần nhiều vòng; phải tune cùng nhau |
| Boosting | độ sâu/số lá, `min_samples_leaf` | Cây cơ sở phức tạp hơn học tương tác mạnh hơn, dễ overfit hơn |
| Boosting | `subsample` | Lấy mẫu từng vòng, đổi bias/variance và chi phí |

Trên dữ liệu bảng lớn, có thể xem các biến thể histogram-based trong scikit-learn; chúng không giống hoàn toàn `GradientBoostingClassifier` cổ điển về hỗ trợ feature và hyperparameter. Kiểm tài liệu của estimator được chọn trước khi áp dụng một công thức tuning chung. [1]

**Chi phí:** forest có thể fit cây song song; boosting thường xây tuần tự nên thời gian fit tăng theo số vòng. Inference phải đi qua nhiều cây, nên đo latency theo kích thước mô hình. Không kết luận boosting luôn hơn forest: kiểm trên validation cùng feature, split và ngân sách.

### So sánh Toàn diện: Random Forest vs. Gradient Boosting

| Tiêu chí kỹ thuật | Random Forest | Gradient Boosting (GBM / XGBoost / LightGBM) |
|---|---|---|
| **Cơ chế cốt lõi** | Bagging (Song song, độc lập). | Boosting (Tuần tự, phụ thuộc). |
| **Mục tiêu tối ưu** | Giảm thiểu Variance. | Giảm thiểu Bias. |
| **Độ sâu cây cơ sở** | Cây sâu tối đa (`max_depth = None` hoặc lớn). | Cây rất nông (`max_depth = 3-6` hoặc `max_leaves = 31`). |
| **Rủi ro Overfitting** | Không bị overfit khi tăng số cây $`B`$ (chỉ hội tụ). | Rất dễ overfit nếu boost quá nhiều vòng $`\implies`$ Bắt buộc Early Stopping. |
| **Khả năng song song hóa** | Cực kỳ xuất sắc (Huấn luyện $`B`$ cây hoàn toàn độc lập). | Khó song song hóa theo chuỗi cây (chỉ song song ở phép tìm split nút). |
| **Độ nhạy với Outlier/Nhiễu** | Rất bền bỉ (Robust) nhờ cơ chế bỏ phiếu trung bình. | Rất nhạy cảm: các vòng sau sẽ tập trung học vào các điểm nhiễu. |
| **Hiệu năng thực tế** | Hoạt động tốt ngay với siêu tham số mặc định. | Cần tinh chỉnh siêu tham số kỹ lưỡng để đạt đỉnh cao (SOTA dữ liệu bảng). |

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Ensemble** | Mô hình gộp nhiều mô hình cơ sở để tạo dự đoán cuối. | Gộp nhiều dự đoán để giảm lỗi so với một mô hình đơn | Khung chung của forest và boosting |
| **Bagging** | Fit nhiều mô hình trên các mẫu dữ liệu rút có hoàn lại rồi gộp kết quả. | Giảm phương sai bằng các mô hình fit song song trên mẫu khác nhau | Cơ chế tạo đa dạng giữa các cây trong random forest |
| **Bootstrap sample** | Mẫu rút từ train **có hoàn lại**, nên một hàng có thể xuất hiện nhiều lần. | Tạo dữ liệu train khác nhau cho từng cây | Bước lấy mẫu khi huấn luyện bagging và forest |
| **OOB / out-of-bag** | Các hàng không có trong bootstrap của một cây; dùng dự đoán của các cây bỏ qua hàng đó để ước lượng lỗi. | Ước lượng chất lượng bằng mẫu cây chưa nhìn thấy | Tín hiệu đánh giá nội bộ của forest; không thay test cuối |
| **Random feature selection** | Mỗi split của forest chỉ xem một tập con feature để giảm tương quan giữa cây. | Làm các cây bớt giống nhau | Bước chọn candidate feature tại mỗi split trong forest |
| **Boosting** | Xây mô hình tuần tự, mỗi bước điều chỉnh dự đoán ensemble đang có. | Sửa dần sai số của ensemble qua các vòng | Cơ chế huấn luyện tuần tự của gradient boosting |
| **Pseudo-residual** | Hướng giảm loss tại dự đoán hiện tại; với squared error liên quan $`y-\hat{y}`$. | Chỉ hướng cần chỉnh dự đoán để giảm loss | Target tạm cho cây tiếp theo trong gradient boosting |
| **Learning rate $`\eta`$** | Hệ số thu nhỏ đóng góp của cây mới trong boosting. | Giảm bước đóng góp mỗi cây, thường cần nhiều cây hơn | Hyperparameter đi cùng số cây khi tune boosting |
| **Early stopping** | Dừng thêm cây khi metric trên validation không còn cải thiện theo quy tắc đã định. | Ngăn thêm cây sau khi validation ngừng cải thiện | Quy tắc chọn số vòng boosting bằng dữ liệu validation |
| **Functional Gradient Descent** | Giải thuật hạ dốc trong không gian hàm số thay vì không gian tham số. | Đặt nền tảng toán học giải tích chặt chẽ cho Gradient Boosting. | Đột phá lý thuyết của Jerome Friedman (2001). |

---

## Nguồn

- [1] [scikit-learn: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html)
- [2] [Breiman: Random Forests](https://doi.org/10.1023/A:1010933404324)
- [3] [Friedman: Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451)
- [4] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [5] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [6] [scikit-learn: RandomForestClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [7] [scikit-learn: OOB errors for Random Forest](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)
- [8] [scikit-learn: Gradient Boosting OOB estimates](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html)
- [9] Breiman, L. (1996). *Bagging predictors*. Machine Learning, 24(2), 123-140.
- [10] Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
