# Chương 2 — Học giám sát (Supervised Learning)

> Bản ghi chú tiếng Việt dựa trên [ML_Chapter2_new.pdf](../Tài liệu học tập/ML_Chapter2_new.pdf) (58 trang), bài giảng của TS. Nguyễn Thị Khánh Tiên. Số trang là số trang PDF. Nội dung được hệ thống hóa và diễn giải lại; các lưu ý hiệu đính ở cuối giúp tránh hiểu sai một số ví dụ trong slide.

## 1. Mục tiêu và đầu vào/đầu ra (tr. 2–6)

Học giám sát dùng các cặp dữ liệu đã gán nhãn `(xᵢ, yᵢ)` để học một hàm `f(x)` có thể dự đoán cho `x` chưa thấy. `x` là vector feature, `y` là nhãn hoặc giá trị đích. Chất lượng phải đo trên dữ liệu chưa dùng để học và chọn mô hình; điểm train đẹp chỉ cho biết mô hình khớp các mẫu đã thấy.

| Bài toán | Target `y` | Output của mô hình | Ví dụ | Metric thường dùng |
|---|---|---|---|---|
| Classification | Lớp rời rạc | Nhãn, score hoặc xác suất lớp | Spam/không spam; loài hoa Iris | Accuracy, precision, recall, F1, AUC |
| Regression | Giá trị số liên tục | Giá trị dự đoán | Giá nhà, doanh số | MAE, MSE, RMSE, R² |

**Chuỗi cơ bản:** thu thập và gán nhãn → kiểm/làm sạch dữ liệu → chia train/validation/test → fit tiền xử lý và mô hình → chọn cấu hình trên validation/CV → đánh giá test → triển khai và theo dõi. Slide tr. 4 mô tả từng bước; khi thực hiện cần bảo đảm phép biến đổi có học tham số chỉ nhìn train. Classification không chỉ có hai lớp: binary và multiclass được giải thích ở tr. 17.

**Ưu điểm** (tr. 5): có mục tiêu rõ nhờ nhãn, dễ đo sai số trên tập giữ riêng, dùng được cho nhiều dự đoán thực tế. **Hạn chế:** nhãn có thể đắt hoặc sai; dữ liệu lịch sử có thể chứa bias; mô hình có thể overfit hoặc kém khi phân phối mới khác dữ liệu train.

## 2. Chia dữ liệu và vòng train đúng thứ tự (tr. 4, 11–16)

- **Train:** học trọng số, thống kê tiền xử lý, cấu trúc cây hoặc mẫu tham chiếu của kNN.
- **Validation/CV:** so mô hình, chọn hyperparameter và ngưỡng quyết định. Trong `k`-fold CV, mỗi vòng fit trên `k-1` fold và kiểm trên fold còn lại; mọi preprocessing phải được fit lại bên trong từng vòng.
- **Test:** phép kiểm cuối sau khi đã khóa toàn bộ lựa chọn. Không dùng test để quyết định `k`, `C`, feature, threshold hoặc chọn lại thuật toán.

Với dữ liệu có nhiều hàng của cùng người dùng/bệnh nhân, cần split theo group nếu mục tiêu là dự đoán cho người mới. Với chuỗi thời gian, train phải nằm trước validation/test. Tỷ lệ 70/30 trong ví dụ Iris ở slide là một minh họa, không phải quy tắc chung.

Các sơ đồ scikit-learn ở tr. 7–10 gom API theo chức năng: `train_test_split`/CV để chia và chọn mô hình, transformer như `StandardScaler` để tiền xử lý, estimator có `fit`/`predict` để học và dự đoán, `sklearn.metrics` để đánh giá. `fit` học tham số từ dữ liệu; `transform` áp dụng phép biến đổi đã học; `predict` tạo dự đoán. Bộ API thống nhất giúp thử nhiều thuật toán trong cùng một quy trình.

### Ví dụ Iris trong bài giảng (tr. 11–16)

- **Input:** 4 phép đo của hoa Iris (chiều dài/rộng đài hoa và cánh hoa).
- **Target:** một trong 3 loài `setosa`, `versicolor`, `virginica`.
- **Quy trình:** `load_iris` → split → `StandardScaler` fit trên train → `LogisticRegression` fit trên train → dự đoán và báo cáo confusion matrix/metric trên test.
- **Tuning:** thử các giá trị `C` bằng `GridSearchCV` **chỉ trên train**, rồi dùng estimator tốt nhất để kiểm test. Slide trình bày bước đánh giá test trước bước tuning để minh họa công cụ; trong thí nghiệm thật nên tuning trước và chỉ xem test khi đã khóa lựa chọn.

Đoạn code sau thể hiện thứ tự an toàn. Đây là ví dụ học tập, không có điểm số thực nghiệm được giả định trước:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid={"model__C": [0.1, 1.0, 10.0]},
    scoring="accuracy",
    cv=cv,
)
search.fit(X_train, y_train)

pred = search.predict(X_test)
print("C được chọn trên train/CV:", search.best_params_)
print("Accuracy test:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=load_iris().target_names))
```

`Pipeline` giúp scaler được fit trong từng fold train của `GridSearchCV`. Dùng `stratify=y` để giữ tỷ lệ lớp gần nhau trong split. Code không dùng `solver='liblinear'` của slide cho ví dụ multiclass vì khả năng hỗ trợ multiclass phụ thuộc solver và phiên bản scikit-learn.

## 3. Đánh giá classification (tr. 17–20)

Với một lớp dương được xác định trước, confusion matrix gồm:

| | Dự đoán dương | Dự đoán âm |
|---|---:|---:|
| Thật dương | TP | FN |
| Thật âm | FP | TN |

| Metric | Công thức | Trả lời câu hỏi |
|---|---|---|
| Accuracy | `(TP+TN)/(TP+FP+FN+TN)` | Tỷ lệ đúng chung là bao nhiêu? |
| Precision | `TP/(TP+FP)` | Trong các dự đoán dương, bao nhiêu là đúng? |
| Recall / TPR | `TP/(TP+FN)` | Trong các mẫu dương thật, tìm được bao nhiêu? |
| F1 | `2PR/(P+R)` với `P=precision`, `R=recall` | Cân bằng precision và recall theo trung bình điều hòa |
| Specificity / TNR | `TN/(TN+FP)` | Trong các mẫu âm thật, loại đúng bao nhiêu? |
| FPR | `FP/(FP+TN)` | Tỷ lệ báo động giả trên các mẫu âm là bao nhiêu? |

**Ví dụ:** nếu `TP=30`, `FP=10`, `FN=20`, `TN=940`, accuracy là 97% nhưng recall dương chỉ 60%. Vì vậy accuracy cao không đủ khi lớp dương hiếm hoặc bỏ sót rất tốn kém. Với multiclass như Iris, xem metric của từng lớp và macro average, không chỉ accuracy tổng.

**ROC curve** vẽ TPR theo FPR khi thay ngưỡng; ROC-AUC đo khả năng xếp hạng trên nhiều ngưỡng. Nó không chọn giúp ngưỡng vận hành. Khi lớp dương hiếm, nên xem thêm precision–recall curve và metric tại ngưỡng được chọn. F1 không tự động là lựa chọn đúng cho mọi bài toán: cần xét chi phí FP/FN và khả năng xử lý dự đoán dương.

Slide hình ở tr. 18 còn nhắc **Fβ-score**: `Fβ=(1+β²)PR/(β²P+R)`. Khi `β>1`, recall được nhấn mạnh hơn; khi `β<1`, precision được nhấn mạnh hơn. Đây vẫn là một metric thay thế chi phí thực tế, không tự xác định ngưỡng tốt nhất. Nếu mẫu số của precision/recall bằng 0, phải ghi quy ước xử lý.

## 4. Các thuật toán classification (tr. 21–41)

### 4.1. K-Nearest Neighbors — kNN (tr. 22–25)

**Ý tưởng:** lưu tập train; với mẫu mới, tính khoảng cách tới các mẫu train, lấy `k` láng giềng gần nhất rồi gộp nhãn. Với feature số, khoảng cách Euclidean là `d(x,z)=√Σⱼ(xⱼ-zⱼ)²`; Manhattan là `Σⱼ|xⱼ-zⱼ|`. `k` nhỏ dễ nhạy nhiễu, `k` lớn dễ làm mượt quá mức.

**Input/output:** vector feature đã có thang đo hợp lý; đầu ra là lớp từ đa số láng giềng hoặc xác suất dựa trên tần suất/trọng số. Cần scale trong pipeline vì feature có đơn vị lớn có thể lấn át khoảng cách. **Trade-off:** fit nhanh, dễ hiểu nhưng suy luận tốn hơn khi tập train lớn và khoảng cách ít ý nghĩa ở chiều cao. Chọn `k`, metric khoảng cách và weighting bằng CV.

### 4.2. Logistic Regression (tr. 26–28)

**Ý tưởng nhị phân:** tính `z=b+wᵀx`, sau đó `p(y=1|x)=1/(1+e^(-z))`; chọn lớp bằng một threshold. Train thường tối thiểu hóa log loss có regularization. Trong scikit-learn, `C` nhỏ tương ứng regularization mạnh hơn.

**Input/output:** feature số/đã encode, nhãn lớp; đầu ra gồm score, xác suất ước lượng và nhãn. **Trade-off:** baseline nhanh, hệ số có thể diễn giải theo log-odds, nhưng ranh giới tuyến tính nếu không tạo thêm feature. Cần kiểm calibration nếu xác suất dùng để ra quyết định; threshold 0.5 không phải luôn tối ưu.

### 4.3. Support Vector Machine — SVM (tr. 29–34)

**Ý tưởng:** tìm ranh giới có margin lớn, cho phép một số vi phạm qua tham số `C`. Các điểm ảnh hưởng đến ranh giới là support vectors. Kernel tuyến tính, đa thức hoặc RBF tạo các ranh giới khác nhau; với kernel RBF, `gamma` ảnh hưởng độ cục bộ của ranh giới.

**Input/output:** vector số cần scale; classifier trả nhãn/decision score, không mặc nhiên là xác suất đáng tin. **Trade-off:** hữu ích khi ranh giới phức tạp hoặc feature nhiều chiều, nhưng kernel SVM có thể tốn train/inference khi số mẫu lớn. SVM không tự động chống outlier: kết quả phụ thuộc `C`, kernel và dữ liệu.

### 4.4. Naive Bayes (tr. 35–37)

Từ Bayes, `P(y|x) ∝ P(y)P(x|y)`; giả định các feature độc lập có điều kiện theo lớp cho `P(x|y)=∏ⱼP(xⱼ|y)`. Fit prior và thống kê từng feature theo lớp, sau đó chọn lớp có posterior cao nhất.

| Biến thể | Dữ liệu thường dùng | Điều kiện đáng chú ý |
|---|---|---|
| GaussianNB | Feature số liên tục | Mô hình hóa phân phối Gaussian theo lớp |
| MultinomialNB | Count/tần suất không âm, như Bag-of-Words | Smoothing để tránh xác suất 0 |
| BernoulliNB | Feature nhị phân | Dùng sự xuất hiện/vắng mặt của feature |

Ưu điểm là train/predict nhanh; hạn chế là giả định độc lập thường không đúng và xác suất có thể quá tự tin. Missing value vẫn cần xử lý theo biến thể/implementation; không nên hiểu slide là Naive Bayes tự giải quyết mọi dữ liệu thiếu.

### 4.5. Decision Tree và Random Forest (tr. 38–41)

**Decision Tree:** ở mỗi nút, tìm feature và ngưỡng tách làm giảm độ hỗn tạp của lớp, sau đó lặp đệ quy đến lá. Có thể dùng Gini `1-Σₖpₖ²` hoặc entropy. Cây sâu dễ overfit; điều chỉnh `max_depth`, `min_samples_leaf` hoặc pruning.

**Random Forest:** fit nhiều cây trên các bootstrap sample; mỗi lần tách chỉ xét một tập con feature, rồi gộp dự đoán. Với `RandomForestClassifier` của scikit-learn, lớp cuối dựa trên **xác suất trung bình qua cây**, không nhất thiết trùng với majority vote được nêu đơn giản trên slide. Rừng thường ổn định hơn một cây nhưng tốn bộ nhớ và khó đọc như một tập quy tắc duy nhất. [Tài liệu scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

**Gradient Boosting** cũng được nhắc tới ở tr. 21 và 57: cây được thêm tuần tự để giảm loss hiện tại, khác với cách forest xây nhiều cây bootstrap tương đối độc lập. Không có thuật toán nào được bảo đảm tốt nhất cho mọi tập dữ liệu.

## 5. Hyperparameter, loss và overfitting (tr. 42–48)

**Parameter** được học từ train, ví dụ hệ số `w` của linear/logistic regression. **Hyperparameter** được chọn ngoài vòng fit, ví dụ `k` của kNN, độ sâu cây, `C` của SVM/logistic hoặc learning rate. [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html) thử một lưới giá trị; random search lấy mẫu các cấu hình. Bayesian optimization và evolutionary search được slide nhắc đến như kỹ thuật tổng quát, không nên hiểu là đều có sẵn trực tiếp trong API cốt lõi của scikit-learn.

Một objective phổ biến có dạng `J(w)=(1/n)Σᵢ L(yᵢ,f_w(xᵢ)) + λR(w)`. `L` là loss, `R` là penalty, `λ` kiểm soát regularization. Với regression, MSE phạt lỗi lớn mạnh hơn MAE; Huber loss dùng dạng bậc hai cho lỗi nhỏ và tuyến tính cho lỗi lớn. Dạng Huber thông dụng với residual `e` là `0.5e²` khi `|e|≤δ`, còn `δ(|e|-0.5δ)` khi `|e|>δ`. Công thức in ở tr. 44 có hệ số chưa nhất quán; nên dùng công thức chuẩn khi tính toán.

| Dấu hiệu trên train/validation | Cách hiểu ban đầu | Việc cần kiểm |
|---|---|---|
| Cả hai cùng kém | Underfitting hoặc dữ liệu/nhãn có vấn đề | Baseline, feature, nhãn, năng lực mô hình |
| Train tốt, validation kém | Overfitting hoặc lệch phân phối/leakage | Split, preprocessing, regularization, độ sâu |
| Cả hai tốt | Có thể tổng quát hóa trong phạm vi dữ liệu tương tự | Kiểm test cuối, nhóm hiếm, dữ liệu mới |

Các biện pháp slide nêu: L1/L2 regularization, cross-validation, early stopping, feature engineering và giảm độ phức tạp. Chúng không thay thế việc kiểm chất lượng nhãn hay dữ liệu sai thời điểm.

## 6. Regression và metric (tr. 44, 49–50)

Regression dự đoán giá trị liên tục. Đặt `eᵢ=yᵢ-ŷᵢ`, `n` là số mẫu đánh giá:

| Metric | Công thức | Cách đọc và giới hạn |
|---|---|---|
| MSE | `(1/n)Σeᵢ²` | Phạt mạnh sai số lớn; đơn vị là bình phương đơn vị target |
| RMSE | `√MSE` | Cùng đơn vị target, nhạy outlier |
| MAE | `(1/n)Σ|eᵢ|` | Sai số tuyệt đối trung bình, dễ diễn giải |
| R² | `1-Σeᵢ²/Σ(yᵢ-ȳ)²` | So với trung bình của `y` trong tập đánh giá; có thể âm trên test |
| Adjusted R² | `1-(1-R²)(n-1)/(n-p-1)` | Công thức thống kê cho một số bài toán hồi quy khi `n>p+1`; không thay thế test/CV |
| MAPE | `(100/n)Σ|eᵢ/yᵢ|` | Khó dùng khi `yᵢ=0` hoặc gần 0; cần ghi quy ước xử lý |

Với **weighted MSE**, có nhiều quy ước chuẩn hóa. Một lựa chọn là `Σwᵢeᵢ²/Σwᵢ` cho `wᵢ≥0`; nếu viết `(1/n)Σwᵢeᵢ²` như trên slide thì ý nghĩa độ lớn phụ thuộc cách đặt tổng trọng số. Ghi rõ quy ước khi báo cáo. Metric đánh giá và loss dùng để train có thể khác nhau, nhưng phải khớp mục tiêu quyết định.

## 7. Các thuật toán regression (tr. 51–57)

| Thuật toán | Cách dự đoán/fit | Khi nên thử | Giới hạn chính |
|---|---|---|---|
| Linear Regression / OLS | `ŷ=b+wᵀx`, giảm `Σ(y-ŷ)²` | Baseline dễ hiểu cho dữ liệu bảng | Quan hệ phi tuyến cần feature bổ sung |
| Ridge | OLS + penalty L2 `λΣwⱼ²` | Feature tương quan, hệ số không ổn định | Không tự chọn feature về đúng 0 |
| Lasso | OLS + penalty L1 `λΣ|wⱼ|` | Muốn nghiệm thưa/chọn feature | Có thể chọn không ổn định khi feature tương quan |
| Elastic Net | Phối hợp L1 và L2 | Vừa muốn thưa vừa muốn ổn định hơn | Cần tune thêm tỷ lệ L1/L2 |
| Polynomial Regression | Tạo `x²`, `x³`, tương tác rồi fit mô hình tuyến tính trên feature mới | Quan hệ cong với số feature vừa phải | Bậc cao dễ overfit và khó extrapolate |
| SVR | Tối ưu hàm dự đoán với vùng sai số `ε` và regularization | Quan hệ phức tạp, tập vừa | Nhạy scale/hyperparameter, kernel có thể tốn kém |
| Decision Tree Regressor | Tách dữ liệu thành lá và dự đoán từ target ở lá | Quan hệ phi tuyến, tương tác | Dễ overfit; dự đoán dạng bậc thang |
| Random Forest / Boosting Regressor | Gộp nhiều cây hoặc thêm cây tuần tự | Dữ liệu bảng phi tuyến | Tốn tài nguyên, khó giải thích hơn mô hình tuyến tính |

**OLS** ở tr. 52–53 tìm `w,b` giảm tổng bình phương residual. Với một feature, `ŷ=mx+b`; với nhiều feature, `ŷ=b+Σⱼwⱼxⱼ`. **Polynomial Regression** vẫn có thể tuyến tính theo hệ số sau khi biến đổi feature, dù quan hệ với `x` là đường cong. **SVR** không đơn giản là “tìm hyperplane có margin lớn giữa các điểm” như phân loại SVM; nó dùng một vùng `ε` quanh hàm dự đoán và phạt điểm nằm ngoài vùng đó.

## 8. Ghi chú hiệu đính và giới hạn của slide

| Trang | Điều dễ hiểu sai | Cách dùng đúng |
|---|---|---|
| 13, 16 | Đánh giá test rồi mới tuning, dùng `liblinear` trong ví dụ nhiều lớp | Tuning trên train/CV trước; test sau cùng. Kiểm solver theo phiên bản thư viện |
| 17 | “Customer segmentation” được liệt kê cùng ứng dụng classification | Segmentation/gom nhóm không nhãn thường là unsupervised; classification chỉ khi có lớp định nghĩa sẵn |
| 19 | F1 gắn với vài điều kiện cố định | Chọn metric theo chi phí FP/FN, phân bố lớp và mục tiêu; không có điều kiện chung như “TN cao” làm F1 luôn phù hợp |
| 34 | SVM được mô tả là robust với outlier và hiệu quả trên tập lớn | Không bảo đảm; phụ thuộc `C`, kernel, số mẫu và outlier |
| 37 | Naive Bayes được mô tả là chịu được missing data | Vẫn cần kiểm và xử lý missing theo biến thể/implementation |
| 40 | Forest classifier luôn majority vote | Implementation scikit-learn dùng trung bình xác suất qua cây |
| 44 | Biểu thức Huber loss trên slide chưa nhất quán hệ số | Dùng công thức Huber chuẩn ở mục 5 |
| 50 | MAPE, weighted MSE được viết ngắn | Kiểm `y=0` cho MAPE; ghi quy ước chuẩn hóa trọng số |
| 51, 56 | Mô tả khái quát ở tr. 51 có thể khiến SVR bị hiểu như SVM phân loại; hình tr. 56 có vùng `ε` | Phân biệt vùng sai số `ε` của regression với margin phân loại |

Các ghi chú này là giải thích kỹ thuật bổ sung, không thay đổi chủ đề chính của bài giảng.

## 9. Câu hỏi tự kiểm tra

1. Vì sao `StandardScaler` phải fit trên train fold khi dùng cross-validation?
2. Với `TP=40`, `FP=10`, `FN=20`, `TN=930`, hãy tính accuracy, precision, recall và F1. Metric nào có thể che việc bỏ sót lớp dương?
3. Nêu điểm khác nhau giữa `k` của kNN, `C` của logistic/SVM và hệ số `w` của mô hình tuyến tính.
4. Khi nào nên dùng MAE thay vì RMSE? MAPE gặp vấn đề gì khi target bằng 0?
5. Vì sao chọn threshold trên test làm kết quả test không còn là đánh giá độc lập?

## Nguồn và đọc tiếp

- Nguồn chính: [ML_Chapter2_new.pdf](../Tài liệu học tập/ML_Chapter2_new.pdf), tr. 2–57.
- Tài liệu chính thức để kiểm tra ví dụ và API: [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html), [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- Bộ ghi chú chi tiết hơn theo từng thuật toán đã có trong [hoc_giam_sat](00_tong_quan.md).
