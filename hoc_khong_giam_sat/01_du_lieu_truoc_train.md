# Dữ liệu và quy trình trước train

## Purpose

Biến dữ liệu thô chưa gán nhãn thành ma trận đặc trưng $`X`$ đáng tin cậy và có cấu trúc hình học chuẩn mực trước khi đưa vào bất kỳ thuật toán phân cụm hay giảm chiều nào. Trong học có giám sát, nếu một đặc trưng bị lệch thang đo, mô hình vẫn có thể dùng nhãn $`y`$ để bù đắp trọng số. Ngược lại, trong **Học không giám sát**, sai lầm ở khâu tiền xử lý là **không thể cứu vãn**: khoảng cách hình học giữa các điểm sẽ bị bóp méo hoàn toàn, dẫn đến các cụm được tìm ra chỉ là ảo ảnh của việc lệch đơn vị đo lường. [1][2]

**Ví dụ dẫn nhập:** Trong bài toán phân khúc khách hàng ngân hàng dựa trên `Số dư tài khoản` (từ 0 đến 5 tỷ VNĐ) và `Số năm gắn bó` (từ 1 đến 15 năm). Nếu không chuẩn hóa thang đo, chênh lệch vài trăm triệu số dư giữa hai khách hàng sẽ áp đảo hoàn toàn chênh lệch 10 năm gắn bó khi tính khoảng cách Euclid. Thuật toán phân cụm thực chất chỉ gom nhóm theo số dư và biến `Số năm gắn bó` hoàn toàn bị triệt tiêu giá trị thông tin.

Ký hiệu: tập dữ liệu ban đầu gồm $`n`$ quan sát và $`p`$ biến số, ma trận $`X \in \mathbb{R}^{n \times p}`$. Không có biến phụ thuộc $`y`$. Cần xây dựng pipeline tiền xử lý biến đổi $`X \to X_{\text{scaled}}`$ sao cho mỗi thuộc tính đóng góp công bằng vào độ đo khoảng cách không gian. [1]

**Cách đọc lần đầu:** nắm ví dụ trên rồi đi tiếp qua các mục Input → Output → Evaluation metrics → How it works. Phần phân tích toán học về các họ khoảng cách bên dưới giải thích cơ sở hình học; có thể quay lại sau khi hiểu quy trình làm sạch dữ liệu.

### Định đề "Garbage In, Garbage Out" trong Không gian Metric (Geometric Intuition)

Hầu hết các thuật toán học không giám sát đều xây dựng trên khái niệm **Không gian Metric $`(\mathcal{X}, D)`$**, nơi $`D(x, z)`$ là hàm khoảng cách thỏa mãn 4 tiên đề Euclid:
1. Tính không âm: $`D(x, z) \ge 0`$.
2. Tính đồng nhất: $`D(x, z) = 0 \iff x = z`$.
3. Tính đối xứng: $`D(x, z) = D(z, x)`$.
4. Bất đẳng thức tam giác: $`D(x, z) \le D(x, u) + D(u, z)`$.

Nếu một thuộc tính chưa được làm sạch chứa các giá trị ngoại lai cực đoan (Outliers), nó sẽ kéo giãn không gian metric theo chiều đó, phá vỡ cấu trúc mật độ tự nhiên và làm cho các thuật toán phân cụm dựa trên tâm (như K-Means) bị kéo lệch hoàn toàn khỏi trọng tâm thực sự của dữ liệu.

---

## Input

- Dữ liệu thô từ database hoặc log hệ thống, định nghĩa từng trường thông tin, thời điểm ghi nhận và nguồn gốc dữ liệu.
- Bảng đặc tả thuộc tính: cột nào là số liên tục, cột nào là số rời rạc, cột nào là định danh (Categorical) hoặc văn bản.
- Tỷ lệ giá trị khuyết (Missing Rate), các giá trị dị biệt (như số âm ở trường tuổi, số 999999 đại diện cho missing).
- Yêu cầu về độ trễ xử lý và loại khoảng cách mục tiêu (Euclid, Manhattan hay Cosine).

### Bảng Kiểm toán Cấu trúc Dữ liệu (Unsupervised Data Audit Matrix)

Trước khi thực hiện phân cụm, kỹ sư dữ liệu cần lập ma trận kiểm toán cho toàn bộ các thuộc tính:

| Tên trường (Feature) | Kiểu vật lý | Kiểu thống kê | Miền giá trị | Tỷ lệ Null | Chiến lược Imputation | Phương pháp Chuẩn hóa |
|---|---|---|---|---|---|---|
| `customer_id` | String | Identifier | Mã định danh duy nhất | 0% | Loại bỏ khỏi $X$ | Không (Khóa chính) |
| `age` | Integer | Numerical | $[18, 95]$ | $< 1\%$ | Impute median | `RobustScaler` |
| `annual_income` | Float | Numerical | $\ge 0$ (Lệch phải mạnh) | $5\%$ | Impute median + Yeo-Johnson | `StandardScaler` |
| `spending_score` | Integer | Numerical | $[1, 100]$ | 0% | Không | `MinMaxScaler` |
| `gender` | String | Categorical | `{'Male', 'Female'}` | 0% | One-Hot Encoding | Cần cân nhắc trọng số |

---

## Output

- Ma trận đặc trưng đã làm sạch và chuẩn hóa: $`X_{\text{clean}} \in \mathbb{R}^{n \times p}`$.
- Pipeline tiền xử lý hoàn chỉnh (gồm Imputer, Scaler, Transformer) có thể tái sử dụng để biến đổi các mẫu dữ liệu mới trong môi trường suy luận (Inference).
- Báo cáo khám phá dữ liệu (EDA Report): ma trận tương quan giữa các biến, phân phối độ lệch (Skewness), biểu đồ phân tán 2D sau khi chiếu sơ bộ.

### Bảo toàn Cấu trúc Hình học Cục bộ (Local Geometry Preservation)
Một pipeline tiền xử lý được coi là chuẩn mực trong học không giám sát khi và chỉ khi:
1. **Không tạo ra tương quan giả:** Phép điền khuyết không làm sai lệch ma trận hiệp phương sai của các biến.
2. **Không làm méo mó tỉ lệ khoảng cách tương đối:** Nếu điểm $`A`$ gần điểm $`B`$ hơn điểm $`C`$ trong bản chất vật lý thực tế, thì sau khi chuẩn hóa, quan hệ $`D(A, B) < D(A, C)`$ vẫn phải được bảo toàn.

---

## Evaluation metrics

Chất lượng tiền xử lý trong học không giám sát được đánh giá qua các chỉ số thống kê nội tại của ma trận $X$:

1. **Tỷ số Phương sai giữa các Biến (Variance Ratio Balance):**
   Trước khi chuẩn hóa, tỷ số giữa biến có phương sai lớn nhất và nhỏ nhất có thể lên đến hàng triệu ($`10^6`$). Sau khi chuẩn hóa qua `StandardScaler` hoặc `RobustScaler`, tỷ số này phải được đưa về gần mức $`1.0`$:

$$
\frac{\max_j \mathrm{Var}(X_{*, j})}{\min_j \mathrm{Var}(X_{*, j})} \approx 1
$$

2. **Hệ số Bất đối xứng (Skewness Score):**
   Đo lường mức độ lệch của phân phối. Các đặc trưng có độ lệch tuyệt đối $`\lvert \text{Skewness} \rvert > 1.5`$ cần được nén bằng logarit hoặc biến đổi Box-Cox/Yeo-Johnson để tránh tạo ra các cụm nhân tạo chỉ gồm một vài điểm đuôi dài.
3. **Chỉ số Tương quan Đa cộng tuyến (Multicollinearity / VIF):**
   Nếu hai biến có tương quan tuyến tính gần tuyệt đối ($`r > 0.95`$), chúng sẽ nhân đôi trọng số của khía cạnh đó trong khoảng cách Euclid. Cần loại bỏ một trong hai biến hoặc dùng PCA để trực giao hóa.

---

## How it works

Quy trình 5 bước tiền xử lý dữ liệu chuẩn mực cho học không giám sát:

```text
[Dữ liệu thô X] ──> [1. Audit & Impute Missing] ──> [2. Biến đổi phân phối lệch]
                                                              │
[5. X_scaled sẵn sàng] <── [4. Feature Scaling] <── [3. Khử tương quan cao]
```

1. **Xử lý giá trị khuyết (Missing Values):**
   * Trong học không giám sát, hầu hết các thuật toán tính khoảng cách không chấp nhận giá trị `NaN`.
   * Nếu hàng thiếu quá nhiều thuộc tính: loại bỏ hàng.
   * Nếu cột thiếu $< 10\%$: điền bằng Trung vị (`SimpleImputer(strategy='median')`) để kháng ngoại lai, hoặc dùng `KNNImputer` để ước lượng dựa trên các láng giềng gần nhất.
2. **Xử lý phân phối lệch (Skewed Distribution Handling):**
   * Dữ liệu tài chính, doanh thu thường có phân phối lệch phải cực mạnh. Áp dụng biến đổi logarit $`\ln(1 + x)`$ hoặc Yeo-Johnson để đưa phân phối về dạng đối xứng hình chuông.
3. **Lựa chọn hàm khoảng cách phù hợp (Distance Metric Selection):**
   * *Euclidean ($`L_2`$):* Thích hợp cho dữ liệu số liên tục, ít chiều ($`p < 20`$), các cụm dạng hình cầu.
   * *Manhattan ($`L_1`$):* Bền vững hơn trước ngoại lai, phù hợp khi số chiều lớn hơn.
   * *Cosine Distance:* Đo lường góc lệch thay vì độ dài, chuẩn mực cho dữ liệu văn bản (TF-IDF), đếm từ hoặc vector embedding.
4. **Chuẩn hóa thang đo (Feature Scaling):**
   * Nếu dữ liệu ít ngoại lai: dùng `StandardScaler()` (đưa về trung bình 0, phương sai 1).
   * Nếu dữ liệu có nhiều ngoại lai: dùng `RobustScaler()` (chuẩn hóa dựa trên Trung vị và Khoảng tứ phân vị IQR).
5. **Kiểm tra Lời nguyền số chiều (Curse of Dimensionality):**
   * Nếu số chiều $`p > 50`$, áp dụng PCA hoặc TruncatedSVD sơ bộ để giảm xuống không gian compact hơn trước khi phân cụm.

---

## Data used

Bảng đối chiếu sự khác biệt giữa dữ liệu thực hành trong bài tập và dữ liệu thực tế ở giai đoạn trước huấn luyện:

| Thuộc tính | Dữ liệu Bài tập / Slide môn học | Dữ liệu Hệ thống Thực tế (Production) |
|---|---|---|
| **Dung lượng & Số chiều** | Nhỏ ($`n < 1.000`$, $`p = 2 \sim 4`$), như tập `Iris` hay `Mall_Customers`. | Rất lớn ($`n > 10^6`$, $`p`$ từ vài chục đến hàng trăm đặc trưng). |
| **Giá trị khuyết (Missing)**| Không có missing hoặc missing được tạo nhân tạo rất ít. | Tỷ lệ missing cao (10% - 40%), phân bố rải rác trên nhiều cột. |
| **Độ lệch thang đo** | Thường đã được chuẩn hóa sẵn hoặc các biến cùng đơn vị đo. | Thang đo lệch nhau hàng triệu lần (ví dụ: tuổi vs số tiền giao dịch). |
| **Ngoại lai (Outliers)** | Dữ liệu sạch, không có lỗi nhập liệu hay lỗi cảm biến. | Chứa nhiều điểm dị biệt cực đoan do tài khoản ảo, bot cào dữ liệu hoặc lỗi phần cứng. |
| **Loại biến** | Thuần số (Numerical) giúp dễ vẽ biểu đồ phân tán. | Hỗn hợp phức tạp (Số, Phân loại, Tọa độ GPS, Thời gian, Chuỗi văn bản). |

---

## Design

### Pipeline Tiền xử lý Hoàn chỉnh với Scikit-learn
Đóng gói quy trình tiền xử lý cho dữ liệu hỗn hợp (gồm cả cột số và cột danh mục) vào `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Phân loại danh sách cột
num_features = ['age', 'annual_income', 'spending_score']
cat_features = ['gender', 'membership_type']

# Pipeline xử lý cột số: Impute median -> Scale kháng ngoại lai
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Pipeline xử lý cột phân loại: Impute mode -> One-Hot encoding
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Ghép nối bộ tiền xử lý toàn diện
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Fit trên dữ liệu X
X_transformed = preprocessor.fit_transform(X)
```

### Key words nền tảng

| Thuật ngữ tiếng Anh | Thuật ngữ tiếng Việt | Định nghĩa & Trực giác cốt lõi |
|---|---|---|
| **Feature Scaling** | Chuẩn hóa đặc trưng | Đưa các biến về cùng một thang đo để khoảng cách không bị chi phối bởi đơn vị đo. |
| **RobustScaler** | Bộ chuẩn hóa bền vững | Chuẩn hóa dựa trên Trung vị ($`Q_2`$) và Khoảng tứ phân vị ($`IQR`$), miễn nhiễm ngoại lai. |
| **Euclidean Distance** | Khoảng cách Euclid | Khoảng cách thẳng đường chim bay giữa hai điểm trong không gian $`\mathbb{R}^p`$. |
| **Manhattan Distance** | Khoảng cách Manhattan | Tổng độ lệch tuyệt đối theo từng trục tọa độ (khoảng cách ô cờ). |
| **Cosine Similarity** | Độ tương đồng Cosine | Đo lường góc lệch giữa hai vector, không phụ thuộc vào độ dài vector. |
| **Curse of Dimensionality** | Lời nguyền số chiều | Hiện tượng không gian trở nên cực kỳ thưa thớt khi tăng số chiều thuộc tính. |
| **Imputation** | Điền khuyết | Kỹ thuật ước lượng và thay thế các giá trị bị thiếu bằng giá trị thống kê hợp lý. |

---

## Nguồn

- [1] [ML_Chapter3_new.pdf](../Tài liệu học tập/ML_Chapter3_new.pdf): Bài giảng Unsupervised Learning (Trang 5: Clustering Pipeline; Trang 22: Distance Metrics), TS. Nguyễn Thị Khánh Tiên, ĐH GTVT TP.HCM.
- [2] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (Chapter 14: Unsupervised Learning & Cluster Analysis). Springer.
- [3] [scikit-learn User Guide: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html).
- [4] [scikit-learn User Guide: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html).
- [5] Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). *On the Surprising Behavior of Distance Metrics in High Dimensional Space*. ICDT.
