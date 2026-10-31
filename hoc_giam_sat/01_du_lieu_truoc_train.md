# Dữ liệu và quy trình trước train

## Purpose

Biến yêu cầu thực tế thành tập dữ liệu đáng tin cậy và một phép thử công bằng. Chất lượng split và nhãn quyết định giá trị của toàn bộ kết quả phía sau.

### Định đề "Garbage In, Garbage Out" & Tầm quan trọng của Thiết kế Dữ liệu

Trong kỹ thuật học máy thực chứng, mô hình chỉ là tấm gương phản chiếu cấu trúc thống kê của dữ liệu huấn luyện. Nếu dữ liệu bị ô nhiễm bởi nhãn sai, giá trị bất thường hoặc rò rỉ thông tin tương lai, thuật toán tối ưu hóa dù hiện đại đến đâu cũng chỉ đơn thuần tối ưu hóa việc "học vẹt" những lỗi sai đó (nguyên lý GIGO: *Rác vào thì rác ra*).

Mục tiêu tối thượng của giai đoạn trước huấn luyện (Pre-training Phase) là thiết lập một **khung thử nghiệm độc lập, công bằng và có khả năng kiểm chứng** (Fair, Unbiased & Auditable Testbed). Một quy trình chuẩn bị dữ liệu thất bại sẽ dẫn đến một trong hai thảm họa:
1. **Lạc quan giả tạo (Pessimistic Failure / False Confidence):** Mô hình đạt điểm kiểm tra gần như hoàn hảo trong phòng thí nghiệm (do rò rỉ dữ liệu), nhưng sụp đổ hoàn toàn khi triển khai thực tế.
2. **Đánh giá thiên lệch (Biased Evaluation):** Tập kiểm tra không đại diện đúng cho phân phối thực tế, khiến kỹ sư đưa ra quyết định kiến trúc sai lầm.

---

## Input

- Định nghĩa đơn vị dự đoán, target `y`, thời điểm dự đoán và dữ liệu thực sự có sẵn ở thời điểm đó.
- Dữ liệu thô, schema, nguồn nhãn, mã định danh nhóm, timestamp, thông tin về missing/outlier.
- Metric chính, baseline và ngưỡng chấp nhận đã thống nhất trước train.

### Bảng Kiểm toán Cấu trúc Dữ liệu (Data Audit Schema Matrix)

Trước khi viết bất kỳ dòng mã tiền xử lý nào, kỹ sư nghiên cứu cần lập bảng đặc tả thuộc tính cho từng cột trong tập dữ liệu:

| Tên trường (Feature) | Kiểu vật lý | Kiểu thống kê | Miền giá trị hợp lệ | Tỷ lệ Null | Chiến lược xử lý Null | Nguy cơ Leakage |
|---|---|---|---|---|---|---|
| `user_id` | String/UUID | Identifier | Khóa định danh | 0% | Giữ lại làm Group ID | Rất cao nếu split ngẫu nhiên |
| `transaction_time` | Timestamp | Temporal | Giờ UTC chuẩn | 0% | Khóa phân chia thời gian | Cần chặn mọi feature sau mốc này |
| `age` | Integer | Numerical | $`[18, 100]`$ | $`< 2\%`$ | Impute median trong fold | Thấp |
| `income` | Float | Numerical | $`\ge 0`$ (Lệch phải) | $`15\%`$ | Median + Thêm cờ `income_isna` | Thấp |
| `account_status` | String | Categorical | `{'active', 'locked'}` | 0% | One-Hot / Ordinal Encoding | Trung bình (cần kiểm thời điểm đổi status) |
| `target_churn` | Binary | Target ($`y`$) | $`\{0, 1\}`$ | 0% | Xác nhận sau 30 ngày | **Cấm đưa vào ma trận $`X`$** |

---

## Output

- Train/validation/test không chồng lặp; mô tả quy tắc split và tỷ lệ lớp/nhóm/thời gian.
- Pipeline tiền xử lý có thể `fit` trên train và `transform` trên validation/test/inference.
- Báo cáo khám phá dữ liệu: số mẫu, tỉ lệ nhãn, missing, trùng lặp, biến bất thường, nguy cơ leakage.

### Tính Độc lập Không gian Toán học của các Tập Phân chia

Một phép phân chia dữ liệu $`D`$ thành tập huấn luyện ($`D_{\text{train}}`$), tập kiểm định ($`D_{\text{val}}`$) và tập kiểm tra ($`D_{\text{test}}`$) được định nghĩa là chuẩn mực khi và chỉ khi thỏa mãn đồng thời 2 điều kiện tiên quyết:

1. **Tính rời rạc tuyệt đối (Mutual Exclusivity):**

$$
D_{\text{train}} \cap D_{\text{val}} = \emptyset, \quad D_{\text{train}} \cap D_{\text{test}} = \emptyset, \quad D_{\text{val}} \cap D_{\text{test}} = \emptyset
$$

2. **Tính độc lập thực thể (Entity/Group Independence):**
   Nếu mẫu dữ liệu gắn liền với thực thể $`g \in G`$ (ví dụ: bệnh nhân, khách hàng):

$$
\mathrm{Group}(D_{\text{train}}) \cap \mathrm{Group}(D_{\text{test}}) = \emptyset
$$

   Đảm bảo mô hình được đánh giá trên khả năng tổng quát hóa cho **thực thể hoàn toàn mới**, chứ không phải nhận diện lại đặc tính của thực thể cũ.

---

## Evaluation metrics

- Trước khi có mô hình: tỉ lệ nhãn thiếu/sai, tỉ lệ giá trị thiếu, tỉ lệ trùng, độ phủ lớp/nhóm/thời gian; đây là **kiểm tra dữ liệu**, không phải điểm mô hình.
- Baseline: majority class/stratified/random cho classification; mean/median cho regression, dùng cùng metric chính với mô hình. [1]
- Kiểm tra test có đủ ví dụ ở từng nhóm trọng yếu; không có tỷ lệ split cố định phù hợp mọi dữ liệu. [2]

### Thước đo Đánh giá Chất lượng Dữ liệu Thô (Data Health Metrics)

1. **Tỷ lệ Mất cân bằng Nhãn (Imbalance Ratio - $`IR`$):**

$$
IR = \frac{N_{\text{majority}}}{N_{\text{minority}}}
$$

   * Nếu $`IR > 10`$: Phải áp dụng phân tầng (*Stratified Split*), tuyệt đối không dùng Accuracy làm metric chính; chuyển sang PR-AUC, F1-macro hoặc Brier Score.
2. **Hệ số Phóng đại Phương sai (Variance Inflation Factor - VIF):**
   Đo lường mức độ đa cộng tuyến giữa các biến độc lập $`x_j`$:

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2}
$$

   Trong đó $`R_j^2`$ là hệ số xác định khi hồi quy $`x_j`$ theo toàn bộ $`p-1`$ biến còn lại. $`\mathrm{VIF}_j > 5`$ hoặc $`10`$ báo hiệu tương quan cao, làm mất ổn định hệ số trong các mô hình tuyến tính.
3. **Chỉ số Cardinality (Số lượng giá trị phân biệt):**
   Với biến định danh, tỷ lệ giữa số giá trị phân biệt và tổng số hàng. Biến có cardinality quá cao (như ID, địa chỉ IP) nếu không xử lý đúng sẽ gây overfit nghiêm trọng.

---

## How it works

1. Đóng khung bài toán: classification hay regression, nhãn được biết sau bao lâu, có sai lệch chọn mẫu hoặc nhãn không.
2. Audit schema, đơn vị đo, phân phối, missing, trùng, nhãn hiếm và lỗi gán nhãn. Loại feature xuất hiện **sau** thời điểm dự đoán.
3. Chia dữ liệu **trước khi fit** imputer, scaler, encoder, selector hay target encoding. IID: split ngẫu nhiên; lớp hiếm: cân nhắc stratification; người dùng/bệnh nhân lặp lại: split theo group; dự đoán tương lai: split theo thời gian. [3][4][5]
4. Fit mọi thống kê tiền xử lý trên train hoặc trong từng fold CV; áp dụng nguyên trạng sang validation/test. Giữ phép biến đổi trong `Pipeline` để tránh leakage. [3]
5. Train dummy baseline và mô hình đơn giản, ghi lại seed, phiên bản dữ liệu, schema và quy tắc split.

**Ví dụ chia theo thời gian:** dự đoán khách hàng rời dịch vụ vào ngày 1/7 từ thông tin có đến 30/6. Nếu nhãn chỉ xác nhận sau 30 ngày, feature từ tháng 7 hoặc số liệu tổng hợp đến tháng 8 là leakage. Train trên các tháng cũ, validation trên tháng kế tiếp, test trên giai đoạn muộn hơn; mọi feature cần được dựng theo thời điểm dự đoán của từng hàng.

**Thứ tự an toàn cho một fold:** `split fold → fit imputer/encoder/scaler trên fold train → transform fold validation → fit model trên fold train → score fold validation`. Lặp lại toàn bộ chuỗi cho từng fold. Không được `fit_transform` trên toàn bộ `X` trước cross-validation. [3]

### Phân tích Ba Cơ chế Rò rỉ Dữ liệu (Data Leakage Taxonomies)

Rò rỉ dữ liệu xảy ra khi thông tin ngoài luồng từ bên ngoài tập huấn luyện vô tình được đưa vào quá trình xây dựng mô hình:

```text
                               DATA LEAKAGE
                     _______________|_______________
                    |               |               |
             Train-Test       Temporal / Time     Target
            Contamination         Leakage        Leakage
```

1. **Train-Test Contamination (Nhiễm chéo Train-Test):**
   * *Bản chất:* Tính toán các giá trị thống kê (Mean, Standard Deviation, Max, Min, IDF trong NLP, hoặc danh mục Categorical) trên toàn bộ tập dữ liệu trước khi chia train/test.
   * *Hậu quả:* Tập train đã "biết trước" phân phối của tập test. Độ lệch chuẩn và trung bình của tập test đã ngấm ngầm định hình các tham số chuẩn hóa của mô hình.
2. **Temporal / Time Leakage (Rò rỉ Thời gian):**
   * *Bản chất:* Dùng dữ liệu sinh ra tại thời điểm $`t + \Delta t`$ để dự đoán sự kiện tại thời điểm $`t`$.
   * *Ví dụ:* Đưa đặc trưng "Số cuộc gọi khiếu nại trong tháng 8" vào mô hình dự đoán khách hàng có hủy dịch vụ vào ngày 31/07 hay không.
3. **Target Leakage (Rò rỉ Nhãn mục tiêu):**
   * *Bản chất:* Một biến đặc trưng được tạo ra bằng cách biến đổi trực tiếp từ nhãn mục tiêu hoặc là hệ quả trực tiếp của nhãn mục tiêu.
   * *Ví dụ:* Dự đoán bệnh nhân có mắc bệnh cúm không, nhưng feature lại bao gồm "Đã được kê đơn thuốc đặc trị cúm Tamiflu". Khi đó Tamiflu xuất hiện sau khi bác sĩ đã chẩn đoán, biến này mang bản chất của nhãn chứ không phải triệu chứng dự đoán.

### Ví dụ Số học Minh họa: Thảm họa Fit Scaler Ngoài Fold

Xét một tập dữ liệu gồm 4 mẫu một chiều $`X = [10, 20, 30, 100]`$:
* Giả sử ta chia: $`X_{\text{train}} = [10, 20, 30]`$, $`X_{\text{test}} = [100]`$.

* **Cách làm SAI (Fit trên toàn bộ dữ liệu trước split):**
  * $`\mu_{\text{all}} = \frac{10 + 20 + 30 + 100}{4} = 40`$, $`\sigma_{\text{all}} \approx 35.35`$.
  * Mẫu huấn luyện thứ ba ($`30`$) được biến đổi thành:

$$
z_3 = \frac{30 - 40}{35.35} \approx -0.28
$$

  * *Nhận xét:* Giá trị $`z_3`$ của train bị kéo dịch chuyển sang âm vì có sự hiện diện của giá trị ngoại lai $`100`$ bên tập test! Tập train đã gián tiếp biết tập test có giá trị lớn.

* **Cách làm ĐÚNG CHUẨN KHOA HỌC (Fit CHỈ trên train):**
  * $`\mu_{\text{train}} = \frac{10 + 20 + 30}{3} = 20`$, $`\sigma_{\text{train}} \approx 8.16`$.
  * Mẫu huấn luyện thứ ba ($`30`$) được biến đổi chuẩn tắc:

$$
z_3 = \frac{30 - 20}{8.16} \approx +1.22
$$

  * Sau đó, áp dụng nguyên vẹn tham số đã fit sang test:

$$
z_{\text{test}} = \frac{100 - \mu_{\text{train}}}{\sigma_{\text{train}}} = \frac{100 - 20}{8.16} \approx +9.80
$$

  * *Nhận xét:* Tập train hoàn toàn độc lập với sự tồn tại của điểm $`100`$, phản ánh chân thực khả năng tiếp nhận dữ liệu ngoại lai khi triển khai thực tế.

---

## Data used

Ví dụ bảng giá nhà: `X` gồm diện tích, vị trí, năm xây; `y` là giá giao dịch. Nếu dùng giá chốt sau ngày dự đoán hoặc thống kê toàn bộ tập trước split, mô hình đã nhìn thấy thông tin tương lai. Với bệnh nhân có nhiều lần khám, mọi lần khám của cùng bệnh nhân cần ở cùng phía split khi mục tiêu là bệnh nhân mới. [3][4]

---

## Design

- Chuẩn hóa số thường hữu ích cho kNN, SVM và mô hình tuyến tính có regularization; cây thường ít cần scale. Mã hóa biến phân loại dựa vào train, xác định cách xử lý giá trị chưa thấy khi inference.
- Resampling lớp hiếm, feature selection và tuning đều phải nằm **bên trong fold train** khi cross-validation. Không dùng validation/test để học thống kê hoặc chọn feature. [3]
- Split theo thời gian cần tuân thủ trật tự và cân nhắc khoảng cách giữa tập train/test nếu nhãn hoặc feature bị trễ. [5]
- Failure modes: leakage, nhãn không đại diện, trùng mẫu, class imbalance, dữ liệu thiếu theo cơ chế liên quan đến target.

**Checklist dữ liệu:**

| Kiểm tra | Câu hỏi cụ thể | Hành động khi lỗi |
|---|---|---|
| Target | Nhãn được tạo bởi ai, khi nào, có bất đồng không? | Audit mẫu nhãn, sửa quy tắc gán nhãn |
| Đơn vị quan sát | Một người/tài khoản có nhiều hàng không? | Group split nếu mục tiêu là người/tài khoản mới |
| Thời gian | Feature có sẵn tại thời điểm dự đoán không? | Loại feature tương lai, time split |
| Missing | Thiếu do ngẫu nhiên hay do quy trình nghiệp vụ? | Tạo cờ missing nếu có ý nghĩa, impute trong pipeline |
| Trùng lặp | Có cùng bản ghi ở train và test không? | Gộp/lọc theo định danh phù hợp trước khi đánh giá |
| Lệch lớp | Có đủ mẫu lớp hiếm ở từng tập không? | Stratification khi hợp lý; báo cáo số mẫu theo lớp |

**Lưu ý:** Không áp dụng một tỷ lệ 70/15/15 máy móc. Tập test phải đủ lớn cho kết luận cần đưa ra, còn cách chia phải mô phỏng dữ liệu sẽ gặp lúc dùng mô hình. [2]

### Cơ chế Giá trị Khuyết (Missing Data Mechanisms)

Hiểu rõ bản chất toán học của dữ liệu khuyết giúp lựa chọn phương pháp xử lý chính xác:

1. **MCAR (Missing Completely at Random - Khuyết hoàn toàn ngẫu nhiên):**
   * Xác suất một giá trị bị khuyết không phụ thuộc vào chính nó và cũng không phụ thuộc vào bất kỳ biến quan sát nào khác: $`P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M)`$.
   * *Cách xử lý:* Có thể loại bỏ dòng (Drop row) nếu tỷ lệ nhỏ, hoặc impute đơn giản bằng Mean/Median.
2. **MAR (Missing at Random - Khuyết ngẫu nhiên có điều kiện):**
   * Xác suất bị khuyết phụ thuộc vào các biến quan sát được nhưng không phụ thuộc vào giá trị chưa quan sát: $`P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M \mid Y_{\text{obs}})`$.
   * *Ví dụ:* Nam giới ít khi trả lời câu hỏi về thu nhập hơn nữ giới, nhưng trong nhóm nam giới thì việc trả lời không phụ thuộc vào mức lương cao hay thấp.
   * *Cách xử lý:* KNN Imputer, Iterative Imputer (MICE) dựa trên các biến đã quan sát.
3. **MNAR (Missing Not at Random - Khuyết không ngẫu nhiên):**
   * Xác suất bị khuyết phụ thuộc trực tiếp vào chính giá trị bị thiếu đó: $`P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) \neq P(M \mid Y_{\text{obs}})`$.
   * *Ví dụ:* Những người có thu nhập cực kỳ cao thường cố tình từ chối kê khai thu nhập.
   * *Cách xử lý:* Bắt buộc phải tạo thêm một biến nhị phân chỉ báo thiếu (`income_is_missing = 1`), kết hợp với mô hình hóa đặc thù.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Data Leakage** | Rò rỉ thông tin từ test/tương lai vào train. | Tạo hiệu năng cao giả tạo trong lúc train. | Hiểm họa số 1 cần triệt tiêu trước khi chọn thuật toán. |
| **Stratified Split** | Chia dữ liệu bảo toàn nguyên vẹn tỷ lệ các lớp. | Giữ nguyên phân phối mục tiêu giữa train và test. | Bắt buộc đối với bài toán mất cân bằng nhãn. |
| **Group Split** | Giữ mọi quan sát của cùng một thực thể trong cùng một fold. | Kiểm tra năng lực tổng quát hóa trên thực thể mới. | Bắt buộc khi dữ liệu có cấu trúc phân cụm (bệnh nhân, thiết bị). |
| **TimeSeriesSplit** | Chia dữ liệu theo dòng thời gian tịnh tiến. | Ngăn chặn việc học quy luật tương lai để dự đoán quá khứ. | Bắt buộc đối với dữ liệu chuỗi thời gian, tài chính, giao dịch. |
| **Purged & Embargo** | Kỹ thuật loại bỏ các mẫu trùng lặp thời gian nhãn. | Triệt tiêu triệt để rò rỉ nhãn có độ trễ thời gian dài. | Sử dụng trong các mô hình tài chính định lượng cao cấp. |
| **Pipeline Encapsulation** | Đóng gói toàn bộ biến đổi vào đối tượng Pipeline. | Đảm bảo tính nhất quán giữa train, validation và serving. | Chuẩn mực kỹ thuật phần mềm bắt buộc trong Data Science. |

---

## Nguồn

- [1] [scikit-learn: Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
- [2] [Google: Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [3] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [4] [scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [5] [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [6] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in data mining: Formulation, detection, and avoidance*. ACM Transactions on Knowledge Discovery from Data (TKDD).
- [7] López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons. (Chương 7: Cross-Validation in Finance & Purged K-Fold).
