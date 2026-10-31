# Training phase — giai đoạn huấn luyện mô hình học giám sát

> Giai đoạn này đi từ dữ liệu đã định nghĩa đến một **pipeline có thể dùng để dự đoán**. Bài giảng [Chương 1](<../Tài liệu học tập/ML_Chapter1_new.pdf>) trình bày quy trình dự án (tr. 16–22); [Chương 2](<../Tài liệu học tập/ML_Chapter2_new.pdf>) trình bày học giám sát, split, train, tuning và đánh giá (tr. 2–16, 42–48). Tài liệu này triển khai chi tiết riêng phần training; xem [inference phase](17_inference_phase.md) cho bước dùng mô hình đã học.

## Purpose

Training phase học tham số/quy tắc từ những ví dụ có nhãn `(X, y)`, chọn mô hình và cấu hình phù hợp bằng dữ liệu phát triển, rồi **khóa** pipeline trước khi đánh giá cuối. Mục tiêu không phải đạt điểm train cao nhất, mà là dự đoán tốt trên dữ liệu mới theo metric và chi phí lỗi đã xác định. [1][2]

Phân biệt ba việc: **fit tham số** (ví dụ hệ số `w`, nút cây), **chọn hyperparameter** (ví dụ `C`, `k`, `max_depth`) và **đánh giá cuối**. Mỗi việc dùng dữ liệu với quyền truy cập khác nhau để hạn chế overfitting vào tập đánh giá.

### Ranh giới Giữa Thử nghiệm Khoa học và Kỹ nghệ Đóng gói (From Research to Engineering)

Một sai lầm phổ biến trong đào tạo là chỉ xem huấn luyện như việc gọi hàm `model.fit(X, y)` trong một Jupyter Notebook. Trong kỹ nghệ học máy chuyên nghiệp, Training Phase là một **dây chuyền sản xuất tự động, có thể tái lập và có thể kiểm toán được (Reproducible & Auditable Production Line)**.

Giai đoạn này đòi hỏi sự tách biệt tuyệt đối giữa:
1. **Môi trường Khám phá (Exploration Environment):** Nơi nhà nghiên cứu thử nghiệm các giả thuyết, vẽ đồ thị phân tích và so sánh các họ thuật toán khác nhau.
2. **Dây chuyền Đóng gói (Packaging Pipeline):** Nơi toàn bộ các bước biến đổi dữ liệu (từ vector hóa, điền khuyết, chuẩn hóa đến mô hình dự đoán) được kết nối thành một đối tượng bất biến (Immutable Pipeline Artifact), đảm bảo rằng khi chuyển giao sang môi trường suy luận, dữ liệu thô mới sẽ trải qua **chính xác 100% cùng một chuỗi các phép toán số học**.

---

## Input

| Đầu vào | Nội dung cần có | Vì sao cần |
|---|---|---|
| Định nghĩa bài toán | Đơn vị dự đoán, thời điểm dự đoán, target, lớp dương nếu binary | Ngăn feature/nhãn bị xác định sau thời điểm cần dự đoán |
| Dữ liệu có nhãn | `X` feature, `y` target, nguồn và quy tắc tạo nhãn | Xác định mô hình đang học từ điều gì |
| Quy tắc split | Train, validation/CV, test; group/timestamp nếu có | Mô phỏng dữ liệu sẽ gặp khi inference |
| Tiền xử lý | Imputation, encoding, scaling, feature selection | Biến dữ liệu thô thành đầu vào hợp lệ của estimator |
| Tiêu chí chọn | Metric chính, baseline, chi phí FP/FN hoặc sai số số học, giới hạn latency | Tránh đổi mục tiêu sau khi xem kết quả |
| Cấu hình thí nghiệm | Seed, search space, ngân sách tính toán, phiên bản dữ liệu/code/thư viện | Tái lập và so sánh công bằng |

Ví dụ Iris ở Chương 2: `X` gồm bốn phép đo hoa, `y` là một trong ba loài. Với dữ liệu nghiệp vụ, phải thêm ID nhóm hoặc timestamp nếu các hàng không độc lập hay mục tiêu là dự đoán tương lai. [1][3]

### Quản lý Phiên bản Dữ liệu & Mã nguồn (Data & Code Lineage)

Để đảm bảo tính tái lập trong nghiên cứu khoa học, mọi đợt huấn luyện bắt buộc phải ghi nhận ba hàm băm (Hash Fingerprints):
* **Data Hash:** Băm mã hóa (SHA-256) của tập dữ liệu huấn luyện (sử dụng các công cụ như DVC - Data Version Control).
* **Code Hash:** Git commit hash chính xác của kho mã nguồn thực hiện huấn luyện.
* **Environment Manifest:** Tệp khóa cấu hình thư viện chính xác (ví dụ: `requirements.txt` với số phiên bản cố định của `numpy`, `scipy`, `scikit-learn`).

---

## Output

**Artifact chính:** pipeline đã `fit` gồm mọi bước tiền xử lý và estimator. Nếu bài toán cần xác suất/nhãn cuối, artifact hoặc cấu hình kèm theo còn phải chứa calibrator, threshold và thứ tự nhãn lớp. Không lưu mô hình riêng mà quên scaler/encoder đã học. [4]

**Hồ sơ đi kèm:** schema đầu vào, danh sách/ý nghĩa feature, phiên bản dữ liệu và code, tham số/hyperparameter, seed, kết quả CV/validation, baseline, báo cáo test cuối, lỗi theo nhóm, ngày train và giới hạn áp dụng. Đầu ra của training phase là **đầu vào của inference phase**; một người chỉ có dữ liệu mới và artifact này phải tạo được dự đoán theo đúng cùng phép biến đổi.

### Tiêu chuẩn Thẻ Báo cáo Mô hình (Model Card Standard)

Theo chuẩn mực nghiên cứu quốc tế (Mitchell et al., 2019), một artifact mô hình hoàn chỉnh phải đi kèm một tài liệu `MODEL_CARD.md` tóm lược:
* **Mục đích thiết kế (Intended Use):** Mô hình được thiết kế cho tác vụ nào? Trong điều kiện hoạt động nào?
* **Cảnh báo giới hạn (Out-of-Scope Use Cases):** Những trường hợp nào tuyệt đối không được phép sử dụng mô hình (ví dụ: dữ liệu ngoại suy ngoài miền tuổi $`18-60`$)?
* **Phân tích Công bằng & Đạo đức (Fairness & Bias Disclosures):** Hiệu năng của mô hình trên các nhóm thiểu số hoặc nhóm nhạy cảm.

---

## Evaluation metrics

| Loại đo | Ví dụ | Dùng lúc nào | Không nên nhầm với |
|---|---|---|---|
| Training loss | MSE, log loss, hinge loss | Tối ưu tham số và theo dõi hội tụ | Metric nghiệp vụ cuối cùng |
| Validation/CV metric | F1, recall, AP, MAE, RMSE | Chọn mô hình và hyperparameter | Điểm test độc lập |
| Test metric | Cùng metric chính đã định nghĩa trước | Đánh giá cuối sau khi khóa lựa chọn | Công cụ tuning tiếp |
| Kiểm tra bổ sung | Train–validation gap, calibration, latency, lỗi theo nhóm | Tìm failure mode và trade-off | Một điểm tổng thay thế mọi phân tích |

Với classification, chọn metric theo chi phí sai: accuracy có thể che lớp hiếm; precision/recall phụ thuộc threshold; AP/ROC-AUC đánh giá khả năng xếp hạng nhưng không tự xác định ngưỡng. Với regression, MAE dễ diễn giải, RMSE nhấn mạnh lỗi lớn, R² cho góc nhìn bổ sung. Luôn so với **dummy baseline** trên cùng split. [5]

Ví dụ nếu bỏ sót ca dương tính rất tốn kém, có thể chọn recall làm metric chính và đặt ràng buộc precision tối thiểu. Ngưỡng cụ thể phải được chọn trên dữ liệu phát triển, không được chọn bằng test.

---

## How it works

1. **Khóa định nghĩa:** xác định `X`, `y`, thời điểm dự đoán, metric và baseline. Kiểm nhãn thiếu/sai, mẫu trùng, feature chỉ có sau sự kiện cần dự đoán.
2. **Chia dữ liệu:** tách test trước; trên phần phát triển, chọn validation hoặc CV. Dùng group split khi cùng thực thể có nhiều hàng; dùng time split khi dự đoán tương lai. Tỷ lệ split là lựa chọn theo lượng dữ liệu và mục tiêu kiểm định, không là hằng số. [2]
3. **Tạo pipeline:** ở từng train fold, `fit` imputer/encoder/scaler/feature selector **chỉ** trên fold đó; áp dụng `transform` cho validation fold. `Pipeline` giúp giữ đúng thứ tự này. [3][4]
4. **Fit baseline và ứng viên:** đo dummy baseline, rồi mô hình đơn giản; chỉ thử mô hình phức tạp hơn khi có lý do từ lỗi validation. Tối ưu loss theo thuật toán: gradient/solver, tách nút cây hoặc lưu mẫu với kNN.
5. **Chọn cấu hình:** CV/grid/random search trên train để so hyperparameter. Early stopping dùng validation phù hợp, không dùng test. Ghi cả trung bình metric, độ biến thiên theo fold và chi phí train/inference. [2][6]
6. **Calibration/threshold nếu cần:** calibrator phải học từ dự đoán trên dữ liệu không dùng để fit classifier hoặc từ dự đoán out-of-fold đúng cách. Threshold chọn trên tập dành riêng cho bước đó hoặc quy trình CV hợp lệ. Nếu làm cả hai, tránh dùng cùng một holdout để tối ưu mọi bước rồi báo cáo chính holdout ấy như đánh giá độc lập. [7][8]
7. **Khóa và test:** quyết định preprocessing, estimator, hyperparameter, calibrator và threshold; đánh giá một lần trên test chưa dùng để chọn. Phân tích nhầm lẫn/residual theo lớp, nhóm và thời gian. Nếu sửa thiết kế sau khi xem test, cần một đánh giá độc lập mới.
8. **Đóng gói và kiểm tra bàn giao:** lưu pipeline, schema, phiên bản và báo cáo. Chọn vài bản ghi cố định (gồm trường hợp thông thường và trường hợp biên), lưu dự đoán trước khi serialize, rồi nạp lại artifact trong môi trường inference và so nhãn/giá trị/xác suất theo sai số số học chấp nhận được. Kiểm cả thứ tự lớp và phép biến đổi feature; nếu kết quả khác, không bàn giao artifact. [9]

```text
Raw labeled data
  → kiểm nhãn/schema/thời điểm
  → split train | validation/CV | test
  → [fit preprocessing + fit model] trên train fold
  → score validation/CV, chọn cấu hình
  → (calibration/threshold trên dữ liệu hợp lệ nếu cần)
  → khóa artifact
  → test cuối → báo cáo + artifact cho inference
```

### Quy trình Kiểm thử Bất biến Đầu ra (Golden Samples Invariant Verification)

Trước khi chuyển giao artifact mô hình cho đội ngũ kỹ nghệ triển khai, bắt buộc phải thực hiện bài kiểm tra tính toàn vẹn serialization:

1. **Trích xuất Golden Dataset ($`D_{\text{gold}}`$):** Chọn từ $`5`$ đến $`10`$ mẫu dữ liệu đại diện, bao gồm:
   * Các mẫu điển hình ở trung tâm phân phối.
   * Các mẫu có giá trị khuyết (Missing Values).
   * Các mẫu ở biên cực trị (Edge Cases / Outliers).
2. **Ghi nhận Baseline Inference trước khi lưu:**

$$
\hat{Y}_{\text{pre}} = \text{Pipeline}_{\text{memory}}.\text{predict\_proba}(D_{\text{gold}})
$$

3. **Tuần tự hóa (Serialize):** Lưu pipeline ra đĩa (ví dụ: định dạng `.joblib`, `.pickle`, hoặc chuẩn mở `ONNX`).
4. **Giải tuần tự hóa (Deserialize) trong tiến trình độc lập:** Nạp lại pipeline từ file đã lưu:

$$
\hat{Y}_{\text{post}} = \text{Pipeline}_{\text{disk}}.\text{predict\_proba}(D_{\text{gold}})
$$

5. **Kiểm định sai số số học (Assertion):**

$$
\max \lvert \hat{Y}_{\text{pre}} - \hat{Y}_{\text{post}} \rvert \le 10^{-6}
$$

   Nếu sai số lớn hơn ngưỡng dung sai số thực hoặc thứ tự các cột nhãn lớp bị đảo lộn, **hủy bỏ đợt phát hành ngay lập tức**.

---

## Data used

| Tập dữ liệu | Được phép dùng để | Không được dùng để |
|---|---|---|
| Train/fold train | Fit tiền xử lý và mô hình | Tự báo cáo như hiệu năng dữ liệu mới |
| Validation/fold validation | Chọn mô hình, hyperparameter, early stopping | Fit preprocessing của fold train hoặc giả làm test cuối |
| Calibration/threshold holdout hoặc OOF | Fit calibrator/chọn ngưỡng theo thiết kế | Dùng như điểm test độc lập sau khi đã tối ưu trên nó |
| Test giữ riêng | Báo cáo hiệu năng cuối của pipeline đã khóa | Chọn feature, mô hình, hyperparameter, calibrator hay threshold |

Trường hợp dùng CV hoặc refit trên train+validation phải ghi rõ mẫu nào đã được dùng để fit classifier. Một validation set đã trở thành dữ liệu fit classifier không còn là holdout độc lập cho calibration. [2][7]

---

## Design

- **Ưu tiên baseline:** dummy và mô hình giải thích được là mốc; mô hình phức tạp phải chứng minh cải thiện ổn định và phù hợp chi phí. [5]
- **Chống leakage:** feature selection, oversampling, imputation, encoding và scaling có học tham số phải nằm trong train fold. Không học từ test hoặc dữ liệu tương lai. [3]
- **Tính tái lập:** lưu seed, split, snapshot dữ liệu, code, dependency, search space và log; seed cố định không loại bỏ mọi nguồn bất định.
- **Kiểm tra artifact:** bộ mẫu cố định (golden samples) và kết quả dự đoán kỳ vọng cần đi kèm phiên bản mô hình để kiểm trước/sau lưu–nạp và khi thay đổi môi trường serving.
- **Tương thích inference:** cùng schema, đơn vị đo, thứ tự cột và cách xử lý giá trị mới. Kiểm artifact có thể load trong môi trường được hỗ trợ; không nạp artifact kiểu pickle/joblib từ nguồn không tin cậy và không giả định tương thích giữa các phiên bản scikit-learn. [9]
- **Failure modes:** nhãn nhiễu, lớp hiếm không đủ mẫu, group/time leakage, tuning lặp quá lâu trên một validation set, metric không gắn chi phí, train-serving skew sau triển khai.

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Fit** | Học tham số mô hình hoặc thống kê của transformer từ dữ liệu train. | Rút ra tham số hoặc thống kê từ dữ liệu | Bước học trên train; transformer phải fit trong fold tương ứng |
| **Validation / CV** | Dữ liệu hoặc các fold dùng chọn cấu hình; CV luân phiên nhiều split train/validation. | Ước lượng hiệu quả cấu hình trên dữ liệu chưa dùng để fit vòng đó | Bước chọn mô hình và hyperparameter trước test cuối |
| **Hyperparameter** | Thiết lập chọn trước mỗi lần fit, như `k`, `C`, độ sâu cây. | Điều khiển cách mô hình được học chứ không được gradient học trực tiếp | Đầu vào của quá trình tìm kiếm cấu hình bằng validation |
| **Baseline** | Mốc đơn giản để biết mô hình phức tạp có cải thiện thực sự không. | Đặt mốc chất lượng tối thiểu cần vượt | Bước kiểm tra giá trị thực của mô hình phức tạp hơn |
| **Data leakage** | Thông tin không hợp lệ từ test/tương lai/nhãn đi vào train hoặc lựa chọn mô hình. | Làm đánh giá quá lạc quan vì mô hình thấy thông tin không hợp lệ | Rủi ro phải chặn ở split, preprocessing và tuning |
| **Calibration** | Fit/kiểm sự phù hợp giữa xác suất dự đoán và tần suất thật trên dữ liệu hợp lệ. | Làm xác suất đầu ra gần với tần suất quan sát | Bước bổ sung sau fit nếu quyết định cần xác suất đáng tin |
| **Artifact** | Pipeline, tham số và cấu hình đã khóa để inference dùng lại. | Đảm bảo dùng đúng preprocessing và tham số khi triển khai | Đầu ra được lưu sau training để inference nạp lại |
| **Model Card** | Bản tài liệu học thuật đặc tả nguồn gốc, hiệu năng và giới hạn sử dụng mô hình. | Đảm bảo tính minh bạch, đạo đức và khả năng kiểm toán hệ thống. | Chuẩn mực chuyển giao từ phòng nghiên cứu sang sản xuất. |
| **ONNX** | Chuẩn định dạng mở đại diện cho các mô hình học máy (Open Neural Network Exchange). | Tách biệt mô hình khỏi framework Python, tối ưu tốc độ thực thi đa nền tảng. | Định dạng lưu trữ tối ưu cho triển khai công nghiệp. |

---

## Nguồn

- [1] [PDF Chương 2: Supervised Learning](<../Tài liệu học tập/ML_Chapter2_new.pdf>)
- [2] [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [3] [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [4] [scikit-learn: Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [5] [scikit-learn: Metrics and dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [6] [scikit-learn: Hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [7] [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [8] [scikit-learn: Decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [9] [scikit-learn: Model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [10] Mitchell, M., et al. (2019). *Model Cards for Model Reporting*. Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*), 220-229.
