# AdaBoost — Adaptive Boosting

> PDF [Chương 2](<../Tài liệu học tập/ML_Chapter2_new.pdf>) nhắc AdaBoost trong nhóm boosting ở tr. 57. AdaBoost khác Gradient Boosting: bản kinh điển tăng trọng số các mẫu bị phân loại sai, rồi cộng có trọng số các weak learners. [1]

## Purpose

Kết hợp nhiều mô hình yếu (thường là cây rất nông) thành classifier mạnh hơn bằng cách tập trung dần vào các mẫu khó. Đây là một họ ensemble hữu ích để hiểu boosting tuần tự; cần so với cây đơn, Random Forest và Gradient Boosting trên cùng dữ liệu. [1][2]

## Input

`X` dữ liệu có nhãn lớp `y`, một base estimator hỗ trợ trọng số mẫu theo yêu cầu implementation, số vòng tối đa `n_estimators`, `learning_rate` và metric chính. Với bài toán nhị phân trong công thức kinh điển, mã hóa $`y\in \{-1,+1\}`$. Dữ liệu missing/categorical phải được xử lý theo khả năng estimator được dùng.

## Output

Danh sách weak learners $`h_{1},\ldots ,h_M`$ và trọng số đóng góp của từng learner. Dự đoán nhị phân kinh điển dùng dấu của tổng $`\sum _m \alpha _m h_m(x)`$; implementation đa lớp như `AdaBoostClassifier` có quy tắc SAMME và có thể xuất score/xác suất theo API. Xác suất ước lượng cần được kiểm calibration nếu dùng để quyết định theo rủi ro. [1][3]

## Evaluation metrics

Classification: precision/recall/F1 theo lớp, confusion matrix, AP/ROC-AUC nếu có score thích hợp, calibration khi dùng xác suất. So baseline và cây đơn trên validation/CV; xem chênh train–validation khi tăng số vòng. Theo dõi thời gian fit/inference vì dự đoán phải gộp nhiều weak learners. [4]

## How it works

**AdaBoost nhị phân kinh điển:** khởi tạo $`w_i=1/n`$. Ở vòng `m`, fit weak learner theo trọng số $`w_i`$; tính lỗi có trọng số $`e_m=\frac{\sum_i w_i\,\mathbf{1}[h_m(x_i)\ne y_i]}{\sum_i w_i}`$; khi $`0<e_m<0.5`$, đặt $`\alpha_m=\frac{1}{2}\ln\left(\frac{1-e_m}{e_m}\right)`$; cập nhật $`w_i\leftarrow w_i \exp (-\alpha _m y_i h_m(x_i))`$, rồi chuẩn hóa. Mẫu bị dự đoán sai được tăng trọng số tương đối. Nếu $`e_m=0`$, công thức log có biên vô hạn và thuật toán thường dừng vì đã fit hoàn hảo; nếu learner không tốt hơn ngẫu nhiên ($`e_m\ge 0.5`$), cần dừng hoặc xử lý theo quy tắc của implementation thay vì áp công thức máy móc. Công thức này mô tả **binary AdaBoost cổ điển**; không thay thế công thức của mọi phiên bản đa lớp/API. [1][2][3]

**Ví dụ một vòng:** bốn mẫu cùng trọng số `0.25`; weak learner sai đúng một mẫu nên $`e=0.25`$, $`\alpha=\frac{1}{2}\ln 3\approx 0.549`$. Sau cập nhật và chuẩn hóa, mẫu sai có trọng số `0.5`, ba mẫu đúng mỗi mẫu khoảng `1/6`. Vòng kế tiếp vì thế chú ý mẫu sai nhiều hơn. Ví dụ này giải thích fit, không là điểm benchmark.

```text
khởi tạo sample weights → lặp:
    fit weak learner theo weights
    tính lỗi có trọng số, xác định trọng số learner
    tăng trọng số tương đối của mẫu sai, chuẩn hóa
predict: gộp các learner theo trọng số đã học
```

AdaBoost tăng trọng số **mẫu**, còn Gradient Boosting thường fit cây theo pseudo-residual/gradient của loss. Cả hai đều tuần tự, khác Random Forest dùng bootstrap và feature randomness để gộp nhiều cây. [1][2]

## Data used

Dữ liệu bảng classification với nhãn tương đối đáng tin, sau tiền xử lý hợp lệ; Iris có thể minh họa cơ chế, nhưng không đại diện production. Nếu nhãn sai hoặc outlier khó, AdaBoost có thể tiếp tục tăng chú ý vào chính các mẫu nhiễu đó. Với lớp hiếm, chọn metric theo chi phí lỗi và xem lỗi từng lớp; không kết luận accuracy chung là đủ.

## Design

| Quyết định | Vai trò | Cần kiểm |
|---|---|---|
| Base estimator/độ sâu cây | Năng lực của mỗi weak learner | Cây quá mạnh tăng chi phí và nguy cơ fit nhiễu |
| `n_estimators` | Số vòng tối đa | Validation curve và latency |
| `learning_rate` | Mức đóng góp của mỗi learner | Tune cùng số vòng; quá lớn có thể kém ổn định |

Chọn trên CV/train; test chỉ dùng sau khi khóa cấu hình. Cần lưu toàn bộ ensemble và preprocessing cho inference. Khi nhãn nhiễu nhiều, thử mô hình có regularization/robustness phù hợp và kiểm các mẫu bị tăng trọng số; không tự cho rằng boosting luôn tốt hơn forest. `AdaBoostClassifier` trong scikit-learn có chi tiết API thay đổi theo phiên bản, nên xem tài liệu của phiên bản đang sử dụng. [1][3]

### Key words

| Thuật ngữ | Là gì? | Tác dụng trực tiếp | Vai trò trong quy trình |
|---|---|---|---|
| **Weak learner** | Mô hình cơ sở đơn giản, ví dụ cây một tầng; các learner được gộp thành ensemble. | Tạo dự đoán đơn giản để ensemble cải thiện qua nhiều vòng | Mô hình cơ sở được fit ở mỗi vòng AdaBoost |
| **Sample weight $`w_i`$** | Mức ảnh hưởng của mẫu `i` khi fit learner ở vòng hiện tại. | Làm mẫu khó có ảnh hưởng lớn hơn ở vòng tiếp theo | Trọng số truyền từ vòng trước vào learner hiện tại |
| **Weighted error $`e_m`$** | Tỷ lệ lỗi của learner sau khi tính theo trọng số mẫu. | Đo lỗi có xét mức quan trọng của từng mẫu | Căn cứ tính hệ số learner trong công thức AdaBoost cổ điển |
| **Learner weight $`\alpha _m`$** | Mức đóng góp của learner `m` vào dự đoán ensemble trong công thức cổ điển. | Quy định learner nào ảnh hưởng mạnh tới dự đoán chung | Trọng số gộp đầu ra của các learner |
| **Boosting round** | Một vòng fit learner mới và cập nhật trọng số mẫu. | Thêm một learner và cập nhật mức chú ý vào mẫu | Đơn vị lặp của quá trình huấn luyện AdaBoost |
| **Stump** | Cây quyết định chỉ có một phép tách, thường dùng làm weak learner. | Tạo quy tắc phân lớp đơn giản, dễ làm weak learner | Base estimator thường gặp của AdaBoost |
| **Learning rate** | Hệ số điều chỉnh đóng góp của learner theo API, cần tune cùng số vòng. | Co mức đóng góp mỗi vòng để kiểm soát việc học | Hyperparameter tune cùng số learner |
| **SAMME** | Biến thể AdaBoost đa lớp; công thức trọng số khác công thức nhị phân minh họa ở trên. | Mở rộng cách gộp learner cho bài toán nhiều lớp | Biến thể thuật toán cần phân biệt với công thức nhị phân |

## Nguồn

- [1] [scikit-learn: Ensembles — AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
- [2] [Freund & Schapire: A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://doi.org/10.1006/jcss.1997.1504)
- [3] [scikit-learn: AdaBoostClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- [4] [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
