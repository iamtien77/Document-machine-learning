# Chương 4 — Học Bán Giám Sát (Semi-Supervised Learning)

> Bản ghi chú tiếng Việt hệ thống hóa toàn diện dựa trên bài giảng [ML_Chapter4_new.pdf](<../Tài liệu học tập/ML_Chapter4_new.pdf>) (10 trang slide), biên soạn bởi **TS. Nguyễn Thị Khánh Tiên** (`tienntk@ut.edu.vn`), Trường Đại học Giao thông Vận tải TP.HCM (UTH). Số trang trong ngoặc đơn tương ứng với số trang slide trong tệp PDF gốc. Nội dung được cấu trúc hóa logic, chuẩn hóa ký hiệu toán học và bổ sung phân tích chuyên sâu phục vụ học tập, nghiên cứu và thực hành.

---

## 1. Khái Niệm và Bản Chất của Học Bán Giám Sát (tr. 2)

Học bán giám sát (Semi-Supervised Learning - SSL) là phân nhánh của học máy kết hợp đồng thời cả **dữ liệu có nhãn (Labeled Data)** và **dữ liệu không nhãn (Unlabeled Data)** trong quá trình huấn luyện nhằm xây dựng mô hình dự đoán chính xác và khái quát hóa tốt hơn so với việc chỉ học thuần túy trên tập nhãn hạn chế.

Trong thực tế, việc thu thập dữ liệu thô (ảnh, văn bản, bản ghi âm, tín hiệu cảm biến) thường rất dễ dàng và chi phí thấp, nhưng công đoạn gán nhãn chính xác bởi chuyên gia (bác sĩ, kỹ sư, nhà ngôn ngữ học) lại vô cùng tốn kém, mất thời gian và khó mở rộng quy mô. SSL tận dụng nguồn dữ liệu không nhãn khổng lồ để nhận diện hình học phân phối tiềm ẩn `$ p(x) $`, qua đó định hình lại ranh giới quyết định của hàm phân loại `$ p(y \mid x) $`.

### Các Khái Niệm Then Chốt (Key Concepts - tr. 2)
1. **Dữ liệu có nhãn (Labeled Data):**
   
$$
\mathcal{D}_L = \left\lbrace (x_1, y_1), (x_2, y_2), \dots, (x_{N_L}, y_{N_L}) \right\rbrace
$$

   Các điểm dữ liệu đi kèm với nhãn mục tiêu đích thực tế `$ y_i $`. Số lượng `$ N_L $` thường rất nhỏ.

2. **Dữ liệu không nhãn (Unlabeled Data):**
   
$$
\mathcal{D}_U = \left\lbrace u_1, u_2, \dots, u_{N_U} \right\rbrace
$$

   Các điểm dữ liệu chỉ có vector đặc trưng đầu vào, hoàn toàn không có nhãn đích. Số lượng `$ N_U \gg N_L $`.

3. **Mô hình học bán giám sát (SSL Model):**
   Mô hình học máy tiếp nhận cả hai nguồn dữ liệu `$ \mathcal{D}_L \cup \mathcal{D}_U $`, tối ưu hóa hàm mục tiêu liên hợp bao gồm thành phần giám sát và thành phần điều quy không giám sát.

---

### Quy Trình 5 Bước Vận Hành Chuẩn Mực của SSL (tr. 2)

Quá trình học bán giám sát vận hành qua chu trình lặp khép kín gồm 5 giai đoạn:

```
[Dữ liệu có nhãn D_L] ──> [1. Huấn luyện Ban đầu (Initial Training)]
                                       │
                                       ▼
                       [2. Khai thác Dữ liệu Không nhãn D_U]
                                       │
                                       ▼
                       [3. Tinh chỉnh Lặp (Iterative Refinement)]
                                       │
                                       ▼
                       [4. Lan truyền Nhãn / Gán Giả nhãn (Label Propagation)]
                                       │
                                       ▼
                       [5. Huấn luyện lại Mô hình (Model Retraining)]
```

1. **Huấn luyện ban đầu (Initial Training):** Mô hình học máy cơ sở được huấn luyện sơ bộ trên lượng nhỏ dữ liệu có nhãn `$ \mathcal{D}_L $`.
2. **Khai thác dữ liệu không nhãn (Leveraging Unlabeled Data):** Mô hình sử dụng tập `$ \mathcal{D}_U $` để khảo sát cấu trúc đa tạp, vùng tập trung mật độ cao và phân phối biên xác suất `$ p(x) $`.
3. **Tinh chỉnh lặp (Iterative Refinement):** Mô hình liên tục tinh chỉnh các dự đoán trên tập dữ liệu không nhãn qua các vòng lặp hoặc các bước cập nhật gradient.
4. **Lan truyền nhãn (Label Propagation):** Gán giả nhãn (pseudo-labels) cho các mẫu không nhãn dựa trên độ tự tin dự đoán cao nhất hoặc dựa trên liên kết đồ thị lân cận.
5. **Huấn luyện lại mô hình (Model Retraining):** Mô hình được huấn luyện lại trên tập dữ liệu mở rộng kết hợp giữa nhãn gốc chuẩn xác và các giả nhãn tự tin mới được bổ sung.

---

## 2. Bốn Kỹ Thuật Bán Giám Sát Cốt Lõi (tr. 3)

Slide bài giảng xác định 4 trụ cột thuật toán kinh điển trong học bán giám sát:

| Kỹ thuật SSL | Nguyên lý vận hành chính | Phạm vi áp dụng tối ưu |
| :--- | :--- | :--- |
| **1. Self-Training** | Mô hình tự dự đoán trên tập không nhãn, chọn các mẫu tự tin nhất làm giả nhãn rồi tự huấn luyện lại. | Dữ liệu dạng bảng (Tabular), bài toán phân loại tổng quát. |
| **2. Co-Training** | Huấn luyện nhiều mô hình trên các góc nhìn độc lập (Views) của dữ liệu, các mô hình gán nhãn chéo cho nhau. | Dữ liệu đa góc nhìn (Multi-view: Web, Văn bản + Ảnh, Đa cảm biến). |
| **3. Semi-Supervised SVM (SS-SVM)** | Mở rộng SVM truyền thống, tối đa hóa khoảng cách lề (margin) đồng thời đẩy ranh giới phân chia ra khỏi vùng dữ liệu không nhãn dày đặc. | Dữ liệu bảng, không gian đặc trưng chiều cao, tập dữ liệu quy mô nhỏ/vừa. |
| **4. Generative Models (VAEs, GANs)** | Mô hình hóa phân phối đồng thời $p(x, y)$, dùng mạng sinh sinh dữ liệu tổng hợp hoặc trích xuất biểu diễn tiềm ẩn. | Dữ liệu hình ảnh, âm thanh, phân tích mật độ phân phối phức tạp. |

---

### Các Lĩnh Vực Ứng Dụng Tiêu Biểu (tr. 3)
- **Phân loại hình ảnh (Image Classification):** Huấn luyện bộ phân loại thị giác với số lượng nhãn rất ít bằng cách khai thác hàng triệu ảnh không gán nhãn trên Internet.
- **Xử lý ngôn ngữ tự nhiên (NLP):** Cải thiện các mô hình ngôn ngữ lớn (LLMs), trích xuất thực thể và dịch máy dựa trên kho văn bản khổng lồ chưa chú thích.
- **Phân tích hình ảnh y tế (Medical Image Analysis):** Chẩn đoán phim chụp X-ray, CT-scan, MRI trong điều kiện bác sĩ chuyên khoa chỉ có thể chú thích một số lượng hữu hạn ca bệnh.
- **Phát hiện bất thường và gian lận (Anomaly Detection):** Nhận diện giao dịch gian lận tài chính hoặc lỗ hổng bảo mật mạng dựa trên việc mô hình hóa phân phối của các hành vi bình thường không nhãn.

---

## 3. Tiềm Năng của SSL trong Phân Tích Cảm Xúc (Sentiment Analysis - tr. 4)

Phân tích cảm xúc (Sentiment Analysis) từ văn bản đánh giá khách hàng (Reviews, Tweets) là một trong những ứng dụng thành công rực rỡ nhất của SSL:
- **Đặc thù bài toán:** Mỗi ngày có hàng triệu bình luận mới được tạo ra trên mạng xã hội và sàn thương mại điện tử. Việc gán nhãn cảm xúc thủ công (Tích cực / Tiêu cực / Trung tính) cho toàn bộ là bất khả thi.
- **Cơ chế SSL phát huy hiệu quả:**
  1. Sử dụng một tập hạt nhân (seed set) gồm vài trăm bình luận được chuyên gia gán nhãn kỹ lưỡng.
  2. Mô hình ngôn ngữ (hoặc Co-Training giữa góc nhìn từ ngữ N-grams và góc nhìn cấu trúc cú pháp POS tags) tiến hành mở rộng nhãn sang kho bình luận thô.
  3. Mô hình học thêm được nhiều từ lóng (slang), cách biểu đạt địa phương mới xuất hiện mà không cần con người cập nhật từ điển thủ công.

---

## 4. Phân Loại Hệ Thống Các Phương Pháp SSL (Taxonomy - tr. 5)

Dựa trên cấu trúc cây phân loại học bán giám sát tổng thể, các phương pháp được chia thành các nhánh lớn:

```
                       CÁC PHƯƠNG PHÁP BÁN GIÁM SÁT (SSL)
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  [INDUCTIVE METHODS]           [TRANSDUCTIVE METHODS]        [GENERATIVE METHODS]
         │                              │                              │
 ┌───────┴───────┐              ┌───────┴───────┐              ┌───────┴───────┐
 ▼               ▼              ▼               ▼              ▼               ▼
[Wrapper]    [Direct/Model] [Graph-based]    [Large-margin]  [Gaussian GMM]  [VAE & SGAN]
(Self-Train,  (Consistency, (Label Prop,      (TSVM, S3VM)    (EM with L)     (p(x, y))
 Co-Train)     FixMatch)     Label Spread)
```

1. **Phương pháp Quy nạp (Inductive Methods):** Học một hàm dự đoán tổng quát `$ f: \mathcal{X} \to \mathcal{Y} $` có khả năng dự đoán tức thời cho bất kỳ điểm dữ liệu mới nào ngoài tập huấn luyện.
2. **Phương pháp Quy nạp chuyển vị (Transductive Methods):** Tập trung gán nhãn tối ưu cho chính các điểm trong tập không nhãn hiện có `$ \mathcal{D}_U $`, tận dụng cấu trúc hình học của toàn bộ tập dữ liệu.

---

## 5. Phương Pháp Tự Huấn Luyện (Self-Training - tr. 6–7)

*(Dẫn chiếu nghiên cứu khảo sát toàn diện: Self-Training: A Survey, arXiv:2202.12040)*

Self-Training là kỹ thuật đóng gói (Wrapper Algorithm) đơn giản và phổ biến nhất trong lịch sử bán giám sát:
1. **Khởi tạo:** Huấn luyện bộ phân loại cơ sở `$ f_\theta $` trên `$ \mathcal{D}_L $`.
2. **Dự đoán:** Đưa toàn bộ mẫu không nhãn `$ u \in \mathcal{D}_U $` qua mô hình để thu được vector xác suất `$ p_\theta(y \mid u) $`.
3. **Chọn lọc tự tin (Confidence Selection):**
   
$$
\mathcal{S} = \left\lbrace u \in \mathcal{D}_U \;\middle|\; \max_{c} p_\theta(y=c \mid u) \ge \tau \right\rbrace
$$

   Chỉ những mẫu có xác suất vượt ngưỡng `$ \tau $` (ví dụ `$ \tau = 0.90 $`) mới được chuyển thành nhãn cứng `$ \hat{y} = \arg\max_c p_\theta(y=c \mid u) $`.
4. **Cập nhật:** Hợp nhất `$ \mathcal{D}_L \leftarrow \mathcal{D}_L \cup \mathcal{S} $`, loại bỏ `$ \mathcal{S} $` khỏi `$ \mathcal{D}_U $`.
5. **Lặp lại:** Huấn luyện lại `$ f_\theta $` trên tập dữ liệu mở rộng cho đến khi không còn mẫu nào vượt ngưỡng hoặc đạt số vòng lặp tối đa.

> **Hạn chế then chốt:** Rủi ro **Confirmation Bias (Tích lũy sai lệch)** — nếu mô hình dự đoán sai một nhãn và tự nạp vào tập huấn luyện với độ tự tin cao, sai lầm này sẽ được phóng đại trong các vòng lặp sau.

---

## 6. Học Bán Giám Sát Dựa Trên Đồ Thị (Graph-Based Label Propagation - tr. 8)

*(Dẫn chiếu công trình kinh điển: Zhu & Ghahramani, CMU Technical Report CMU-CALD-02-107, 2002)*

Phương pháp đồ thị coi mỗi điểm dữ liệu (cả có nhãn và không nhãn) là một đỉnh (node) trên đồ thị, và các cạnh biểu thị độ tương đồng hình học giữa chúng:

### Ma Trận Tương Đồng Gauss (RBF Kernel Weight Matrix)
Trọng số liên kết giữa điểm `$ x_i $` và `$ x_j $`:

$$
W_{ij} = \exp\left( -\frac{\|x_i - x_j\|^2}{2\sigma^2} \right) \quad \text{với } W_{ii} = 0
$$

### Quá Trình Lan Truyền Nhãn
Ma trận chuyển trạng thái xác suất `$ T_{ij} = P(j \to i) = \frac{W_{ij}}{\sum_k W_{kj}} $`.
Tại mỗi bước lặp `$ t $`:
1. Lan truyền phân phối nhãn qua các cạnh đồ thị:
   
$$
Y^{(t)} = T \cdot Y^{(t-1)}
$$

2. **Kẹp chặt nhãn gốc (Clamping Condition):** Đặt lại các hàng tương ứng với dữ liệu có nhãn thật về đúng phân phối ban đầu:
   
$$
Y_L^{(t)} = Y_L^{(0)}
$$

3. Quá trình hội tụ về nghiệm giải tích dừng duy nhất, phản ánh nguyên lý làm mịn hàm trên đa tạp dữ liệu.

---

## 7. Phương Pháp Đồng Huấn Luyện (Co-Training - tr. 9)

*(Dẫn chiếu công trình nền tảng: Blum & Mitchell, COLT 1998)*

Co-Training khai thác tính đa góc nhìn (Multi-view) tự nhiên của bài toán:
- **Giả định phân rã đặc trưng:** Không gian đặc trưng phân rã thành hai góc nhìn độc lập điều kiện `$ X = (X_1, X_2) $`. Ví dụ:
  - Phân loại trang web: `$ X_1 $` là từ ngữ xuất hiện trên trang web; `$ X_2 $` là văn bản liên kết (anchor text) từ các trang khác trỏ về.
- **Cơ chế hoạt động:**
  1. Huấn luyện bộ phân loại `$ f_1 $` trên View 1 và `$ f_2 $` trên View 2 bằng tập nhãn ban đầu.
  2. Mỗi mô hình dự đoán trên tập dữ liệu không nhãn.
  3. `$ f_1 $` chọn ra các mẫu mà nó tự tin nhất để gán nhãn và nạp vào tập huấn luyện của `$ f_2 $`.
  4. Ngược lại, `$ f_2 $` chọn các mẫu nó tự tin nhất để bổ sung cho `$ f_1 $`.
  5. Hai mô hình hỗ trợ sửa sai chéo (Cross-validation) lẫn nhau, khắc phục hiệu quả rủi ro confirmation bias của Self-Training đơn lẻ.

---

## 8. Máy Vectơ Hỗ Trợ Bán Giám Sát (Semi-Supervised SVM / S3VM - tr. 10)

*(Dẫn chiếu bài toán thực hành chuẩn: A simple semi-supervised strategy SVM Wine, Kaggle)*

S3VM (còn gọi là Transductive SVM - TSVM) mở rộng bài toán tối ưu của SVM bằng cách tìm đồng thời siêu phẳng phân chia và phép gán nhãn nhị phân `$ y_u \in \lbrace -1, +1 \rbrace $` cho toàn bộ mẫu không nhãn sao cho độ rộng lề hình học (geometric margin) là lớn nhất:

### Hàm Mục Tiêu Tối Ưu Hóa S3VM

$$
\min_{w, b, y_u} \frac{1}{2} \|w\|^2 + C_L \sum_{i=1}^{N_L} \xi_i + C_U \sum_{j=1}^{N_U} \xi_j^*
$$

Thỏa mãn các ràng buộc:

$$
\begin{aligned}
y_i (w^\top x_i + b) &\ge 1 - \xi_i, \quad &\xi_i \ge 0 \quad &\forall i \in \mathcal{D}_L \\
y_j (w^\top u_j + b) &\ge 1 - \xi_j^*, \quad &\xi_j^* \ge 0 \quad &\forall j \in \mathcal{D}_U
\end{aligned}
$$

### Ý Nghĩa Hình Học
Siêu phẳng tối ưu buộc phải thỏa mãn:
1. Phân loại đúng các điểm có nhãn với sai số lề nhỏ nhất.
2. Tuyệt đối không cắt ngang qua các cụm dữ liệu không nhãn dày đặc, mà phải luồn lách qua các vùng khe thưa thớt mật độ (Low-density Separation Principle).

---

## 9. Tổng Kết và Khuyến Nghị Thực Hành Xuyên Suốt

1. **Hiểu rõ giả định dữ liệu:** Học bán giám sát chỉ phát huy tác dụng khi dữ liệu tuân thủ Giả định Cụm (Cluster), Giả định Đa tạp (Manifold) hoặc Giả định Trơn (Smoothness). Nếu dữ liệu phân bố ngẫu nhiên không có cấu trúc cụm, SSL không thể cải thiện hiệu năng so với học có giám sát.
2. **Khởi đầu từ Baseline đơn giản:** Luôn xây dựng mô hình có giám sát làm mốc so sánh chuẩn trước khi áp dụng SSL. Chỉ đưa SSL vào sử dụng khi chỉ số đánh giá vượt trội rõ rệt (`$ \Delta_{\mathrm{SSL}} \ge 2\% $`).
3. **Ưu tiên thuật toán phù hợp dạng dữ liệu:**
   - Dữ liệu dạng bảng: Dùng **Self-Training** kết hợp LightGBM hoặc **Tri-Training**.
   - Dữ liệu đồ thị/mạng lưới: Dùng **Label Spreading** với hệ số kẹp nhãn `$ \alpha = 0.2 $`.
   - Dữ liệu văn bản / Đa cảm biến: Dùng **Co-Training**.
   - Dữ liệu hình ảnh sâu: Dùng **FixMatch / FlexMatch**.
4. **Kết hợp với Active Learning:** Khi ngân sách cho phép, kết hợp giữa Active Learning (người gán nhãn các mẫu bất định gần ranh giới) và SSL (máy tự gán nhãn các mẫu tự tin sâu trong cụm) để tối ưu hóa triệt để chi phí gán nhãn.

---

## 10. Tài Liệu Tham Khảo Nền Tảng Trong Slide

- [1] **Nguyen Thi Khanh Tien (Ph.D.)**. *Chapter 4: Semi-Supervised Learning*, University of Transport and Communications (UTH).
- [2] **Self-Training Survey**: *Self-Training: A Survey*, arXiv:2202.12040 (2022).
- [3] **Zhu, X., & Ghahramani, Z. (2002)**: *Learning from labeled and unlabeled data with label propagation*. CMU-CALD-02-107, Carnegie Mellon University.
- [4] **Blum, A., & Mitchell, T. (1998)**: *Combining labeled and unlabeled data with co-training*. Proceedings of COLT'98.
- [5] **Kaggle Code Reference**: *A simple semi-supervised strategy SVM Wine*, [Kaggle Notebook](https://www.kaggle.com/code/saptarsi/a-simple-semi-supervised-strategy-svm-wine).
