# Lựa Chọn Thuật Toán Học Bán Giám Sát

## Purpose

Tài liệu này cung cấp khung hướng dẫn có cấu trúc và cây quyết định chuẩn hóa (Decision Framework) nhằm giúp kỹ sư máy học lựa chọn chính xác giải thuật bán giám sát phù hợp nhất cho bài toán thực tế.

Học bán giám sát không phải là giải pháp vạn năng. Khi áp dụng sai thuật toán hoặc khi phân phối dữ liệu vi phạm các giả định nền tảng (Smoothness, Cluster, Manifold, Multi-view), mô hình bán giám sát có nguy cơ suy giảm hiệu năng nghiêm trọng (Degradation / Negative Learning) so với baseline giám sát thuần túy. Khung hướng dẫn này giúp:
- Xác định điều kiện tiên quyết trước khi quyết định đầu tư vào bán giám sát.
- Phân loại giải thuật dựa trên bản chất dữ liệu, quy mô mẫu, giả định toán học và ràng buộc tài nguyên tính toán.
- Đưa ra giải pháp dự phòng (Fallback Strategy) an toàn trong môi trường sản xuất.

---

## Input

Việc lựa chọn thuật toán đòi hỏi phân tích toàn diện 4 chiều thông tin đầu vào của dự án:

1. **Quy mô và Tỷ lệ Dữ liệu:**
   - Số lượng mẫu có nhãn $`N_L`$ và số lượng mẫu không nhãn $`N_U`$.
   - Tỷ lệ gán nhãn:
   
$$
\eta = \frac{N_L}{N_L + N_U}
$$

   - Số chiều đặc trưng $`d`$ (chiều thấp $`d < 50`$, chiều trung bình $`50 \le d \le 1000`$, hay siêu cao chiều $`d > 1000`$).

2. **Dạng Thức Dữ liệu (Data Modality):**
   - **Dữ liệu dạng bảng (Tabular):** Chứa các đặc trưng số và phân loại, quan hệ phi tuyến phức tạp nhưng không có cấu trúc không gian/thời gian.
   - **Dữ liệu hình ảnh / Âm thanh:** Cấu trúc không gian dày đặc, dễ dàng áp dụng các phép biến đổi tăng cường dữ liệu (Data Augmentation) mà không làm đổi nhãn ngữ nghĩa.
   - **Dữ liệu văn bản (NLP):** Rời rạc, nhạy cảm với ngữ cảnh, phụ thuộc vào mô hình biểu diễn ngôn ngữ (Word Embeddings, Transformers).
   - **Dữ liệu đồ thị / Mạng lưới (Graph/Network):** Đã có sẵn ma trận kề biểu diễn quan hệ liên kết tự nhiên giữa các thực thể.

3. **Tính Khả Thi của các Giả Định Toán Học:**
   - *Smoothness Assumption:* Hai điểm gần nhau trong không gian đặc trưng có xác suất cùng nhãn cao?
   - *Cluster Assumption:* Ranh giới phân chia có đi qua vùng thưa thớt mật độ?
   - *Manifold Assumption:* Dữ liệu nằm trên đa tạp con số chiều thấp hơn?
   - *Multi-view Independence:* Các đặc trưng có thể tách thành 2 góc nhìn độc lập điều kiện không?

4. **Chế Độ Phục Vụ Mục Tiêu (Inference Target):**
   - **Quy nạp (Inductive):** Cần phục vụ API thời gian thực dự đoán nhãn cho các mẫu mới chưa từng thấy trong tương lai.
   - **Quy nạp chuyển vị (Transductive):** Chỉ cần gán nhãn cho chính tập dữ liệu không nhãn $`\mathcal{D}_U`$ hiện có (ví dụ: làm giàu cơ sở dữ liệu nội bộ).

---

## Output

Khung quyết định cung cấp đầu ra rõ ràng:

1. **Thuật toán chính khuyến nghị (Primary Recommended Algorithm):** Giải thuật tối ưu nhất về mặt hiệu năng và độ ổn định cho cấu hình bài toán.
2. **Cấu hình siêu tham số khởi đầu (Default Hyperparameters):** Các giá trị an toàn được kiểm nghiệm thực nghiệm để làm mốc so sánh đầu tiên.
3. **Mô hình dự phòng an toàn (Fallback Baseline):** Mô hình có giám sát thuần túy kèm kiểm định thống kê (Statistical Hypothesis Testing) để quyết định có đưa mô hình bán giám sát vào vận hành hay không.

---

## Evaluation metrics

Các chỉ số dùng để đưa ra quyết định chấp thuận (Acceptance Gate):

1. **Hệ số Tăng trưởng Bán giám sát (SSL Gain):**
   
$$
\Delta_{\mathrm{SSL}} = \mathrm{Metric}(\mathcal{M}_{\mathrm{SSL}}) - \mathrm{Metric}(\mathcal{M}_{\mathrm{Supervised}})
$$

   Nguyên tắc: Chỉ chấp thuận mô hình bán giám sát nếu $`\Delta_{\mathrm{SSL}} \ge 2.0\%`$ và vượt qua kiểm định t-test với $`p\text{-value} < 0.05`$.

2. **Độ ổn định đa mầm ngẫu nhiên (Seed Stability Variance):**
   Phương sai của điểm số đánh giá qua 5 lần chia dữ liệu độc lập. Mô hình tốt phải có $`\sigma^2 \le 0.005`$.

3. **Hệ số Chi phí Tính toán (Computational Overhead Ratio):**
   
$$
\rho_{\mathrm{comp}} = \frac{T_{\mathrm{train}}(\mathcal{M}_{\mathrm{SSL}})}{T_{\mathrm{train}}(\mathcal{M}_{\mathrm{Supervised}})}
$$

   Nếu $`\rho_{\mathrm{comp}} > 20`$ nhưng $`\Delta_{\mathrm{SSL}} < 1\%`$, dự án nên ưu tiên thu thập thêm nhãn thủ công thay vì triển khai SSL phức tạp.

---

## How it works

### 1. Cây Quyết Định Lựa Chọn Thuật Toán Bán Giám Sát

```
                             [BẮT ĐẦU PHÂN TÍCH BÀI TOÁN]
                                          │
                   Có sẵn cấu trúc đồ thị hoặc quan hệ liên kết?
                                    ┌─────┴─────┐
                                  [CÓ]        [KHÔNG]
                                   │             │
                    Số lượng mẫu N > 50,000?    Dạng dữ liệu chính là gì?
                        ┌──────────┴──────────┐  │
                      [CÓ]                  [KHÔNG]
                       │                      │
           [Label Spreading k-NN]   [Label Propagation]
                                              │
                      ┌───────────────────────┼────────────────────────┐
                      ▼                       ▼                        ▼
                [Hình ảnh / Âm thanh]     [Văn bản / NLP]      [Dữ liệu Bảng Tabular]
                      │                       │                        │
            Có thể chạy GPU sâu?      Có pre-trained LLM?    Có 2 tập đặc trưng độc lập?
                ┌─────┴─────┐           ┌─────┴─────┐            ┌─────┴─────┐
              [CÓ]        [KHÔNG]     [CÓ]        [KHÔNG]      [CÓ]        [KHÔNG]
               │             │         │             │          │             │
          [FixMatch]     [Self-Train] [UDA]     [Self-Train] [Co-Train]  N_U < 10,000?
                                                                         ┌────┴────┐
                                                                       [CÓ]      [KHÔNG]
                                                                        │           │
                                                                     [S3VM]   [Self-Train]
```

---

### 2. Ma Trận So Sánh Toàn Diện Các Giải Thuật Bán Giám Sát

| Thuật toán | Loại hình học | Dạng dữ liệu tối ưu | Quy mô mẫu ($N$) | Ưu điểm cốt lõi | Rủi ro chính |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Self-Training** | Inductive | Bảng, Văn bản | $N > 1,000,000$ | Dễ cài đặt, tương thích mọi base classifier | Tích lũy lỗi (Confirmation bias) |
| **Co-Training** | Inductive | Web, Đa cảm biến | $N < 100,000$ | Hai view sửa lỗi chéo lẫn nhau | Đòi hỏi điều kiện chia view độc lập khắt khe |
| **Tri-Training** | Inductive | Bảng, Y tế | $N < 100,000$ | Không cần chia view, cơ chế bỏ phiếu 2/3 an toàn | Chi phí nhân 3 thời gian huấn luyện |
| **Label Propagation** | Transductive | Đồ thị, Mạng xã hội | $N < 20,000$ | Rất nhanh, hội tụ nghiệm đóng | Nhạy cảm với nhiễu nhãn ban đầu |
| **Label Spreading** | Transductive | Đồ thị, Phân cụm | $N < 30,000$ | Kháng nhiễu cực tốt nhờ hệ số kẹp $\alpha$ | Tốn bộ nhớ $O(N^2)$ ma trận kề |
| **S3VM / TSVM** | Transductive | Bảng, Hóa học | $N < 5,000$ | Cực đại hóa lề bao gồm cả mẫu không nhãn | Độ phức tạp tổ hợp NP-hard, chậm với $N$ lớn |
| **Generative SSL** | Inductive | Mật độ, Ảnh, Tabular | $N < 50,000$ | Tận dụng mô hình sinh phân phối $p(x)$ | Dễ sụp đổ nếu giả định phân phối sai lệch |
| **Consistency / FixMatch**| Inductive | Ảnh, Âm thanh, Text | $N > 10,000$ | Hiệu năng SOTA, tận dụng tối đa dữ liệu lớn | Đòi hỏi tài nguyên GPU, thiết kế augmentation mạnh |

---

## Data used

Bảng hướng dẫn ánh xạ các bài toán công nghiệp điển hình vào giải thuật SSL phù hợp:

| Miền ứng dụng | Đặc điểm bài toán | Dữ liệu sẵn có | Thuật toán khuyến nghị |
| :--- | :--- | :--- | :--- |
| **Y tế & Sinh học** | Ảnh X-ray/MRI ít nhãn do chuyên gia bận | 100 ảnh có nhãn, 5,000 ảnh không nhãn | **FixMatch** hoặc **Mean Teacher** |
| **Phát hiện Gian lận** | Giao dịch tài chính dạng bảng, tỷ lệ lừa đảo cực thấp | 500 ca gian lận xác nhận, 1,000,000 ca chưa rõ | **Self-Training** kết hợp XGBoost/LightGBM |
| **Phân loại Trang web** | Mỗi trang có URL, Anchor text và Nội dung văn bản | 200 trang phân loại, 50,000 trang thu thập web | **Co-Training** (View 1: Text, View 2: Inbound links) |
| **Hệ thống Gợi ý** | Người dùng và sản phẩm trên đồ thị quan hệ tương tác | 1% người dùng điền khảo sát sở thích | **Label Spreading** trên đồ thị User-Item |
| **Phân tích Cảm xúc** | Bình luận khách hàng trên mạng xã hội | 1,000 bình luận gắn nhãn, 100,000 bình luận thô | **UDA** hoặc **Consistency Regularization** trên RoBERTa |

---

## Design

### Hệ Thống Cố Vấn Tự Động (SSL Algorithm Advisor Module)

Đoạn mã dưới đây triển khai một module chuyên gia bằng Python, hỗ trợ thẩm định và đề xuất giải thuật bán giám sát tối ưu dựa trên thông số dự án:

```python
class SSLAlgorithmAdvisor:
    """Hệ thống cố vấn lựa chọn thuật toán Học Bán Giám Sát chuẩn hóa."""
    
    @staticmethod
    def advise(
        modality: str,            # 'tabular', 'image', 'text', 'graph'
        n_labeled: int,
        n_unlabeled: int,
        has_multi_view: bool = False,
        is_transductive_only: bool = False,
        gpu_available: bool = False
    ) -> dict:
        total_n = n_labeled + n_unlabeled
        label_ratio = n_labeled / total_n if total_n > 0 else 0
        
        # 1. Kiểm tra an toàn tỷ lệ nhãn
        if label_ratio > 0.5:
            return {
                "decision": "USE_SUPERVISED",
                "primary": "Standard Supervised Model",
                "reason": f"Tỷ lệ nhãn cao ({label_ratio*100:.1f}%). Dữ liệu có giám sát đã đủ mạnh, SSL không mang lại lợi ích biên đáng kể."
            }
            
        # 2. Xử lý dữ liệu Đồ thị (Graph)
        if modality == 'graph' or is_transductive_only:
            if total_n <= 30000:
                return {
                    "decision": "USE_GRAPH_SSL",
                    "primary": "Label Spreading",
                    "fallback": "Label Propagation",
                    "params": {"kernel": "rbf", "alpha": 0.2, "gamma": 0.1},
                    "reason": "Dữ liệu có quan hệ liên kết hình học hoặc bài toán quy nạp chuyển vị quy mô vừa."
                }
            else:
                return {
                    "decision": "USE_SPARSE_GRAPH_SSL",
                    "primary": "Label Spreading (k-NN Sparse Graph)",
                    "params": {"kernel": "knn", "n_neighbors": 7, "alpha": 0.2},
                    "reason": "Quy mô lớn cần dùng đồ thị thưa k-NN để tránh quá tải bộ nhớ O(N^2)."
                }
                
        # 3. Xử lý dữ liệu Hình ảnh / Âm thanh
        if modality == 'image':
            if gpu_available and n_unlabeled >= 1000:
                return {
                    "decision": "USE_DEEP_SSL",
                    "primary": "FixMatch",
                    "fallback": "MixMatch",
                    "params": {"threshold": 0.95, "mu": 7, "lambda_u": 1.0},
                    "reason": "Dữ liệu ảnh giàu tiềm năng tăng cường (Weak-Strong Augmentation), GPU đáp ứng tốt."
                }
            else:
                return {
                    "decision": "USE_SELF_TRAINING",
                    "primary": "SelfTrainingClassifier (Base: ResNet/Pretrained)",
                    "params": {"threshold": 0.90, "max_iter": 10},
                    "reason": "Hạn chế tài nguyên phần cứng, sử dụng trích xuất đặc trưng có sẵn kết hợp Self-Training."
                }
                
        # 4. Xử lý dữ liệu Bảng (Tabular)
        if modality == 'tabular':
            if has_multi_view:
                return {
                    "decision": "USE_MULTI_VIEW",
                    "primary": "Co-Training",
                    "fallback": "Tri-Training",
                    "reason": "Dữ liệu thỏa mãn điều kiện phân rã thành 2 góc nhìn độc lập điều kiện."
                }
            elif total_n <= 3000:
                return {
                    "decision": "USE_TSVM_OR_TRI_TRAINING",
                    "primary": "Tri-Training",
                    "fallback": "Transductive SVM (TSVM)",
                    "reason": "Quy mô nhỏ, cơ chế bỏ phiếu 3 mô hình bảo vệ khỏi tích lũy lỗi gán nhãn sai."
                }
            else:
                return {
                    "decision": "USE_SELF_TRAINING",
                    "primary": "SelfTrainingClassifier (Base: LightGBM / Logistic Regression)",
                    "fallback": "Supervised Baseline with Class Weighting",
                    "params": {"threshold": 0.85, "k_best": 50},
                    "reason": "Dữ liệu bảng quy mô lớn, tương thích tối đa với mô hình cây quyết định gradient boosting."
                }
                
        # 5. Mặc định
        return {
            "decision": "USE_SELF_TRAINING",
            "primary": "SelfTrainingClassifier",
            "fallback": "Supervised Baseline"
        }

# Ví dụ minh họa thực thi
if __name__ == "__main__":
    cases = [
        {"name": "Ảnh Y tế", "modality": "image", "n_labeled": 50, "n_unlabeled": 5000, "gpu_available": True},
        {"name": "Bảng Gian lận", "modality": "tabular", "n_labeled": 200, "n_unlabeled": 50000, "gpu_available": False},
        {"name": "Phân loại Web", "modality": "tabular", "n_labeled": 100, "n_unlabeled": 2000, "has_multi_view": True},
    ]
    
    print("=== TƯ VẤN THUẬT TOÁN BÁN GIÁM SÁT TỰ ĐỘNG ===")
    for c in cases:
        rec = SSLAlgorithmAdvisor.advise(
            c["modality"], c["n_labeled"], c["n_unlabeled"],
            has_multi_view=c.get("has_multi_view", False),
            gpu_available=c.get("gpu_available", False)
        )
        print(f"\nDự án: {c['name']}")
        print(f"-> Quyết định: {rec['decision']}")
        print(f"-> Thuật toán chính: {rec['primary']}")
        print(f"-> Lý do: {rec.get('reason', '')}")
```

---

## Nguồn

- [1] Zhu, X. (2005). *Semi-supervised learning literature survey*. Computer Sciences Technical Report 1530, University of Wisconsin-Madison.
- [2] Li, Y. F., & Zhou, Z. H. (2014). *Towards making unlabeled data never hurt*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(1), 175-188.
- [3] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I. (2018). *Realistic evaluation of deep semi-supervised learning algorithms*. Advances in Neural Information Processing Systems (NeurIPS), 31, 3235-3246.

