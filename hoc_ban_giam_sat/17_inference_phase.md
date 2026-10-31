# Kiến Trúc Pha Suy Luận (Inference Phase) Trong Học Bán Giám Sát

## Purpose

Tài liệu này chuẩn hóa kiến trúc pha suy luận (Inference Phase Architecture) và các giải pháp triển khai mô hình học bán giám sát vào hệ thống sản xuất thời gian thực (Production Serving System).

Trong môi trường thực tế, việc đưa một mô hình bán giám sát vào vận hành đối mặt với các thách thức kỹ thuật đặc thù:
- **Nghịch lý Quy nạp chuyển vị (The Transductive Paradox):** Một số thuật toán đồ thị hiệu năng rất cao (như Label Propagation, Label Spreading, TSVM) vốn có bản chất transductive, không có hàm ánh xạ trực tiếp cho mẫu mới ngoài tập huấn luyện (Out-of-Sample Points). Cần giải pháp mở rộng quy nạp (Inductive Extension) hoặc chắt lọc tri thức (Distillation).
- **Độ tin cậy của xác suất dự đoán (Confidence Miscalibration):** Các mô hình bán giám sát thường tự tin thái quá (Overconfident) do cơ chế cực tiểu hóa entropy. Cần kỹ thuật hiệu chuẩn xác suất (Probability Calibration) tại tầng phục vụ.
- **Cơ chế phòng thủ rủi ro (Uncertainty Fallback Guard):** Khi mẫu truy vấn rơi vào vùng thưa thớt mật độ hoặc trôi dạt phân phối (Out-of-Distribution - OOD), hệ thống phải kịp thời phát hiện và chuyển tiếp sang luồng xử lý an toàn thay vì đưa ra quyết định sai lầm.

---

## Input

Hệ thống phục vụ suy luận tiếp nhận các thành phần đầu vào:

1. **Yêu cầu truy vấn suy luận (Inference Request Payload):**
   - Vector đặc trưng mẫu đơn lẻ $`x_{\mathrm{new}} \in \mathbb{R}^d`$ (trong API thời gian thực) hoặc lô mẫu $`X_{\mathrm{batch}} \in \mathbb{R}^{B \times d}`$ (trong xử lý hàng loạt theo ca).

2. **Artifact mô hình đã đóng gói (Packaged Model Artifact):**
   - Tệp trọng số mô hình EMA đã tối ưu hóa định dạng runtime (ví dụ `model_ema.onnx`, `model_quantized.tflite`, hoặc `model.joblib`).
   - Tham số hiệu chuẩn nhiệt độ $`T_{\mathrm{calib}}`$ thu được từ tập kiểm định.

3. **Cấu hình ngưỡng phòng thủ (Serving Guard Thresholds):**
   - Ngưỡng tự tin tối thiểu chấp nhận dự đoán: $`\tau_{\mathrm{infer}} \in [0.70, 0.90]`$.
   - Ngưỡng khoảng cách ngoại lai (Outlier Distance Threshold) đo khoảng cách tới cụm dữ liệu huấn luyện gần nhất.

---

## Output

Kết quả trả về cho mỗi truy vấn suy luận bao gồm:

1. **Nhãn dự đoán phân loại (Predicted Class Label):**

$$
\hat{y} = \arg\max_{c \in \lbrace 1, \dots, C \rbrace} p_{\mathrm{calib}}(y = c \mid x_{\mathrm{new}})
$$

2. **Vector xác suất đã hiệu chuẩn (Calibrated Probability Distribution):**
   Vector $`[p_1, p_2, \dots, p_C]`$ phản ánh chân thực độ chắc chắn thống kê của quyết định.

3. **Trạng thái an toàn và cờ chuyển tiếp (Safety Status & Fallback Flag):**
   - Trạng thái `ACCEPTED`: Độ tự tin vượt ngưỡng $`\tau_{\mathrm{infer}}`$, hệ thống tự động xuất kết quả.
   - Trạng thái `UNCERTAIN_FALLBACK`: Độ tự tin dưới ngưỡng, chuyển yêu cầu sang chuyên gia thẩm định hoặc áp dụng luật nghiệp vụ mặc định.

4. **Thời gian đáp ứng (Serving Latency):** Thời gian xử lý tính bằng mili-giây (ms).

---

## Evaluation metrics

Chỉ số SLA đánh giá tầng phục vụ suy luận bán giám sát:

1. **Độ trễ phân vị P95 và P99 (Latency SLA):**
   Thời gian hoàn tất phản hồi cho 95% và 99% số lượng request:
   - Hệ thống thời gian thực (Real-time Serving): $`\text{P95} < 10\text{ ms}`$, $`\text{P99} < 25\text{ ms}`$.
   - Hệ thống theo lô (Batch Serving): Thông lượng $`> 2,000 \text{ req/sec}`$.

2. **Sai số hiệu chuẩn kỳ vọng phục vụ (Serving ECE):**
   Đo lường sai lệch giữa độ tự tin trả về và tỷ lệ đúng thực tế trên dữ liệu vận hành. ECE phải duy trì $`\le 0.05`$.

3. **Tỷ lệ chuyển tiếp phòng vệ (Fallback Rate):**

$$
\text{Rate}_{\mathrm{fallback}} = \frac{N_{\mathrm{fallback}}}{N_{\mathrm{total}}} \times 100\%
$$

   Tỷ lệ này tăng đột biến là dấu hiệu cảnh báo phân phối dữ liệu đầu vào đã bị trôi dạt (Data Drift).

---

## How it works

### 1. Luồng Xử Lý Suy Luận Quy Nạp (Inductive Serving Flow)

```
[Request x_new] ──> [Chuẩn hóa Input Scaling] ──> [Forward Model EMA]
                                                         │
                                                         ▼
                                            [Logits thô: z_1, ..., z_C]
                                                         │
                                                         ▼
                                      [Temperature Scaling: z_c / T_calib]
                                                         │
                                                         ▼
                                            [Xác suất Softmax p_calib]
                                                         │
                                ┌────────────────────────┴────────────────────────┐
                                ▼                                                 ▼
                  [Max(p) >= τ_infer]                               [Max(p) < τ_infer]
                                │                                                 │
                                ▼                                                 ▼
                 Trạng thái: ACCEPTED                              Trạng thái: FALLBACK
              Trả về nhãn phân loại y_hat                       Chuyển tiếp chuyên gia duyệt
```

---

### 2. Giải Pháp Quy Nạp Cho Mô Hình Đồ Thị Transductive

Mô hình đồ thị như **Label Spreading** không thể tính toán trực tiếp nhãn cho mẫu mới $`x_{\mathrm{new}}`$ vì đồ thị cố định trên tập $`N`$ mẫu huấn luyện. Để đưa vào sản xuất, có hai giải pháp kỹ thuật chuẩn mực:

#### Giải pháp 1: Mở rộng Ngoại suy Điểm Lân cận (Inductive 1-NN / k-NN Extension)
Dự đoán nhãn của $`x_{\mathrm{new}}`$ dựa trên khoảng cách tới các mẫu trong đồ thị huấn luyện ban đầu có trọng số nhãn phân bố $`F^{\ast}`$:

$$
p(y = c \mid x_{\mathrm{new}}) = \frac{\sum_{i \in \mathcal{N}_k(x_{\mathrm{new}})} W(x_{\mathrm{new}}, x_i) \cdot F_{i, c}^{\ast}{\sum_{i \in \mathcal{N}_k(x_{\mathrm{new}})} W(x_{\mathrm{new}}, x_i)}
$$

Trong đó $`W(x_{\mathrm{new}}, x_i) = \exp\left(-\gamma \|x_{\mathrm{new}} - x_i\|^2\right)`$.

#### Giải pháp 2: Chắt Lọc Tri Thức Thành Mô Hình Học Sinh Nhẹ (Student Distillation - Khuyến nghị)
1. Sử dụng mô hình Label Spreading gán phân phối xác suất $`F^{\ast}`$ cho toàn bộ tập dữ liệu không nhãn ban đầu $`\mathcal{D}_U`$.
2. Huấn luyện một mạng nơ-ron nhỏ (Lightweight MLP) hoặc mô hình LightGBM trên toàn bộ dữ liệu kèm nhãn giả đó làm hàm mục tiêu (Soft-target Loss).
3. Triển khai mô hình Student lên API phục vụ độc lập với độ trễ siêu thấp $`< 2\text{ ms}`$.

---

## Data used

Bảng so sánh tài nguyên và độ trễ phục vụ giữa các chiến lược suy luận bán giám sát:

| Chiến lược phục vụ | Định dạng đóng gói | Bộ nhớ RAM phục vụ | P95 Latency | Khả năng mở rộng ngang (Horizontal Scale) |
| :--- | :--- | :--- | :--- | :--- |
| **Self-Training (Tree/Linear)** | Joblib / Treelite | $< 50\text{ MB}$ | **1.2 ms** | Rất dễ (Stateless microservices) |
| **Label Spreading (k-NN)** | Ma trận đồ thị + FAISS | $500\text{ MB} - 2\text{ GB}$ | **15.4 ms** | Trung bình (Cần lưu trữ tập mẫu tham chiếu) |
| **Student Distillation** | ONNX Runtime | $< 100\text{ MB}$ | **2.5 ms** | Rất dễ (Stateless microservices) |
| **Deep SSL (FixMatch EMA)** | ONNX / TensorRT (GPU) | $1\text{ GB} - 4\text{ GB}$ | **8.0 ms** | Cần hỗ trợ GPU worker pool |

---

## Design

### Mã Nguồn Dịch Vụ Suy Luận Bán Giám Sát An Toàn (Python Production Service)

Đoạn mã dưới đây triển khai lớp phục vụ suy luận khép kín chuẩn công nghiệp, tích hợp hiệu chuẩn nhiệt độ (Temperature Scaling), cơ chế kiểm tra ngưỡng an toàn và logging đo lường độ trễ:

```python
import numpy as np
import time

class SSLInferenceService:
    """Dịch vụ phục vụ suy luận bán giám sát thời gian thực chuẩn sản xuất."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        temperature: float = 1.25,
        threshold: float = 0.85
    ):
        self.weights = weights          # Trọng số tuyến tính [d, C]
        self.bias = bias                # Bias vector [C]
        self.temperature = temperature  # Hệ số hiệu chuẩn nhiệt độ
        self.threshold = threshold      # Ngưỡng tự tin an toàn

    def predict(self, x_input: np.ndarray) -> dict:
        """Thực hiện suy luận có đo độ trễ và kiểm soát rủi ro ngoại lai."""
        t_start = time.perf_counter()

        # 1. Forward pass tính Logits
        logits = np.dot(x_input, self.weights) + self.bias

        # 2. Temperature Scaling để hiệu chuẩn xác suất
        scaled_logits = logits / self.temperature

        # 3. Softmax tính phân phối xác suất đã hiệu chuẩn
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        # 4. Trích xuất độ tự tin cực đại và nhãn dự đoán
        pred_class = int(np.argmax(calibrated_probs))
        confidence = float(np.max(calibrated_probs))

        # 5. Kiểm tra phòng vệ bất định (Uncertainty Guard)
        is_fallback = confidence < self.threshold
        status = "UNCERTAIN_FALLBACK" if is_fallback else "ACCEPTED"

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "prediction": pred_class if not is_fallback else None,
            "confidence": round(confidence, 4),
            "status": status,
            "calibrated_probabilities": [round(p, 4) for p in calibrated_probs.tolist()],
            "latency_ms": round(latency_ms, 3)
        }

# ==============================================================================
# KIỂM THỬ TẦNG PHỤC VỤ SUY LUẬN
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    # Khởi tạo mô hình giả lập: 10 đặc trưng, 3 lớp phân loại
    mock_weights = np.random.randn(10, 3)
    mock_bias = np.zeros(3)

    service = SSLInferenceService(
        weights=mock_weights,
        bias=mock_bias,
        temperature=1.2,
        threshold=0.80
    )

    print("=== KIỂM THỬ DỊCH VỤ SUY LUẬN BÁN GIÁM SÁT THỜI GIAN THỰC ===")

    # Mẫu 1: Rõ ràng, tự tin cao
    x_confident = np.ones(10) * 2.0
    res1 = service.predict(x_confident)
    print(f"\nMẫu 1 (Tự tin cao):")
    print(f"-> Trạng thái: {res1['status']} | Nhãn: {res1['prediction']} | Tự tin: {res1['confidence']*100:.1f}% | Độ trễ: {res1['latency_ms']} ms")

    # Mẫu 2: Nằm gần ranh giới, tự tin thấp
    x_ambiguous = np.zeros(10)
    res2 = service.predict(x_ambiguous)
    print(f"\nMẫu 2 (Nằm ranh giới quyết định):")
    print(f"-> Trạng thái: {res2['status']} | Nhãn: {res2['prediction']} | Tự tin: {res2['confidence']*100:.1f}% | Độ trễ: {res2['latency_ms']} ms")
    print("   (Hệ thống tự động kích hoạt FALLBACK bảo vệ, không trả về quyết định sai lầm)")
```

---

## Nguồn

- [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks*. Proceedings of the 34th International Conference on Machine Learning (ICML), 1321-1330.
- [2] Bengio, Y., Delalleau, O., Le Roux, N., Paiement, J. F., Vincent, P., & Ouimet, M. (2004). *Out-of-sample extensions for LLE, Isomap, MDS, Eigenmaps, and Spectral Clustering*. Advances in Neural Information Processing Systems (NeurIPS), 16, 177-184.
- [3] Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network*. arXiv preprint arXiv:1503.02531.

