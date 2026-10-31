# Kiến Trúc Pha Huấn Luyện (Training Phase) Trong Học Bán Giám Sát

## Purpose

Tài liệu này chuẩn hóa kiến trúc pha huấn luyện (Training Phase Architecture) cho các mô hình học bán giám sát (Semi-Supervised Learning - SSL).

Khác biệt căn bản giữa huấn luyện có giám sát truyền thống và huấn luyện bán giám sát nằm ở sự phối hợp động giữa hai luồng dữ liệu có nhãn $`\mathcal{D}_L`$ và không nhãn $`\mathcal{D}_U`$. Nếu không có cơ chế điều phối khoa học, quá trình huấn luyện bán giám sát rất dễ gặp các hiện tượng lỗi nghiêm trọng:
- **Tích lũy sai lệch (Confirmation Bias):** Mô hình tự củng cố các giả nhãn sai ban đầu, dẫn đến việc tối ưu hóa trên dữ liệu rác.
- **Sụp đổ phân phối (Class Collapse / Representation Collapse):** Mô hình gán toàn bộ dữ liệu không nhãn vào một lớp dễ học duy nhất để giảm thiểu cực tiểu hàm mất mát cục bộ.
- **Bất ổn định gradient đầu kỳ (Early Gradient Instability):** Gradient phát sinh từ các mẫu không nhãn có độ tự tin giả tạo lấn át hoàn toàn tín hiệu giám sát từ tập có nhãn thật.

Khung kiến trúc này thiết lập quy trình chuẩn mực từ lập lịch trọng số (Loss Scheduling), cân bằng kích thước batch, cập nhật trọng số trung bình động (EMA), đến cơ chế dừng sớm an toàn (Safe Early Stopping).

---

## Input

Pha huấn luyện bán giám sát tiếp nhận các thành phần đầu vào:

1. **Bộ nạp dữ liệu kép (Dual Dataloader Pipeline):**
   - **Luồng có nhãn (Labeled Loader):** Tạo các mini-batch kích thước $`B_L`$. Dữ liệu được lặp tuần hoàn (Cycling/Repeat) để đảm bảo không bị gián đoạn khi duyệt hết epoch.
   - **Luồng không nhãn (Unlabeled Loader):** Tạo các mini-batch kích thước $`B_U = \mu \cdot B_L`$, trong đó $`\mu \ge 1`$ là tỷ lệ mở rộng không nhãn (thường $`\mu \in [3, 7]`$).

2. **Mô hình học máy và Mạng bóng EMA:**
   - Mạng nơ-ron chính $`f_\theta`$ với tham số $`\theta`$ được cập nhật qua đạo hàm riêng (Stochastic Gradient Descent).
   - Mô hình trung bình động hàm mũ (Exponential Moving Average) $`f_{\theta_{\mathrm{EMA}}}`$ với hệ số phân rã $`\beta_{\mathrm{EMA}} \in [0.999, 0.9999]`$.

3. **Chiến lược lập lịch hàm mất mát (Loss Ramp-up Schedule):**
   - Hàm tăng dần hệ số không nhãn $`\lambda_U(t)`$ theo bước huấn luyện $`t`$.
   - Thời gian khởi động (Warm-up Steps $`T_{\mathrm{warm}}`$) trong đó mô hình chỉ tập trung học từ nhãn thật.

4. **Tập kiểm định độc lập (Hold-out Validation Set $`\mathcal{D}_{\mathrm{val}}`$):**
   Gồm các mẫu có nhãn chuẩn xác 100%, độc lập tuyệt đối với cả $`\mathcal{D}_L`$ và $`\mathcal{D}_U`$, dùng để giám sát khách quan quá trình hội tụ.

---

## Output

Kết thúc pha huấn luyện, hệ thống bàn giao các thành phần đóng gói (Training Artifacts):

1. **Trọng số tối ưu của mô hình EMA ($`\theta_{\mathrm{EMA}}^{\ast}`$):** Mô hình đã được làm mịn phương sai, có khả năng tổng quát hóa cao nhất trên tập kiểm định.
2. **Nhật ký huấn luyện định lượng (Training Logs):**
   - Lịch sử suy giảm của hàm mất mát có giám sát $`\mathcal{L}_s`$ và mất mát không nhãn $`\mathcal{L}_u`$.
   - Tỷ lệ kích hoạt mặt nạ tự tin (Mask Utilization Rate) theo từng epoch.
   - Độ chính xác thực tế của giả nhãn (Pseudo-Label Accuracy Tracking).
3. **Thẻ siêu tham số và phiên bản (Model Metadata Artifact):** Ghi nhận cấu hình $`\mu, \tau, \lambda_U, \beta_{\mathrm{EMA}}`$, thời gian thực thi và đường dẫn checkpoint.

---

## Evaluation metrics

Các chỉ số giám sát trực tiếp trong pha huấn luyện:

1. **Mất mát kiểm định có giám sát (Validation Cross-Entropy Loss):**
   Chỉ số quyết định để lưu Checkpoint và kích hoạt dừng sớm:

$$
\mathcal{L}_{\mathrm{val}}(\theta) = -\frac{1}{N_{\mathrm{val}}} \sum_{i=1}^{N_{\mathrm{val}}} \log f_\theta(x_i^{\mathrm{val}})_{y_i^{\mathrm{val}}}
$$

2. **Tỷ lệ tận dụng mặt nạ (Mask Utilization Rate $`\mathcal{U}_t`$):**
   Tỷ lệ phần trăm các mẫu không nhãn vượt qua ngưỡng cắt tự tin $`\tau`$:

$$
\mathcal{U}_t = \frac{1}{B_U} \sum_{b=1}^{B_U} \mathbb{I}\left(\max_{c} q_{b, c}^{(t)} \ge \tau\right) \times 100\%
$$

   Quỹ đạo lý tưởng: Tăng dần từ $0\% \to 5\%$ trong giai đoạn warm-up lên $70\% \to 90\%$ khi mô hình hội tụ.

3. **Tỷ lệ phân bổ lớp của giả nhãn (Class Distribution Balance):**
   Đo lường độ lệch chuẩn của số lượng giả nhãn gán cho từng lớp. Nếu một lớp chiếm $`> 60\%`$ tổng số giả nhãn trong khi dữ liệu gốc cân bằng, hệ thống sẽ cảnh báo nguy cơ sụp đổ phân phối.

---

## How it works

### 1. Sơ đồ Luồng Hoạt Động của Pha Huấn Luyện

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CHU KỲ HUẤN LUYỆN BÁN GIÁM SÁT                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Labeled Loader]      [Unlabeled Loader]                                   │
│         │                       │                                           │
│  Mini-batch B_L          Mini-batch B_U (μ * B_L)                           │
│         │                       │                                           │
│         ▼                       ▼                                           │
│  Compute Loss L_s        Weak/Strong Transforms                             │
│         │                       │                                           │
│         │                Predict & Filter: Max(q) >= τ                      │
│         │                       │                                           │
│         │                Compute Loss L_u with Mask                         │
│         │                       │                                           │
│         └───────────────┬───────┘                                           │
│                         ▼                                                   │
│            Total Loss: L = L_s + λ_u(t) * L_u                               │
│                         │                                                   │
│                         ▼                                                   │
│            Backpropagation (SGD / AdamW)                                    │
│                         │                                                   │
│                         ▼                                                   │
│            EMA Weights Update: θ_ema = β * θ_ema + (1-β) * θ                │
│                         │                                                   │
│                         ▼                                                   │
│         Periodic Evaluation on D_val -> Save Best Model Checkpoint          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Các Cơ Chế Cốt Lõi Trong Pha Huấn Luyện

#### A. Hàm Lập Lịch Tăng Dần Hệ Số Không Nhãn (Ramp-up Scheduling)
Trong những bước đầu tiên của quá trình huấn luyện, mô hình chưa có khả năng phân loại chuẩn xác. Nếu áp dụng toàn bộ mất mát không nhãn ngay lập tức, các giả nhãn sai sẽ phá vỡ không gian đặc trưng. Do đó, hệ số $`\lambda_U(t)`$ được điều chỉnh tăng dần:

$$
\lambda_U(t) = \lambda_{\max} \cdot \exp\left( -5 \cdot \left(1 - \frac{t}{T_{\mathrm{warm}}}\right)^2 \right) \quad \text{với } t < T_{\mathrm{warm}}
$$

Khi $`t \ge T_{\mathrm{warm}}`$, hệ số $`\lambda_U(t) = \lambda_{\max}`$.

#### B. Cân Bằng Tỷ Lệ Batch Dữ Liệu ($\mu$)
Tập dữ liệu không nhãn thường có quy mô lớn hơn nhiều so với tập có nhãn. Việc chọn tỷ lệ $`\mu = B_U / B_L \in [3, 7]`$ mang lại hai lợi ích:
1. Đảm bảo mỗi bước cập nhật gradient đều có sự hiện diện của tín hiệu chuẩn từ tập $`\mathcal{D}_L`$.
2. Khai thác đa dạng nhiều mẫu không nhãn cùng lúc để ước lượng gradient không nhãn có phương sai thấp.

#### C. Cập Nhật Trọng Số Trung Bình Động Hàm Mũ (EMA Updating)
Mô hình chính $`\theta`$ dao động liên tục giữa các bước do nhiễu từ giả nhãn. Mô hình EMA lưu trữ trung bình lịch sử của các trọng số:

$$
\theta_{\mathrm{EMA}}^{(t)} = \beta_{\mathrm{EMA}} \cdot \theta_{\mathrm{EMA}}^{(t-1)} + (1 - \beta_{\mathrm{EMA}}) \cdot \theta^{(t)}
$$

Trọng số $`\theta_{\mathrm{EMA}}`$ có bề mặt mất mát (loss landscape) phẳng hơn và khả năng chống quá khớp vượt trội.

---

## Data used

Bảng tham số cấu hình pha huấn luyện chuẩn mực cho các quy mô bài toán khác nhau:

| Quy mô bài toán | Kích thước $ `B_L` $ | Tỷ lệ $\mu$ | Kích thước $ `B_U` $ | Tốc độ học (lr) | Warm-up Steps | Ngưỡng $\tau$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nhỏ (Small Tabular)** | 32 | 2 | 64 | 1e-3 | 500 steps | 0.80 |
| **Vừa (Medium Vision)** | 64 | 5 | 320 | 3e-4 | 2,000 steps | 0.90 |
| **Lớn (Large SOTA)** | 64 | 7 | 448 | 3e-4 (Cosine) | 5,000 steps | 0.95 |

---

## Design

### Mã Nguồn Module Huấn Luyện Bán Giám Sát Chuẩn Hóa (Python / PyTorch-style)

Đoạn mã sau mô phỏng toàn bộ logic điều phối pha huấn luyện bán giám sát, bao gồm hàm lập lịch ramp-up, bộ cập nhật EMA, bộ theo dõi chỉ số và cơ chế kiểm soát mặt nạ:

```python
import numpy as np
import time

class SSLTrainingCoordinator:
    """Module điều phối pha huấn luyện bán giám sát chuẩn công nghiệp."""

    def __init__(
        self,
        max_steps: int = 10000,
        warmup_steps: int = 2000,
        lambda_max: float = 1.0,
        threshold: float = 0.95,
        ema_decay: float = 0.999
    ):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.lambda_max = lambda_max
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.history = []

    def get_current_lambda_u(self, current_step: int) -> float:
        """Hàm tính hệ số phạt không nhãn theo lịch trình Sigmoid Ramp-up."""
        if current_step >= self.warmup_steps:
            return self.lambda_max
        # Công thức Sigmoid Ramp-up kinh điển
        phase = 1.0 - (current_step / self.warmup_steps)
        return float(self.lambda_max * np.exp(-5.0 * phase * phase))

    def update_ema_weights(self, student_weights: np.ndarray, teacher_weights: np.ndarray) -> np.ndarray:
        """Cập nhật trọng số trung bình động hàm mũ (EMA)."""
        return self.ema_decay * teacher_weights + (1.0 - self.ema_decay) * student_weights

    def compute_mask_utilization(self, confidence_scores: np.ndarray) -> tuple:
        """Lọc mẫu tự tin và tính toán tỷ lệ kích hoạt mặt nạ."""
        mask = confidence_scores >= self.threshold
        utilization_rate = np.mean(mask)
        return mask, utilization_rate

    def simulate_training_loop(self):
        """Mô phỏng chu kỳ huấn luyện 5 epochs để kiểm tra động thái chỉ số."""
        print("=== BẮT ĐẦU MÔ PHỎNG PHA HUẤN LUYỆN BÁN GIÁM SÁT ===")
        print(f"Ngưỡng tự tin tau: {self.threshold} | Warmup Steps: {self.warmup_steps} | Lambda Max: {self.lambda_max}\n")

        # Giả lập tham số mô hình
        student_w = np.ones(5)
        teacher_w = np.ones(5)

        steps = [500, 1000, 2000, 5000, 10000]
        for step in steps:
            lambda_u = self.get_current_lambda_u(step)

            # Giả lập độ tự tin của mô hình tăng dần theo số step
            simulated_conf = np.random.beta(a=1 + step/2000, b=2, size=100)
            mask, util_rate = self.compute_mask_utilization(simulated_conf)

            # Cập nhật trọng số
            student_w += np.random.normal(0, 0.01, size=5)
            teacher_w = self.update_ema_weights(student_w, teacher_w)

            # Giả lập loss
            loss_s = max(0.1, 2.0 - step / 6000)
            loss_u = max(0.05, 1.5 - step / 8000)
            total_loss = loss_s + lambda_u * loss_u

            print(f"Step {step:05d} | Lambda_u: {lambda_u:.4f} | Mask Rate: {util_rate*100:>5.1f}% | Loss_s: {loss_s:.4f} | Loss_u: {loss_u:.4f} | Total Loss: {total_loss:.4f}")

# Thực thi kiểm thử
if __name__ == "__main__":
    coordinator = SSLTrainingCoordinator(
        max_steps=10000,
        warmup_steps=2000,
        lambda_max=1.0,
        threshold=0.90
    )
    coordinator.simulate_training_loop()
    print("\n=> Pha huấn luyện được thiết kế khép kín, ổn định gradient và bảo toàn cấu trúc tối ưu.")
```

---

## Nguồn

- [1] Laine, S., & Aila, T. (2017). *Temporal ensembling for semi-supervised learning*. International Conference on Learning Representations (ICLR).
- [2] Tarvainen, A., & Valpola, H. (2017). *Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results*. Advances in Neural Information Processing Systems (NeurIPS), 30, 1195-1204.
- [3] Sohn, K., et al. (2020). *FixMatch: Simplifying semi-supervised learning with consistency and confidence*. Advances in Neural Information Processing Systems (NeurIPS), 33, 596-608.
