# Học Bán Giám Sát Sâu (Deep Semi-Supervised Learning)

## Purpose

Deep Semi-Supervised Learning (Deep SSL) là sự kết hợp có hệ thống giữa các kiến trúc mạng nơ-ron sâu (Deep Neural Networks - DNNs) với các nguyên lý học bán giám sát tiên tiến nhằm khai phá triệt để phân phối biểu diễn từ lượng lớn dữ liệu không nhãn, giảm thiểu tối đa sự phụ thuộc vào dữ liệu gán nhãn thủ công tốn kém.

Trong thị giác máy tính (Computer Vision), xử lý ngôn ngữ tự nhiên (NLP) và tín hiệu âm thanh, việc thu thập hàng triệu mẫu dữ liệu thô rất dễ dàng nhưng chi phí gán nhãn chính xác bởi chuyên gia là rào cản lớn nhất. Deep SSL giải quyết nghịch lý này bằng cách thiết lập các hàm mục tiêu đa thành phần:
- Giám sát chuẩn (Supervised Loss) trên tập dữ liệu có nhãn giới hạn.
- Ràng buộc nhất quán (Consistency Regularization) chống lại các nhiễu loạn tăng cường dữ liệu thực tế (Augmentation Perturbations).
- Cực tiểu hóa độ hỗn loạn (Entropy Minimization) thông qua cơ chế lọc giả nhãn tự tin (Confidence Thresholded Pseudo-Labeling).
- Ràng buộc cấu trúc biểu diễn trung gian (Representation Regularization) như biến đổi MixUp trên không gian tiềm ẩn.

Bảng dưới đây tổng hợp tiến trình tiến hóa của các phương pháp Deep SSL hiện đại:

| Thuật toán | Năm | Cơ chế cốt lõi | Đột phá chính | Hạn chế |
| :--- | :--- | :--- | :--- | :--- |
| **Pseudo-Label** (Lee) | 2013 | Hard Pseudo-labeling sau mỗi epoch | Đơn giản, dễ tích hợp | Dễ tích lũy lỗi (confirmation bias) khi mạng chưa hội tụ |
| **$\Pi$-Model** (Laine & Aila) | 2016 | Consistency giữa 2 lần feedforward cùng mẫu | Khởi xướng ràng buộc nhất quán thời gian thực | Kháng nhiễu yếu, tốc độ hội tụ chậm |
| **Temporal Ensembling** | 2016 | So sánh dự đoán hiện tại với moving average lịch sử | Giảm phương sai dự đoán | Cập nhật chậm qua từng epoch, tốn bộ nhớ lưu trữ |
| **Mean Teacher** (Tarvainen) | 2017 | EMA trọng số mô hình sinh target nhất quán | Cập nhật mục tiêu liên tục theo từng step huấn luyện | Đòi hỏi điều chỉnh hệ số phân rã EMA nhạy cảm |
| **MixMatch** (Berthelot et al.) | 2019 | Tăng cường $K$ lần + Sharpening + MixUp đa tầng | Đồng thời tối ưu tính trơn và cực tiểu entropy | Chi phí tính toán cao ($K$ lần suy luận cho mỗi mẫu) |
| **UDA** (Xie et al.) | 2020 | Unsupervised Data Augmentation + RandAugment/TSA | Chứng minh vai trò sống còn của Strong Augmentation | Yêu cầu chính sách tăng cường nâng cao tinh vi |
| **FixMatch** (Sohn et al.) | 2020 | Weak-Strong Augmentation + Hard Thresholding | Đơn giản tối đa, đạt hiệu năng SOTA kỷ lục | Ngưỡng tự tin cố định gây bất lợi cho lớp khó/ít mẫu |
| **FlexMatch** (Zhang et al.) | 2021 | Curriculum Pseudo-Labeling (CPL) ngưỡng động | Tự điều chỉnh ngưỡng theo độ khó và tiến độ từng lớp | Phức tạp hơn trong cài đặt theo dõi động thái lớp |

---

## Input

Quá trình huấn luyện một hệ thống Deep SSL tiếp nhận các nguồn dữ liệu và siêu tham số sau:

1. **Tập dữ liệu có nhãn nhỏ (Labeled Dataset):**

$$
\mathcal{D}_L = \left\lbrace (x_i, y_i) \right\rbrace_{i=1}^{N_L}
$$

   Trong đó $`x_i \in \mathcal{X}`$ là mẫu đầu vào (hình ảnh, văn bản), $`y_i \in \lbrace 1, 2, \dots, C \rbrace`$ là nhãn thực tế dạng one-hot vector hoặc chỉ số lớp. Số lượng mẫu $`N_L`$ thường rất nhỏ (ví dụ 4 đến 25 mẫu mỗi lớp).

2. **Tập dữ liệu không nhãn lớn (Unlabeled Dataset):**

$$
\mathcal{D}_U = \left\lbrace u_j \right\rbrace_{j=1}^{N_U}
$$

   Trong đó $`u_j \in \mathcal{X}`$ cùng phân phối biên $`p(x)`$ với $`\mathcal{D}_L`$, với quy mô $`N_U \gg N_L`$ (thường gấp từ 10 đến 100 lần).

3. **Cặp phép biến đổi tăng cường dữ liệu (Augmentation Operators):**
   - **Phép biến đổi yếu (Weak Augmentation $`\alpha(\cdot)`$):** Các phép biến đổi bảo toàn cấu trúc không gian nhẹ, ví dụ lật ngang ngẫu nhiên (Random Horizontal Flip) và dịch chuyển tịnh tiến nhỏ (Random Crop/Translation).
   - **Phép biến đổi mạnh (Strong Augmentation $`\mathcal{A}(\cdot)`$):** Các phép biến đổi hình thái học và màu sắc cường độ cao không bảo toàn ngữ cảnh pixel, ví dụ RandAugment, CTAugment, AutoAugment hoặc CutOut.

4. **Kiến trúc mạng nơ-ron sâu (Backbone Architecture):**
   Mạng nơ-ron tích chập hoặc Transformer tham số hóa bởi vector trọng số $`\theta`$:

$$
f_\theta: \mathcal{X} \to \Delta^{C-1}
$$

   Ánh xạ đầu vào sang không gian phân phối xác suất trên $`C`$ lớp đầu ra (thông qua hàm kích hoạt Softmax).

5. **Bộ siêu tham số điều khiển (Hyperparameters):**
   - Tỷ lệ kích thước batch không nhãn so với có nhãn: $`\mu = B_U / B_L`$ (thường chọn $`\mu \in [5, 7]`$).
   - Ngưỡng cắt tự tin (Confidence Threshold): $`\tau`$ (trong FixMatch mặc định $`\tau = 0.95`$).
   - Hệ số cân bằng hàm mất mát không nhãn: $`\lambda_U`$ (thường chọn $`\lambda_U = 1.0`$).
   - Nhiệt độ làm sắc nét phân phối: $`T`$ (trong MixMatch, thường $`T \in [0.5, 0.8]`$).
   - Hệ số suy giảm trung bình động trọng số (EMA decay rate): $`\beta_{\mathrm{EMA}} \in [0.999, 0.9999]`$.

---

## Output

Mô hình Deep SSL sau quá trình tối ưu cung cấp các kết quả đầu ra:

1. **Bộ trọng số mạng tối ưu $`\theta^*`$ và Mô hình EMA $`\theta_{\mathrm{EMA}}`$:**
   Mạng nơ-ron đã học được không gian đặc trưng phân tách rõ nét giữa các cụm mật độ cao, tuân thủ giả định cụm (Cluster Assumption) và giả định đa tạp (Manifold Assumption).

2. **Bộ phân loại quy nạp (Inductive Classifier):**
   Có khả năng dự đoán phân phối xác suất chuẩn xác $`p_\theta(y \mid x_{\mathrm{new}})`$ cho bất kỳ mẫu thử nghiệm mới nào ngoài phân phối huấn luyện mà không cần chạy lại thuật toán:

$$
\hat{y} = \arg\max_{c \in \lbrace 1, \dots, C \rbrace} f_{\theta^*}(x_{\mathrm{new}})_c
$$

3. **Tập giả nhãn chất lượng cao cho toàn bộ tập dữ liệu không nhãn ban đầu:**
   Kèm theo độ tự tin định lượng $`\max_c p_\theta(y=c \mid u_j)`$, phục vụ các quy trình chắt lọc tri thức (Knowledge Distillation) hoặc làm giàu kho nhãn dữ liệu.

---

## Evaluation metrics

Đánh giá mô hình Deep SSL đòi hỏi các chỉ số đa chiều từ độ chính xác tổng quát hóa đến chất lượng phân bổ giả nhãn nội tại:

1. **Tỷ lệ lỗi kiểm thử (Top-1 Test Error Rate %):**
   Chỉ số tiêu chuẩn trên các tập benchmark (CIFAR, SVHN, ImageNet):

$$
\mathrm{Error} = \frac{1}{N_{\mathrm{test}}} \sum_{k=1}^{N_{\mathrm{test}}} \mathbb{I}(\hat{y}_k \neq y_k^*) \times 100\%
$$

2. **Chỉ số F1-Score vĩ mô (Macro F1-Score):**
   Đo lường sự cân bằng dự đoán trên các lớp, đặc biệt nhạy cảm với hiện tượng sụp đổ phân phối (class collapse) khi mô hình thiên vị lớp dễ:

$$
\mathrm{Macro\text{ }F1} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot \mathrm{Precision}_c \cdot \mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}
$$

3. **Tỷ lệ tận dụng mẫu không nhãn (Mask Utilization Rate):**
   Tỷ lệ phần trăm các mẫu không nhãn trong batch vượt qua ngưỡng tự tin $`\tau`$ để đóng góp vào gradient cập nhật:

$$
\mathrm{Utilization} = \frac{1}{B_U} \sum_{b=1}^{B_U} \mathbb{I}(\max_c q_{b, c} \ge \tau)
$$

   Tỷ lệ này tăng dần theo quá trình huấn luyện phản ánh sự tiến bộ và tự tin của mô hình.

4. **Độ chính xác của giả nhãn (Pseudo-Label Accuracy):**
   Tỷ lệ giả nhãn chính xác trong số các mẫu đã vượt qua ngưỡng tự tin:

$$
\mathrm{PL\text{-}Acc} = \frac{\sum_{b=1}^{B_U} \mathbb{I}(\max_c q_{b, c} \ge \tau \land \hat{y}_b = y_b^*)}{\sum_{b=1}^{B_U} \mathbb{I}(\max_c q_{b, c} \ge \tau)}
$$

   Chỉ số này phải duy trì trên 95% để triệt tiêu hiệu ứng confirmation bias.

5. **Sai số hiệu chuẩn kỳ vọng (Expected Calibration Error - ECE):**
   Đo lường sự chênh lệch giữa độ tự tin dự đoán và độ chính xác thực tế, ngăn ngừa hiện tượng mạng tự tin thái quá (overconfident):

$$
\mathrm{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|
$$

---

## How it works

### 1. Kiến trúc nền tảng 3 trụ cột của Deep SSL

Deep SSL hiện đại vận hành dựa trên sự kết hợp hài hòa của 3 trụ cột toán học:

```
[Mẫu không nhãn u] ──┬──> [Weak Augmentation α(u)] ──> [Model f_θ] ──> [Dự đoán q] ──> [Ngưỡng τ] ──> [Giả nhãn cứng ŷ]
                     │                                                                                      │
                     └──> [Strong Augmentation A(u)] ─> [Model f_θ] ──> [Dự đoán p] <───────────────────────┘
                                                                               │
                                                                               ▼
                                                                  [Cross-Entropy Loss L_u]
```

1. **Consistency Regularization:** Dự đoán của mô hình phải bất biến đối với các phép biến dạng dữ liệu thực tế giữ nguyên ngữ nghĩa: $`f_\theta(\alpha(u)) \approx f_\theta(\mathcal{A}(u))`$.
2. **Entropy Minimization:** Ép ranh giới quyết định (decision boundary) đi qua các vùng thưa thớt dữ liệu bằng cách chuyển đổi phân phối mềm thành nhãn cứng (one-hot pseudo-label).
3. **Data Augmentation Asymmetry:** Sử dụng phép tăng cường yếu $`\alpha`$ để sinh mục tiêu (target pseudo-label) có độ tin cậy cao, và sử dụng phép tăng cường mạnh $`\mathcal{A}`$ để tạo thử thách học tập cho mô hình.

---

### 2. Thuật toán MixMatch

MixMatch hợp nhất nhiều kỹ thuật vào một quy trình thống nhất:
1. **Augmentation:** Với mỗi mẫu không nhãn $`u_b`$, áp dụng phép tăng cường ngẫu nhiên $`K`$ lần để sinh ra $`\lbrace u_{b, 1}, \dots, u_{b, K} \rbrace`$.
2. **Label Guessing:** Dự đoán trung bình phân phối xác suất trên $`K`$ phiên bản:

$$
\bar{q}_b = \frac{1}{K} \sum_{k=1}^K f_\theta(u_{b, k})
$$

3. **Sharpening:** Giảm entropy của phân phối dự đoán thông qua nhiệt độ $`T`$:

$$
\mathrm{Sharpen}(\bar{q}_b, T)_c = \frac{\bar{q}_{b, c}^{1/T}}{\sum_{j=1}^C \bar{q}_{b, j}^{1/T}}
$$

4. **MixUp hỗn hợp:** Trộn tập có nhãn đã tăng cường và tập không nhãn đã làm sắc nét thông qua phép biến đổi MixUp:

$$
x' = \lambda x_1 + (1 - \lambda) x_2, \quad y' = \lambda y_1 + (1 - \lambda) y_2, \quad \lambda \sim \mathrm{Beta}(\alpha, \alpha)
$$

5. **Hàm mất mát kết hợp:**

$$
\mathcal{L}_{\mathrm{MixMatch}} = \mathcal{L}_s(X', Y') + \lambda_U \mathcal{L}_u(U', Q')
$$

   Trong đó $`\mathcal{L}_s`$ dùng Cross-Entropy và $`\mathcal{L}_u`$ dùng sai số bình phương trung bình $`L_2`$.

---

### 3. Thuật toán FixMatch (Chuẩn mực SOTA tối giản)

FixMatch loại bỏ nhu cầu suy luận $`K`$ lần và MixUp phức tạp, đạt hiệu năng vượt trội nhờ cơ chế ghép cặp yếu - mạnh:

#### Bước 1: Tính toán trên mẫu có nhãn
Với mini-batch có nhãn $`\mathcal{B}_L = \left\lbrace (x_b, y_b) \right\rbrace_{b=1}^{B_L}`$, áp dụng weak augmentation $`\alpha(x_b)`$ và tính toán hàm mất mát phân loại có giám sát tiêu chuẩn:

$$
\mathcal{L}_s = \frac{1}{B_L} \sum_{b=1}^{B_L} \mathrm{H}\left(y_b, f_\theta(\alpha(x_b))\right)
$$

Trong đó $`\mathrm{H}(p, q) = -\sum_c p_c \log q_c`$ là hàm mất mát Cross-Entropy.

#### Bước 2: Sinh giả nhãn từ Weak Augmentation
Với mini-batch không nhãn $`\mathcal{B}_U = \left\lbrace u_b \right\rbrace_{b=1}^{B_U}`$, đưa ảnh qua weak augmentation $`\alpha(u_b)`$ để thu được phân phối dự đoán:

$$
q_b = f_\theta(\alpha(u_b))
$$

Xác định độ tự tin cực đại $`\max_c(q_{b, c})`$ và nhãn giả đoán nhận:

$$
\hat{y}_b = \arg\max_c(q_{b, c})
$$

#### Bước 3: Ràng buộc nhất quán trên Strong Augmentation
Mẫu không nhãn $`u_b`$ tiếp tục được đưa qua phép biến đổi mạnh $`\mathcal{A}(u_b)`$. Hàm mất mát bán giám sát chỉ được kích hoạt nếu độ tự tin của nhánh yếu vượt qua ngưỡng định sẵn $`\tau`$:

$$
\mathcal{L}_u = \frac{1}{B_U} \sum_{b=1}^{B_U} \mathbb{I}\left(\max_c q_{b, c} \ge \tau\right) \cdot \mathrm{H}\left(\hat{y}_b, f_\theta(\mathcal{A}(u_b))\right)
$$

Trong đó $`\mathbb{I}(\cdot)`$ là hàm chỉ thị nhị phân (masking). Chú ý rằng gradient không truyền ngược (stop-gradient) qua nhánh sinh giả nhãn $`q_b`$.

#### Bước 4: Tối ưu hóa tổng thể

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_s + \lambda_U \mathcal{L}_u
$$

---

### 4. Thuật toán FlexMatch (Curriculum Pseudo-Labeling)

FixMatch sử dụng ngưỡng tĩnh $`\tau = 0.95`$ cho tất cả các lớp. Điều này gây ra nghịch lý:
- Các lớp dễ học sớm đạt ngưỡng $`\tau`$, sinh nhiều giả nhãn và được củng cố liên tục.
- Các lớp khó học có độ tự tin thấp, bị mask lọc bỏ gần hết, dẫn đến hiện tượng thiếu mẫu và suy giảm độ chính xác trên lớp đó.

FlexMatch giới thiệu cơ chế **Curriculum Pseudo-Labeling (CPL)**:
1. Theo dõi số lượng mẫu vượt ngưỡng ban đầu của từng lớp $`c`$ tại thời điểm huấn luyện $`t`$:

$$
\sigma_t(c) = \sum_{b=1}^{B_U} \mathbb{I}\left(\max_{c'} q_{b, c'} \ge \tau \land \arg\max_{c'} q_{b, c'} = c\right)
$$

2. Chuẩn hóa độ thành thạo của từng lớp:

$$
\beta_t(c) = \frac{\sigma_t(c)}{\max_{c'} \sigma_t(c')}
$$

3. Thiết lập ngưỡng động riêng cho từng lớp $`\tau_t(c)`$:

$$
\tau_t(c) = \frac{\beta_t(c)}{2 - \beta_t(c)} \cdot \tau
$$

   Nhờ đó, các lớp khó có ngưỡng linh hoạt thấp hơn, cho phép mô hình thu nhận giả nhãn dần dần theo lộ trình sư phạm tự nhiên (Curriculum Learning).

---

## Data used

Deep SSL được thiết kế chuyên biệt cho các tập dữ liệu có quy mô không gian lớn và cấu trúc phức tạp:

| Tập dữ liệu | Bản chất dữ liệu | Số lớp | Kích thước ảnh | Cấu hình SSL kinh điển | Hiệu năng FixMatch (Error %) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | Ảnh tự nhiên 10 đối tượng | 10 | $32 \times 32 \times 3$ | 40 nhãn (4 nhãn/lớp)<br>250 nhãn (25 nhãn/lớp)<br>4000 nhãn | **13.81%** (40 nhãn)<br>**5.07%** (250 nhãn)<br>**4.26%** (4000 nhãn) |
| **CIFAR-100** | Đối tượng chi tiết 100 lớp | 100 | $32 \times 32 \times 3$ | 400 nhãn (4 nhãn/lớp)<br>2500 nhãn (25 nhãn/lớp) | **46.40%** (400 nhãn)<br>**28.29%** (2500 nhãn) |
| **SVHN** | Chữ số địa chỉ nhà phố | 10 | $32 \times 32 \times 3$ | 40 nhãn (4 nhãn/lớp)<br>250 nhãn (25 nhãn/lớp) | **3.96%** (40 nhãn)<br>**2.48%** (250 nhãn) |
| **STL-10** | Ảnh tự nhiên độ phân giải cao | 10 | $96 \times 96 \times 3$ | 1000 nhãn có sẵn + 100k ảnh không nhãn ngoài tập | **5.17%** (vượt qua supervised trên toàn bộ tập) |
| **ImageNet-1K** | Phân loại thị giác quy mô lớn | 1000 | $224 \times 224 \times 3$ | 10% có nhãn (~128k ảnh)<br>1% có nhãn (~12.8k ảnh) | **28.5%** Top-1 Error (10% nhãn) |

---

## Design

### 1. Kiến trúc hệ thống huấn luyện FixMatch chuẩn công nghiệp

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             FIXMATCH ENGINE                                 │
├──────────────────────────────────────┬──────────────────────────────────────┤
│           LABELED BRANCH             │          UNLABELED BRANCH            │
│                                      │                                      │
│ Batch Labeled (B_L)                  │ Batch Unlabeled (B_U = μ * B_L)      │
│         │                            │         │                            │
│ Weak Augmentation α(x)               │ ┌───────┴──────────────────┐         │
│         │                            │ │                          │         │
│         ▼                            │ ▼                          ▼         │
│ Model f_θ                            │ Weak Aug α(u)       Strong Aug A(u)  │
│         │                            │ │                          │         │
│         ▼                            │ ▼                          ▼         │
│ Logits z_s                           │ Model f_θ (no grad)   Model f_θ      │
│         │                            │ │                          │         │
│         ▼                            │ ▼                          ▼         │
│ Cross-Entropy Loss L_s               │ Softmax prob q        Logits z_u     │
│         │                            │ │                          │         │
│         │                            │ ├─ Max(q) >= τ ?           │         │
│         │                            │ └─ Pseudo-label ŷ          │         │
│         │                            │           │                │         │
│         │                            │           ▼                ▼         │
│         │                            │      Masked Cross-Entropy Loss L_u   │
│         │                            │                    │                 │
│         └────────────────────┬───────┴────────────────────┘                 │
│                              ▼                                              │
│                 Total Loss: L = L_s + λ_u * L_u                             │
│                              │                                              │
│                     Backpropagation & AdamW                                 │
│                              │                                              │
│                     EMA Weight Update θ_ema                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Triển khai Mã nguồn Hoàn chỉnh (PyTorch Implementation)

Đoạn mã sau cài đặt hoàn chỉnh pipeline FixMatch với mô hình mạng tích chập, bộ tạo dữ liệu tăng cường đôi (Weak/Strong Transform), cơ chế sinh giả nhãn có ngưỡng cắt, và vòng lặp huấn luyện chuẩn xác:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np

# ==============================================================================
# 1. BỘ PHÉP BIẾN ĐỔI DỮ LIỆU TĂNG CƯỜNG ĐÔI (WEAK & STRONG TRANSFORMS)
# ==============================================================================
class TransformFixMatch:
    """Tạo ra cặp ảnh: 1 ảnh qua Weak Augmentation, 1 ảnh qua Strong Augmentation."""
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        self.weak = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        # Mô phỏng Strong Augmentation tương đương RandAugment + Cutout
        self.strong = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        ])

    def __call__(self, x):
        return self.weak(x), self.strong(x)


# ==============================================================================
# 2. KIẾN TRÚC MẠNG NƠ-RON TÍCH CHẬP CHUẨN (CONVNET BACKBONE)
# ==============================================================================
class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2), # 8x8
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits


# ==============================================================================
# 3. MÔ PHỎNG DATASET CHO THỬ NGHIỆM ĐỘC LẬP
# ==============================================================================
class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples, transform=None, is_labeled=True, num_classes=10):
        self.num_samples = num_samples
        self.transform = transform
        self.is_labeled = is_labeled
        # Sinh dữ liệu ảnh giả lập dạng PIL Image tương thích torchvision
        self.data = [T.ToPILImage()(torch.rand(3, 32, 32)) for _ in range(num_samples)]
        if is_labeled:
            self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
            
        if self.is_labeled:
            return img, self.targets[idx]
        return img


# ==============================================================================
# 4. HUẤN LUYỆN FIXMATCH STEP VÀ METRIC TRACKING
# ==============================================================================
def train_fixmatch_one_epoch(
    model, 
    labeled_loader, 
    unlabeled_loader, 
    optimizer, 
    device, 
    threshold=0.95, 
    lambda_u=1.0
):
    model.train()
    total_loss_list = []
    loss_s_list = []
    loss_u_list = []
    mask_rate_list = []

    # Tạo iterator tuần hoàn cho tập unlabeled nếu tỷ lệ kích thước lệch nhau
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx, (x_l, y_l) in enumerate(labeled_loader):
        try:
            (u_w, u_s) = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            (u_w, u_s) = next(unlabeled_iter)

        x_l, y_l = x_l.to(device), y_l.to(device)
        u_w, u_s = u_w.to(device), u_s.to(device)

        # 1. Forward tập có nhãn (Labeled Loss)
        logits_l = model(x_l)
        loss_s = F.cross_entropy(logits_l, y_l)

        # 2. Forward tập không nhãn nhánh yếu để sinh giả nhãn (No Gradient)
        with torch.no_grad():
            logits_u_w = model(u_w)
            probs_u_w = torch.softmax(logits_u_w, dim=-1)
            max_probs, pseudo_labels = torch.max(probs_u_w, dim=-1)
            mask = (max_probs >= threshold).float()

        # 3. Forward tập không nhãn nhánh mạnh (Strong Augmentation)
        logits_u_s = model(u_s)
        
        # 4. Tính Cross-Entropy giữa logits nhánh mạnh và pseudo-label với mask
        loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo_labels, reduction='none')
        loss_u = (loss_u_per_sample * mask).mean()

        # 5. Hàm mất mát tổng hợp
        total_loss = loss_s + lambda_u * loss_u

        # Lan truyền ngược và tối ưu trọng số
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Ghi nhận số liệu
        total_loss_list.append(total_loss.item())
        loss_s_list.append(loss_s.item())
        loss_u_list.append(loss_u.item())
        mask_rate_list.append(mask.mean().item())

    avg_total = np.mean(total_loss_list)
    avg_s = np.mean(loss_s_list)
    avg_u = np.mean(loss_u_list)
    avg_mask = np.mean(mask_rate_list)

    return avg_total, avg_s, avg_u, avg_mask


# ==============================================================================
# 5. KHỞI CHẠY THỬ NGHIỆM ĐỊNH LƯỢNG
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Khởi chạy FixMatch Pipeline trên thiết bị: {device}")

    # Cấu hình siêu tham số
    num_classes = 10
    n_labeled = 100       # 100 mẫu có nhãn (10 mẫu/lớp)
    n_unlabeled = 700     # 700 mẫu không nhãn (tỷ lệ mu = 7)
    batch_size = 16
    tau = 0.90            # Ngưỡng tự tin 90%
    lambda_u = 1.0

    # Dataset & Dataloader
    transform_l = T.Compose([
        T.RandomCrop(size=32, padding=4, padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform_u = TransformFixMatch(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    ds_labeled = SyntheticImageDataset(n_labeled, transform=transform_l, is_labeled=True, num_classes=num_classes)
    ds_unlabeled = SyntheticImageDataset(n_unlabeled, transform=transform_u, is_labeled=False)

    loader_labeled = DataLoader(ds_labeled, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_unlabeled = DataLoader(ds_unlabeled, batch_size=batch_size * 4, shuffle=True, drop_last=True)

    # Model & Optimizer
    model = SimpleConvNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\nTiến hành huấn luyện 3 Epochs minh họa cơ chế:")
    for epoch in range(1, 4):
        total_loss, loss_s, loss_u, mask_rate = train_fixmatch_one_epoch(
            model, loader_labeled, loader_unlabeled, optimizer, device, threshold=tau, lambda_u=lambda_u
        )
        print(f"Epoch {epoch:02d} | Total Loss: {total_loss:.4f} | L_s: {loss_s:.4f} | L_u: {loss_u:.4f} | Mask Rate: {mask_rate*100:.2f}%")

    print("\n=> Pipeline FixMatch vận hành hoàn tất, bảo toàn cấu trúc toán học và quy tắc mask tự tin.")
```

---

## Nguồn

- [1] Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C. A., Cubuk, E. D., Kurakin, A., & Li, C. L. (2020). *FixMatch: Simplifying semi-supervised learning with consistency and confidence*. Advances in Neural Information Processing Systems (NeurIPS), 33, 596-608.
- [2] Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., & Raffel, C. A. (2019). *MixMatch: A holistic approach to semi-supervised learning*. Advances in Neural Information Processing Systems (NeurIPS), 32, 5049-5059.
- [3] Berthelot, D., Carlini, N., Cubuk, E. D., Kurakin, A., Sohn, K., Zhang, H., & Raffel, C. (2020). *ReMixMatch: Semi-supervised learning with distribution matching and augmentation anchoring*. International Conference on Learning Representations (ICLR).
- [4] Xie, Q., Dai, Z., Hovy, E., Luong, T., & Le, Q. (2020). *Unsupervised data augmentation for consistency training*. Advances in Neural Information Processing Systems (NeurIPS), 33, 6256-6268.
- [5] Zhang, B., Wang, Y., Hou, W., Wu, H., Wang, J., Okumura, M., & Shinozaki, T. (2021). *FlexMatch: Boosting semi-supervised learning with curriculum pseudo-labeling*. Advances in Neural Information Processing Systems (NeurIPS), 34, 18408-18419.
- [6] Tarvainen, A., & Valpola, H. (2017). *Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results*. Advances in Neural Information Processing Systems (NeurIPS), 30, 1195-1204.
