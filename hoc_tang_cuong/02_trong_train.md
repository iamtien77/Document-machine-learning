# Trong Huấn Luyện: Động Thái Tương Tác, Thăm Dò vs Khai Thác Và Ổn Định Học Tập

## Purpose

Tài liệu này chuẩn hóa và phân tích chuyên sâu các động thái diễn ra bên trong pha huấn luyện của hệ thống Học tăng cường (RL Training Dynamics).

Huấn luyện RL là một trong những quy trình bất ổn định nhất trong khoa học máy tính bởi vì:
- **Hiện tượng Non-Stationary Target:** Mục tiêu tối ưu (Target Value) thay đổi liên tục theo từng bước cập nhật tham số.
- **Vòng xoáy tự củng cố (Feedback Loop):** Nếu Agent tình cờ học được một hành vi dưới chuẩn (Suboptimal) và ngừng thăm dò quá sớm, nó sẽ liên tục thu thập dữ liệu thiên lệch củng cố cho hành vi sai lầm đó.
- **Nghịch lý Thăm dò và Khai thác (Exploration vs Exploitation):** Quyết định chọn hành động ngẫu nhiên để mở rộng hiểu biết hay chọn hành động tốt nhất hiện tại để tối đa hóa điểm số.

Tài liệu này cung cấp các nguyên lý lập lịch thăm dò ($`\epsilon`$-decay, Softmax/Boltzmann, UCB), kỹ thuật kiểm soát lỗi sai phân thời gian (TD-Error), và cơ chế giám sát phát hiện sớm hiện tượng sụp đổ chính sách (Policy Collapse).

---

## Input

Pha huấn luyện tiếp nhận cấu hình điều khiển động thái:

1. **Bộ tham số lập lịch thăm dò ($`\epsilon`$-Greedy Schedule):**
   - Giá trị thăm dò khởi đầu: $`\epsilon_{\mathrm{start}} = 1.0`$ (100% ngẫu nhiên tại bước 0).
   - Giá trị thăm dò tối thiểu: $`\epsilon_{\mathrm{min}} \in [0.01, 0.05]`$ (duy trì mức thăm dò nhẹ để thích ứng khi môi trường biến đổi).
   - Hệ số phân rã (Decay Rate $`\lambda`$ hoặc Decay Steps $`T_{\mathrm{decay}}`$).

2. **Hệ số chiết khấu tương lai (Discount Factor $`\gamma \in [0.90, 0.999]`$):**
   Quyết định tầm nhìn chiến lược của Agent qua thời gian.

3. **Tốc độ học (Learning Rate $`\alpha`$):**
   - Trong bảng Q (Tabular): $`\alpha \in [0.05, 0.2]`$.
   - Trong Deep RL (AdamW): $`\alpha \in [10^{-4}, 3 \times 10^{-4}]`$.

4. **Giới hạn số bước mỗi Episode (Max Steps per Episode):**
   Ngăn ngừa Agent bị kẹt trong các vòng lặp vô tận khi chưa biết đường đi.

---

## Output

Kết quả sinh ra trong và sau pha huấn luyện:

1. **Quỹ đạo đường cong học tập (Learning Curves):**
   - Đồ thị Episode Return theo thời gian thực.
   - Đồ thị Episode Length (số bước để hoàn thành nhiệm vụ).
   - Đồ thị độ lớn của sai số TD (TD-Error Magnitude $`|\delta_t|`$).

2. **Lịch sử biến thiên của hàm giá trị ($`\max_a Q(s, a)`$):**
   Dấu hiệu chẩn đoán hiện tượng ước lượng thổi phồng giá trị (Overestimation Bias).

3. **Bộ trọng số chính sách tốt nhất (Best Checkpoint Policy):**
   Được lưu trữ dựa trên điểm số trung bình trượt (Rolling Average Return) cao nhất trên tập kiểm định độc lập.

---

## Evaluation metrics

Các chỉ số giám sát trực tiếp theo từng epoch/episode:

1. **Sai số sai phân thời gian trung bình (Mean Absolute TD-Error):**

$$
\bar{\delta}_k = \frac{1}{T_k} \sum_{t=1}^{T_k} \left| r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right|
$$

   Quỹ đạo chuẩn: Rất lớn ở giai đoạn đầu ($`\delta \gg 1`$) và giảm dần hội tụ về vùng lân cận $`0`$.

2. **Tỷ lệ thăm dò hiệu dụng (Effective Exploration Rate $`\epsilon_t`$):**
   Tần suất các hành động ngẫu nhiên thực sự được thực hiện trong batch hiện tại.

3. **Chỉ số cảnh báo sụp đổ chính sách (Policy Collapse Indicator):**
   Độ sụt giảm phần thưởng tức thời so với trung bình lịch sử:

$$
\text{Drop} = \frac{\bar{G}_{\mathrm{recent}} - \bar{G}_{\mathrm{peak}}}{\bar{G}_{\mathrm{peak}}}
$$

   Nếu $`\text{Drop} < -30\%`$, hệ thống kích hoạt cảnh báo bất ổn định.

---

## How it works

### 1. Chiến Lược Lập Lịch Thăm Dò $`\epsilon`$-Greedy

Tại mỗi bước $`t`$, Agent tung đồng xu với xác suất $`\epsilon_t`$:
- Với xác suất $`\epsilon_t`$: Chọn một hành động ngẫu nhiên từ $`\mathcal{A}`$ (Thăm dò - Exploration).
- Với xác suất $`1 - \epsilon_t`$: Chọn hành động có giá trị $`Q`$ cao nhất (Khai thác - Exploitation).

Phân phối xác suất chọn hành động $`a`$:

$$
\pi(a \mid s) = \begin{cases}
1 - \epsilon_t + \frac{\epsilon_t}{|\mathcal{A}|} & \text{nếu } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon_t}{|\mathcal{A}|} & \text{nếu } a \neq \arg\max_{a'} Q(s, a')
\end{cases}
$$

#### Hai mô hình suy giảm $`\epsilon`$ phổ biến:
1. **Phân rã tuyến tính (Linear Decay):**

$$
\epsilon_t = \max\left( \epsilon_{\min}, \epsilon_{\mathrm{start}} - \frac{\epsilon_{\mathrm{start}} - \epsilon_{\min}}{T_{\mathrm{decay}}} \cdot t \right)
$$

2. **Phân rã hàm mũ (Exponential Decay - Mượt mà hơn):**

$$
\epsilon_t = \max\left( \epsilon_{\min}, \epsilon_{\mathrm{start}} \cdot e^{-\lambda t} \right)
$$

---

### 2. Thăm Dò Dựa Trên Phân Phối Nhiệt Độ Boltzmann (Softmax Exploration)

Điểm yếu của $`\epsilon`$-greedy là khi thăm dò, nó đối xử với hành động tệ hại nhất ngang bằng với hành động tốt thứ nhì. Thăm dò Boltzmann gán xác suất tỉ lệ thuận với giá trị hàm $`Q`$ thông qua hệ số nhiệt độ $`\tau > 0`$:

$$
\pi(a \mid s) = \frac{\exp(Q(s, a) / \tau)}{\sum_{b \in \mathcal{A}} \exp(Q(s, b) / \tau)}
$$

- Khi nhiệt độ $`\tau \to \infty`$: Xác suất phân bổ đều (hoàn toàn ngẫu nhiên).
- Khi nhiệt độ $`\tau \to 0`$: Phân phối suy biến thành hành động tham lam tuyệt đối (Greedy).

---

### 3. Vòng Đời Cập Nhật Và Động Thái Của TD-Error

```
[Trạng thái s_t] ──> [Hành động a_t] ──> [Nhận thưởng r_{t+1}, Trạng thái s_{t+1}]
                                                      │
                                                      ▼
                      [Tính TD-Target: r_{t+1} + γ * max Q(s_{t+1}, a')]
                                                      │
                                                      ▼
                          [Tính TD-Error: δ_t = TD-Target - Q(s_t, a_t)]
                                                      │
                                                      ▼
                           [Cập nhật: Q(s_t, a_t) += α * δ_t]
```

- Nếu $`\delta_t > 0`$: Kết quả thực tế tốt hơn kỳ vọng, giá trị $`Q(s_t, a_t)`$ được điều chỉnh tăng lên.
- Nếu $`\delta_t < 0`$: Kết quả thực tế gây thất vọng, giá trị $`Q(s_t, a_t)`$ bị kéo giảm xuống.

---

## Data used

Cấu trúc bản ghi chuyển dịch cơ bản (Transition Tuple) được sinh ra liên tục trong pha huấn luyện:

$$
e_t = \left( s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1} \right)
$$

Trong đó $`d_{t+1} \in \lbrace 0, 1 \rbrace`$ là biến chỉ thị trạng thái kết thúc (Terminal Flag). Biến này đóng vai trò sống còn trong công thức Bellman:

$$
\text{TD-Target} = r_{t+1} + (1 - d_{t+1}) \cdot \gamma \max_{a'} Q(s_{t+1}, a')
$$

Nếu $`d_{t+1} = 1`$ (Episode kết thúc), thành phần tương lai bị triệt tiêu về $`0`$, ngăn chặn việc lan truyền giá trị từ episode này sang episode kế tiếp.

---

## Design

### Mã Nguồn Module Giám Sát Pha Huấn Luyện (RL Training Monitor & Exploration Scheduler)

Đoạn mã dưới đây triển khai lớp điều phối và giám sát huấn luyện chuẩn công nghiệp, tích hợp lịch trình suy giảm $`\epsilon`$, tính toán rolling metric và cơ chế cảnh báo sụt giảm hiệu năng:

```python
import numpy as np

class RLTrainingMonitor:
    """Bộ điều phối thăm dò và giám sát động thái huấn luyện Học tăng cường."""

    def __init__(
        self,
        n_episodes: int = 500,
        eps_start: float = 1.0,
        eps_min: float = 0.05,
        eps_decay: float = 0.995,
        rolling_window: int = 30
    ):
        self.n_episodes = n_episodes
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rolling_window = rolling_window

        self.reward_history = []
        self.td_error_history = []
        self.best_rolling_reward = -np.inf

    def update_epsilon(self):
        """Cập nhật giá trị epsilon theo quy luật phân rã hàm mũ."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return self.eps

    def record_episode(self, episode_idx: int, total_reward: float, avg_td_error: float):
        """Ghi nhận chỉ số cuối mỗi episode và kiểm tra cảnh báo sụt giảm."""
        self.reward_history.append(total_reward)
        self.td_error_history.append(avg_td_error)

        # Tính toán điểm trung bình trượt
        if len(self.reward_history) >= self.rolling_window:
            rolling_reward = np.mean(self.reward_history[-self.rolling_window:])
        else:
            rolling_reward = np.mean(self.reward_history)

        # Cảnh báo sụp đổ chính sách (Policy Collapse Warning)
        if rolling_reward > self.best_rolling_reward:
            self.best_rolling_reward = rolling_reward
            is_best = True
        else:
            is_best = False
            drop = (rolling_reward - self.best_rolling_reward) / (abs(self.best_rolling_reward) + 1e-6)
            if drop < -0.35 and episode_idx > 100:
                print(f"[CẢNH BÁO] Episode {episode_idx:03d}: Phát hiện sụt giảm hiệu năng nghiêm trọng ({drop*100:.1f}%)!")

        return rolling_reward, is_best

# ==============================================================================
# MÔ PHỎNG ĐỘNG THÁI PHA HUẤN LUYỆN
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    monitor = RLTrainingMonitor(n_episodes=200, eps_start=1.0, eps_min=0.05, eps_decay=0.98)

    print("=== MÔ PHỎNG ĐỘNG THÁI GIÁM SÁT PHA HUẤN LUYỆN RL ===")
    print(f"{'Episode':<8} | {'Epsilon':<8} | {'Raw Return':<12} | {'Rolling Reward':<16} | {'Avg TD-Error':<14} | {'Ghi chú'}")
    print("-" * 75)

    for ep in range(1, 201):
        eps_val = monitor.update_epsilon()

        # Giả lập reward tăng dần và TD-Error giảm dần theo quy luật học
        simulated_reward = -20.0 + (ep / 200.0) * 35.0 + np.random.normal(0, 3.0)
        simulated_td_error = max(0.05, 5.0 * np.exp(-ep / 50.0) + np.random.normal(0, 0.1))

        # Đưa vào nhiễu gây sụt giảm giả định tại episode 120
        if 120 <= ep <= 130:
            simulated_reward -= 18.0

        rolling_ret, is_best = monitor.record_episode(ep, simulated_reward, simulated_td_error)

        if ep % 25 == 0 or ep == 1:
            note = "[BEST MODEL SAVED]" if is_best else ""
            print(f"{ep:<8} | {eps_val:>6.4f} | {simulated_reward:>+10.2f} | {rolling_ret:>+14.2f} | {simulated_td_error:>12.4f} | {note}")

    print("\n=> Hệ thống giám sát vận hành ổn định, theo dõi mượt mà quá trình hội tụ của Agent.")
```

---

## Nguồn

- [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 2: Multi-armed Bandits). MIT Press.
- [2] Thrun, S. B. (1992). *The role of exploration in learning control*. Handbook of intelligent control: Neural, fuzzy, and adaptive approaches, 527-559.
- [3] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). *Deep reinforcement learning that matters*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).

