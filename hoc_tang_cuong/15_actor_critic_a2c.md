# Kiến Trúc Actor-Critic Và Thuật Toán Advantage Actor-Critic (A2C)

Trong không gian các phương pháp Học tăng cường hiện đại, kiến trúc Actor-Critic đóng vai trò cầu nối kết hợp sức mạnh tối ưu của hai trường phái: Policy-Based (Actor - trực tiếp tối ưu hóa chính sách hành động) và Value-Based (Critic - học hàm giá trị trạng thái để thẩm định chất lượng hành động). Tài liệu này phân tích chi tiết cơ sở toán học của kiến trúc Actor-Critic, nguyên lý triệt tiêu phương sai thông qua hàm Lợi thế (Advantage Function), và thuật toán đồng bộ Advantage Actor-Critic (A2C) trên nền tảng vector hóa đa môi trường.

---

## Purpose

1. **Khắc phục nhược điểm cốt tử của phương pháp REINFORCE thuần túy:** Thuật toán REINFORCE dựa trên ước lượng Monte Carlo đòi hỏi phải chờ toàn bộ episode kết thúc mới có thể cập nhật trọng số, đồng thời có phương sai cực lớn. Kiến trúc Actor-Critic áp dụng cơ chế Bootstrapping của Temporal Difference (TD), cho phép cập nhật mô hình trực tuyến từng bước ($`1`$-step) hoặc sau mỗi chu kỳ ngắn ($`n`$-step).
2. **Triệt tiêu phương sai bằng hàm Lợi thế (Advantage Function $`A(s, a)`$):** Thay vì sử dụng phần thưởng hoàn trả tích lũy dài hạn $`G_t`$ đầy nhiễu loạn, Actor được dẫn hướng bởi đại lượng so sánh tương đối: hành động vừa thực hiện tốt hơn hay kém hơn mức kỳ vọng trung bình của trạng thái hiện tại.
3. **Hiện đại hóa kiến trúc song song (A2C thay thế A3C):** Khác với thuật toán bất đồng bộ Asynchronous Advantage Actor-Critic (A3C) chạy trên nhiều luồng CPU rời rạc với độ trễ khóa tham số, A2C thực thi đồng bộ hóa hoàn toàn trên GPU (Synchronous Batching), tối đa hóa thông lượng tính toán (Throughput) và loại bỏ hoàn toàn hiện tượng cập nhật trên các phiên bản chính sách cũ (stale policies).
4. **Cơ chế biểu diễn chung (Shared Representation):** Tận dụng một mạng nơ-ron tích hợp chia sẻ các tầng trích xuất đặc trưng cơ sở (Backbone), vừa giảm thiểu chi phí bộ nhớ vừa thúc đẩy việc học các biểu diễn trạng thái có tính khái quát hóa cao.

---

## Input

Quá trình huấn luyện một hệ thống A2C tiếp nhận các luồng dữ liệu và tham số:

1. **Trạng thái môi trường quan sát ($`s_t \in \mathcal{S}`$):** Tensor đa chiều biểu diễn trạng thái của $`N_{\mathrm{envs}}`$ môi trường chạy song song, kích thước $`(N_{\mathrm{envs}}, d_{\mathrm{state}})`$.
2. **Tập hành động tương tác ($`a_t \sim \pi_\theta(\cdot|s_t)`$):** Mảng chỉ số hành động rời rạc hoặc vector điều khiển liên tục.
3. **Phần thưởng phản hồi tức thời ($`r_{t+1}`$):** Vector phần thưởng kích thước $`N_{\mathrm{envs}}`$.
4. **Cờ báo trạng thái kết thúc ($`d_{t+1} \in \lbrace 0, 1 \rbrace`$):** Xác định các quỹ đạo đã chạm đích hoặc sập đổ.
5. **Bộ siêu tham số điều khiển thuật toán:**
   - Hệ số chiết khấu tương lai: $`\gamma \in [0.99, 0.999]`$.
   - Chu kỳ cập nhật bước chuyển tiếp (Rollout steps): $`n_{\mathrm{steps}} \in [5, 20]`$.
   - Hệ số trọng số mất mát Critic (Value Loss Coefficient): $`c_v \in [0.25, 0.5]`$.
   - Hệ số khuyến khích Entropy (Entropy Coefficient): $`c_e \in [0.001, 0.05]`$.
   - Ngưỡng cắt chuẩn gradient (Gradient Clipping Norm): $`\text{max\_norm} \in [0.5, 1.0]`$.
   - Tốc độ học tối ưu (Learning Rate): $`\alpha \in [10^{-4}, 10^{-3}]`$ (thường sử dụng optimizer Adam hoặc RMSprop).

---

## Output

1. **Mạng nơ-ron Actor tối ưu ($`\pi_{\theta^{\ast}}(a|s)`$):** Ánh xạ từ trạng thái sang phân phối xác suất hành động.
2. **Mạng nơ-ron Critic tối ưu ($`V_{\phi^{\ast}}(s)`$):** Xấp xỉ hàm giá trị trạng thái chuẩn xác $`V^{\ast}(s)`$.
3. **Chính sách suy luận tối hậu:**
   - Khi vận hành thực tế (Exploitation): $`a^{\ast} = \arg\max_{a} \pi_{\theta^{\ast}}(a|s)`$ (đối với không gian rời rạc) hoặc $`a^{\ast} = \boldsymbol{\mu}(s; \theta^{\ast})`$ (đối với không gian liên tục).
4. **Ước lượng sai số TD và hàm lợi thế ($`\delta_t, A_t`$):** Dữ liệu phục vụ chẩn đoán mức độ bất ngờ và tiến bộ học tập của hệ thống.

---

## Evaluation metrics

1. **Tổng hàm tổn thất tổng hợp A2C (Total Joint Loss):**

$$
\mathcal{L}_{\mathrm{total}}(\theta, \phi) = \mathcal{L}_{\mathrm{policy}}(\theta) + c_v \mathcal{L}_{\mathrm{value}}(\phi) - c_e \mathcal{H}(\pi_\theta)
$$

2. **Tổn thất chính sách có trọng số lợi thế (Policy Advantage Loss):**

$$
\mathcal{L}_{\mathrm{policy}}(\theta) = -\frac{1}{B} \sum_{i=1}^B \log \pi_\theta(a_i \mid s_i) \cdot \widehat{A}(s_i, a_i)
$$

3. **Sai số toàn phương trung bình của Critic (Critic Value MSE Loss):**

$$
\mathcal{L}_{\mathrm{value}}(\phi) = \frac{1}{B} \sum_{i=1}^B \left( R_i^{\mathrm{target}} - V_\phi(s_i) \right)^2
$$

4. **Độ đo Entropy phân phối hành động (Policy Entropy):**

$$
\mathcal{H}(\pi_\theta(\cdot \mid s)) = -\sum_{a \in \mathcal{A}} \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)
$$

   *(Entropy cao đảm bảo tác tử tiếp tục khám phá; khi chính sách hội tụ về nghiệm tất định, Entropy giảm dần về 0)*.
5. **Tỷ lệ giải thích phương sai (Explained Variance of Value Function):** Đo lường mức độ khớp của Critic:

$$
\mathcal{E}_V = 1 - \frac{\mathrm{Var}(R^{\mathrm{target}} - V_\phi(s))}{\mathrm{Var}(R^{\mathrm{target}})}
$$

   *(Nếu Critic hội tụ tốt, $`\mathcal{E}_V \to 1.0`$; nếu Critic dự đoán kém hơn mốc trung bình, $`\mathcal{E}_V \le 0`$)*.

---

## How it works

### 1. Phân Công Trách Nhiệm Giữa Actor Và Critic

Mô hình thiết lập cơ chế tương tác hai chiều khép kín:

```
                      +-------------------+
                      |    Môi trường     |
                      +---------+---------+
                                |
                   Trạng thái s_t | Phần thưởng r_t
                                v
           +--------------------+--------------------+
           |                                         |
           v                                         v
+---------------------+                   +---------------------+
|        ACTOR        |                   |       CRITIC        |
| Policy: pi_theta    |                   | Value: V_phi        |
+----------+----------+                   +----------+----------+
           |                                         |
           | Hành động a_t                           | Giá trị V(s_t)
           v                                         v
   Thực thi môi trường                +--------------+--------------+
                                      |   Ước lượng sai số TD:      |
                                      | delta = r + gamma V(s') - V |
                                      +--------------+--------------+
                                                     |
                                                     | Tín hiệu Advantage
                                                     v
                                       Dẫn hướng cập nhật Actor
```

- **Actor (Người hành động):** Tham số hóa chính sách $`\pi_\theta(a|s)`$. Nhiệm vụ duy nhất là lựa chọn hành động dựa trên trạng thái quan sát.
- **Critic (Người thẩm định):** Tham số hóa hàm giá trị trạng thái $`V_\phi(s)`$. Nhiệm vụ là đánh giá xem trạng thái hiện tại có tiềm năng mang lại nhiều phần thưởng trong tương lai hay không.

---

### 2. Hàm Lợi Thế (Advantage Function) Và Sai Số TD

Hàm Lợi thế đo lường sự vượt trội của một hành động cụ thể so với mức trung bình của trạng thái:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

Nếu trực tiếp huấn luyện thêm một mạng để ước lượng $`Q(s, a)`$, ta sẽ tốn gấp đôi tài nguyên. Định lý học tăng cường chứng minh rằng **Sai số Temporal Difference 1-bước chính là một ước lượng không chệch của Hàm Lợi thế**:

$$
\delta_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)
$$

Chứng minh:

$$
\mathbb{E}\left[ \delta_t \mid S_t = s, A_t = a \right] = \mathbb{E}\left[ R_{t+1} + \gamma V_\phi(S_{t+1}) \mid S_t = s, A_t = a \right] - V_\phi(s) = Q^\pi(s, a) - V^\pi(s) = A^\pi(s, a)
$$

Ý nghĩa: Chỉ cần duy trì **duy nhất một mạng ước lượng trạng thái $`V_\phi(s)`$**, ta vừa có thể cập nhật Critic bằng quy hoạch động TD, vừa lập tức suy ra giá trị Advantage để dẫn hướng Actor!

---

### 3. Ước Lượng Lợi Thế $`n`$-Bước ($`n`$-step Advantage)

Trong thực tế triển khai A2C, tác tử thu thập dữ liệu qua $`n`$ bước liên tiếp trước khi cập nhật. Mục tiêu $`n`$-bước được tính bằng tổng phần thưởng chiết khấu cộng với giá trị Bootstrap tại bước $`t+n`$:

$$
G_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V_\phi(S_{t+n})
$$

Khi đó, hàm lợi thế $`n`$-bước là:

$$
\widehat{A}_t^{(n)} = G_{t:t+n} - V_\phi(S_t)
$$

Kỹ thuật này cho phép cân bằng tối ưu giữa Bias (đến từ mạng nơ-ron xấp xỉ $`V_\phi`$) và Variance (đến từ chuỗi phần thưởng thực nghiệm).

---

### 4. Thuật Toán Đồng Bộ Hóa Song Song (Synchronous Vectorized Execution)

Trong A2C, $`M`$ bản sao môi trường chạy song song độc lập. Tại mỗi chu kỳ:
1. Tất cả $`M`$ môi trường thực hiện bước đi đồng thời, thu thập ma trận chuyển tiếp kích thước $`(n_{\mathrm{steps}}, M)`$.
2. Toàn bộ tensor dữ liệu được duỗi phẳng thành batch kích thước $`B = n_{\mathrm{steps}} \times M`$.
3. Lan truyền tiến (Forward Pass) trên GPU để tính toán đồng thời các phân phối $\pi$, giá trị $V$ và hàm mất mát tổng hợp.
4. Lan truyền ngược (Backward Pass) duy nhất một lần, cập nhật toàn bộ hệ thống bằng gradient chuẩn hóa.

---

## Data used

1. **Tensor Rollout Đa Môi Trường:** Cấu trúc dữ liệu lưu trữ các batch chuyển tiếp song song gồm tensor trạng thái $`(n_{\mathrm{steps}}, N_{\mathrm{envs}}, d)`$, tensor hành động, phần thưởng, log-xác suất và cờ kết thúc.
2. **Môi trường mô phỏng thử nghiệm:** Gymnasium `CartPole-v1` (bài toán cân bằng con lắc rời rạc) và `LunarLander-v2` (bài toán hạ cánh tàu vũ trụ với không gian phần thưởng phức tạp).

---

## Design

Dưới đây là mã nguồn Python chuẩn mực triển khai kiến trúc Actor-Critic tích hợp (Shared Feature Backbone), lớp quản lý vector hóa môi trường song song và thuật toán huấn luyện đồng bộ A2C bằng PyTorch.

```python
"""Module triển khai thuật toán Advantage Actor-Critic (A2C) trên nền tảng PyTorch:

- ActorCriticNetwork: Mạng nơ-ron chia sẻ Backbone với hai đầu ra Actor & Critic
- RolloutStorage: Bộ đệm tensor phục vụ tính toán n-step returns và Advantage
- Vòng lặp huấn luyện đồng bộ A2C với điều hòa Entropy và Gradient Clipping
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ==============================================================================
# 1. KIẾN TRÚC MẠNG NƠ-RON ACTOR-CRITIC TÍCH HỢP
# ==============================================================================
class ActorCriticNetwork(nn.Module):
    """Mạng tích hợp: Tầng trích xuất đặc trưng chung, 2 đầu ra Actor & Critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        # Backbone trích xuất đặc trưng trạng thái chung
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Đầu ra Actor: Logits phân phối xác suất hành động
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Đầu ra Critic: Giá trị trạng thái vô hướng V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """Trả về phân phối hành động Categorical và giá trị trạng thái V(s)."""
        features = self.feature_extractor(state)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Thẩm định hành động phục vụ tính toán gradient của batch."""
        dist, value = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy


# ==============================================================================
# 2. BỘ ĐỆM LƯU TRỮ ROLLOUT PHỤC VỤ A2C BATCHING
# ==============================================================================
class RolloutBuffer:
    """Bộ đệm lưu trữ quỹ đạo đa môi trường n-bước."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        state_dim: int,
        device: torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.states = torch.zeros((num_steps, num_envs, state_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.step_idx = 0

    def insert(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.states[self.step_idx].copy_(state)
        self.actions[self.step_idx].copy_(action)
        self.rewards[self.step_idx].copy_(reward)
        self.dones[self.step_idx].copy_(done)
        self.values[self.step_idx].copy_(value)
        self.step_idx = (self.step_idx + 1) % self.num_steps

    def compute_returns_and_advantages(
        self, next_value: torch.Tensor, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tính toán n-step returns và hàm lợi thế Advantage chuẩn hóa."""
        returns = torch.zeros_like(self.rewards)
        running_return = next_value

        for t in reversed(range(self.num_steps)):
            # Nếu done = 1, ngắt tín hiệu giá trị tương lai
            running_return = self.rewards[t] + gamma * running_return * (1.0 - self.dones[t])
            returns[t] = running_return

        advantages = returns - self.values
        # Chuẩn hóa Advantages trên toàn bộ batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


# ==============================================================================
# 3. THUẬT TOÁN HUẤN LUYỆN A2C TỔNG HỢP
# ==============================================================================
def train_a2c_epoch(
    model: ActorCriticNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> Tuple[float, float, float]:
    """Thực hiện một chu kỳ cập nhật trọng số A2C trên toàn bộ rollout."""
    # 1. Tính toán Returns và Advantages
    returns, advantages = buffer.compute_returns_and_advantages(next_value, gamma)

    # 2. Duỗi phẳng tensor thành batch 1 chiều (Num_Steps * Num_Envs)
    b_states = buffer.states.view(-1, buffer.states.size(-1))
    b_actions = buffer.actions.view(-1)
    b_returns = returns.view(-1)
    b_advantages = advantages.view(-1)

    # 3. Lan truyền tiến và thẩm định trên mô hình hiện tại
    log_probs, values, entropy = model.evaluate_actions(b_states, b_actions)

    # 4. Tính toán các thành phần hàm tổn thất
    policy_loss = -(log_probs * b_advantages.detach()).mean()
    value_loss = F.mse_loss(values, b_returns)
    entropy_loss = -entropy.mean()

    # Tổng hàm mất mát hỗn hợp
    total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

    # 5. Tối ưu hóa trọng số với Gradient Clipping
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item()), float(entropy_loss.item())


# ==============================================================================
# 4. MINH HỌA VẬN HÀNH THỰC NGHIỆM
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Siêu tham số
    state_dim = 4
    action_dim = 2
    num_envs = 8        # 8 môi trường chạy song song
    num_steps = 10      # Thu thập 10 bước mỗi chu kỳ -> Batch 80 mẫu
    gamma = 0.99

    model = ActorCriticNetwork(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = RolloutBuffer(num_steps, num_envs, state_dim, device)

    print("Mô phỏng 3 chu kỳ cập nhật đồng bộ A2C đa môi trường:")
    # Giả lập trạng thái ban đầu của 8 môi trường
    curr_states = torch.randn(num_envs, state_dim, device=device)

    for cycle in range(1, 4):
        # Pha thu thập dữ liệu (Rollout Phase)
        with torch.no_grad():
            for step in range(num_steps):
                dist, values = model(curr_states)
                actions = dist.sample()

                # Giả lập phản hồi từ môi trường
                rewards = torch.ones(num_envs, device=device)
                dones = (torch.rand(num_envs, device=device) > 0.95).float()
                next_states = torch.randn(num_envs, state_dim, device=device)

                buffer.insert(curr_states, actions, rewards, dones, values)
                curr_states = next_states

            # Ước lượng giá trị Bootstrap tại bước tiếp theo
            _, next_val = model(curr_states)

        # Pha cập nhật tham số (Update Phase)
        pol_loss, val_loss, ent_loss = train_a2c_epoch(
            model, optimizer, buffer, next_val, gamma=gamma
        )

        print(
            f"Chu kỳ {cycle:02d} | Policy Loss: {pol_loss:.4f} | "
            f"Value MSE Loss: {val_loss:.4f} | Entropy Loss: {ent_loss:.4f}"
        )

    print("=> Hoàn tất cập nhật đồng bộ A2C bảo toàn nguyên lý Advantage.")
```

---

## Nguồn

- [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 13: Policy Gradient Methods, Section 13.5: Actor-Critic Methods). MIT Press.
- [2] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). *Asynchronous methods for deep reinforcement learning*. International Conference on Machine Learning (ICML), PMLR, 48, 1928-1937.
- [3] Wu, Y., Mansimov, E., Grosse, R. B., Liao, S., & Ba, J. (2017). *Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation*. Advances in Neural Information Processing Systems (NeurIPS), 30, 5279-5288.
- [4] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-dimensional continuous control using generalized advantage estimation*. International Conference on Learning Representations (ICLR).
- [5] Konda, V. R., & Tsitsiklis, J. N. (2000). *Actor-critic algorithms*. Advances in Neural Information Processing Systems (NeurIPS), 12, 1008-1014.
