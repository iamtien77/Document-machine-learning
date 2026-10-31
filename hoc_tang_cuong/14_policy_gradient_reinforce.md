# Gradient Chính Sách Trực Tiếp Và Thuật Toán REINFORCE

Trong khi các phương pháp dựa trên giá trị (Value-Based như Q-Learning, DQN) phải tối ưu hóa hàm giá trị $`Q(s, a)`$ trước rồi mới suy ra chính sách hành động thông qua toán tử $\arg\max$, các phương pháp Gradient Chính sách (Policy Gradient Methods) tham số hóa và tối ưu hóa trực tiếp hàm chính sách $`\pi_\theta(a|s)`$. Tài liệu này phân tích chi tiết nền tảng lý thuyết của Định lý Gradient Chính sách (Policy Gradient Theorem), thuật toán kinh điển REINFORCE của Ronald Williams (1992), và kỹ thuật triệt tiêu phương sai bằng hàm Baseline.

---

## Purpose

1. **Tối ưu hóa trực tiếp trong không gian chính sách (Direct Policy Search):** Học trực tiếp hàm phân phối hành động $`\pi_\theta(a|s)`$ mà không cần thông qua bảng giá trị trung gian, đảm bảo sự mượt mà và hội tụ ổn định hơn khi xấp xỉ hàm phi tuyến.
2. **Làm chủ không gian hành động liên tục và đa chiều ($`\mathcal{A} \subseteq \mathbb{R}^m`$):** Trong các bài toán điều khiển robot, lái xe tự hành hoặc tay máy công nghiệp, việc giải $`\arg\max_{a} Q(s, a)`$ trên không gian liên tục tại mỗi bước thời gian là bất khả thi về mặt tính toán. Policy Gradient xuất trực tiếp các tham số phân phối xác suất (kỳ vọng $`\boldsymbol{\mu}`$ và độ lệch chuẩn $`\boldsymbol{\sigma}`$) chỉ qua một lần lan truyền tiến (Forward Pass).
3. **Học chính sách ngẫu nhiên tối ưu (Optimal Stochastic Policies):** Trong các môi trường thông tin không hoàn hảo (Partially Observable Markov Decision Processes - POMDP) hoặc trò chơi đối kháng (Game Theory như Oẳn tù tì, Poker), chính sách tất định (Deterministic Policy) dễ bị đối phương khai thác hoặc rơi vào bế tắc. Policy Gradient tự nhiên học được phân phối xác suất tối ưu hỗn hợp (Mixed Strategy).
4. **Hạ thấp phương sai ước lượng thông qua Baseline:** Khắc phục nhược điểm cố hữu của phương pháp Monte Carlo (phương sai cao do tích lũy phần thưởng dài hạn) bằng cách trừ đi hàm chuẩn nền $`b(s)`$ mà vẫn bảo toàn tính không chệch của gradient.

---

## Input

Quá trình huấn luyện thuật toán REINFORCE tiếp nhận các luồng dữ liệu sau:

1. **Trạng thái môi trường quan sát ($`s_t \in \mathcal{S}`$):** Vector trạng thái $`\mathbb{R}^d`$.
2. **Hành động lấy mẫu từ chính sách ($`a_t \sim \pi_\theta(\cdot|s_t)`$):** Chỉ số hành động rời rạc $`\lbrace 0, 1, \dots, |\mathcal{A}|-1 \rbrace`$ hoặc vector lực liên tục.
3. **Phần thưởng phản hồi tức thời ($`r_{t+1} \in \mathbb{R}`$):** Tín hiệu vô hướng từ môi trường tại mỗi bước thời gian.
4. **Quỹ đạo tập hợp hoàn chỉnh (Full Episode Trajectory $`\tau`$):**

$$
\tau = \left( s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T, s_T \right)
$$

5. **Bộ siêu tham số thuật toán:**
   - Tốc độ học mạng chính sách (Policy Learning Rate): $`\alpha_\theta \in [10^{-4}, 10^{-3}]`$.
   - Tốc độ học mạng Baseline (Value Learning Rate): $`\alpha_w \in [10^{-3}, 10^{-2}]`$.
   - Hệ số chiết khấu: $`\gamma \in [0.99, 0.999]`$.
   - Hệ số khuyến khích Entropy (Entropy Regularization): $`\beta_{\mathrm{ent}} \in [0.001, 0.05]`$.

---

## Output

1. **Bộ trọng số tối ưu mạng chính sách ($`\theta^{\ast}`$):** Tham số hóa phân phối hành động tối ưu $`\pi_{\theta^{\ast}}(a|s)`$.
2. **Bộ trọng số mạng Baseline trạng thái ($`w^{\ast}`$):** Xấp xỉ hàm giá trị $`V_w(s) \approx \mathbb{E}[G_t \mid S_t = s]`$.
3. **Chính sách hành động suy luận:**
   - Trong quá trình khai thác tất định: $`a^{\ast} = \arg\max_a \pi_{\theta^{\ast}}(a|s)`$ (rời rạc) hoặc $`a^{\ast} = \boldsymbol{\mu}(s; \theta^{\ast})`$ (liên tục).
   - Trong quá trình lấy mẫu ngẫu nhiên: $`a \sim \pi_{\theta^{\ast}}(\cdot|s)`$.
4. **Log quỹ đạo huấn luyện:** Lịch sử tổn thất chính sách, giá trị phần thưởng hoàn trả $`G_t`$, và độ đo Entropy phân phối.

---

## Evaluation metrics

1. **Tổng phần thưởng không chiết khấu mỗi tập (Episode Undiscounted Return):**

$$
R(\tau) = \sum_{t=0}^{T-1} r_{t+1}
$$

2. **Entropy của phân phối chính sách (Policy Entropy):** Giám sát mức độ phong phú trong hành vi thăm dò:

$$
\mathcal{H}(\pi_\theta(\cdot|s)) = -\sum_{a \in \mathcal{A}} \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

   *(Nếu Entropy sụt giảm về 0 quá nhanh, chính sách bị sụp đổ sớm vào nghiệm cục bộ dưới mức tối ưu)*.
3. **Phương sai của ước lượng Gradient (Gradient Variance):**

$$
\mathrm{Var}\left( \widehat{\nabla_\theta J(\theta)} \right) = \mathbb{E}\left[ \left\| \widehat{g} - \mathbb{E}[\widehat{g}] \right\|_2^2 \right]
$$

   *(Baseline tốt phải kéo giảm chỉ số này xuống nhiều bậc độ lớn)*.
4. **Sai số toàn phương trung bình của mạng Baseline (Value MSE Loss):**

$$
\mathcal{L}(w) = \frac{1}{T} \sum_{t=0}^{T-1} \left( G_t - V_w(s_t) \right)^2
$$

---

## How it works

### 1. Hàm Mục Tiêu Và Thách Thức Toán Học

Mục tiêu của học tăng cường là tìm bộ trọng số $`\theta`$ cực đại hóa tổng phần thưởng kỳ vọng trên phân phối các quỹ đạo:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \int P(\tau; \theta) R(\tau) d\tau
$$

Trong đó xác suất xuất hiện một quỹ đạo $`\tau`$ phụ thuộc vào cả chính sách của tác tử và xác suất chuyển trạng thái của môi trường:

$$
P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
$$

Thách thức cốt tử: Chúng ta **không biết trước mô hình chuyển động của môi trường** $`P(s_{t+1}|s_t, a_t)`$ (Model-Free), nên không thể trực tiếp tính đạo hàm giải tích $`\nabla_\theta P(\tau; \theta)`$ theo cách giải tích thông thường.

---

### 2. Thủ Thuật Tỷ Số Hợp Lý (Likelihood Ratio / Log-Derivative Trick)

Sử dụng đồng nhất thức giải tích $`\nabla_x f(x) = f(x) \nabla_x \log f(x)`$:

$$
\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau; \theta) R(\tau) d\tau = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]
$$

Lấy log tự nhiên của biểu thức xác suất quỹ đạo:

$$
\log P(\tau; \theta) = \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1}|s_t, a_t)
$$

Khi lấy đạo hàm theo $`\theta`$, tất cả các số hạng không phụ thuộc vào $`\theta`$ (phân phối khởi đầu $`\mu(s_0)`$ và hàm động học môi trường $`P(s_{t+1}|s_t, a_t)`$) **đều biến mất hoàn toàn**:

$$
\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

Kết quả mang tính đột phá: Gradient kỳ vọng có thể ước lượng hoàn toàn thông qua việc lấy mẫu trải nghiệm thực nghiệm từ chính sách!

---

### 3. Nguyên Lý Nhân Quả (Causality) Và Thuật Toán REINFORCE

Trong công thức gốc, hành động tại bước $`t`$ bị nhân với tổng phần thưởng của toàn bộ tập $`R(\tau)`$, bao gồm cả những phần thưởng đã xảy ra trong quá khứ ($`t' < t`$). Theo nguyên lý nhân quả vật lý, hành động ở hiện tại không thể tác động tới phần thưởng trong quá khứ.

Do đó, ta thay thế $`R(\tau)`$ bằng tổng phần thưởng tích lũy trong tương lai (Reward-to-go hay Return $`G_t`$):

$$
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} R_{k+1}
$$

Định lý Gradient Chính sách của Sutton et al. (1999) chứng minh:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t|S_t) G_t \right]
$$

Quy tắc cập nhật trọng số của thuật toán REINFORCE:

$$
\theta \leftarrow \theta + \alpha_\theta \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t|S_t)
$$

Ý nghĩa trực quan: Nếu một hành động dẫn đến phần thưởng tương lai $`G_t > 0`$ cao, ta tăng xác suất lựa chọn hành động đó trong tương lai thông qua hướng dốc $`\nabla_\theta \log \pi_\theta(a_t|s_t)`$. Ngược lại, nếu $`G_t`$ thấp hoặc âm, xác suất sẽ bị giảm xuống.

---

### 4. Kỹ Thuật Giảm Phương Sai Bằng Hàm Baseline

Điểm yếu lớn nhất của REINFORCE là phần thưởng Monte Carlo $`G_t`$ có phương sai cực lớn do tích lũy nhiều biến ngẫu nhiên qua các bước thời gian dài.

Ta có thể trừ đi một hàm chuẩn nền $`b(S_t)`$ bất kỳ khỏi $`G_t`$, miễn là $`b(S_t)`$ **không phụ thuộc vào hành động $`A_t`$**. Bổ đề toán học chứng minh kỳ vọng của gradient hoàn toàn không bị lệch (Unbiased):

$$
\mathbb{E}_{A_t \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(A_t|S_t) b(S_t) \right] = b(S_t) \sum_{a \in \mathcal{A}} \pi_\theta(a|S_t) \frac{\nabla_\theta \pi_\theta(a|S_t)}{\pi_\theta(a|S_t)} = b(S_t) \nabla_\theta \left( \sum_{a \in \mathcal{A}} \pi_\theta(a|S_t) \right) = b(S_t) \nabla_\theta (1) = 0
$$

Lựa chọn tối ưu nhất cho $`b(S_t)`$ là hàm giá trị trạng thái $`V(S_t)`$. Khi đó, đại lượng $`G_t - V(S_t)`$ chính là ước lượng của hàm lợi thế (Advantage $`A(S_t, A_t)`$):
- Nếu $`G_t > V(S_t)`$: Hành động $`A_t`$ mang lại kết quả tốt hơn mức trung bình của trạng thái $`S_t`$ $`\to`$ Tăng xác suất.
- Nếu $`G_t < V(S_t)`$: Hành động $`A_t`$ mang lại kết quả kém hơn kỳ vọng trung bình $`\to`$ Giảm xác suất.

Quy tắc cập nhật REINFORCE with Baseline:

$$
\theta \leftarrow \theta + \alpha_\theta \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t|S_t) \left( G_t - V_w(S_t) \right)
$$

---

## Data used

1. **Bộ đệm quỹ đạo tập (Episode Rollout Buffer):** Cấu trúc danh sách trong bộ nhớ lưu trữ các vector trạng thái $`s_t`$, chỉ số hành động $`a_t`$, log-xác suất $`\log \pi(a_t|s_t)`$, và phần thưởng tức thời $`r_{t+1}`$ cho đến khi tập kết thúc.
2. **Vector phần thưởng chiết khấu chuẩn hóa (Normalized Returns):** Mảng $`G_t`$ được chuẩn hóa $`(G_t - \mu_G) / (\sigma_G + \epsilon)`$ trên từng batch tập để ổn định kích thước bước gradient.
3. **Môi trường thực nghiệm chuẩn:** Gymnasium CartPole-v1 (bài toán giữ thăng bằng thanh con lắc ngược trên xe đẩy với phần thưởng $`+1`$ mỗi bước, tối đa 500 bước).

---

## Design

Dưới đây là mã nguồn Python hoàn chỉnh sử dụng PyTorch triển khai mạng chính sách Policy Network, mạng Baseline Value Network, và thuật toán huấn luyện REINFORCE with Baseline đạt độ ổn định cao.

```python
"""Module triển khai thuật toán REINFORCE with Baseline trên nền tảng PyTorch:

- PolicyNetwork (Mạng chính sách xuất Softmax phân phối hành động)
- ValueNetwork (Mạng Baseline xấp xỉ hàm giá trị trạng thái V(s))
- Vòng lặp huấn luyện Monte Carlo Policy Gradient với chuẩn hóa Return
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ==============================================================================
# 1. KIẾN TRÚC MẠNG NƠ-RON CHÍNH SÁCH VÀ BASELINE
# ==============================================================================
class PolicyNetwork(nn.Module):
    """Mạng chính sách tham số hóa pi_theta(a|s) cho không gian hành động rời rạc."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        """Trả về phân phối xác suất hành động Categorical."""
        logits = self.net(state)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Mạng Baseline tham số hóa V_w(s) nhằm triệt tiêu phương sai Monte Carlo."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Ước lượng giá trị vô hướng V(s)."""
        return self.net(state).squeeze(-1)


# ==============================================================================
# 2. BỘ ĐỆM QUỸ ĐẠO EPISODE (ROLLOUT BUFFER)
# ==============================================================================
class EpisodeBuffer:
    """Lưu trữ toàn bộ quỹ đạo của một episode phục vụ tính toán Monte Carlo."""

    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []

    def push(
        self, state: np.ndarray, action: int, reward: float, log_prob: torch.Tensor
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ==============================================================================
# 3. TÍNH TOÁN DISCOUNTED RETURNS
# ==============================================================================
def compute_discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """Tính toán chuỗi phần thưởng tương lai G_t = sum_{k=t}^T gamma^{k-t} R_{k+1}."""
    t_steps = len(rewards)
    returns = np.zeros(t_steps, dtype=np.float32)
    running_return = 0.0

    for t in reversed(range(t_steps)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


# ==============================================================================
# 4. THUẬT TOÁN HUẤN LUYỆN REINFORCE WITH BASELINE
# ==============================================================================
class REINFORCEAgent:
    """Tác tử REINFORCE tích hợp mạng Baseline và chuẩn hóa lợi thế."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_policy: float = 1e-3,
        lr_value: float = 2e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)

        self.buffer = EpisodeBuffer()

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Lấy mẫu hành động từ phân phối chính sách hiện tại."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        dist = self.policy_net(state_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob

    def update(self) -> Tuple[float, float, float]:
        """Thực hiện một bước cập nhật gradient chính sách trên toàn bộ episode."""
        if len(self.buffer) == 0:
            return 0.0, 0.0, 0.0

        # Tính toán chuỗi Return G_t
        returns_np = compute_discounted_returns(self.buffer.rewards, self.gamma)
        returns_t = torch.tensor(returns_np, dtype=torch.float32, device=self.device)
        states_t = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        log_probs_t = torch.stack(self.buffer.log_probs)

        # 1. Dự đoán Baseline V(s_t) từ mạng giá trị
        values = self.value_net(states_t)

        # 2. Ước lượng hàm lợi thế A(s_t, a_t) = G_t - V(s_t)
        advantages = returns_t - values.detach()

        # Chuẩn hóa Advantages giúp ổn định quá trình tối ưu
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Tính Policy Loss với hệ số điều hòa Entropy
        dist = self.policy_net(states_t)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs_t * advantages).mean() - self.entropy_coef * entropy

        # 4. Tính Value Loss (Mean Squared Error)
        value_loss = F.mse_loss(values, returns_t)

        # Cập nhật mạng Policy
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer_policy.step()

        # Cập nhật mạng Value
        self.optimizer_value.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.optimizer_value.step()

        # Xóa bộ đệm sau mỗi episode (thuật toán On-Policy Monte Carlo)
        ep_return = sum(self.buffer.rewards)
        self.buffer.clear()

        return float(policy_loss.item()), float(value_loss.item()), float(ep_return)


# ==============================================================================
# 5. KHỞI CHẠY THỬ NGHIỆM ĐỊNH LƯỢNG
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 4
    action_dim = 2
    agent = REINFORCEAgent(state_dim, action_dim, device=device)

    print("Mô phỏng 3 episodes huấn luyện REINFORCE with Baseline:")
    for ep in range(1, 4):
        # Giả lập một quỹ đạo có độ dài 50 bước
        s = np.random.randn(state_dim).astype(np.float32)
        for t in range(50):
            a, log_prob = agent.select_action(s)
            r = 1.0  # Phần thưởng cố định như CartPole
            agent.buffer.push(s, a, r, log_prob)
            s = np.random.randn(state_dim).astype(np.float32)

        pol_loss, val_loss, ep_ret = agent.update()
        print(
            f"Episode {ep:02d} | Return: {ep_ret:.1f} | "
            f"Policy Loss: {pol_loss:.4f} | Value Loss: {val_loss:.4f}"
        )

    print("=> Hoàn tất cập nhật tham số REINFORCE bảo toàn nguyên lý Likelihood Ratio.")
```

---

## Nguồn

- [1] Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. Machine Learning, 8(3-4), 229-256.
- [2] Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy gradient methods for reinforcement learning with function approximation*. Advances in Neural Information Processing Systems (NeurIPS), 12, 1057-1063.
- [3] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). *High-dimensional continuous control using generalized advantage estimation*. International Conference on Learning Representations (ICLR).
- [4] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). *Asynchronous methods for deep reinforcement learning*. International Conference on Machine Learning (ICML), PMLR, 48, 1928-1937.
- [5] Kakade, S. M. (2001). *A natural policy gradient*. Advances in Neural Information Processing Systems (NeurIPS), 14, 1531-1538.

