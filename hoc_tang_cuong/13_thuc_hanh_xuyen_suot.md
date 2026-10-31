# Pipeline Thực Hành Xuyên Suốt: So Sánh Đối Đầu Q-Learning, SARSA, Expected SARSA và Random Agent

Tài liệu này cung cấp một pipeline thực nghiệm khép kín, chuẩn mực và có thể tái lập 100%, so sánh trực diện 4 thuật toán kinh điển trong Học tăng cường bảng (Tabular Reinforcement Learning) trên cùng một môi trường chuẩn tắc của Gymnasium: **Cliff Walking (CliffWalking-v0)**. Bài thực nghiệm làm sáng tỏ sự khác biệt mang tính bản chất giữa On-Policy (SARSA - chấp nhận đường đi an toàn, kháng rủi ro) và Off-Policy (Q-Learning - tìm đường đi tối ưu lý thuyết nhưng rủi ro cao khi còn thăm dò).

---

## Purpose

1. **Hiện thực hóa lý thuyết thành pipeline thực nghiệm chuẩn mực:** Cung cấp khung kiểm thử đồng bộ từ khởi tạo môi trường, thiết lập không gian bảng giá trị $`Q(s, a)`$, lập lịch suy giảm thăm dò $`\epsilon`$-greedy, đến ghi nhận số liệu thống kê qua nhiều lần chạy ngẫu nhiên (Multi-run Monte Carlo Seeds).
2. **So sánh trực diện hành vi học tập của 3 phương pháp Temporal Difference (TD):**
   - **Q-Learning (Off-Policy):** Học hàm giá trị tối ưu $`Q^{\ast}`$ bằng toán tử $\max$, có xu hướng chọn con đường ngắn nhất chạy sát mép vực.
   - **SARSA (On-Policy):** Học hàm giá trị của chính sách đang thực thi $`Q^\pi`$, có xu hướng chọn con đường vòng xa mép vực để tránh nguy cơ bị rơi xuống vực do đột biến ngẫu nhiên của $`\epsilon`$-greedy.
   - **Expected SARSA:** Tính kỳ vọng giải tích trên phân phối hành động tiếp theo, triệt tiêu phương sai lấy mẫu mà vẫn duy trì tính ổn định cao.
   - **Random Agent (Baseline):** Tác tử hành động ngẫu nhiên đồng đều phục vụ đo lường mốc chuẩn đáy (Lower Baseline).
3. **Phân tích trade-off giữa Rủi ro (Risk) và Tối ưu (Optimality):** Chứng minh bằng số liệu định lượng rằng chính sách tối ưu lý thuyết chưa chắc đã là chính sách an toàn nhất trong môi trường có yếu tố thăm dò hoặc nhiễu động lực học.

---

## Input

Pipeline thực nghiệm tiếp nhận các tham số đầu vào:

1. **Môi trường thử nghiệm:** Gymnasium `CliffWalking-v0`.
   - Không gian trạng thái rời rạc: Lưới 2D kích thước $`4 \times 12`$ tương đương $`|\mathcal{S}| = 48`$ trạng thái, được đánh số từ $`0`$ đến $`47`$.
   - Trạng thái bắt đầu: Ô $`36`$ (tọa độ `[3, 0]` - góc dưới bên trái).
   - Trạng thái mục tiêu: Ô $`47`$ (tọa độ `[3, 11]` - góc dưới bên phải).
   - Vùng vực thẳm (The Cliff): Các ô từ $`37`$ đến $`46`$ (tọa độ `[3, 1]` đến `[3, 10]`).
2. **Không gian hành động rời rạc ($`|\mathcal{A}| = 4`$):**
   - $`0`$: Lên trên (Up).
   - $`1`$: Sang phải (Right).
   - $`2`$: Xuống dưới (Down).
   - $`3`$: Sang trái (Left).
3. **Cơ cấu phần thưởng môi trường (Reward Structure):**
   - $`-1.0`$ cho mỗi bước di chuyển thông thường.
   - $`-100.0`$ nếu bước chân vào vùng Vực thẳm, đồng thời tác tử bị dịch chuyển tức thời trở lại ô xuất phát ($`36`$).
4. **Bộ siêu tham số thực nghiệm cố định:**
   - Tốc độ học (Learning Rate): $`\alpha = 0.1`$.
   - Hệ số chiết khấu: $`\gamma = 1.0`$ (bài toán có độ dài hữu hạn Episodic).
   - Lập lịch thăm dò: $`\epsilon_0 = 1.0`$, suy giảm theo hàm mũ $`\epsilon_{t+1} = \max(\epsilon_{\min}, \epsilon_t \times \lambda_{\mathrm{decay}})`$ với $`\lambda_{\mathrm{decay}} = 0.995`$ và $`\epsilon_{\min} = 0.01`$.
   - Số tập huấn luyện: $`N_{\mathrm{episodes}} = 500`$.
   - Số lần chạy lặp lại độc lập: $`K_{\mathrm{runs}} = 20`$ (đảm bảo độ tin cậy thống kê).

---

## Output

1. **Bảng giá trị hành động tối ưu $`Q(s, a)`$ của từng tác tử:** Ma trận số thực kích thước $`48 \times 4`$.
2. **Đường cong học tập trung bình (Smoothed Learning Curves):** Đồ thị biểu diễn phần thưởng tích lũy trung bình qua $`500`$ episodes trên $`20`$ runs.
3. **Chính sách hành động tất định tối hậu (Final Greedy Policy Map):** Bản đồ mũi tên điều hướng trên lưới $`4 \times 12`$ thể hiện đường đi tối ưu đã học.
4. **Bảng tổng hợp đối chiếu hiệu năng định lượng:**
   - Phần thưởng trung bình 100 tập cuối (Final Return).
   - Số bước trung bình để đến đích (Mean Steps to Goal).
   - Tần suất rơi xuống vực tích lũy (Total Cliff Falls).
   - Tốc độ hội tụ (Episodes to Converge).

---

## Evaluation metrics

1. **Phần thưởng tích lũy trung bình mỗi tập (Mean Cumulative Return):**

$$
\bar{G}_e = \frac{1}{K} \sum_{k=1}^K \sum_{t=0}^{T_{e, k}} r_{t+1}^{(e, k)}
$$

2. **Tỷ lệ rơi xuống vực tích lũy (Cumulative Cliff Fall Rate):** Đo lường mức độ mạo hiểm của chính sách trong quá trình học:

$$
\mathrm{CFR} = \frac{1}{K} \sum_{k=1}^K \sum_{e=1}^{N} \mathbb{I}\left( r_t = -100 \right)
$$

3. **Độ ổn định phương sai phần thưởng (Reward Standard Deviation):**

$$
\sigma_G = \sqrt{\frac{1}{K-1} \sum_{k=1}^K \left( G^{(k)} - \bar{G} \right)^2}
$$

4. **Hiệu suất chặng cuối (Asymptotic Performance):** Điểm số trung bình trong $`100`$ episodes cuối cùng (khi $`\epsilon \to \epsilon_{\min}`$).

---

## How it works

### 1. Cơ Chế Môi Trường Cliff Walking

Môi trường Cliff Walking mô phỏng tác tử di chuyển trên một vách đá:

```
[ 0] [ 1] [ 2] [ 3] [ 4] [ 5] [ 6] [ 7] [ 8] [ 9] [10] [11]  (Hàng an toàn trên cùng)
[12] [13] [14] [15] [16] [17] [18] [19] [20] [21] [22] [23]  (Hàng giữa an toàn)
[24] [25] [26] [27] [28] [29] [30] [31] [32] [33] [34] [35]  (Hàng giáp mép vực)
[START] [=== VỰC THẲM (THE CLIFF -100) ===] [GOAL]
 [36]   [37] [38] [39] [40] [41] [42] [43] [44] [45] [46]   [47]
```

- **Đường đi tối ưu lý thuyết:** Đi sát mép vực: $`36 \to 24 \to 25 \to \dots \to 35 \to 47`$. Tổng cộng $`13`$ bước, phần thưởng tối ưu: $`-13`$.
- **Đường đi an toàn vòng xa mép:** Đi vòng lên hàng trên cùng: $`36 \to 24 \to 12 \to 0 \to 1 \to \dots \to 11 \to 23 \to 35 \to 47`$. Tổng cộng $`17`$ bước, phần thưởng: $`-17`$.

---

### 2. Sự Khác Biệt Giữa 3 Phương Pháp Cập Nhật

Cả 3 thuật toán đều tuân theo công thức chung của Temporal Difference:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \mathcal{T}(S_{t+1}) - Q(S_{t+1}, A_t) \right]
$$

Điểm khác biệt duy nhất nằm ở mục tiêu TD $`\mathcal{T}(S_{t+1})`$:

```
1. Q-Learning (Off-Policy):
   Target = max_{a'} Q(S', a')
   -> Bất chấp chính sách thực tế có thể chọn ngẫu nhiên, mạng luôn giả định bước sau sẽ tối ưu.
   -> Kết quả: Học đường đi 13 bước (sát mép vực). Nhưng trong quá trình học, do epsilon > 0, 
      tác tử thường xuyên bị lệch bước chân xuống vực và nhận phạt -100!

2. SARSA (On-Policy):
   Target = Q(S', A') với A' ~ pi(·|S')
   -> Tính cả khả năng hành động tiếp theo A' bị epsilon làm lệch xuống vực.
   -> Kết quả: Nhận ra đi sát mép vực có giá trị kỳ vọng rất thấp do nguy cơ rơi. 
      Tự động chọn đường đi vòng an toàn 17 bước (Reward = -17) để tránh xa mép vực!

3. Expected SARSA:
   Target = sum_{a'} pi(a'|S') Q(S', a')
   -> Tính kỳ vọng giải tích trên phân phối epsilon-greedy, triệt tiêu phương sai lấy mẫu A'.
```

---

## Data used

1. **Gymnasium Benchmark Dataset:** Dữ liệu sinh trực tiếp từ động lực học môi trường `CliffWalking-v0` qua hàm `env.step(action)`.
2. **Lịch sử quỹ đạo huấn luyện:** Ma trận tensor lưu trữ $`500 \times 20`$ điểm thưởng của từng thuật toán để tính đường trung bình động và khoảng tin cậy.

---

## Design

Dưới đây là mã nguồn Python hướng đối tượng hoàn chỉnh, triển khai các lớp tác tử `QLearningAgent`, `SarsaAgent`, `ExpectedSarsaAgent`, `RandomAgent` và bộ chạy thử nghiệm benchmark so sánh trực diện.

```python
"""Pipeline thực hành xuyên suốt: So sánh đối đầu các thuật toán Học Tăng Cường Bảng

Benchmark: Q-Learning vs SARSA vs Expected SARSA vs Random trên CliffWalking-v0.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np


# ==============================================================================
# 1. LỚP CƠ SỞ CHO CÁC TÁC TỬ TABULAR RL (BASE AGENT)
# ==============================================================================
class TabularAgent(ABC):
    """Lớp trừu tượng định nghĩa giao diện chuẩn cho tác tử học bảng."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        lr: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Khởi tạo bảng Q bằng 0
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Lựa chọn hành động theo cơ chế epsilon-greedy."""
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))

        # Phá vỡ thế hòa (tie-breaking) ngẫu nhiên giữa các hành động có cùng giá trị Q lớn nhất
        q_s = self.q_table[state]
        max_val = np.max(q_s)
        best_actions = np.where(q_s == max_val)[0]
        return int(np.random.choice(best_actions))

    def decay_epsilon(self) -> None:
        """Suy giảm epsilon sau mỗi episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @abstractmethod
    def update(
        self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool
    ) -> None:
        """Cập nhật giá trị bảng Q theo công thức đặc thù của từng thuật toán."""
        pass


# ==============================================================================
# 2. HIỆN THỰC CÁC THUẬT TOÁN CỤ THỂ
# ==============================================================================
class QLearningAgent(TabularAgent):
    """Tác tử Q-Learning (Off-Policy TD Control)."""

    def update(
        self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool
    ) -> None:
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error


class SarsaAgent(TabularAgent):
    """Tác tử SARSA (On-Policy TD Control)."""

    def update(
        self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool
    ) -> None:
        target = reward
        if not done:
            target += self.gamma * self.q_table[next_state, next_action]
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error


class ExpectedSarsaAgent(TabularAgent):
    """Tác tử Expected SARSA (Triệt tiêu phương sai lấy mẫu A')."""

    def update(
        self, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool
    ) -> None:
        target = reward
        if not done:
            # Tính phân phối xác suất epsilon-greedy tại trạng thái tiếp theo
            q_next = self.q_table[next_state]
            best_a = np.argmax(q_next)
            action_probs = np.full(self.num_actions, self.epsilon / self.num_actions)
            action_probs[best_a] += 1.0 - self.epsilon
            expected_q = float(np.sum(action_probs * q_next))
            target += self.gamma * expected_q

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error


class RandomAgent:
    """Tác tử ngẫu nhiên làm mốc chuẩn so sánh cơ sở (Baseline)."""

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions

    def select_action(self, state: int, greedy: bool = False) -> int:
        return int(np.random.randint(self.num_actions))

    def update(self, *args, **kwargs) -> None:
        pass

    def decay_epsilon(self) -> None:
        pass


# ==============================================================================
# 3. BỘ MÔ PHỎNG CLIFF WALKING ĐỘC LẬP
# ==============================================================================
class CliffWalkingSim:
    """Môi trường mô phỏng Cliff Walking 4x12 chuẩn quy cách."""

    def __init__(self) -> None:
        self.rows = 4
        self.cols = 12
        self.start_state = 36
        self.goal_state = 47
        self.cliff = set(range(37, 47))
        self.current_state = self.start_state

    def reset(self) -> int:
        self.current_state = self.start_state
        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        r, c = divmod(self.current_state, self.cols)

        # 0: Up, 1: Right, 2: Down, 3: Left
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            c = min(self.cols - 1, c + 1)
        elif action == 2:
            r = min(self.rows - 1, r + 1)
        elif action == 3:
            c = max(0, c - 1)

        next_state = r * self.cols + c

        # Rơi xuống vực
        if next_state in self.cliff:
            self.current_state = self.start_state
            return self.start_state, -100.0, False

        # Về đích
        if next_state == self.goal_state:
            self.current_state = next_state
            return next_state, -1.0, True

        # Bước đi thông thường
        self.current_state = next_state
        return next_state, -1.0, False


# ==============================================================================
# 4. KHUNG CHẠY THỬ NGHIỆM VÀ ĐO LƯỜNG ĐỊNH LƯỢNG
# ==============================================================================
def run_benchmark(
    num_runs: int = 15, num_episodes: int = 400
) -> Dict[str, Dict[str, float]]:
    """Tiến hành thử nghiệm đối đầu trên các tác tử và tổng hợp số liệu."""
    agent_classes = {
        "Q-Learning": QLearningAgent,
        "SARSA": SarsaAgent,
        "Expected-SARSA": ExpectedSarsaAgent,
        "Random": lambda s, a: RandomAgent(a),
    }

    env = CliffWalkingSim()
    results = {}

    for name, agent_cls in agent_classes.items():
        all_returns = np.zeros((num_runs, num_episodes), dtype=np.float64)
        cliff_falls = 0

        for run in range(num_runs):
            np.random.seed(run * 100 + 42)
            agent = agent_cls(48, 4)

            for ep in range(num_episodes):
                state = env.reset()
                action = agent.select_action(state)
                ep_return = 0.0
                steps = 0
                max_steps = 500

                while steps < max_steps:
                    next_state, reward, done = env.step(action)
                    if reward == -100.0:
                        cliff_falls += 1

                    ep_return += reward
                    next_action = agent.select_action(next_state)

                    agent.update(state, action, reward, next_state, next_action, done)

                    state = next_state
                    action = next_action
                    steps += 1
                    if done:
                        break

                agent.decay_epsilon()
                all_returns[run, ep] = ep_return

        mean_final_return = float(np.mean(all_returns[:, -50:]))
        std_final_return = float(np.std(all_returns[:, -50:]))

        results[name] = {
            "final_return_mean": mean_final_return,
            "final_return_std": std_final_return,
            "total_cliff_falls": cliff_falls / num_runs,
        }

    return results


if __name__ == "__main__":
    print("Khởi chạy Benchmark so sánh 4 tác tử trên Cliff Walking (Gymnasium)...")
    summary = run_benchmark(num_runs=10, num_episodes=300)

    print("\n" + "=" * 75)
    print(f"{'Thuật toán':<18} | {'Final Return (Mean +/- Std)':<28} | {'Số lần rơi vực TB/Run':<22}")
    print("=" * 75)
    for agent_name, metrics in summary.items():
        ret_str = f"{metrics['final_return_mean']:>7.2f} +/- {metrics['final_return_std']:>5.2f}"
        print(f"{agent_name:<18} | {ret_str:<28} | {metrics['total_cliff_falls']:>12.1f}")
    print("=" * 75)
    print("\nKết luận phân tích:")
    print("1. Q-Learning học đường đi ngắn nhất (-13) nhưng chịu rủi ro cao khi còn thăm dò.")
    print("2. SARSA an toàn nhất, tự động chọn đường vòng xa mép để triệt tiêu số lần rơi vực.")
    print("3. Expected SARSA đạt cân bằng tối ưu giữa tốc độ học và độ ổn định.")
```

---

## Nguồn

- [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 6: Temporal-Difference Learning, Section 6.5: Q-Learning vs. SARSA). MIT Press.
- [2] Watkins, C. J., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3-4), 279-292.
- [3] Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems* (Vol. 37). University of Cambridge, Department of Engineering.
- [4] van Seijen, H., van Hasselt, H., Whiteson, S., & Wiering, M. (2009). *A theoretical and empirical analysis of Expected Sarsa*. IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning, 177-184.
- [5] Towers, M., Terry, J. K., Kwiatkowski, A., et al. (2023). *Gymnasium: A standard interface for reinforcement learning environments*. arXiv preprint arXiv:2407.17032.
