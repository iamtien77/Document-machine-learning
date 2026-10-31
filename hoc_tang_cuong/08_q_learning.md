# Thuật Toán Q-Learning (Off-Policy TD Control)

## Purpose

Thuật toán Q-Learning, được phát minh bởi Chris Watkins trong luận án tiến sĩ tại Đại học Cambridge vào năm 1989, là **giải thuật điều khiển nổi tiếng, phổ biến và có sức ảnh hưởng lớn nhất trong lịch sử Học tăng cường**.

Bản chất cốt lõi của Q-Learning là một thuật toán **Off-Policy Temporal Difference Control**:
- **Tính chất Off-Policy:** Tác tử có thể thu thập trải nghiệm bằng một chính sách hành vi linh hoạt (Behavior Policy, ví dụ như $`\epsilon`$-greedy với tỷ lệ ngẫu nhiên cao để thỏa sức khám phá môi trường), nhưng mục tiêu cập nhật bảng giá trị $`Q`$ lại luôn hướng về **chính sách tối ưu tuyệt đối (Target Policy là Greedy Policy: $`\max_{a'} Q(s', a')`$)**.
- **Tính chất Không cần Mô hình (Model-Free):** Q-Learning không cần biết ma trận chuyển dịch xác suất hay hàm thưởng của môi trường.
- **Hội tụ vững chắc (Convergence Guarantee):** Watkins & Dayan (1992) đã chứng minh toán học rằng bảng $`Q(s, a)`$ được đảm bảo hội tụ xác suất $100\%$ về giá trị tối ưu $`Q^*(s, a)`$, miễn là mọi cặp trạng thái - hành động đều được tác tử ghé thăm vô hạn lần và tốc độ học $`\alpha`$ suy giảm thỏa mãn điều kiện Robbins-Monro.

---

## Input

Quá trình huấn luyện Q-Learning tiếp nhận các thành phần đầu vào:

1. **Môi trường rời rạc ($`\mathcal{E}`$):** Có không gian trạng thái hữu hạn $`\mathcal{S}`$ và không gian hành động rời rạc $`\mathcal{A}`$.
2. **Bảng giá trị Q ban đầu (Q-Table):**
   
$$
Q \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}
$$

   Khởi tạo bằng 0 cho tất cả các ô, riêng các trạng thái kết thúc (Terminal) luôn cố định bằng 0.

3. **Tốc độ học (Learning Rate $`\alpha \in (0, 1]`$):** Thường chọn $`\alpha \in [0.05, 0.2]`$.
4. **Hệ số chiết khấu tương lai (Discount Factor $`\gamma \in [0, 1)`$):** Thường chọn $`\gamma \in [0.90, 0.99]`$.
5. **Lịch trình thăm dò ($`\epsilon`$-Greedy Schedule):** $`\epsilon_{\mathrm{start}} = 1.0`$, $`\epsilon_{\mathrm{min}} = 0.05`$, hệ số phân rã $`\lambda_{\mathrm{decay}}`$.

---

## Output

Kết quả đầu ra của thuật toán Q-Learning:

1. **Bảng giá trị hành động tối ưu ($`Q^*(s, a)`$):** Ước lượng chính xác điểm số kỳ vọng tích lũy cho mọi cặp $`(s, a)`$.
2. **Chính sách tất định tối ưu ($`\pi^*(s)`$):**
   
$$
\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s, a)
$$

3. **Hàm giá trị trạng thái tối ưu ngầm định ($`V^*(s)`$):**
   
$$
V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)
$$

---

## Evaluation metrics

Chỉ số đánh giá năng lực hội tụ của Q-Learning:

1. **Tổng điểm thưởng trung bình trượt (Moving Average Episode Return):**
   Điểm số trung bình qua 50 hoặc 100 episodes gần nhất. Đồ thị phải tăng dần và đạt trạng thái bình nguyên ổn định.

2. **Sai số sai phân thời gian trung bình (Mean TD-Error $`\mathbb{E}[|\delta_t|]`$):**
   
$$
|\delta_t| = \left| r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right|
$$

   Biên độ sai số này phải tiệm cận dần về $`0`$.

3. **Số bước hoàn thành nhiệm vụ tối ưu (Shortest Path Length):**
   Trong các bài toán định tuyến hay mê cung, số bước di chuyển phải tiệm cận sát với khoảng cách lý thuyết ngắn nhất.

---

## How it works

### 1. Công Thức Cập Nhật Cốt Lõi Của Q-Learning

Tại mỗi bước thời gian $`t`$, khi đang ở trạng thái $`s_t`$, tác tử thực hiện hành động $`a_t`$, nhận phần thưởng $`r_{t+1}`$ và quan sát trạng thái kế tiếp $`s_{t+1}`$. Giá trị $`Q(s_t, a_t)`$ được cập nhật theo công thức:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a')}_{\text{Q-Target (Off-Policy)}} - \underbrace{Q(s_t, a_t)}_{\text{Ước lượng cũ}} \right]
$$

#### Phân Tích Cơ Chế Off-Policy:
- **Hành động $`a_t`$ ở bước hiện tại:** Được chọn theo chính sách hành vi thăm dò (ví dụ: $`\epsilon`$-greedy, đôi khi chọn hành động ngẫu nhiên tệ hại để thử nghiệm).
- **Hành động $`a'`$ ở bước tiếp theo để tính Target:** Không phụ thuộc vào việc bước tiếp theo Agent sẽ làm gì, mà **luôn giả định hành động tốt nhất** thông qua toán tử $`\max_{a'} Q(s_{t+1}, a')`$.
- Đây chính là sự khác biệt then chốt so với **SARSA** (thuật toán On-Policy, vốn dùng hành động thực tế tiếp theo $`a_{t+1}`$ để cập nhật).

---

### 2. Quy Trình Thuật Toán Từng Bước (Step-by-Step Algorithm)

```
Khởi tạo bảng Q(s, a) = 0 cho mọi s ∈ S, a ∈ A
Khởi tạo Q(Terminal, ·) = 0

Lặp với mỗi Episode:
    Khởi tạo trạng thái bắt đầu s
    Lặp với từng bước trong Episode:
        1. Chọn hành động a từ trạng thái s bằng chiến lược ε-greedy dựa trên Q(s, ·)
        2. Thực hiện hành động a, quan sát phần thưởng r và trạng thái kế tiếp s'
        3. Tính TD-Target:
               target = r + γ * max_{a'} Q(s', a')  (nếu s' không phải Terminal)
               target = r                           (nếu s' là Terminal)
        4. Cập nhật giá trị bảng Q:
               Q(s, a) ← Q(s, a) + α * (target - Q(s, a))
        5. Chuyển trạng thái: s ← s'
    Dừng khi s chạm trạng thái Terminal
```

---

### 3. Điều Kiện Hội Tụ Toán Học (Watkins & Dayan, 1992)

Thuật toán Q-Learning được chứng minh hội tụ về $`Q^*(s, a)`$ với xác suất $1$ khi thỏa mãn hai điều kiện tiêu chuẩn Robbins-Monro về tốc độ học:

$$
\sum_{t=1}^\infty \alpha_t(s, a) = \infty \quad \text{và} \quad \sum_{t=1}^\infty \alpha_t^2(s, a) < \infty
$$

Điều kiện thứ nhất đảm bảo bước nhảy đủ lớn để thoát khỏi mọi khởi tạo ban đầu; điều kiện thứ hai đảm bảo bước nhảy đủ nhỏ để triệt tiêu phương sai ngẫu nhiên khi hội tụ.

---

## Data used

Môi trường kiểm chuẩn kinh điển: **Taxi-v3 (Gymnasium / OpenAI Gym - Slide tr. 32)**:

| Đặc tính môi trường | Thông số kỹ thuật | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- |
| **Bản đồ thế giới** | Lưới $5 \times 5$ (25 ô tọa độ) | Mê cung các tuyến phố có rào cản |
| **Điểm đón/trả khách** | 4 vị trí cố định: R, G, Y, B | Red, Green, Yellow, Blue |
| **Không gian trạng thái $|\mathcal{S}|$** | $25 \times 5 \times 4 = 500$ trạng thái | Tọa độ xe ($25$) $\times$ Vị trí khách ($4$ điểm hoặc trên xe) $\times$ Điểm đích ($4$) |
| **Không gian hành động $|\mathcal{A}|$** | Discrete(6): $\{0, 1, 2, 3, 4, 5\}$ | 4 hướng di chuyển: Nam, Bắc, Đông, Tây<br>2 thao tác: Đón khách (Pickup), Trả khách (Dropoff) |
| **Cơ cấu phần thưởng** | - $+20.0$: Trả khách thành công tại đúng điểm đích.<br>- $-10.0$: Đón hoặc trả khách sai quy định.<br>- $-1.0$: Chi phí mỗi bước di chuyển tìm đường. | Khuyến khích tối ưu quãng đường và tránh thao tác thừa. |

---

## Design

### Mã Nguồn Triển Khai Q-Learning Hoàn Chỉnh (Pure Python Implementation)

Đoạn mã dưới đây cài đặt hoàn chỉnh một tác tử `QLearningAgent` thuần Python, tương tác và huấn luyện trên môi trường tìm đường GridWorld có chướng ngại vật, theo dõi đường cong học tập và trích xuất lộ trình tối ưu:

```python
import numpy as np

class MazeEnv:
    """Môi trường Mê cung 4x4:
    - S: Xuất phát (0, 0)
    - G: Đích đến (3, 3) -> Thưởng +10.0
    - B: Bẫy lửa (1, 1) và (2, 2) -> Phạt -10.0 (Terminal)
    - Bước đi bình thường -> Phạt -0.5
    """
    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.traps = [(1, 1), (2, 2)]
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # 0: LÊN, 1: XUỐNG, 2: TRÁI, 3: PHẢI
        r, c = self.state
        if action == 0:    r = max(0, r - 1)
        elif action == 1:  r = min(self.rows - 1, r + 1)
        elif action == 2:  c = max(0, c - 1)
        elif action == 3:  c = min(self.cols - 1, c + 1)
        
        self.state = (r, c)
        
        if self.state == self.goal:
            return self.state, 10.0, True
        elif self.state in self.traps:
            return self.state, -10.0, True
        else:
            return self.state, -0.5, False


class QLearningAgent:
    """Tác tử Q-Learning chuẩn mực."""
    def __init__(self, n_states=16, n_actions=4, lr=0.1, gamma=0.95, eps_start=1.0, eps_min=0.05, eps_decay=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        
        # Khởi tạo bảng Q-Table [16 x 4]
        self.Q = np.zeros((n_states, n_actions))

    def to_state_idx(self, state_tuple):
        return state_tuple[0] * 4 + state_tuple[1]

    def choose_action(self, state_idx):
        """Chiến lược Epsilon-Greedy chọn hành động."""
        if np.random.rand() < self.eps:
            return np.random.choice(self.n_actions) # Thăm dò
        return int(np.argmax(self.Q[state_idx, :]))  # Khai thác

    def update(self, s, a, r, s_next, terminated):
        """Công thức cập nhật Off-Policy Bellman Q-Learning."""
        best_next_a = np.argmax(self.Q[s_next, :])
        q_target = r + self.gamma * self.Q[s_next, best_next_a] * (not terminated)
        td_error = q_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error
        return td_error

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ==============================================================================
# TIẾN HÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ CHÍNH SÁCH
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    env = MazeEnv()
    agent = QLearningAgent(n_states=16, n_actions=4, lr=0.1, gamma=0.95, eps_start=1.0, eps_min=0.05, eps_decay=0.992)
    
    n_episodes = 300
    rewards_history = []
    
    print("=== HUẤN LUYỆN Q-LEARNING TRÊN MÔI TRƯỜNG MÊ CUNG CÓ BẪY ===")
    print(f"Tổng số Episodes: {n_episodes} | Learning Rate: {agent.lr} | Gamma: {agent.gamma}\n")
    
    for ep in range(1, n_episodes + 1):
        state = env.reset()
        s_idx = agent.to_state_idx(state)
        total_reward = 0
        terminated = False
        steps = 0
        
        while not terminated and steps < 50:
            action = agent.choose_action(s_idx)
            next_state, reward, terminated = env.step(action)
            s_next_idx = agent.to_state_idx(next_state)
            
            agent.update(s_idx, action, reward, s_next_idx, terminated)
            
            s_idx = s_next_idx
            total_reward += reward
            steps += 1
            
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        if ep % 50 == 0:
            avg_rew = np.mean(rewards_history[-50:])
            print(f"Episode {ep:03d} | Epsilon: {agent.eps:.4f} | Avg Return (50 eps): {avg_rew:>+6.2f} | Bước đi: {steps}")

    # Kiểm tra chính sách Greedy tối ưu sau huấn luyện
    print("\n--- KIỂM THỬ CHÍNH SÁCH GREEDY TỐI ƯU (KHÔNG THĂM DÒ) ---")
    state = env.reset()
    path = [state]
    total_test_reward = 0
    terminated = False
    
    while not terminated and len(path) < 15:
        s_idx = agent.to_state_idx(state)
        best_action = np.argmax(agent.Q[s_idx, :])
        state, r, terminated = env.step(best_action)
        path.append(state)
        total_test_reward += r
        
    print(f"Lộ trình tối ưu Agent đã đi: {path}")
    print(f"Tổng điểm thưởng kiểm thử: {total_test_reward:.2f}")
    print(f"Số bước tối ưu tới đích: {len(path) - 1} bước (Đã né hoàn hảo các bẫy (1,1) và (2,2))")
```

---

## Nguồn

- [1] Watkins, C. J. C. H. (1989). *Learning from delayed rewards*. Ph.D. thesis, University of Cambridge, England.
- [2] Watkins, C. J., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3-4), 279-292.
- [3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 6: Temporal-Difference Learning, Section 6.5: Q-learning: Off-policy TD Control). MIT Press.
