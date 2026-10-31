# Tổng Quan Về Học Tăng Cường (Reinforcement Learning)

## Purpose

Học Tăng Cường (Reinforcement Learning - RL) là một trong ba phân nhánh nền tảng của học máy (bên cạnh Học có giám sát và Học không giám sát), tập trung vào bài toán: **Làm thế nào để một tác tử tính toán (Agent) học được cách ra quyết định tuần tự (Sequential Decision Making) thông qua tương tác thử - sai (Trial and Error) với môi trường (Environment) nhằm tối đa hóa phần thưởng tích lũy lâu dài**.

Khác với Học có giám sát (vốn dựa trên nhãn đúng sai tức thì từ tập dữ liệu tĩnh) hay Học không giám sát (khám phá cấu trúc hình học ẩn), Học tăng cường giải quyết bài toán hành động trong thế giới động với các đặc thù:
- **Tín hiệu đánh giá trễ (Evaluative & Delayed Feedback):** Hành động được thực hiện ở thời điểm hiện tại có thể không mang lại phần thưởng ngay lập tức, mà tác động sâu sắc đến trạng thái tương lai và chỉ nhận được kết quả (thắng/thua, sống sót/va chạm) sau hàng trăm bước tiếp theo.
- **Dữ liệu phụ thuộc vào hành động (Non-i.i.d Interactive Data):** Dữ liệu không có sẵn dưới dạng tập tĩnh độc lập cùng phân phối, mà được sinh ra tuần tự dựa trên chính các quyết định của Agent.
- **Nghịch lý Thăm dò và Khai thác (Exploration vs. Exploitation Dilemma):** Agent phải liên tục cân bằng giữa việc *khai thác* những hành động đã biết là mang lại điểm số cao, và việc *thăm dò* những hành động mới chưa từng thử nhằm tìm kiếm chiến lược vượt trội hơn.

Bảng so sánh 3 trụ cột của Học máy:

| Tiêu chí | Học có giám sát (Supervised) | Học không giám sát (Unsupervised) | Học tăng cường (Reinforcement) |
| :--- | :--- | :--- | :--- |
| **Dữ liệu đầu vào** | Cặp đặc trưng và nhãn $`(x_i, y_i)`$ | Chỉ có đặc trưng $`x_i`$ | Chuỗi tương tác $`(s_t, a_t, r_{t+1}, s_{t+1})`$ |
| **Tín hiệu phản hồi** | Chỉ dẫn chính xác nhãn đúng (Ground Truth) | Hoàn toàn không có tín hiệu phản hồi | Điểm số phần thưởng định lượng (Scalar Reward) |
| **Bản chất phản hồi** | Tức thì, hướng dẫn trực tiếp (Instructive) | Không có | Có độ trễ, mang tính đánh giá (Evaluative) |
| **Mục tiêu học** | Xấp xỉ hàm ánh xạ $`f: X \to Y`$ | Khám phá cấu trúc, mật độ, phân cụm | Tối đa hóa tổng phần thưởng tích lũy $`\sum r_t`$ |
| **Môi trường** | Tĩnh, độc lập thống kê (i.i.d) | Tĩnh, độc lập thống kê (i.i.d) | Động, trạng thái hiện tại phụ thuộc bước trước |

---

## Input

Một bài toán Học tăng cường tiêu chuẩn tiếp nhận các thành phần đầu vào:

1. **Môi trường mô phỏng hoặc thực tế (Environment $`\mathcal{E}`$):**
   Thế giới mà Agent tồn tại, tiếp nhận hành động và phản hồi lại thông tin trạng thái kèm điểm thưởng.

2. **Không gian trạng thái (State Space $`\mathcal{S}`$):**
   Tập hợp tất cả các trạng thái hợp lệ của môi trường. Trạng thái $`s_t \in \mathcal{S}`$ biểu diễn đầy đủ thông tin cần thiết tại thời điểm $`t`$ (ví dụ: góc nghiêng và vận tốc của con lắc, vị trí bàn cờ, hoặc khung hình ảnh camera).

3. **Không gian hành động (Action Space $`\mathcal{A}`$):**
   Tập hợp các hành động mà Agent được phép thực hiện:
   - Rời rạc (Discrete): $`\mathcal{A} = \left\lbrace \text{LÊN}, \text{XUỐNG}, \text{TRÁI}, \text{PHẢI} \right\rbrace`$.
   - Liên tục (Continuous): $`\mathcal{A} \subset \mathbb{R}^d`$ (ví dụ: góc bẻ lái xe $`[-30^\circ, +30^\circ]`$, lực nhấn chân ga $`[0, 1]`$).

4. **Tín hiệu phần thưởng vô hướng (Scalar Reward Signal $`r_t \in \mathbb{R}`$):**
   Giá trị số học môi trường gửi cho Agent sau mỗi bước chuyển trạng thái, mã hóa mục tiêu nghiệp vụ cần đạt.

5. **Hệ số chiết khấu tương lai (Discount Factor $`\gamma \in [0, 1)`$):**
   Xác định mức độ coi trọng phần thưởng tương lai so với phần thưởng trước mắt.

---

## Output

Quá trình huấn luyện Học tăng cường cung cấp các thành phần đầu ra tối ưu:

1. **Chính sách hành động tối ưu (Optimal Policy $`\pi^*`$):**
   Chiến lược chỉ dẫn Agent đưa ra quyết định tại mỗi trạng thái:
   - Chính sách tất định (Deterministic Policy):
   
$$
a_t = \pi^*(s_t)
$$

   - Chính sách xác suất (Stochastic Policy):
   
$$
\pi^*(a \mid s) = P(A_t = a \mid S_t = s)
$$

2. **Hàm giá trị trạng thái tối ưu (Optimal State-Value Function $`V^*(s)`$):**
   Kỳ vọng tổng phần thưởng tích lũy có chiết khấu nếu bắt đầu từ trạng thái $`s`$ và tuân thủ chính sách tối ưu:
   
$$
V^*(s) = \max_\pi \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \;\middle|\; S_t = s \right]
$$

3. **Hàm giá trị hành động tối ưu (Optimal Action-Value Function $`Q^*(s, a)`$):**
   Kỳ vọng tổng phần thưởng tích lũy nếu thực hiện hành động $`a`$ tại trạng thái $`s`$ và sau đó hành động tối ưu:
   
$$
Q^*(s, a) = \max_\pi \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k r_{t+k+1} \;\middle|\; S_t = s, A_t = a \right]
$$

---

## Evaluation metrics

Đánh giá một hệ thống Học tăng cường đòi hỏi các thước đo định lượng chuyên biệt:

1. **Tổng phần thưởng tích lũy mỗi Episode (Undiscounted Return $`G`$):**
   Tổng điểm số thực tế thu được trong một lượt chơi/thực nghiệm từ trạng thái đầu đến trạng thái kết thúc (Terminal State):
   
$$
G = \sum_{t=0}^{T-1} r_{t+1}
$$

2. **Trung bình động phần thưởng (Moving Average Return qua 100 episodes):**
   Làm mịn dao động ngẫu nhiên để đánh giá xu hướng hội tụ thực sự của đường cong học tập:
   
$$
\bar{G}_k = \frac{1}{100} \sum_{i=k-99}^k G_i
$$

3. **Độ dài trung bình Episode (Average Episode Length / Steps to Goal):**
   Số bước tính toán để Agent hoàn thành nhiệm vụ. Trong các bài toán tìm đường (Navigation), số bước càng thấp phản ánh chiến lược càng tối ưu.

4. **Tỷ lệ thành công (Success Rate %):**
   Tỷ lệ phần trăm các episodes Agent đạt tới trạng thái đích an toàn mà không rơi vào trạng thái bẫy hoặc hết thời gian (Time-out).

5. **Độ ổn định phương sai (Policy Stability Variance):**
   Đo lường độ lệch chuẩn của phần thưởng khi triển khai chính sách tất định (Greedy Policy) sau huấn luyện.

---

## How it works

### 1. Chu Trình Tương Tác Cơ Bản Giữa Agent và Environment

Học tăng cường vận hành theo một vòng lặp tương tác khép kín trên trục thời gian rời rạc $`t = 0, 1, 2, \dots`$:

```
                       ┌──────────────────────────────────────────────┐
                       │                 ENVIRONMENT                  │
                       │             (Môi trường tương tác)           │
                       └──────────────┬────────────────────────▲──────┘
                                      │                        │
       Trạng thái s_{t+1}             │                        │  Hành động a_t
       Phần thưởng r_{t+1}            │                        │  (Action)
                                      ▼                        │
                       ┌───────────────────────────────────────┴──────┐
                       │                    AGENT                     │
                       │           (Tác tử đưa ra quyết định)         │
                       └──────────────────────────────────────────────┘
```

1. Tại thời điểm $`t`$, Agent quan sát trạng thái hiện tại $`s_t \in \mathcal{S}`$.
2. Dựa trên chính sách $`\pi(a_t \mid s_t)`$, Agent lựa chọn và phát ra hành động $`a_t \in \mathcal{A}`$.
3. Môi trường tiếp nhận $`a_t`$, chuyển dịch sang trạng thái mới $`s_{t+1}`$ theo hàm xác suất chuyển dịch $`P(s_{t+1} \mid s_t, a_t)`$.
4. Môi trường phát tín hiệu phần thưởng vô hướng $`r_{t+1} \in \mathbb{R}`$ phản hồi về Agent.
5. Agent quan sát cặp $`(s_{t+1}, r_{t+1})`$, cập nhật tri thức nội tại (bảng Q, trọng số mạng nơ-ron hoặc chính sách), và lặp lại bước 1.

---

### 2. Giả Thuyết Phần Thưởng và Công Thức Chiết Khấu (Discounted Return)

**Giả thuyết Phần thưởng (The Reward Hypothesis):**
> *Tất cả các mục tiêu và ý định của bài toán đều có thể được chuẩn hóa dưới dạng tối đa hóa giá trị kỳ vọng của tổng tích lũy tín hiệu phần thưởng vô hướng nhận được từ môi trường.* (Sutton & Barto, 2018).

Để đảm bảo tổng phần thưởng không tiến tới vô cùng trong các bài toán lặp vô hạn (Continuing Tasks), và phản ánh tính không chắc chắn của tương lai xa, tổng phần thưởng tích lũy tại thời điểm $`t`$ được định nghĩa kèm hệ số chiết khấu $`\gamma \in [0, 1)`$:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k r_{t+k+1}
$$

Hệ thức truy hồi Bellman nền tảng liên kết giữa các bước:

$$
G_t = r_{t+1} + \gamma G_{t+1}
$$

- Khi $`\gamma = 0`$: Agent có tầm nhìn thiển cận (Myopic), chỉ quan tâm tối đa hóa phần thưởng ngay bước kế tiếp $`r_{t+1}`$.
- Khi $`\gamma \to 1`$: Agent có tầm nhìn xa trông rộng (Far-sighted), sẵn sàng chấp nhận chịu phạt ngắn hạn để thu hoạch phần thưởng lớn trong tương lai.

---

### 3. Phân Loại Hệ Thống Các Phương Pháp Học Tăng Cường

Cây phân loại tổng thể các giải thuật RL hiện đại:

```
                            CÁC PHƯƠNG PHÁP HỌC TĂNG CƯỜNG (RL)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
             [MODEL-FREE RL]                                 [MODEL-BASED RL]
         (Không cần mô hình động)                        (Học/Biết trước mô hình P, R)
                      │                                             │
         ┌────────────┴────────────┐                        ┌───────┴───────┐
         ▼                         ▼                        ▼               ▼
   [VALUE-BASED]            [POLICY-BASED]             [Dyna-Q]       [AlphaZero/MCTS]
(Học hàm Q(s, a))         (Học trực tiếp π(a|s))
   │         │               │          │
   ▼         ▼               ▼          ▼
[Q-Learn]  [DQN]       [REINFORCE]   [PPO/TRPO]
   │                         │
   └────────────┬────────────┘
                ▼
         [ACTOR-CRITIC]
          (A2C, SAC)
```

1. **Model-Based RL:** Agent học hoặc được cung cấp trước ma trận xác suất chuyển trạng thái $`P(s' \mid s, a)`$ và hàm thưởng $`R(s, a)`$. Agent có thể lập kế hoạch (Planning) trong tâm trí trước khi hành động (ví dụ: Dynamic Programming, AlphaZero).
2. **Model-Free RL:** Agent không biết trước cơ chế vận hành của môi trường. Agent hoàn toàn học thông qua trải nghiệm thực tế (Trial-and-Error). Đây là nhóm giải thuật phổ biến nhất trong thế giới thực.
3. **Value-Based:** Ước lượng hàm giá trị $`Q(s, a)`$ rồi chọn hành động tham lam $`a = \arg\max Q(s, a)`$ (Q-Learning, SARSA, DQN).
4. **Policy-Based:** Tham số hóa trực tiếp hàm chính sách $`\pi_\theta(a \mid s)`$ và leo đồi gradient để cực đại hóa kỳ vọng phần thưởng (Policy Gradient, REINFORCE).
5. **Actor-Critic:** Kết hợp sức mạnh của cả hai — Actor chọn hành động, Critic đánh giá chất lượng hành động để giảm phương sai (A2C, A3C, PPO, Soft Actor-Critic).

---

## Data used

Dữ liệu trong RL không phải là các bảng tĩnh cố định, mà là các chuỗi quỹ đạo động (Trajectories / Rollouts):

$$
\tau = \left( s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T, s_T \right)
$$

Các môi trường chuẩn hóa kiểm thử quốc tế (Gymnasium / OpenAI Gym):

| Môi trường | Kiểu trạng thái $\mathcal{S}$ | Kiểu hành động $\mathcal{A}$ | Mục tiêu bài toán | Ngưỡng giải quyết chuẩn |
| :--- | :--- | :--- | :--- | :--- |
| **GridWorld** | Tọa độ rời rạc $(x, y)$ | 4 hướng $\{0, 1, 2, 3\}$ | Đi từ ô xuất phát đến ô đích, tránh bẫy | Tìm đường ngắn nhất |
| **CartPole-v1** | Liên tục 4 chiều (vị trí xe, góc con lắc...) | 2 hướng (Đẩy trái / Phải) | Giữ thanh con lắc đứng thẳng trên xe trượt | Đạt điểm 500 liên tục |
| **Taxi-v3** | Rời rạc 500 trạng thái | 6 hành động rời rạc | Đón khách và trả đúng điểm đích trong mê cung | Điểm trung bình $> 8.0$ |
| **LunarLander-v2**| Liên tục 8 chiều (vị trí, góc, vận tốc) | 4 hành động động cơ đẩy | Hạ cánh tàu vũ trụ an toàn xuống bãi đáp | Điểm trung bình $> 200.0$ |
| **Atari Pong/Breakout**| Ảnh 4 khung hình $(84 \times 84 \times 4)$ | Phím điều khiển joystick | Đánh bại đối thủ trong trò chơi điện tử cổ điển | Vượt qua kỷ lục con người |

---

## Design

### Mã Nguồn Mô Phỏng Vòng Lặp Học Tăng Cường Tự Chế (Pure Python Simulation)

Đoạn mã dưới đây cài đặt hoàn chỉnh một môi trường GridWorld $1 \times 5$ đơn giản, một Random Agent và một Q-Learning Agent cơ bản, trực quan hóa cách thức chuỗi tương tác sinh ra dữ liệu và tối ưu hóa chính sách:

```python
import numpy as np

# ==============================================================================
# 1. THIẾT KẾ MÔI TRƯỜNG CHUẨN (LINE WORLD ENVIRONMENT: 1x5)
# ==============================================================================
class LineWorldEnv:
    """Môi trường 1D gồm 5 ô: [0, 1, 2, 3, 4]
    - Ô 0: Trạng thái xuất phát (Start).
    - Ô 4: Trạng thái đích (Goal, Terminal) -> Thưởng +10.
    - Mỗi bước di chuyển tốn chi phí -> Thưởng -1.
    """
    def __init__(self, length=5):
        self.length = length
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Action: 0 (TRÁI), 1 (PHẢI)
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(self.length - 1, self.state + 1)

        # Kiểm tra trạng thái kết thúc
        terminated = (self.state == self.length - 1)
        reward = 10.0 if terminated else -1.0
        return self.state, reward, terminated


# ==============================================================================
# 2. VÒNG LẶP HUẤN LUYỆN BẢNG Q-LEARNING CƠ SỞ
# ==============================================================================
def train_simple_q_learning():
    env = LineWorldEnv(length=5)
    n_states = 5
    n_actions = 2
    
    # Khởi tạo bảng Q-Table kích thước [5 x 2] bằng 0
    Q = np.zeros((n_states, n_actions))
    
    lr = 0.1          # Tốc độ học (alpha)
    gamma = 0.9       # Hệ số chiết khấu
    epsilon = 0.2     # Tỷ lệ thăm dò (exploration)
    n_episodes = 100
    
    print("=== BẮT ĐẦU VÒNG LẶP HỌC TĂNG CƯỜNG (LINE WORLD) ===")
    
    returns_history = []
    for ep in range(1, n_episodes + 1):
        s = env.reset()
        episode_reward = 0
        terminated = False
        steps = 0
        
        while not terminated and steps < 20:
            # Chiến lược Epsilon-Greedy chọn hành động
            if np.random.rand() < epsilon:
                a = np.random.choice(n_actions) # Thăm dò (Exploration)
            else:
                a = np.argmax(Q[s, :])          # Khai thác (Exploitation)
                
            s_next, r, terminated = env.step(a)
            episode_reward += r
            steps += 1
            
            # Cập nhật giá trị Q theo công thức Bellman Equation
            best_next_a = np.argmax(Q[s_next, :])
            td_target = r + gamma * Q[s_next, best_next_a] * (not terminated)
            td_error = td_target - Q[s, a]
            Q[s, a] += lr * td_error
            
            s = s_next
            
        returns_history.append(episode_reward)
        if ep % 20 == 0:
            avg_ret = np.mean(returns_history[-20:])
            print(f"Episode {ep:03d} | Average Return (20 eps): {avg_ret:>6.2f} | Bước đi: {steps}")

    print("\n=> BẢNG Q-TABLE TỐI ƯU SAU HUẤN LUYỆN:")
    print("Trạng thái | Q(s, TRÁI) | Q(s, PHẢI) | Hành động khuyến nghị")
    print("-" * 55)
    actions_map = {0: "<- SANG TRÁI", 1: "-> SANG PHẢI"}
    for s_idx in range(n_states - 1):
        best_a = np.argmax(Q[s_idx, :])
        print(f"Ô {s_idx:<8} | {Q[s_idx, 0]:>10.3f} | {Q[s_idx, 1]:>10.3f} | {actions_map[best_a]}")

if __name__ == "__main__":
    np.random.seed(42)
    train_simple_q_learning()
```

---

## Nguồn

- [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.
- [2] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533.
- [3] Kaelbling, L. P., Littman, M. L., & Moore, A. W. (1996). *Reinforcement learning: A survey*. Journal of Artificial Intelligence Research, 4, 237-285.
