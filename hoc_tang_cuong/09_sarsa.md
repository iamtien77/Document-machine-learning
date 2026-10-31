# Thuật Toán SARSA (On-Policy TD Control)

## Purpose

Thuật toán SARSA (được đề xuất bởi Rummery & Niranjan năm 1994 và được đặt tên chính thức bởi Richard Sutton năm 1996) là giải thuật điều khiển **On-Policy Temporal Difference Control** kinh điển nhất trong Học tăng cường.

Tên gọi **SARSA** xuất phát từ chính chuỗi ngũ hợp dữ liệu trải nghiệm mà thuật toán xử lý tại mỗi bước lặp:

$$
\left( S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1} \right) \quad \longrightarrow \quad \mathbf{S} \text{-} \mathbf{A} \text{-} \mathbf{R} \text{-} \mathbf{S} \text{-} \mathbf{A}
$$

Sự khác biệt mang tính triết học và ứng dụng sâu sắc giữa SARSA (On-Policy) và Q-Learning (Off-Policy):
- **Q-Learning (Tối ưu nhưng liều lĩnh):** Cập nhật hàm giá trị dựa trên giả định rằng ở bước tiếp theo, tác tử sẽ luôn hành động tối ưu tuyệt đối ($`\max_{a'} Q(s', a')`$). Do đó, Q-Learning tìm ra con đường ngắn nhất trên lý thuyết, nhưng trong quá trình huấn luyện khi còn thăm dò ngẫu nhiên ($`\epsilon`$-greedy), nó rất dễ rơi vào hiểm họa chết người nếu con đường tối ưu nằm sát bờ vực.
- **SARSA (Thực tế và nhận thức rủi ro - Risk-Averse):** Cập nhật hàm giá trị dựa trên chính hành động thực tế tiếp theo $`A_{t+1}`$ mà tác tử sẽ thực thi (bao gồm cả khả năng chọn hành động ngẫu nhiên do thăm dò). Do đó, SARSA học được rằng việc đi sát mép vực là cực kỳ nguy hiểm và chủ động chọn con đường vòng xa hơn nhưng an toàn tuyệt đối.

---

## Input

Quá trình huấn luyện SARSA tiếp nhận:

1. **Môi trường rời rạc ($`\mathcal{E}`$):** Không gian trạng thái $`\mathcal{S}`$ và hành động $`\mathcal{A}`$.
2. **Bảng giá trị hành động khởi đầu ($`Q \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}`$):** Khởi tạo bằng 0 cho mọi cặp trạng thái - hành động.
3. **Chính sách hành vi mềm ($`\pi(a \mid s)`$):** Thường sử dụng chiến lược $`\epsilon`$-greedy.
4. **Tốc độ học (Learning Rate $`\alpha \in (0, 1]`$):** Khống chế mức độ cập nhật sai số.
5. **Hệ số chiết khấu ($`\gamma \in [0, 1)`$):** Cân bằng tầm nhìn phần thưởng tương lai.

---

## Output

Kết quả đầu ra của thuật toán SARSA:

1. **Bảng giá trị hành động On-policy tối ưu ($`Q^\pi(s, a)`$):** Phản ánh kỳ vọng điểm số thực tế khi tác tử tuân thủ chính sách hiện tại (kể cả rủi ro thăm dò).
2. **Chính sách kiểm soát an toàn ($`\pi^{\ast}(s)`$):**

$$
\pi^{\ast}(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)
$$

3. **Lộ trình hành động có tính toán rủi ro:** Tránh xa các trạng thái nguy hiểm có thể bị kích hoạt ngẫu nhiên bởi thăm dò.

---

## Evaluation metrics

Chỉ số so sánh thực nghiệm giữa SARSA và Q-Learning:

1. **Điểm thưởng tích lũy trong huấn luyện (Online Training Return):**
   Trong các môi trường có hình phạt lớn (bẫy, vực), SARSA đạt tổng điểm huấn luyện vượt trội so với Q-Learning nhờ ít khi bị sảy chân.

2. **Điểm thưởng kiểm thử không thăm dò (Asymptotic Test Return):**
   Khi tắt hoàn toàn thăm dò ($`\epsilon = 0`$), Q-Learning thường đạt điểm tối ưu ngắn nhất, trong khi SARSA đạt con đường dài hơn nhưng lề an toàn cao hơn.

3. **Tần suất kích hoạt bẫy / rơi xuống vực (Catastrophic Fall Rate %):**
   Tỷ lệ phần trăm các bước di chuyển tác tử rơi vào trạng thái nguy hiểm.

---

## How it works

### 1. Công Thức Cập Nhật Của Thuật Toán SARSA

Tại thời điểm $`t`$, tác tử ở trạng thái $`S_t`$ và đã chọn hành động $`A_t`$. Sau khi thực hiện $`A_t`$, môi trường trả về phần thưởng $`R_{t+1}`$ và trạng thái mới $`S_{t+1}`$.

Điểm then chốt: **Trước khi cập nhật bảng Q, tác tử sử dụng chính sách hiện tại để chọn luôn hành động tiếp theo $`A_{t+1}`$ từ $`S_{t+1}`$**.

Công thức cập nhật:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})}_{\text{SARSA Target (On-Policy)}} - Q(S_t, A_t) \right]
$$

Sai số TD của SARSA:

$$
\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
$$

---

### 2. So Sánh Bản Chất: SARSA vs Q-Learning vs Expected SARSA

```
SARSA (On-Policy)              Q-Learning (Off-Policy)           Expected SARSA
       S_t                            S_t                             S_t
        │                              │                               │
        ▼                              ▼                               ▼
       A_t                            A_t                             A_t
        │                              │                               │
        ▼                              ▼                               ▼
   R_{t+1}, S_{t+1}               R_{t+1}, S_{t+1}                R_{t+1}, S_{t+1}
        │                              │                               │
        ▼ (Theo chính sách hiện tại)   ▼ (Luôn chọn max)               ▼ (Kỳ vọng xác suất)
      A_{t+1}                    max_{a'} Q(S', a')          ∑ π(a'|S') Q(S', a')
```

| Thuật toán | Công thức mục tiêu TD (TD Target) | Bản chất chính sách | Mức độ chấp nhận rủi ro |
| :--- | :--- | :--- | :--- |
| **SARSA** | $`R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})`$ | On-Policy | **Thận trọng (Risk-Averse)**: Tránh xa ranh giới nguy hiểm |
| **Q-Learning** | $`R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')`$ | Off-Policy | **Lạc quan (Optimistic)**: Bám sát mép vực để tìm đường ngắn nhất |
| **Expected SARSA**| $`R_{t+1} + \gamma \sum_{a'} \pi(a' \mid S_{t+1}) Q(S_{t+1}, a')`$ | Off-Policy / On-Policy | **Cân bằng**: Triệt tiêu phương sai lấy mẫu $`A_{t+1}`$ |

---

### 3. Quy Trình Thuật Toán SARSA Từng Bước (Step-by-Step)

```
Khởi tạo bảng Q(s, a) tùy ý, Q(Terminal, ·) = 0

Lặp với mỗi Episode:
    Khởi tạo trạng thái ban đầu s
    Chọn hành động ban đầu a từ s bằng chính sách ε-greedy dựa trên Q(s, ·)

    Lặp với từng bước cho đến khi s là Terminal:
        1. Thực hiện hành động a, nhận phần thưởng r và trạng thái mới s'
        2. Chọn hành động tiếp theo a' từ s' bằng chính sách ε-greedy dựa trên Q(s', ·)
        3. Cập nhật bảng Q:
               Q(s, a) ← Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
        4. Chuyển dịch: s ← s', a ← a'
```

---

## Data used

Môi trường kiểm chuẩn kinh điển: **Vách Đá Tử Thần (The Cliff Walking - Sutton & Barto Ví dụ 6.6)**:

```
    0       1       2       3       4       5       6       7       8       9       10      11
  ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
0 │       │       │       │       │       │       │       │       │       │       │       │       │
  ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
1 │       │       │       │       │       │       │       │       │       │       │       │       │
  ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
2 │       │       │       │       │       │       │       │       │       │       │       │       │
  ├───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤
3 │ START │ VỰC 1 │ VỰC 2 │ VỰC 3 │ VỰC 4 │ VỰC 5 │ VỰC 6 │ VỰC 7 │ VỰC 8 │ VỰC 9 │ VỰC 10│ GOAL  │
  └───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

- **Kích thước lưới:** $4 \times 12$ ô ($48$ trạng thái).
- **Điểm xuất phát:** Ô $(3, 0)$ (Góc dưới bên trái).
- **Điểm đích (Goal):** Ô $(3, 11)$ (Góc dưới bên phải).
- **Vách đá (The Cliff):** Các ô nằm giữa xuất phát và đích: $(3, 1)$ đến $(3, 10)$.
  - Rơi xuống vách đá: Nhận phạt cực nặng $R = -100.0$ và **bị đưa ngược trở lại điểm xuất phát** $(3, 0)$.
- **Chi phí bước đi:** Mọi bước di chuyển khác nhận thưởng $R = -1.0$.

---

## Design

### Mã Nguồn Cài Đặt và Đối Chiếu Thực Nghiệm SARSA vs Q-Learning (Pure Python)

Đoạn mã dưới đây cài đặt hoàn chỉnh môi trường Cliff Walking và thực thi đồng thời hai thuật toán SARSA và Q-Learning, minh họa trực quan sự khác biệt về hành vi nhận thức rủi ro:

```python
import numpy as np

class CliffWalkingEnv:
    """Môi trường Vách đá Cliff Walking 4x12 (Sutton & Barto Ví dụ 6.6)."""
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, c) for c in range(1, 11)]
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        r, c = self.state
        if action == 0:    r = max(0, r - 1)
        elif action == 1:  r = min(self.rows - 1, r + 1)
        elif action == 2:  c = max(0, c - 1)
        elif action == 3:  c = min(self.cols - 1, c + 1)

        self.state = (r, c)

        if self.state in self.cliff:
            # Rơi xuống vực: Phạt -100 và quay về điểm xuất phát
            return self.start, -100.0, False
        elif self.state == self.goal:
            # Cán đích an toàn
            return self.state, -1.0, True
        else:
            # Bước đi thông thường
            return self.state, -1.0, False


def epsilon_greedy(Q, state, eps, n_actions=4):
    if np.random.rand() < eps:
        return np.random.choice(n_actions)
    return int(np.argmax(Q[state[0], state[1], :]))


def train_sarsa_vs_q_learning(n_episodes=500, alpha=0.1, gamma=1.0, eps=0.1):
    env_sarsa = CliffWalkingEnv()
    env_ql = CliffWalkingEnv()

    Q_sarsa = np.zeros((4, 12, 4))
    Q_ql = np.zeros((4, 12, 4))

    sarsa_rewards = []
    ql_rewards = []

    for ep in range(1, n_episodes + 1):
        # ----------------------------------------------------------------------
        # 1. Chạy một Episode SARSA (On-Policy)
        # ----------------------------------------------------------------------
        s = env_sarsa.reset()
        a = epsilon_greedy(Q_sarsa, s, eps)
        ep_rew_sarsa = 0
        while True:
            s_next, r, term = env_sarsa.step(a)
            ep_rew_sarsa += r
            if term:
                Q_sarsa[s[0], s[1], a] += alpha * (r - Q_sarsa[s[0], s[1], a])
                break
            a_next = epsilon_greedy(Q_sarsa, s_next, eps)
            td_target = r + gamma * Q_sarsa[s_next[0], s_next[1], a_next]
            Q_sarsa[s[0], s[1], a] += alpha * (td_target - Q_sarsa[s[0], s[1], a])
            s, a = s_next, a_next
        sarsa_rewards.append(ep_rew_sarsa)

        # ----------------------------------------------------------------------
        # 2. Chạy một Episode Q-Learning (Off-Policy)
        # ----------------------------------------------------------------------
        s = env_ql.reset()
        ep_rew_ql = 0
        while True:
            a = epsilon_greedy(Q_ql, s, eps)
            s_next, r, term = env_ql.step(a)
            ep_rew_ql += r
            if term:
                Q_ql[s[0], s[1], a] += alpha * (r - Q_ql[s[0], s[1], a])
                break
            best_a = np.argmax(Q_ql[s_next[0], s_next[1], :])
            td_target = r + gamma * Q_ql[s_next[0], s_next[1], best_a]
            Q_ql[s[0], s[1], a] += alpha * (td_target - Q_ql[s[0], s[1], a])
            s = s_next
        ql_rewards.append(ep_rew_ql)

    return Q_sarsa, Q_ql, sarsa_rewards, ql_rewards

# ==============================================================================
# ĐỐI CHIẾU THỰC NGHIỆM VÀ ĐÁNH GIÁ LỘ TRÌNH
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    Q_sarsa, Q_ql, rew_sarsa, rew_ql = train_sarsa_vs_q_learning(n_episodes=500, eps=0.1)

    print("=== KẾT QUẢ SO SÁNH SARSA VS Q-LEARNING TRÊN CLIFF WALKING ===")
    print(f"Điểm thưởng trung bình trong huấn luyện (100 episodes cuối):")
    print(f"-> SARSA (On-Policy)     : {np.mean(rew_sarsa[-100:]):>7.2f} (Ít rơi xuống vực hơn nhiều)")
    print(f"-> Q-Learning (Off-Policy): {np.mean(rew_ql[-100:]):>7.2f} (Thường xuyên rơi vực do thăm dò)")

    # Kiểm tra lộ trình Greedy tối ưu sau khi tắt thăm dò
    def extract_path(Q):
        env = CliffWalkingEnv()
        s = env.reset()
        path = [s]
        for _ in range(20):
            a = np.argmax(Q[s[0], s[1], :])
            s, _, term = env.step(a)
            path.append(s)
            if term:
                break
        return path

    print("\nLỘ TRÌNH SUY LUẬN SAU HUẤN LUYỆN (GREEDY POLICY):")
    print(f"-> Q-Learning : {len(extract_path(Q_ql)) - 1} bước. Đi sát mép hàng 2-3 (Đường ngắn nhất nhưng mép vực)")
    print(f"-> SARSA      : {len(extract_path(Q_sarsa)) - 1} bước. Đi vòng lên hàng 0-1 (Đường an toàn, xa vực)")
    print("\n=> SARSA bảo toàn tính mạng trong huấn luyện, Q-Learning tối ưu quãng đường lý thuyết.")
```

---

## Nguồn

- [1] Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems*. Technical Report CUED/F-INFENG/TR 166, Cambridge University Engineering Department.
- [2] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 6: Temporal-Difference Learning, Section 6.4: Sarsa: On-policy TD Control). MIT Press.
- [3] Van Seijen, H., Van Hasselt, H., Whiteson, S., & Wiering, M. (2009). *A theoretical and empirical analysis of Expected Sarsa*. IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning, 177-184.
