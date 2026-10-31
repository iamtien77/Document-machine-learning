# Quá Trình Quyết Định Markov (Markov Decision Process - MDP)

## Purpose

Quá trình Quyết định Markov (Markov Decision Process - MDP) là khung toán học chuẩn tắc (Formal Mathematical Framework) để mô hình hóa mọi bài toán ra quyết định tuần tự trong điều kiện có yếu tố ngẫu nhiên (Stochastic Sequential Decision Making), tạo nên nền móng lý thuyết vững chắc cho toàn bộ lĩnh vực Học tăng cường (Reinforcement Learning).

Mọi bài toán RL — từ trò chơi cờ vua, điều khiển cánh tay robot công nghiệp đến điều hướng xe tự hành — đều có thể được quy về việc giải một MDP hoặc một biến thể mở rộng của nó.

Điểm mấu chốt của khung lý thuyết này là **Tính chất Markov (The Markov Property)**:
> *"Tương lai độc lập với quá khứ khi đã biết hiện tại."*

Điều này có nghĩa là trạng thái hiện tại $`s_t`$ phải tóm tắt đầy đủ toàn bộ lịch sử các tương tác trong quá khứ sao cho việc dự báo trạng thái tiếp theo $`s_{t+1}`$ không cần biết thêm bất kỳ thông tin nào về các bước đi trước đó.

---

## Input

Một Quá trình Quyết định Markov hữu hạn được định nghĩa đầy đủ bởi một bộ 5 thành phần toán học (5-tuple):

$$
\mathcal{M} = \left\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \right\rangle
$$

1. **Không gian trạng thái hữu hạn ($`\mathcal{S}`$):**
   Tập hợp tất cả các trạng thái khả dĩ mà môi trường có thể có: $`\mathcal{S} = \lbrace s_1, s_2, \dots, s_{|\mathcal{S}|} \rbrace`$.

2. **Không gian hành động hữu hạn ($`\mathcal{A}`$):**
   Tập hợp tất cả các hành động mà tác tử có thể thực hiện: $`\mathcal{A} = \lbrace a_1, a_2, \dots, a_{|\mathcal{A}|} \rbrace`$.

3. **Mô hình xác suất chuyển trạng thái ($`\mathcal{P}`$):**
   Ten-xơ 3 chiều xác định xác suất chuyển sang trạng thái $`s'`$ nếu Agent thực hiện hành động $`a`$ tại trạng thái $`s`$:

$$
\mathcal{P}_{ss'}^a = P(S_{t+1} = s' \mid S_t = s, A_t = a)
$$

   Thỏa mãn điều kiện chuẩn hóa xác suất:

$$
\sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a = 1, \quad \forall s \in \mathcal{S}, \forall a \in \mathcal{A}
$$

4. **Hàm phần thưởng kỳ vọng ($`\mathcal{R}`$):**
   Giá trị phần thưởng trung bình nhận được khi thực hiện hành động $`a`$ tại trạng thái $`s`$ và chuyển sang $`s'`$:

$$
\mathcal{R}_s^a = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a] = \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \cdot \mathcal{R}(s, a, s')
$$

5. **Hệ số chiết khấu tương lai ($`\gamma`$):**
   Hệ số làm giảm giá trị của phần thưởng theo thời gian: $`\gamma \in [0, 1)`$.

---

## Output

Giải một bài toán MDP mang lại hai kết quả đầu ra cốt lõi:

1. **Chính sách tối ưu (Optimal Policy $`\pi^{\ast}`$):**
   Bảng chỉ dẫn hành động tối ưu cho từng trạng thái:

$$
\pi^{\ast}(s) = \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V^{\ast}(s') \right)
$$

2. **Hàm giá trị trạng thái tối ưu (Optimal Value Function $`V^{\ast}(s)`$):**
   Giá trị lớn nhất có thể đạt được từ mỗi trạng thái $`s`$, thỏa mãn phương trình tối ưu Bellman:

$$
V^{\ast}(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V^{\ast}(s') \right)
$$

---

## Evaluation metrics

Chỉ số đánh giá mức độ chính xác khi mô hình hóa bài toán dưới dạng MDP:

1. **Sai số kiểm định tính chất Markov (Markov Violation Residual):**
   Đo lường mức độ chênh lệch giữa phân phối xác suất khi xét lịch sử dài và khi chỉ xét bước hiện tại:

$$
\Delta_{\mathrm{Markov}} = \max_{s_t, a_t, s_{t-1}} \left| P(s_{t+1} \mid s_t, a_t) - P(s_{t+1} \mid s_t, a_t, s_{t-1}) \right|
$$

   Một hệ thống mô hình hóa chuẩn phải có $`\Delta_{\mathrm{Markov}} \approx 0`$.

2. **Dung sai hội quy hoạch động (Value Bellman Residual):**
   Sai số lớn nhất giữa hai vòng lặp giá trị liên tiếp:

$$
\Delta_k = \max_{s \in \mathcal{S}} |V_{k+1}(s) - V_k(s)|
$$

   Thuật toán dừng khi $`\Delta_k < \theta`$ với ngưỡng $`\theta = 10^{-6}`$.

---

## How it works

### 1. Tính Chất Markov và Giản Đồ Chuyển Dịch Trạng Thái

Toàn bộ lý thuyết dựa trên công thức toán học:

$$
P(S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = P(S_{t+1} = s_{t+1}, R_{t+1} = r_{t+1} \mid S_t, A_t)
$$

Nghĩa là: Mọi thông tin cần thiết về lịch sử đã được cô đọng hoàn toàn bên trong vector trạng thái $`S_t`$.

---

### 2. Mô Hình GridWorld $3 \times 4$ Kinh Điển (Slide tr. 15–16)

Mô hình GridWorld của Russell & Norvig là ví dụ chuẩn mực trong giảng dạy MDP:

```
    ┌───────┬───────┬───────┬───────┐
  3 │ (1,3) │ (2,3) │ (3,3) │ +1.0  │ <- GOAL (Terminal)
    ├───────┼───────┼───────┼───────┤
  2 │ (1,2) │ TƯỜNG │ (3,2) │ -1.0  │ <- BẪY (Terminal)
    ├───────┼───────┼───────┼───────┤
  1 │ START │ (2,1) │ (3,1) │ (4,1) │
    └───────┴───────┴───────┴───────┘
        1       2       3       4
```

- **Không gian trạng thái:** 11 ô hợp lệ (ô $`(2,2)`$ là bức tường cản, tác tử không thể bước vào).
- **Trạng thái kết thúc:**
  - Ô $`(4,3)`$: Đích đến thành công, nhận thưởng $`+1.0`$.
  - Ô $`(4,2)`$: Hố bẫy tử thần, nhận phạt $`-1.0`$.
- **Chi phí bước đi:** Mỗi bước di chuyển bình thường nhận thưởng $`r = -0.04`$ (thúc đẩy Agent tìm con đường ngắn nhất tới đích).
- **Tính bất định của chuyển động (Stochastic Transitions):**
  Hành động không phải lúc nào cũng thành công 100%. Nếu Agent ra lệnh đi thẳng:
  - **80%** xác suất đi đúng hướng ra lệnh.
  - **10%** xác suất trượt lệch sang bên trái $90^\circ$.
  - **10%** xác suất trượt lệch sang bên phải $90^\circ$.
  - Nếu bước đi va vào mép lưới hoặc bức tường cản, Agent nảy ngược lại và giữ nguyên vị trí.

---

### 3. Mô Hình Two-State MDP (Slide tr. 17)

Ví dụ tối giản nhất để minh họa cấu trúc mã nguồn MDP:
- Hai trạng thái: $`\mathcal{S} = \lbrace s_0, s_1 \rbrace`$.
- Hai hành động: $`\mathcal{A} = \lbrace a_0, a_1 \rbrace`$.
- Động thái:
  - Tại $`s_0`$, hành động $`a_0`$ giữ nguyên tại $`s_0`$ với thưởng $+1$, hành động $`a_1`$ chuyển sang $`s_1`$ với thưởng $0$.
  - Tại $`s_1`$, hành động $`a_0`$ chuyển về $`s_0`$ với thưởng $+2$, hành động $`a_1`$ giữ nguyên tại $`s_1`$ với thưởng $-1$.

---

## Data used

Bảng cấu trúc ma trận chuyển dịch cho môi trường GridWorld $3 \times 4$:

| Trạng thái $`s`$ | Hành động $`a`$ | Trạng thái kế tiếp $`s'`$ | Xác suất $`\mathcal{P}_{ss'}^a`$ | Thưởng $`R(s, a, s')`$ |
| :--- | :--- | :--- | :--- | :--- |
| `(1,1)` | `UP` | `(1,2)` (Đúng hướng) | **0.8** | $`-0.04`$ |
| `(1,1)` | `UP` | `(1,1)` (Đập mép trái) | **0.1** | $`-0.04`$ |
| `(1,1)` | `UP` | `(2,1)` (Lệch phải) | **0.1** | $`-0.04`$ |
| `(3,3)` | `RIGHT` | `(4,3)` (Cán đích GOAL) | **0.8** | $`+1.00`$ |
| `(3,3)` | `RIGHT` | `(3,2)` (Rơi xuống BẪY) | **0.1** | $`-1.00`$ |

---

## Design

### Mã Nguồn Giải Bài Toán MDP GridWorld $3 \times 4$ Bằng Value Iteration (Pure Python)

Đoạn mã sau cài đặt đầy đủ môi trường MDP GridWorld $3 \times 4$ với xác suất chuyển động bất định $80\% - 10\% - 10\%$, và giải nghiệm tối ưu giải tích bằng thuật toán Value Iteration:

```python
import math

class GridWorldMDP:
    """Cài đặt chuẩn toán học cho bài toán GridWorld 3x4 (Russell & Norvig).
    Thuần Python 100%, không phụ thuộc thư viện bên ngoài.
    """
    def __init__(self, discount=0.99, step_cost=-0.04):
        self.gamma = discount
        self.step_cost = step_cost
        self.rows = 3
        self.cols = 4

        # Danh sách trạng thái hợp lệ (bỏ ô tường (1,1) theo chỉ số 0-based)
        self.states = [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if (r, c) != (1, 1) # Bức tường ở ô (1, 1)
        ]

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.terminals = {(2, 3): 1.0, (1, 3): -1.0} # Ô đích (+1) và Ô bẫy (-1)

        # Bản đồ các hướng quay vuông góc
        self.action_effects = {
            'UP': {'main': (1, 0), 'left': (0, -1), 'right': (0, 1)},
            'DOWN': {'main': (-1, 0), 'left': (0, 1), 'right': (0, -1)},
            'LEFT': {'main': (0, -1), 'left': (-1, 0), 'right': (1, 0)},
            'RIGHT': {'main': (0, 1), 'left': (1, 0), 'right': (-1, 0)}
        }

    def _next_state(self, s, move):
        nr, nc = s[0] + move[0], s[1] + move[1]
        # Nếu đập vào biên hoặc đập vào tường cản (1,1) -> giữ nguyên vị trí
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or (nr, nc) == (1, 1):
            return s
        return (nr, nc)

    def get_transitions(self, s, a):
        """Trả về danh sách các bộ (xác suất, trạng thái mới, phần thưởng)."""
        if s in self.terminals:
            return [(1.0, s, 0.0)]

        effects = self.action_effects[a]
        transitions = [
            (0.8, self._next_state(s, effects['main'])),
            (0.1, self._next_state(s, effects['left'])),
            (0.1, self._next_state(s, effects['right']))
        ]

        results = []
        for prob, s_next in transitions:
            reward = self.terminals.get(s_next, self.step_cost)
            results.append((prob, s_next, reward))
        return results

    def solve_value_iteration(self, theta=1e-5):
        """Giải nghiệm giá trị tối ưu V*(s) và chính sách tối ưu pi*(s)."""
        V = {s: 0.0 for s in self.states}
        for term, r_val in self.terminals.items():
            V[term] = r_val

        while True:
            delta = 0.0
            new_V = V.copy()
            for s in self.states:
                if s in self.terminals:
                    continue
                q_values = []
                for a in self.actions:
                    q = sum(
                        prob * (reward + self.gamma * V[s_next])
                        for prob, s_next, reward in self.get_transitions(s, a)
                    )
                    q_values.append(q)
                best_q = max(q_values)
                delta = max(delta, abs(V[s] - best_q))
                new_V[s] = best_q
            V = new_V
            if delta < theta:
                break

        # Trích xuất chính sách tối ưu pi*(s)
        policy = {}
        for s in self.states:
            if s in self.terminals:
                policy[s] = "TERMINAL"
                continue
            best_a = None
            best_q = float('-inf')
            for a in self.actions:
                q = sum(
                    prob * (reward + self.gamma * V[s_next])
                    for prob, s_next, reward in self.get_transitions(s, a)
                )
                if q > best_q:
                    best_q = q
                    best_a = a
            policy[s] = best_a

        return V, policy


# ==============================================================================
# CHẠY GIẢI NGHIỆM MDP VÀ HIỂN THỊ KẾT QUẢ
# ==============================================================================
if __name__ == "__main__":
    mdp = GridWorldMDP(discount=0.99, step_cost=-0.04)
    V_opt, policy_opt = mdp.solve_value_iteration()

    print("=== NGHIỆM TỐI ƯU CỦA MDP GRIDWORLD 3x4 (VALUE ITERATION) ===")
    print("\n1. BẢNG GIÁ TRỊ TỐI ƯU V*(s):")
    for r in range(mdp.rows - 1, -1, -1):
        row_str = " | ".join(
            f"{V_opt.get((r, c), 0.0):>6.3f}" if (r, c) != (1, 1) else " TƯỜNG "
            for c in range(mdp.cols)
        )
        print(f"Hàng {r+1} | {row_str}")

    print("\n2. CHÍNH SÁCH TỐI ƯU pi*(s):")
    arrows = {'UP': '^ UP ', 'DOWN': 'v DOWN', 'LEFT': '< LEFT', 'RIGHT': '> RGHT', 'TERMINAL': ' GOAL '}
    for r in range(mdp.rows - 1, -1, -1):
        row_str = " | ".join(
            f"{arrows.get(policy_opt.get((r, c), '')):<6}" if (r, c) != (1, 1) else " TƯỜNG "
            for c in range(mdp.cols)
        )
        print(f"Hàng {r+1} | {row_str}")

    print("\n=> Khung toán học MDP giải quyết trọn vẹn yếu tố bất định và cung cấp đường đi tối ưu an toàn.")
```

---

## Nguồn

- [1] Bellman, R. (1957). *A Markovian decision process*. Journal of Mathematics and Mechanics, 679-684.
- [2] Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed., Chapter 17: Making Complex Decisions). Pearson.
- [3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 3: Finite Markov Decision Processes). MIT Press.
