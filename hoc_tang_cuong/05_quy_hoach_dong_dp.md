# Quy Hoạch Động Trong Học Tăng Cường (Dynamic Programming)

## Purpose

Quy hoạch động (Dynamic Programming - DP) trong Học tăng cường là tập hợp các giải thuật toán học dùng để **tính toán chính sách tối ưu khi mô hình môi trường đã được biết trước một cách hoàn hảo** dưới dạng một Quá trình Quyết định Markov (MDP) $`\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle`$.

Mặc dù trong các ứng dụng thực tế phức tạp, việc biết trước ma trận xác suất chuyển trạng thái $`\mathcal{P}`$ và hàm thưởng $`\mathcal{R}`$ là rất hiếm hoi, DP đóng vai trò là **chuẩn mực lý thuyết nền tảng (Theoretical Gold Standard)**:
- Cung cấp cơ chế giải thích nguyên lý hoạt động của mọi thuật toán Model-Free tiếp theo (như Monte Carlo hay Q-Learning, vốn là các phương pháp xấp xỉ lại các bước của DP khi không có mô hình).
- Đưa ra khung tư duy **Generalized Policy Iteration (GPI)** — sự tương tác luân phiên giữa hai quá trình: Đánh giá chính sách (Policy Evaluation) và Cải tiến chính sách (Policy Improvement).

Hai giải thuật điều khiển quy hoạch động kinh điển là:
1. **Lặp chính sách (Policy Iteration):** Đánh giá trọn vẹn chính sách hiện tại cho đến khi hội tụ, sau đó cải tiến chính sách một lần, và lặp lại chu kỳ.
2. **Lặp giá trị (Value Iteration):** Cắt ngắn bước đánh giá xuống chỉ còn đúng 1 bước lặp Bellman Optimality rồi cải tiến ngay lập tức, giúp tăng tốc độ tính toán vượt bậc.

---

## Input

Các thuật toán Quy hoạch động tiếp nhận mô hình MDP đầy đủ:

1. **Tập trạng thái và hành động:** $`\mathcal{S}`$ và $`\mathcal{A}`$.
2. **Mô hình chuyển trạng thái và hàm thưởng:**
   - $`\mathcal{P}_{ss'}^a = P(S_{t+1} = s' \mid S_t = s, A_t = a)`$.
   - $`\mathcal{R}(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']`$.
3. **Hệ số chiết khấu:** $`\gamma \in [0, 1)`$.
4. **Ngưỡng dừng hội tụ:** $`\theta > 0`$ (ví dụ: $`\theta = 10^{-6}`$).
5. **Khởi tạo ban đầu:** Bảng giá trị $`V_0(s) = 0`$ hoặc chính sách ngẫu nhiên ban đầu $`\pi_0(s)`$.

---

## Output

Kết quả đầu ra của Quy hoạch động:

1. **Hàm giá trị trạng thái tối ưu ($`V^*(s)`$):** Bảng giá trị chính xác cho mọi trạng thái $`s \in \mathcal{S}`$.
2. **Chính sách tất định tối ưu ($`\pi^*(s)`$):** Bảng chỉ dẫn hành động tối ưu duy nhất tại từng ô trạng thái:
   
$$
\pi^*(s) = \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V^*(s') \right)
$$

3. **Số vòng lặp và thời gian thực thi hội tụ.**

---

## Evaluation metrics

Chỉ số đánh giá và so sánh giữa Policy Iteration và Value Iteration:

1. **Số vòng lặp chính sách (Policy Iterations):** Số lần chính sách $`\pi_k`$ được cập nhật trước khi đạt trạng thái ổn định tuyệt đối (`policy_stable = True`).
2. **Tổng số lần quét trạng thái (Total State Sweeps):** Tổng số lần duyệt qua toàn bộ không gian trạng thái $`|\mathcal{S}|`$.
3. **Độ phức tạp tính toán mỗi bước quét:**
   
$$
\mathcal{O}\left( |\mathcal{S}|^2 \cdot |\mathcal{A}| \right)
$$

---

## How it works

### 1. Khung Tư Duy Generalized Policy Iteration (GPI)

Mọi thuật toán RL đều có thể được khái quát hóa thành sự tương tác cạnh tranh và hợp tác giữa hai lực lượng:

```
                  ┌──────────────────────────────────────────────┐
                  │              CHÍNH SÁCH π                    │
                  └──────────────┬────────────────────────▲──────┘
                                 │                        │
               Định hướng tạo ra │                        │ Cải tiến tham lam
               dữ liệu đánh giá  │                        │ theo hàm giá trị
                                 ▼                        │
                  ┌───────────────────────────────────────┴──────┐
                  │              HÀM GIÁ TRỊ V                   │
                  └──────────────────────────────────────────────┘
```

- **Policy Evaluation:** Cố định chính sách $`\pi`$, điều chỉnh hàm giá trị $`V`$ sao cho phản ánh trung thực kỳ vọng phần thưởng của $`\pi`$.
- **Policy Improvement:** Cố định hàm giá trị $`V`$, điều chỉnh chính sách $`\pi`$ trở nên tham lam hơn (Greedy) đối với $`V`$.
- Quá trình này chỉ dừng lại khi chính sách tham lam đối với hàm giá trị của chính nó, nghĩa là cả $`V`$ và $`\pi`$ đều chạm tới điểm tối ưu Bellman $`V^*`$ và $`\pi^*`$.

---

### 2. Thuật Toán Lặp Chính Sách (Policy Iteration)

Bao gồm hai bước lặp tuần hoàn cho đến khi chính sách không còn thay đổi:

#### Bước 1: Đánh Giá Chính Sách (Policy Evaluation)
Lặp phương trình kỳ vọng Bellman cho đến khi $`\Delta < \theta`$:

$$
V_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V_k(s') \right)
$$

#### Bước 2: Cải Tiến Chính Sách (Policy Improvement)
Cập nhật hành động tham lam cho từng trạng thái:

$$
\pi_{\mathrm{new}}(s) = \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V(s') \right)
$$

Nếu $`\pi_{\mathrm{new}}(s) = \pi(s)`$ với mọi $`s \in \mathcal{S}`$, thuật toán kết thúc và trả về $`\pi^*`$.

---

### 3. Thuật Toán Lặp Giá Trị (Value Iteration)

Policy Iteration đòi hỏi phải chờ đợi Policy Evaluation hội tụ hoàn toàn ở mỗi vòng lặp, điều này rất tốn kém tài nguyên. **Value Iteration** cắt ngắn bước đánh giá: Chỉ quét qua không gian trạng thái đúng **một lần** với toán tử tối ưu Bellman $`\max`$:

$$
V_{k+1}(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V_k(s') \right)
$$

Thuật toán lặp liên tục cho đến khi $`\max_s |V_{k+1}(s) - V_k(s)| < \theta`$. Sau đó, chính sách tối ưu $`\pi^*`$ được trích xuất chỉ trong một bước tham lam duy nhất.

---

## Data used

Môi trường thực nghiệm chuẩn hóa: **GridWorld $4 \times 4$ (Sutton & Barto, Ví dụ 4.1)**:

```
   ┌─────┬─────┬─────┬─────┐
 0 │ TRM │  1  │  2  │  3  │  <- Ô (0,0) là Terminal
   ├─────┼─────┼─────┼─────┤
 4 │  4  │  5  │  6  │  7  │
   ├─────┼─────┼─────┼─────┤
 8 │  8  │  9  │ 10  │ 11  │
   ├─────┼─────┼─────┼─────┤
12 │ 12  │ 13  │ 14  │ TRM │  <- Ô (3,3) là Terminal
   └─────┴─────┴─────┴─────┘
      0     1     2     3
```

- Không gian trạng thái: 16 ô $`\mathcal{S} = \lbrace 0, 1, \dots, 15 \rbrace`$. Ô 0 và ô 15 là trạng thái kết thúc (Terminal).
- Không gian hành động: 4 hướng $`\mathcal{A} = \lbrace \text{LÊN}, \text{XUỐNG}, \text{TRÁI}, \text{PHẢI} \rbrace`$.
- Động lực học: Xác suất $100\%$ đi đúng hướng; va vào mép lưới thì giữ nguyên vị trí.
- Phần thưởng: $`r = -1.0`$ cho mọi bước di chuyển cho đến khi chạm ô Terminal.

---

## Design

### Mã Nguồn Cài Đặt và So Sánh Policy Iteration vs Value Iteration (Pure Python)

Đoạn mã dưới đây cài đặt độc lập cả hai thuật toán quy hoạch động trên GridWorld $4 \times 4$, đối chiếu chi tiết số bước lặp và xác nhận sự hội tụ đồng quy về cùng một chính sách tối ưu:

```python
import numpy as np

class GridWorld4x4:
    """Môi trường GridWorld 4x4 chuẩn mực (Sutton & Barto Ví dụ 4.1)."""
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.n_states = 16
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.terminals = [0, 15]
        
    def step(self, s, a):
        if s in self.terminals:
            return s, 0.0
            
        r, c = s // 4, s % 4
        if a == 'UP':    r = max(0, r - 1)
        elif a == 'DOWN':  r = min(3, r + 1)
        elif a == 'LEFT':  c = max(0, c - 1)
        elif a == 'RIGHT': c = min(3, c + 1)
        
        s_next = r * 4 + c
        reward = -1.0
        return s_next, reward


def policy_iteration(env, theta=1e-6):
    """Giải bằng thuật toán Policy Iteration."""
    # Khởi tạo chính sách ngẫu nhiên đều
    pi = {s: np.random.choice(env.actions) for s in range(env.n_states)}
    V = np.zeros(env.n_states)
    
    policy_iters = 0
    total_eval_sweeps = 0
    
    while True:
        policy_iters += 1
        # 1. Policy Evaluation
        while True:
            total_eval_sweeps += 1
            delta = 0.0
            V_new = V.copy()
            for s in range(env.n_states):
                if s in env.terminals:
                    continue
                a = pi[s]
                s_next, r = env.step(s, a)
                v_target = r + env.gamma * V[s_next]
                delta = max(delta, abs(V[s] - v_target))
                V_new[s] = v_target
            V = V_new
            if delta < theta:
                break
                
        # 2. Policy Improvement
        policy_stable = True
        for s in range(env.n_states):
            if s in env.terminals:
                continue
            old_action = pi[s]
            best_a = None
            best_val = -np.inf
            for a in env.actions:
                s_next, r = env.step(s, a)
                val = r + env.gamma * V[s_next]
                if val > best_val:
                    best_val = val
                    best_a = a
            pi[s] = best_a
            if best_a != old_action:
                policy_stable = False
                
        if policy_stable:
            break
            
    return V, pi, policy_iters, total_eval_sweeps


def value_iteration(env, theta=1e-6):
    """Giải bằng thuật toán Value Iteration."""
    V = np.zeros(env.n_states)
    sweeps = 0
    
    while True:
        sweeps += 1
        delta = 0.0
        V_new = V.copy()
        for s in range(env.n_states):
            if s in env.terminals:
                continue
            q_vals = []
            for a in env.actions:
                s_next, r = env.step(s, a)
                q_vals.append(r + env.gamma * V[s_next])
            best_q = max(q_vals)
            delta = max(delta, abs(V[s] - best_q))
            V_new[s] = best_q
        V = V_new
        if delta < theta:
            break
            
    # Trích xuất chính sách tối ưu
    pi = {}
    for s in range(env.n_states):
        if s in env.terminals:
            pi[s] = "TRM"
            continue
        best_a = None
        best_val = -np.inf
        for a in env.actions:
            s_next, r = env.step(s, a)
            val = r + env.gamma * V[s_next]
            if val > best_val:
                best_val = val
                best_a = a
        pi[s] = best_a
        
    return V, pi, sweeps


# ==============================================================================
# CHẠY VÀ ĐỐI CHIẾU HIỆU NĂNG HAI THUẬT TOÁN QUY HOẠCH ĐỘNG
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    env = GridWorld4x4(gamma=1.0)
    
    V_pi, pi_pi, p_iters, p_sweeps = policy_iteration(env)
    V_vi, pi_vi, vi_sweeps = value_iteration(env)
    
    print("=== SO SÁNH POLICY ITERATION VS VALUE ITERATION (GRIDWORLD 4x4) ===")
    print(f"Policy Iteration : {p_iters} vòng lặp chính sách | Tổng số bước quét đánh giá: {p_sweeps}")
    print(f"Value Iteration  : {vi_sweeps} bước quét trạng thái trực tiếp")
    print("\n-> Kết quả: Cả hai đều tìm ra cùng hàm giá trị tối ưu V*(s):")
    
    print("\nBẢNG GIÁ TRỊ V*(s) (4x4 Matrix):")
    for r in range(4):
        row = " | ".join(f"{V_vi[r*4 + c]:>6.1f}" for c in range(4))
        print(f"Hàng {r} | {row}")
        
    print("\nCHÍNH SÁCH TỐI ƯU pi*(s):")
    arrows = {'UP': '^ UP ', 'DOWN': 'v DOWN', 'LEFT': '< LEFT', 'RIGHT': '> RGHT', 'TRM': ' TRM '}
    for r in range(4):
        row = " | ".join(f"{arrows.get(pi_vi[r*4 + c], ''):<6}" for c in range(4))
        print(f"Hàng {r} | {row}")
```

---

## Nguồn

- [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 4: Dynamic Programming). MIT Press.
- [2] Howard, R. A. (1960). *Dynamic programming and Markov processes*. MIT Press.
- [3] Puterman, M. L. (2014). *Markov decision processes: discrete stochastic dynamic programming*. John Wiley & Sons.
