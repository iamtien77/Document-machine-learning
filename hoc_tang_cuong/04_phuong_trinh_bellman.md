# Phương Trình Bellman (Bellman Equations)

## Purpose

Phương trình Bellman (được đặt theo tên nhà toán học Richard Bellman, người khai sinh ra Quy hoạch động vào thập niên 1950) là **trái tim toán học của toàn bộ lý thuyết Học tăng cường**.

Phương trình này thiết lập mối quan hệ đệ quy nền tảng: **Giá trị của trạng thái hiện tại bằng phần thưởng kỳ vọng tức thời cộng với giá trị chiết khấu của trạng thái kế tiếp**.

Toàn bộ các thuật toán RL kinh điển và hiện đại — từ Quy hoạch động (Policy/Value Iteration), Học sai phân thời gian (TD Learning, SARSA, Q-Learning) cho đến Học tăng cường sâu (DQN, Actor-Critic) — đều là các biến thể xấp xỉ hoặc phương pháp giải lặp trực tiếp từ các phương trình Bellman.

Hệ thống lý thuyết Bellman phân rã thành hai nhóm phương trình cốt lõi:
1. **Phương trình kỳ vọng Bellman (Bellman Expectation Equations):** Dùng để *đánh giá* một chính sách cho trước $`\pi`$ (Policy Evaluation).
2. **Phương trình tối ưu Bellman (Bellman Optimality Equations):** Dùng để *tìm kiếm* chính sách và giá trị tốt nhất khả dĩ (Control / Optimization).

---

## Input

Quá trình thiết lập và giải phương trình Bellman tiếp nhận các thành phần của một Markov Decision Process (MDP):

1. **Không gian trạng thái và hành động:** $`\mathcal{S}`$ và $`\mathcal{A}`$.
2. **Chính sách đánh giá ($`\pi(a \mid s)`$):** Phân phối xác suất chọn hành động $`a`$ khi đang ở trạng thái $`s`$.
3. **Mô hình động học của môi trường:**
   - Ma trận xác suất chuyển trạng thái:

$$
\mathcal{P}_{ss'}^a = P(S_{t+1} = s' \mid S_t = s, A_t = a)
$$

   - Hàm phần thưởng kỳ vọng:

$$
\mathcal{R}(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']
$$

4. **Hệ số chiết khấu tương lai ($`\gamma \in [0, 1)`$):** Đảm bảo tính hội tụ toán học và giới hạn giá trị tích lũy vô hạn.

---

## Output

Phương trình Bellman cung cấp nghiệm là các hàm giá trị:

1. **Hàm giá trị trạng thái (State-Value Function $`V^\pi(s)`$ hoặc $`V^{\ast}(s)`$):**
   Ước lượng mức độ tốt/xấu khi Agent đứng ở trạng thái $`s`$.

2. **Hàm giá trị hành động (Action-Value / Q-Function $`Q^\pi(s, a)`$ hoặc $`Q^{\ast}(s, a)`$):**
   Ước lượng mức độ tốt/xấu khi Agent thực hiện hành động $`a`$ tại trạng thái $`s`$.

3. **Nghiệm đóng giải tích dạng ma trận (Matrix Closed-Form Solution):**
   Dành cho bài toán MDP quy mô nhỏ có số trạng thái hữu hạn đã biết trước mô hình:

$$
V^\pi = \left( I - \gamma \mathcal{P}^\pi \right)^{-1} \mathcal{R}^\pi
$$

---

## Evaluation metrics

Chỉ số định lượng chất lượng hội tụ của phương trình Bellman:

1. **Độ dư Bellman vô cùng (Bellman Residual / Bellman Error):**
   Đo lường mức độ vi phạm phương trình Bellman lớn nhất trên toàn bộ không gian trạng thái:

$$
\mathrm{Residual}(V) = \| V - \mathcal{T}^\pi V \|_\infty = \max_{s \in \mathcal{S}} \left| V(s) - \left( \sum_{a} \pi(a \mid s) \sum_{s'} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V(s') \right) \right) \right|
$$

   Khi nghiệm hội tụ hoàn hảo: $`\mathrm{Residual}(V) \to 0`$.

2. **Tốc độ co rút hình học (Contraction Rate):**
   Theo định lý Banach, sai số giảm theo cấp số nhân với công bội $`\gamma`$:

$$
\| V_{k+1} - V^{\ast} \|_\infty \le \gamma \| V_k - V^{\ast} \|_\infty
$$

---

## How it works

### 1. Bốn Phương Trình Bellman Cốt Lõi

```
                       CÁC PHƯƠNG TRÌNH BELLMAN
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[PHƯƠNG TRÌNH KỲ VỌNG]                           [PHƯƠNG TRÌNH TỐI ƯU]
  (Bellman Expectation)                            (Bellman Optimality)
         │                                                 │
 ┌───────┴───────┐                                 ┌───────┴───────┐
 ▼               ▼                                 ▼               ▼
V^π(s)         Q^π(s, a)                         V*(s)           Q*(s, a)
(Tuyến tính)   (Tuyến tính)                      (Phi tuyến max) (Phi tuyến max)
```

#### A. Phương Trình Kỳ Vọng Cho $`V^\pi(s)`$
Giá trị của trạng thái $`s`$ là trung bình có trọng số theo chính sách $`\pi`$ của các hành động, mỗi hành động dẫn tới các trạng thái kế tiếp $`s'`$:

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right)
$$

#### B. Phương Trình Kỳ Vọng Cho $`Q^\pi(s, a)`$
Giá trị khi thực hiện hành động $`a`$ tại $`s`$ bằng phần thưởng tức thì cộng giá trị trung bình của các hành động tiếp theo tại $`s'`$:

$$
Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a') \right)
$$

#### C. Mối Quan Hệ Qua Lại Giữa $`V`$ Và $`Q`$

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)
$$

$$
Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right)
$$

---

### 2. Hai Phương Trình Tối Ưu Bellman (Bellman Optimality Equations)

Thay vì lấy trung bình theo chính sách $`\pi`$, chính sách tối ưu luôn chọn hành động mang lại giá trị cực đại ($`\max`$):

#### A. Phương Trình Tối Ưu Cho $`V^{\ast}(s)`$

$$
V^{\ast}(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma V^{\ast}(s') \right)
$$

#### B. Phương Trình Tối Ưu Cho $`Q^{\ast}(s, a)`$

$$
Q^{\ast}(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \left( \mathcal{R}(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^{\ast}(s', a') \right)
$$

> **Lưu ý then chốt:** Vì có toán tử $`\max`$, phương trình tối ưu Bellman là **hệ phương trình phi tuyến**, không thể giải trực tiếp bằng nghịch đảo ma trận tuyến tính $`(I - \gamma P)^{-1}`$ mà bắt buộc phải giải bằng các phương pháp lặp (Value Iteration, Policy Iteration, Q-Learning).

---

### 3. Nguyên Lý Ánh Xạ Co Banach (Contraction Mapping Theorem)

Định nghĩa toán tử Bellman $`\mathcal{T}^\pi: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}`$:

$$
(\mathcal{T}^\pi V)(s) = \mathcal{R}^\pi(s) + \gamma \sum_{s'} \mathcal{P}^\pi_{ss'} V(s')
$$

**Định lý:** Với $`\gamma < 1`$, toán tử $`\mathcal{T}^\pi`$ là một **ánh xạ co $`\gamma`$-co rút (Contraction Mapping)** theo chuẩn vô cùng ($`L_\infty`$ norm):

$$
\| \mathcal{T}^\pi U - \mathcal{T}^\pi V \|_\infty \le \gamma \| U - V \|_\infty
$$

**Hệ quả:** Theo Định lý Điểm Bất Động Banach (Banach Fixed-Point Theorem):
1. Phương trình $`V = \mathcal{T}^\pi V`$ tồn tại **duy nhất một nghiệm điểm bất động** $`V^\pi`$.
2. Phép lặp $`V_{k+1} = \mathcal{T}^\pi V_k`$ khởi đầu từ bất kỳ vector giá trị ngẫu nhiên nào $`V_0`$ đều được đảm bảo **hội tụ 100%** về nghiệm tối ưu $`V^\pi`$.

---

## Data used

Bảng so sánh phương thức giải phương trình Bellman theo quy mô không gian trạng thái:

| Quy mô trạng thái $|\mathcal{S}|$ | Phương pháp giải Bellman khả thi | Độ phức tạp tính toán | Ứng dụng thực tế |
| :--- | :--- | :--- | :--- |
| **Nhỏ ($|\mathcal{S}| < 1,000$)** | Nghịch đảo ma trận giải tích đóng $(I - \gamma P)^{-1}$ | $O(|\mathcal{S}|^3)$ | Phân tích bài toán đồ chơi (Two-state, GridWorld nhỏ) |
| **Vừa ($|\mathcal{S}| < 10^5$)** | Lặp giá trị Bellman (Dynamic Programming Value Iteration) | $O(K \cdot |\mathcal{S}|^2 \cdot |\mathcal{A}|)$ | Mê cung vừa, quản lý tồn kho, định tuyến mạng |
| **Lớn ($|\mathcal{S}| > 10^6$)** | Lấy mẫu Model-Free (TD Learning, Q-Learning) | Tuyến tính theo bước mẫu | Đồ thị lớn, trò chơi bàn cờ đơn giản |
| **Vô hạn / Liên tục** | Xấp xỉ hàm giá trị tham số (Deep Q-Networks / Bellman Loss) | Phụ thuộc mạng nơ-ron | Robot tự hành, xe tự lái, game Atari |

---

## Design

### Mã Nguồn So Sánh Nghiệm Đóng Giải Tích Và Lặp Bellman (Pure Python)

Đoạn mã dưới đây minh họa giải phương trình kỳ vọng Bellman trên một mô hình Markov 3 trạng thái bằng hai cách: (1) Phép lặp toán tử Bellman (Iterative Bellman Backup) và (2) Nghịch đảo ma trận giải tích đóng (Analytical Solution):

```python
import numpy as np

class BellmanSolver:
    """Giải phương trình Bellman trên mô hình Markov 3 trạng thái."""
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.n_states = 3
        self.state_names = ["Trạng thái 1", "Trạng thái 2", "Trạng thái 3"]

        # Ma trận xác suất chuyển trạng thái P^pi kích thước [3 x 3]
        self.P = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.0, 0.4, 0.6]
        ])

        # Vector phần thưởng kỳ vọng R^pi
        self.R = np.array([1.0, 2.0, -1.0])

    def solve_analytical(self):
        """Giải nghiệm đóng giải tích: V = (I - gamma * P)^(-1) * R."""
        I = np.eye(self.n_states)
        A = I - self.gamma * self.P
        V_exact = np.linalg.inv(A).dot(self.R)
        return V_exact

    def solve_iterative(self, theta=1e-7, max_iters=1000):
        """Giải bằng phép lặp toán tử Bellman: V_{k+1} = R + gamma * P * V_k."""
        V = np.zeros(self.n_states)
        history = []

        for k in range(max_iters):
            # Cập nhật Bellman Backup
            V_new = self.R + self.gamma * self.P.dot(V)

            # Tính độ dư Bellman cực đại
            residual = np.max(np.abs(V_new - V))
            history.append(residual)
            V = V_new

            if residual < theta:
                break

        return V, k + 1, history

# ==============================================================================
# CHẠY VÀ ĐỐI CHIẾU NGHIỆM PHƯƠNG TRÌNH BELLMAN
# ==============================================================================
if __name__ == "__main__":
    solver = BellmanSolver(gamma=0.9)

    # 1. Giải nghiệm đóng giải tích
    V_exact = solver.solve_analytical()

    # 2. Giải bằng phép lặp Bellman Backup
    V_iter, iters, hist = solver.solve_iterative()

    print("=== ĐỐI CHIẾU NGHIỆM PHƯƠNG TRÌNH KỲ VỌNG BELLMAN ===")
    print(f"Hệ số chiết khấu gamma: {solver.gamma} | Ngưỡng hội tụ: 1e-7\n")

    print(f"{'Trạng thái':<15} | {'Nghiệm đóng Ma trận':<20} | {'Nghiệm lặp Bellman':<20} | {'Sai số tuyệt đối':<18}")
    print("-" * 80)
    for i, name in enumerate(solver.state_names):
        err = abs(V_exact[i] - V_iter[i])
        print(f"{name:<15} | {V_exact[i]:>18.6f} | {V_iter[i]:>18.6f} | {err:>18.2e}")

    print("-" * 80)
    print(f"Số vòng lặp Bellman cần thiết để hội tụ: {iters} vòng.")
    print(f"Độ dư Bellman cuối cùng: {hist[-1]:.2e}")
    print("\n=> Định lý ánh xạ co Banach bảo đảm phép lặp hội tụ tuyệt đối về nghiệm giải tích.")
```

---

## Nguồn

- [1] Bellman, R. (1957). *Dynamic programming*. Princeton University Press.
- [2] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 3: Finite Markov Decision Processes, Sections 3.5 & 3.6). MIT Press.
- [3] Bertsekas, D. P. (2012). *Dynamic programming and optimal control* (Vol. II, 4th ed.). Athena Scientific.
