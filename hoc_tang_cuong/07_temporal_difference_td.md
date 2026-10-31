# Học Sai Phân Thời Gian (Temporal Difference Learning - TD)

## Purpose

Học Sai Phân Thời Gian (Temporal Difference Learning - TD) được xem là **đột phá vĩ đại và đặc trưng nhất của lĩnh vực Học tăng cường**, do Richard Sutton phát minh vào năm 1988.

TD là sự kết tinh hoàn hảo giữa hai trường phái:
- **Kế thừa từ Monte Carlo (MC):** Học trực tiếp từ kinh nghiệm tương tác thực tế với môi trường mà **không cần biết trước mô hình toán học** (Model-Free).
- **Kế thừa từ Quy hoạch động (DP):** Sử dụng cơ chế **Bootstrapping** — cập nhật giá trị ước lượng của trạng thái hiện tại dựa trên giá trị ước lượng sẵn có của trạng thái kế tiếp, không cần chờ đợi đến khi toàn bộ episode kết thúc.

Ý nghĩa sinh học và thần kinh học: Khái niệm **Sai số TD (TD-Error $`\delta_t`$)** đã được chứng minh trong khoa học thần kinh (Schultz, Dayan & Montague, Science 1997) là mô hình toán học chính xác nhất giải thích hoạt động của các nơ-ron giải phóng Dopamine trong não bộ con người và động vật có vú khi đối mặt với các tín hiệu kích thích và phần thưởng bất ngờ.

---

## Input

Thuật toán TD(0) tiếp nhận các bước chuyển dịch trực tuyến (Online Transitions) theo từng bước thời gian:

1. **Bộ tứ chuyển dịch từng bước:**
   
$$
(s_t, a_t, r_{t+1}, s_{t+1})
$$

2. **Chính sách đánh giá:** $`\pi(a \mid s)`$.
3. **Hàm giá trị khởi tạo ban đầu:** $`V_0(s)`$ (thường gán bằng 0 cho mọi $`s`$).
4. **Tốc độ học (Step-size parameter):** $`\alpha \in (0, 1]`$.
5. **Hệ số chiết khấu:** $`\gamma \in [0, 1)`$.

---

## Output

Kết quả sinh ra từ quá trình học TD:

1. **Hàm giá trị trạng thái hội tụ ($`V^\pi(s)`$):** Bảng ước lượng giá trị chính xác cho mọi trạng thái.
2. **Dãy sai số sai phân thời gian ($`\delta_t`$):** Giảm dần biên độ và tiệm cận về 0 khi mô hình hội tụ.
3. **Khả năng dự đoán thời gian thực:** Cập nhật ngay sau mỗi bước đi $`t+1`$ mà không cần đợi hết lượt chơi.

---

## Evaluation metrics

Chỉ số đánh giá thuật toán TD:

1. **Sai số căn bậc hai trung bình (Root Mean Squared Error - RMSE):**
   So sánh giá trị ước lượng $`V(s)`$ với giá trị chính xác thực sự $`V_{\mathrm{true}}(s)`$:
   
$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( V(s) - V_{\mathrm{true}}(s) \right)^2}
$$

2. **Độ lớn trung bình trượt của TD-Error ($`\mathrm{MA}(|\delta_t|)`$):**
   Phản ánh mức độ bất ngờ của tác tử trước các diễn biến mới của môi trường.
3. **Hiệu quả mẫu (Sample Efficiency):**
   Số lượng bước tương tác cần thiết để đạt cùng một ngưỡng sai số so với phương pháp Monte Carlo.

---

## How it works

### 1. Quy Tắc Cập Nhật Của Thuật Toán TD(0)

Ngay sau khi tác tử đang ở trạng thái $`s_t`$, thực hiện hành động $`a_t`$, nhận phần thưởng $`r_{t+1}`$ và chuyển dịch sang trạng thái mới $`s_{t+1}`$, giá trị của $`s_t`$ được cập nhật tức thì:

$$
V(s_t) \leftarrow V(s_t) + \alpha \delta_t
$$

Trong đó, **Sai số TD (TD-Error)** được định nghĩa là chênh lệch giữa mục tiêu ước lượng một bước và giá trị hiện tại:

$$
\delta_t = \underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD Target}} - \underbrace{V(s_t)}_{\text{Ước lượng cũ}}
$$

- **TD Target:** $`r_{t+1} + \gamma V(s_{t+1})`$ là mục tiêu mới. Chú ý rằng mục tiêu này có chứa $`V(s_{t+1})`$, nghĩa là nó lấy ước lượng để cập nhật ước lượng (Bootstrapping).
- Nếu $`\delta_t > 0`$: Môi trường tốt hơn mong đợi $\to$ tăng $`V(s_t)`$.
- Nếu $`\delta_t < 0`$: Môi trường tồi hơn mong đợi $\to$ giảm $`V(s_t)`$.

---

### 2. So Sánh Tam Giác Vàng: DP vs MC vs TD

| Đặc tính so sánh | Quy hoạch động (DP) | Monte Carlo (MC) | Học sai phân thời gian (TD) |
| :--- | :--- | :--- | :--- |
| **Cần mô hình môi trường?** | **Bắt buộc** (Model-Based) | **Không cần** (Model-Free) | **Không cần** (Model-Free) |
| **Cơ chế Bootstrapping?** | **Có** (Dùng $V(s')$ lặp lại) | **Không** (Dùng Return thực tế $G$) | **Có** (Dùng $V(s')$ lặp lại) |
| **Cơ chế Lấy mẫu (Sampling)?** | **Không** (Quét hết phân bố) | **Có** (Lấy mẫu ngẫu nhiên) | **Có** (Lấy mẫu từng bước) |
| **Thời điểm cập nhật** | Sau mỗi lượt quét ma trận | **Phải đợi hết episode** | **Ngay sau từng bước chuyển** |
| **Áp dụng cho bài toán vô hạn?**| Có | **Không thể** | **Rất tốt** |
| **Độ chệch và Phương sai** | Chệch phụ thuộc mô hình | **Không chệch, Phương sai cao** | **Có chệch nhẹ, Phương sai thấp** |

---

### 3. Phổ Thống Nhất $`n`$-Step TD: Cầu Nối Giữa TD(0) Và Monte Carlo

TD(0) và Monte Carlo không phải là hai thái cực đối lập mà là hai điểm đầu mút trên một dải phổ liên tục gọi là **$`n`$-step TD**:

```
TD(0)           2-step TD         3-step TD                      Monte Carlo
 (n=1)            (n=2)             (n=3)                           (n=∞)
  │                 │                 │                               │
  ▼                 ▼                 ▼                               ▼
r + γ V(s')   r + γ r + γ² V(s'')   r + γ r + γ² r + γ³ V(s''')    r₁ + γ r₂ + γ² r₃ + ...
```

- $`n = 1`$ (TD(0)): Lấy mẫu 1 bước phần thưởng rồi bootstrap: $`G_{t:t+1} = r_{t+1} + \gamma V(s_{t+1})`$.
- $`n = 2`$: Lấy mẫu 2 bước phần thưởng rồi bootstrap: $`G_{t:t+2} = r_{t+1} + \gamma r_{t+2} + \gamma^2 V(s_{t+2})`$.
- $`n \to \infty`$: Lấy mẫu toàn bộ phần thưởng đến hết episode (chính là Monte Carlo).
- Trong thực tế, các giá trị $`n \in [3, 5]`$ hoặc thuật toán trung bình hóa **TD($`\lambda`$)** thường đạt hiệu năng tối ưu nhất nhờ cân bằng hoàn hảo giữa độ chệch và phương sai.

---

## Data used

Môi trường kiểm chuẩn kinh điển: **Random Walk 5 Trạng Thái (Sutton & Barto, Ví dụ 6.2)**:

```
        ┌───────┬───────┬───────┬───────┬───────┐
 [T0] <─│   A   │   B   │ C(ST) │   D   │   E   │─> [T1]
(R = 0) └───────┴───────┴───────┴───────┴───────┘  (R = +1)
```

- Không gian trạng thái: 5 trạng thái phi kết thúc $`\mathcal{S} = \lbrace A, B, C, D, E \rbrace`$.
- Xuất phát: Luôn bắt đầu tại trạng thái trung tâm $`C`$.
- Động lực học: Tại mỗi bước, xác suất $50\%$ bước sang trái và $50\%$ bước sang phải.
- Kết thúc:
  - Rơi vào đầu mút bên trái (T0): Nhận thưởng $`R = 0`$, kết thúc episode.
  - Rơi vào đầu mút bên phải (T1): Nhận thưởng $`R = +1`$, kết thúc episode.
- Mọi bước di chuyển trung gian nhận thưởng $`0`$. Hệ số chiết khấu $`\gamma = 1.0`$.
- **Nghiệm giải tích chính xác:** Giá trị thực sự của các trạng thái là xác suất kết thúc ở đầu mút bên phải:
  
$$
V_{\mathrm{true}} = \left[ V(A)=\frac{1}{6}, V(B)=\frac{2}{6}, V(C)=\frac{3}{6}, V(D)=\frac{4}{6}, V(E)=\frac{5}{6} \right] \approx [0.167, 0.333, 0.500, 0.667, 0.833]
$$

---

## Design

### Mã Nguồn Cài Đặt và So Sánh TD(0) vs Monte Carlo Trên Random Walk (Pure Python)

Đoạn mã dưới đây cài đặt hoàn chỉnh môi trường Random Walk 5 trạng thái, thực thi đồng thời thuật toán TD(0) và First-Visit Monte Carlo, tính toán sai số RMSE so với nghiệm giải tích thực tế qua 100 episodes:

```python
import numpy as np

class RandomWalkEnv:
    """Môi trường Random Walk 5 trạng thái: A, B, C, D, E (Sutton & Barto Ví dụ 6.2)."""
    def __init__(self):
        # 0: Terminal Trái (R=0), 1: A, 2: B, 3: C (Start), 4: D, 5: E, 6: Terminal Phải (R=1)
        self.state = 3
        
    def reset(self):
        self.state = 3
        return self.state

    def step(self):
        # 50% bước sang trái (-1), 50% bước sang phải (+1)
        action = -1 if np.random.rand() < 0.5 else 1
        self.state += action
        
        terminated = (self.state == 0 or self.state == 6)
        reward = 1.0 if self.state == 6 else 0.0
        return self.state, reward, terminated


def evaluate_td_vs_mc(n_episodes=100, alpha_td=0.1, alpha_mc=0.02):
    env = RandomWalkEnv()
    
    # Nghiệm giải tích thực tế của 5 trạng thái A, B, C, D, E (index 1 đến 5)
    V_true = np.array([1/6, 2/6, 3/6, 4/6, 5/6])
    
    # Khởi tạo giá trị ban đầu là 0.5 (như trong sách giáo khoa)
    V_td = np.full(7, 0.5)
    V_td[0] = 0.0; V_td[6] = 0.0
    
    V_mc = np.full(7, 0.5)
    V_mc[0] = 0.0; V_mc[6] = 0.0
    
    td_rmse_history = []
    mc_rmse_history = []
    
    for ep in range(1, n_episodes + 1):
        # ----------------------------------------------------------------------
        # 1. Chạy một Episode cho TD(0)
        # ----------------------------------------------------------------------
        s = env.reset()
        while True:
            s_next, r, term = env.step()
            # Cập nhật TD(0) ngay từng bước
            td_target = r + 1.0 * V_td[s_next] * (not term)
            V_td[s] += alpha_td * (td_target - V_td[s])
            s = s_next
            if term:
                break
                
        # ----------------------------------------------------------------------
        # 2. Chạy một Episode cho Monte Carlo (Cập nhật khi hết episode)
        # ----------------------------------------------------------------------
        s = env.reset()
        trajectory = [s]
        final_reward = 0.0
        while True:
            s_next, r, term = env.step()
            if term:
                final_reward = r
                break
            trajectory.append(s_next)
            s = s_next
            
        for visited_s in set(trajectory):
            V_mc[visited_s] += alpha_mc * (final_reward - V_mc[visited_s])
            
        # Ghi nhận sai số RMSE so với nghiệm giải tích trên 5 trạng thái A-E
        rmse_td = np.sqrt(np.mean((V_td[1:6] - V_true) ** 2))
        rmse_mc = np.sqrt(np.mean((V_mc[1:6] - V_true) ** 2))
        td_rmse_history.append(rmse_td)
        mc_rmse_history.append(rmse_mc)

    return V_td[1:6], V_mc[1:6], V_true, td_rmse_history, mc_rmse_history

# ==============================================================================
# THỰC THI SO SÁNH THỰC NGHIỆM
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    V_td, V_mc, V_true, rmse_td_hist, rmse_mc_hist = evaluate_td_vs_mc(n_episodes=100)
    
    print("=== SO SÁNH HIỆU NĂNG TEMPORAL DIFFERENCE TD(0) VS MONTE CARLO ===")
    print("Môi trường: Random Walk 5 Trạng Thái | Số Episodes: 100\n")
    
    states_label = ['A', 'B', 'C', 'D', 'E']
    print(f"{'Trạng thái':<12} | {'Nghiệm đúng (V_true)':<22} | {'Ước lượng TD(0)':<18} | {'Ước lượng Monte Carlo':<22}")
    print("-" * 80)
    for i in range(5):
        print(f"Trạng thái {states_label[i]:<2} | {V_true[i]:>20.4f} | {V_td[i]:>16.4f} | {V_mc[i]:>20.4f}")

    print("-" * 80)
    print("\nDIỄN BIẾN SAI SỐ RMSE THEO EPISODE:")
    print(f"{'Episode':<10} | {'RMSE TD(0)':<15} | {'RMSE Monte Carlo':<18} | {'Vượt trội'}")
    print("-" * 60)
    for checkpoint in [1, 10, 25, 50, 100]:
        e_td = rmse_td_hist[checkpoint - 1]
        e_mc = rmse_mc_hist[checkpoint - 1]
        winner = "TD(0) TỐT HƠN" if e_td < e_mc else "MC TỐT HƠN"
        print(f"{checkpoint:<10} | {e_td:>12.4f}  | {e_mc:>15.4f}  | {winner}")

    print("\n=> Kết luận: TD(0) hội tụ nhanh hơn và đạt sai số thấp hơn Monte Carlo nhờ phương sai thấp.")
```

---

## Nguồn

- [1] Sutton, R. S. (1988). *Learning to predict by the methods of temporal differences*. Machine Learning, 3(1), 9-44.
- [2] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 6: Temporal-Difference Learning, Sections 6.1 & 6.2). MIT Press.
- [3] Schultz, W., Dayan, P., & Montague, P. R. (1997). *A neural substrate of prediction and reward*. Science, 275(5306), 1593-1599.
