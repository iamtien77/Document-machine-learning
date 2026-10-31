# Dữ Liệu Trước Huấn Luyện: Không Gian Trạng Thái, Hành Động và Kỹ Thuật Thiết Kế Phần Thưởng

## Purpose

Trong Học tăng cường, "dữ liệu trước huấn luyện" không phải là một tập dữ liệu tĩnh (như bảng CSV hay kho ảnh) mà là **bản thiết kế toán học của môi trường tương tác (Environment Design)**. Sự thành bại của một dự án RL phần lớn được quyết định trước khi thuật toán bắt đầu chạy, phụ thuộc vào:
1. **Mô hình hóa không gian trạng thái ($`\mathcal{S}`$) và không gian hành động ($`\mathcal{A}`$):** Trạng thái phải cung cấp đủ thông tin để thỏa mãn tính chất Markov, nhưng không được quá cồng kềnh gây bùng nổ số chiều.
2. **Kỹ thuật thiết kế phần thưởng (Reward Engineering & Shaping):** Phần thưởng là cơ chế duy nhất để con người truyền đạt mục tiêu cho Agent. Thiết kế sai hàm thưởng sẽ dẫn đến hiện tượng **Reward Hacking (Lách luật phần thưởng)** — Agent tìm ra hành vi quái đản để tích lũy điểm số cực đại mà không hoàn thành nhiệm vụ thực tế.
3. **Định hình phần thưởng dựa trên thế năng (Potential-Based Reward Shaping - PBRS):** Giải quyết bài toán phần thưởng thưa thớt (Sparse Rewards) mà vẫn bảo toàn toán học chính sách tối ưu ban đầu.

---

## Input

Quá trình chuẩn bị bài toán RL tiếp nhận các đặc tả nghiệp vụ:

1. **Các biến số quan sát vật lý / nghiệp vụ:**
   - Các tín hiệu cảm biến thô (Sensors): Vị trí, góc xoay, vận tốc góc, gia tốc kế, hình ảnh camera, lịch sử giá cổ phiếu, lưu lượng truy cập mạng.
   - Ranh giới giá trị vật lý khả dĩ: Tối thiểu ($`\min`$) và tối đa ($`\max`$) của từng kênh đo.

2. **Cơ cấu chấp hành và giới hạn điều khiển:**
   - Các lệnh điều khiển khả thi: Bật/tắt động cơ, lực kéo, góc lái, quyết định Đặt lệnh / Giữ tiền mặt.
   - Bước thời gian mô phỏng rời rạc ($`\Delta t`$).

3. **Mục tiêu tối thượng của bài toán (Task Objective):**
   - Tiêu chí thành công: Đến đích an toàn, đánh bại đối thủ, đạt lợi nhuận tối đa.
   - Tiêu chuẩn an toàn: Không va chạm, không vượt ngưỡng nhiệt độ, không chịu sụt giảm vốn quá mức cho phép.

---

## Output

Giai đoạn thiết kế trước huấn luyện bàn giao một môi trường chuẩn hóa:

1. **Không gian quan sát chuẩn hóa (Normalized Observation Space $`\mathcal{S}`$):**
   Được đóng gói theo chuẩn `gymnasium.spaces.Box` hoặc `Discrete`, với các biến số được chuẩn hóa về dải $`[-1, 1]`$ hoặc $`[0, 1]`$.

2. **Không gian hành động chuẩn hóa (Normalized Action Space $`\mathcal{A}`$):**
   Định rõ kiểu dữ liệu rời rạc (`Discrete(n)`) hoặc liên tục (`Box(low, high)`).

3. **Hàm phần thưởng toán học (Mathematical Reward Function $`R(s, a, s')`$):**
   Công thức tính điểm thưởng cân bằng giữa phần thưởng mục tiêu chính và các thành phần phụ trợ (thế năng dẫn đường, phạt chi phí năng lượng, phạt vi phạm an toàn).

4. **Bộ tiêu chí kết thúc Episode (Termination & Truncation Logic):**
   - `terminated`: Agent đạt đích thành công hoặc rơi vào trạng thái chết/thất bại không thể đảo ngược.
   - `truncated`: Hết giới hạn thời gian quy định (Time Limit Exceeded).

---

## Evaluation metrics

Chỉ số thẩm định chất lượng thiết kế môi trường trước khi đưa vào huấn luyện sâu:

1. **Mật độ phần thưởng (Reward Density Ratio):**
   Tỷ lệ phần trăm các bước chuyển trạng thái có phần thưởng khác không:

$$
\text{Density} = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(r_t \neq 0) \times 100
$$

   - Sparse Reward: Mật độ < 1% (rất khó học nếu không có kỹ thuật hỗ trợ).
   - Dense Reward: Mật độ > 80% (Agent nhận phản hồi liên tục).

2. **Tương quan giữa Phần thưởng tích lũy và Tiến độ thực tế (Alignment Correlation):**
   Hệ số tương quan Pearson giữa tổng điểm thưởng $`G`$ và thước đo hoàn thành nghiệp vụ độc lập $`M_{\mathrm{real}}`$. Phải đạt $`\rho \ge 0.85`$.

3. **Tỷ lệ vi phạm an toàn (Safety Violation Rate):**
   Tần suất Agent chạm vào vùng cấm trong các bước chạy ngẫu nhiên thăm dò.

---

## How it works

### 1. Phân Biệt Không Gian Quan Sát (Observation vs State)

- **Trạng thái hoàn chỉnh (State $`s_t`$):** Mô tả đầy đủ toàn bộ thế giới vật lý. Thỏa mãn tính chất Markov: $`P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, \dots, s_0)`$.
- **Quan sát một phần (Observation $`o_t`$):** Trong thực tế, Agent chỉ nhận được thông tin cục bộ qua camera hoặc cảm biến giới hạn (Partially Observable MDP - POMDP).
- **Kỹ thuật Ghép khung hình (Frame Stacking):** Một ảnh tĩnh đơn lẻ không chứa thông tin về vận tốc và gia tốc. Bằng cách ghép $`m = 4`$ khung hình liên tiếp $`o_t = [I_t, I_{t-1}, I_{t-2}, I_{t-3}]`$, Agent có thể suy diễn được hướng bay và tốc độ của vật thể.

---

### 2. Kỹ Thuật Định Hình Phần Thưởng Bảo Toàn Chính Sách (PBRS)

Việc tùy tiện cộng thêm điểm thưởng phụ (ví dụ: thưởng Agent tiến lại gần đích) thường dẫn đến hiện tượng **Reward Hacking** (Agent chạy tới rồi chạy lui liên tục để tích điểm khoảng cách mà không bao giờ chạm vạch đích).

Định lý kinh điển của **Ng, Harada & Russell (ICML 1999)** chứng minh rằng: Một hàm thưởng bổ sung $`F(s, a, s')`$ được đảm bảo toán học **KHÔNG làm thay đổi tập chính sách tối ưu $`\pi^{\ast}`$** nếu và chỉ nếu nó có dạng sai phân thế năng:

$$
F(s, a, s') = \gamma \Phi(s') - \Phi(s)
$$

Trong đó:
- $`\Phi(s)`$ là hàm thế năng bất kỳ ước lượng khoảng cách từ trạng thái $`s`$ tới mục tiêu.
- $`\gamma`$ là hệ số chiết khấu của bài toán.

Hàm thưởng tổng hợp đưa vào huấn luyện:

$$
R_{\mathrm{shaped}}(s, a, s') = R_{\mathrm{original}}(s, a, s') + \left( \gamma \Phi(s') - \Phi(s) \right)
$$

**Ý nghĩa bảo toàn:** Tổng phần thưởng phụ trợ dọc theo một quỹ đạo từ $`s_0`$ đến $`s_T`$ tạo thành một chuỗi triệt tiêu (Telescoping Sum):

$$
\sum_{t=0}^{T-1} \gamma^t F(s_t, a_t, s_{t+1}) = \gamma^T \Phi(s_T) - \Phi(s_0)
$$

Tổng này hoàn toàn độc lập với con đường đi qua, triệt tiêu hoàn toàn động cơ đi đường vòng để kiếm thêm điểm.

---

### 3. Các Bẫy Thiết Kế Hàm Thưởng Điển Hình (Reward Hacking Traps)

| Bẫy thiết kế | Hành vi lỗi của Agent (Reward Exploitation) | Giải pháp khắc phục chuẩn mực |
| :--- | :--- | :--- |
| **Thưởng sống sót qua từng bước** | Đứng im một chỗ, không chịu mạo hiểm di chuyển tới đích để tránh rủi ro. | Đổi thành phạt chi phí thời gian ($-1$ mỗi bước) để ép tìm đường ngắn nhất. |
| **Thưởng khoảng cách Euclid tới đích** | Chạy dao động tiến-lùi quanh đích để tích lũy điểm thưởng nhiều lần. | Áp dụng Potential-Based Reward Shaping $`\gamma \Phi(s') - \Phi(s)`$. |
| **Đua thuyền nhặt vật phẩm (Coast Runners)** | Xoay vòng tròn đốt thuyền để nhặt vật phẩm hồi sinh liên tục thay vì về đích. | Chỉ thưởng điểm vật phẩm khi hoàn thành chặng đua tương ứng. |
| **Phạt va chạm quá nhỏ** | Chấp nhận đâm xe liên tục nếu tốc độ di chuyển vẫn mang lại điểm dương bù đắp. | Đặt ngưỡng phạt va chạm đủ lớn kèm kết thúc episode ngay lập tức (`terminated`). |

---

## Data used

Bảng đặc tả cấu trúc không gian của các môi trường kinh điển trong nghiên cứu và công nghiệp:

| Tên bài toán | Không gian trạng thái $\mathcal{S}$ | Không gian hành động $\mathcal{A}$ | Bản chất phần thưởng gốc |
| :--- | :--- | :--- | :--- |
| **CartPole-v1** | $\mathbb{R}^4$: $[x, \dot{x}, \theta, \dot{\theta}]$ | Discrete(2): $\{$Đẩy trái, Đẩy phải$\}$ | $+1.0$ mỗi bước con lắc không đổ |
| **MountainCar-v0** | $\mathbb{R}^2$: $[\text{vị trí } x, \text{vận tốc } v]$ | Discrete(3): $\{$Ga trái, Không ga, Ga phải$\}$ | $-1.0$ mỗi bước cho tới khi chạm cờ trên đỉnh |
| **Acrobot-v1** | $\mathbb{R}^6$: Sin/Cos của 2 khớp xoay | Discrete(3): Mô-men xoắn $\{-1, 0, +1\}$ | $-1.0$ mỗi bước cho tới khi nâng chân quá vạch |
| **BipedalWalker-v2** | $\mathbb{R}^{24}$: Góc khớp, vận tốc, tiếp xúc đất | $\mathbb{R}^4$: Mô-men xoắn liên tục 4 khớp | Thưởng quãng đường tiến, phạt tiêu hao năng lượng |
| **Giao dịch Tài chính**| $\mathbb{R}^{30}$: Lịch sử nến, RSI, MACD, vị thế | Discrete(3): $\{$Mua, Bán, Giữ$\}$ | Tỷ suất sinh lời PnL hoặc hệ số Sharpe gia tăng |

---

## Design

### Mã Nguồn Môi Trường Tùy Chỉnh Chuẩn Gymnasium (Custom Environment with PBRS)

Đoạn mã Python dưới đây xây dựng một môi trường tìm đường liên tục 1D có tích hợp cơ chế Potential-Based Reward Shaping, kiểm soát chặt chẽ không gian trạng thái, hành động và triệt tiêu nguy cơ reward hacking:

```python
import numpy as np

class ContinuousNavigationEnv:
    """Môi trường tìm đường 1D liên tục:
    - Trạng thái: x in [-1.0, 1.0] (Vị trí tác tử)
    - Hành động: a in [-0.2, 0.2] (Độ dịch chuyển liên tục)
    - Đích: x_goal = 0.8
    - Sử dụng Potential-Based Reward Shaping để tăng tốc học an toàn.
    """
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.goal_pos = 0.8
        self.max_steps = 100
        self.current_step = 0
        self.state = 0.0

    def potential(self, s):
        """Hàm thế năng Phi(s): Càng gần đích thế năng càng cao."""
        distance = abs(s - self.goal_pos)
        return -distance * 10.0

    def reset(self):
        self.state = -0.8 # Bắt đầu ở bên trái
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # Cắt ngưỡng hành động vật lý an toàn
        action = np.clip(action, -0.2, 0.2)
        old_state = self.state

        # Cập nhật trạng thái vật lý
        self.state = np.clip(self.state + action, -1.0, 1.0)

        # Kiểm tra điều kiện đích
        terminated = abs(self.state - self.goal_pos) < 0.05
        truncated = self.current_step >= self.max_steps

        # 1. Phần thưởng gốc: +100 khi chạm đích, -1 mỗi bước chi phí thời gian
        r_base = 100.0 if terminated else -1.0

        # 2. Thành phần Potential-Based Reward Shaping: gamma * Phi(s') - Phi(s)
        phi_old = self.potential(old_state)
        phi_new = self.potential(self.state) if not terminated else 0.0
        f_shaping = self.gamma * phi_new - phi_old

        # Phần thưởng tổng hợp an toàn
        total_reward = r_base + f_shaping

        return self.state, total_reward, terminated, truncated, {"base_reward": r_base, "shaping": f_shaping}

# ==============================================================================
# KIỂM THỬ ĐỘNG THÁI HÀM THƯỞNG VÀ BẢO TOÀN THẾ NĂNG
# ==============================================================================
if __name__ == "__main__":
    env = ContinuousNavigationEnv(gamma=0.99)
    s, _ = env.reset()

    print("=== KIỂM THỬ THIẾT KẾ MÔI TRƯỜNG VÀ POTENTIAL REWARD SHAPING ===")
    print(f"Vị trí bắt đầu: {s:.2f} | Mục tiêu đích: {env.goal_pos:.2f}\n")

    # Kịch bản 1: Đi đúng hướng về phía đích
    print("--- Kịch bản 1: Đi đúng hướng (Action = +0.2) ---")
    s, r, term, trunc, info = env.step(action=0.2)
    print(f"Trạng thái mới: {s:.2f} | Base Reward: {info['base_reward']:.2f} | Shaping: {info['shaping']:>+6.2f} | Total Reward: {r:>+6.2f}")

    # Kịch bản 2: Đi sai hướng lùi xa đích
    print("\n--- Kịch bản 2: Đi sai hướng (Action = -0.2) ---")
    s, r, term, trunc, info = env.step(action=-0.2)
    print(f"Trạng thái mới: {s:.2f} | Base Reward: {info['base_reward']:.2f} | Shaping: {info['shaping']:>+6.2f} | Total Reward: {r:>+6.2f}")

    print("\n=> Kết luận: PBRS cung cấp gradient phần thưởng dẫn đường tức thì mà không thể bị hack lặp lại.")
```

---

## Nguồn

- [1] Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping*. Proceedings of the Sixteenth International Conference on Machine Learning (ICML), 278-287.
- [2] Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). *Concrete problems in AI safety*. arXiv preprint arXiv:1606.06565.
- [3] Towers, M., et al. (2023). *Gymnasium: A standard interface for reinforcement learning environments*. arXiv preprint arXiv:2407.17032.

