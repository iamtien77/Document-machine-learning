# Phương Pháp Monte Carlo Trong Học Tăng Cường (Monte Carlo Methods)

## Purpose

Phương pháp Monte Carlo (MC) trong Học tăng cường là **nhóm giải thuật đầu tiên học tập hoàn toàn không cần mô hình môi trường (Model-Free Learning)**. Tác tử không cần biết trước ma trận xác suất chuyển trạng thái $`\mathcal{P}`$ hay hàm thưởng $`\mathcal{R}`$, mà học trực tiếp từ kinh nghiệm tương tác mẫu (Sample Experience).

Tư tưởng triết học cốt lõi của Monte Carlo rất tự nhiên: **Ước lượng giá trị kỳ vọng của một trạng thái bằng trung bình cộng phần thưởng thực tế thu được từ nhiều lần trải nghiệm hoàn chỉnh (Law of Large Numbers)**.

Đặc tính then chốt của Monte Carlo:
- **Chỉ áp dụng cho bài toán phân hồi (Episodic Tasks):** Toàn bộ ván chơi/thực nghiệm phải kết thúc ở trạng thái Terminal $`S_T`$ thì thuật toán mới bắt đầu tính toán và cập nhật giá trị.
- **Không có tính chất Bootstrapping:** Giá trị của trạng thái $`s_t`$ được cập nhật hoàn toàn dựa trên tổng phần thưởng thực tế thu về đến hết episode ($`G_t`$), không phụ thuộc vào bất kỳ giá trị ước lượng nào của các trạng thái phía sau ($`V(s_{t+1})`$).
- **Không chệch nhưng phương sai lớn (Zero Bias, High Variance):** Ước lượng của Monte Carlo là ước lượng không chệch đối với hàm giá trị thực sự, nhưng dao động phương sai giữa các episode rất mạnh do tích lũy tính ngẫu nhiên dọc theo toàn bộ chuỗi hành động dài.

---

## Input

Các thuật toán Monte Carlo tiếp nhận dữ liệu từ các chuỗi tương tác hoàn chỉnh:

1. **Môi trường mô phỏng dạng hộp đen (Black-box Simulator):** Có khả năng sinh ra các quỹ đạo tương tác hoàn chỉnh từ trạng thái ban đầu tới trạng thái kết thúc.
2. **Tập các quỹ đạo hoàn chỉnh (Collection of Complete Episodes):**

$$
\tau_k = \left( s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T, s_T \right)
$$

3. **Chính sách đánh giá mục tiêu ($`\pi(a \mid s)`$)** và **Chính sách hành vi sinh dữ liệu ($`b(a \mid s)`$)** (trong trường hợp Off-policy).
4. **Hệ số chiết khấu:** $`\gamma \in [0, 1]`$.

---

## Output

Thuật toán Monte Carlo cung cấp:

1. **Hàm giá trị trạng thái ước lượng ($`V(s)`$):** Bảng giá trị trung bình phần thưởng tích lũy cho từng trạng thái.
2. **Hàm giá trị hành động ($`Q(s, a)`$):** Bảng điểm cho từng cặp trạng thái - hành động phục vụ bài toán điều khiển tối ưu (MC Control).
3. **Chính sách tối ưu ($`\pi^{\ast}(s)`$):** Được cải tiến theo nguyên lý tham lam $`\pi(s) = \arg\max_a Q(s, a)`$.

---

## Evaluation metrics

Chỉ số đánh giá độ tin cậy của thuật toán Monte Carlo:

1. **Sai số bình phương trung bình (Mean Squared Error - MSE):**

$$
\mathrm{MSE}(V) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( V(s) - V_{\mathrm{true}}(s) \right)^2
$$

2. **Phương sai của Return ước lượng ($`\mathrm{Var}[G_t]`$):** Đo lường mức độ biến động của phần thưởng dọc theo các quỹ đạo ngẫu nhiên.
3. **Tốc độ hội tụ mẫu:** Theo Định lý Giới hạn Trung tâm, sai số ước lượng giảm theo tỷ lệ $`\mathcal{O}\left( 1 / \sqrt{N} \right)`$ với $`N`$ là số lần trạng thái được ghé thăm.

---

## How it works

### 1. First-Visit MC vs Every-Visit MC

Trong một episode dài, một trạng thái $`s`$ có thể được tác tử ghé thăm nhiều lần:

- **First-Visit Monte Carlo (Khuyến nghị - Không chệch):** Chỉ tính toán và lấy mẫu phần thưởng $`G_t`$ cho **lần đầu tiên** trạng thái $`s`$ xuất hiện trong episode. Các lần xuất hiện sau của $`s`$ trong cùng episode đó bị bỏ qua.
- **Every-Visit Monte Carlo:** Lấy mẫu phần thưởng cho **mọi lần** trạng thái $`s`$ xuất hiện. Đơn giản hơn trong cài đặt nhưng các mẫu trong cùng một episode có tương quan phụ thuộc lẫn nhau.

---

### 2. Công Thức Cập Nhật Gia Tăng (Incremental Update)

Thay vì phải lưu toàn bộ danh sách phần thưởng của từng trạng thái rồi chia trung bình, ta cập nhật trung bình động trực tiếp sau mỗi episode:

$$
V(S_t) \leftarrow V(S_t) + \alpha \left( G_t - V(S_t) \right)
$$

Trong đó:
- $`G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}`$ là tổng phần thưởng thực tế thu được từ thời điểm $`t`$ đến hết episode.
- $`G_t - V(S_t)`$ là sai số ước lượng (Error).
- $`\alpha = \frac{1}{N(S_t)}`$ (cho trung bình cộng chính xác) hoặc $`\alpha`$ cố định (cho bài toán môi trường biến đổi phi dừng).

---

### 3. Điều Khiển Monte Carlo và Thăm Dò Khởi Tạo (Exploring Starts)

Để tìm chính sách tối ưu bằng Monte Carlo, ta phải ước lượng $`Q(s, a)`$ thay vì $`V(s)`$. Tuy nhiên, nếu chính sách là tất định, tác tử sẽ chỉ thử một hành động duy nhất và không bao giờ biết được giá trị của các hành động khác.

Hai giải pháp kinh điển để đảm bảo thăm dò:
1. **Khởi tạo thăm dò (Exploring Starts - MC-ES):** Bắt buộc mọi cặp $`(s, a)`$ đều có xác suất khác không được chọn làm điểm xuất phát của episode.
2. **Chính sách mềm $`\epsilon`$-Greedy (On-Policy MC Control):** Luôn duy trì xác suất thăm dò $`\epsilon`$ ngẫu nhiên trong suốt quá trình chơi.

---

### 4. Off-Policy Monte Carlo và Lấy Mẫu Tầm Quan Trọng (Importance Sampling)

Học chính sách mục tiêu $`\pi(a \mid s)`$ trong khi thu thập dữ liệu bằng một chính sách hành vi linh hoạt khác $`b(a \mid s)`$. Tỷ số trọng số tầm quan trọng (Importance Sampling Ratio) hiệu chỉnh sự khác biệt phân phối:

$$
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}
$$

Công thức cập nhật có trọng số (Weighted Importance Sampling - Giảm phương sai):

$$
V(s) = \frac{\sum_{k \in \mathcal{T}(s)} \rho_{t(k):T(k)-1} \cdot G_{t(k)}}{\sum_{k \in \mathcal{T}(s)} \rho_{t(k):T(k)-1}}
$$

---

## Data used

Môi trường kiểm chuẩn kinh điển: **Trò chơi Xì dách (Blackjack / 21 Điểm - Sutton & Barto Ví dụ 5.1)**:

| Thành phần MDP | Biểu diễn toán học | Diễn giải chi tiết |
| :--- | :--- | :--- |
| **Không gian trạng thái $`\mathcal{S}`$** | Bộ 3 thành phần: $`(s_1, s_2, s_3)`$ | - $`s_1 \in [12, 21]`$: Tổng điểm hiện tại của người chơi.<br>- $`s_2 \in [1, 10]`$: Điểm lá bài ngửa của nhà cái (1 là Át).<br>- $`s_3 \in \lbrace \text{True}, \text{False} \rbrace`$: Người chơi có lá Át dùng được (Usable Ace = 11 điểm mà không quắc) hay không. |
| **Không gian hành động $`\mathcal{A}`$** | Discrete(2): $`\lbrace 0, 1 \rbrace`$ | - 0: Dừng bài (Stick).<br>- 1: Rút thêm bài (Hit). |
| **Phần thưởng $`R`$** | Điểm số vô hướng cuối ván | $+1.0$ (Thắng), $-1.0$ (Thua / Quắc), $0.0$ (Hòa). Mọi bước rút trung gian nhận thưởng $0$. |
| **Hệ số chiết khấu $`\gamma`$** | $1.0$ | Trò chơi có độ dài hữu hạn ($2 \to 5$ bước), không chiết khấu điểm số. |

---

## Design

### Mã Nguồn Cài Đặt First-Visit Monte Carlo Policy Evaluation (Pure Python)

Đoạn mã dưới đây mô phỏng trò chơi Xì dách tối giản và sử dụng thuật toán First-Visit Monte Carlo để đánh giá giá trị trạng thái của người chơi theo một chính sách cố định (chính sách: Rút tiếp nếu điểm $< 20$, dừng bài nếu điểm $\ge 20$):

```python
import numpy as np

class SimpleBlackjack:
    """Môi trường Blackjack tối giản hóa chuẩn lý thuyết Sutton & Barto."""
    def __init__(self):
        # Bộ bài vô hạn: lá bài từ 1 đến 10 (các lá 10, J, Q, K đều tính là 10)
        self.cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def draw_card(self):
        return np.random.choice(self.cards)

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def score(self, hand):
        """Tính điểm bàn tay có xét lá Át (1 hoặc 11)."""
        total = sum(hand)
        usable_ace = (1 in hand) and (total + 10 <= 21)
        if usable_ace:
            total += 10
        return total, usable_ace

    def play_episode(self, policy_fn):
        """Tạo một ván chơi hoàn chỉnh theo chính sách policy_fn."""
        player_hand = self.draw_hand()
        dealer_hand = self.draw_hand()

        trajectory = [] # Lưu trữ các bộ (state, action, reward)

        # Lượt của Người chơi
        while True:
            p_score, usable_ace = self.score(player_hand)
            if p_score > 21: # Người chơi bị quắc
                break

            state = (p_score, dealer_hand[0], usable_ace)
            action = policy_fn(state) # 0: Stick, 1: Hit
            trajectory.append((state, action))

            if action == 1: # Rút bài
                player_hand.append(self.draw_card())
            else: # Dừng bài
                break

        p_score, _ = self.score(player_hand)

        # Nếu người chơi bị quắc -> Thua ngay lập tức
        if p_score > 21:
            reward = -1.0
        else:
            # Lượt của Nhà cái: Nhà cái bắt buộc rút đến khi điểm >= 17
            while True:
                d_score, _ = self.score(dealer_hand)
                if d_score >= 17:
                    break
                dealer_hand.append(self.draw_card())

            d_score, _ = self.score(dealer_hand)
            if d_score > 21 or p_score > d_score:
                reward = 1.0  # Thắng
            elif p_score < d_score:
                reward = -1.0 # Thua
            else:
                reward = 0.0  # Hòa

        return trajectory, reward


def first_visit_monte_carlo(n_episodes=50000):
    """Thuật toán First-Visit Monte Carlo Policy Evaluation."""
    game = SimpleBlackjack()

    # Chính sách cố định: Dừng nếu điểm >= 20, ngược lại rút tiếp
    def policy(state):
        player_score, _, _ = state
        return 0 if player_score >= 20 else 1

    V = {} # Bảng giá trị V(s)
    returns = {} # Lưu tổng return và số lần xuất hiện

    for ep in range(1, n_episodes + 1):
        trajectory, reward = game.play_episode(policy)

        visited_in_episode = set()
        for state, action in trajectory:
            if state not in visited_in_episode:
                visited_in_episode.add(state)
                if state not in returns:
                    returns[state] = [0.0, 0] # [tổng return, số lượt đếm]
                returns[state][0] += reward
                returns[state][1] += 1
                V[state] = returns[state][0] / returns[state][1]

    return V

# ==============================================================================
# CHẠY ĐÁNH GIÁ FIRST-VISIT MONTE CARLO
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    print("=== HUẤN LUYỆN ĐÁNH GIÁ GIÁ TRỊ FIRST-VISIT MONTE CARLO (BLACKJACK) ===")
    print("Mô phỏng 50,000 ván chơi để xấp xỉ kỳ vọng toán học...\n")

    V_table = first_visit_monte_carlo(n_episodes=50000)

    # Hiển thị mẫu giá trị ước lượng cho các trạng thái người chơi có 20 hoặc 21 điểm
    print(f"{'Điểm người chơi':<18} | {'Lá ngửa nhà cái':<18} | {'Có Át (Usable Ace)':<20} | {'Giá trị ước lượng V(s)':<22}")
    print("-" * 85)

    sample_queries = [
        (21, 10, True), (21, 2, True),
        (20, 10, False), (20, 5, False),
        (19, 10, False), (18, 10, False)
    ]
    for q in sample_queries:
        val = V_table.get(q, 0.0)
        print(f"{q[0]:<18} | {q[1]:<18} | {str(q[2]):<20} | {val:>+18.4f}")

    print("-" * 85)
    print("=> Khi điểm người chơi là 21, giá trị V(s) dương lớn (~ +0.8 đến +0.9).")
    print("=> Khi điểm người chơi là 18 mà buộc phải rút tiếp (theo chính sách dở), V(s) âm nặng (~ -0.7).")
    print("=> Monte Carlo ước lượng chính xác giá trị trạng thái mà không cần biết phân phối xác suất xáo bài.")
```

---

## Nguồn

- [1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 5: Monte Carlo Methods). MIT Press.
- [2] Metropolis, N., & Ulam, S. (1949). *The monte carlo method*. Journal of the American Statistical Association, 44(247), 335-341.
- [3] Singh, S. P., & Sutton, R. S. (1996). *Reinforcement learning with replacing eligibility traces*. Machine Learning, 22(1-3), 123-158.
