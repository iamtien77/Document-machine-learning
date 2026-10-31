# Các Cải Tiến Của Deep Q-Network: Double DQN, Dueling DQN, PER và Rainbow

Tài liệu này hệ thống hóa các cải tiến đột phá trên nền tảng Deep Q-Network (DQN) nhằm khắc phục các hạn chế cốt tử của thuật toán Vanilla DQN: hiện tượng thiên lệch ước lượng quá mức (Overestimation Bias), sự kết hợp lỏng lẻo giữa giá trị trạng thái và hành động, sự lãng phí tài nguyên khi lấy mẫu kinh nghiệm đồng đều, và độ trễ lan truyền tín hiệu phần thưởng đơn bước.

---

## Purpose

1. **Khắc phục hiện tượng Overestimation Bias:** Thuật toán Q-Learning và Vanilla DQN sử dụng toán tử $\max$ đồng thời cho việc lựa chọn hành động và ước lượng giá trị hành động, dẫn đến xu hướng thổi phồng giá trị $`Q(s, a)`$ một cách có hệ thống. Double DQN tách rời hai nhiệm vụ này giữa mạng chính (Online Network) và mạng mục tiêu (Target Network).
2. **Phân tách giá trị trạng thái và giá trị lợi thế (Advantage):** Dueling DQN tái cấu trúc mạng nơ-ron thành hai nhánh tính toán riêng biệt: hàm giá trị trạng thái tổng quát $`V(s)`$ và hàm lợi thế tương đối $`A(s, a)`$, giúp tăng tốc độ hội tụ ngay cả khi nhiều hành động không làm thay đổi đáng kể môi trường.
3. **Tối ưu hóa hiệu quả lấy mẫu chuyển tiếp (Sample Efficiency):** Prioritized Experience Replay (PER) thay thế cơ chế lấy mẫu ngẫu nhiên đồng đều bằng cơ chế lấy mẫu ưu tiên dựa trên độ lớn sai số Temporal Difference ($`|\delta_t|`$), tập trung tài nguyên tính toán vào những trạng thái mà mạng dự đoán sai lệch nhiều nhất.
4. **Hợp nhất sức mạnh toàn diện (Rainbow DQN):** Tích hợp 6 trụ cột cải tiến độc lập (Double DQN, Dueling DQN, PER, Multi-step Bootstrap, Distributional RL, Noisy Networks) tạo nên chuẩn mực hiệu năng tối ưu trên không gian hành động rời rạc.

---

## Input

Quá trình huấn luyện các biến thể cải tiến của DQN tiếp nhận các luồng dữ liệu và tham số:

1. **Trạng thái môi trường quan sát được ($`s_t \in \mathcal{S}`$):** Vector đặc trưng liên tục $`\mathbb{R}^d`$ hoặc khung hình pixel tiền xử lý $`\mathbb{R}^{C \times H \times W}`$.
2. **Bộ chuyển tiếp kinh nghiệm ($`e_t = (s_t, a_t, r_t, s_{t+1}, d_t)`$):** Tuple 5 phần tử lưu trong hàng đợi bộ nhớ có dung lượng cố định $`N`$.
3. **Độ ưu tiên của mẫu kinh nghiệm ($`p_i`$):** Giá trị độ lớn sai số TD gắn liền với mỗi chuyển tiếp trong Replay Buffer.
4. **Bộ siêu tham số điều khiển thuật toán:**
   - Hệ số chiết khấu tương lai: $`\gamma \in [0.95, 0.999]`$.
   - Tốc độ học (Learning rate): $`\alpha_{\mathrm{opt}} \in [10^{-4}, 10^{-3}]`$.
   - Tỷ trọng ưu tiên lấy mẫu (Priority Exponent): $`\alpha \in [0.4, 0.7]`$ (với $`\alpha = 0`$ tương đương lấy mẫu đồng đều).
   - Hệ số bù trừ thiên lệch Importance Sampling: $`\beta_0 \in [0.4, 0.6]`$, tăng dần tuyến tính lên $`1.0`$ khi kết thúc huấn luyện.
   - Hằng số dương nhỏ tránh xác suất bằng 0: $`\epsilon_{\mathrm{per}} = 10^{-6}`$.
   - Chu kỳ cập nhật trọng số mạng mục tiêu: $`C_{\mathrm{target}} \in [500, 10000]`$ bước hoặc hệ số Polyak $`\tau \in [0.001, 0.01]`$.

---

## Output

1. **Bộ trọng số tối ưu của mạng giá trị ($`\theta^*`$):** Tham số hóa xấp xỉ hàm $`Q^*(s, a)`$ đạt chuẩn không thiên lệch hoặc phân tách Dueling.
2. **Chính sách suy luận tối ưu (Deterministic Greedy Policy):**

$$
\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q(s, a; \theta^*)
$$

3. **Cấu trúc cây chỉ mục SumTree:** Phân cấp cây nhị phân lưu trữ tổng phân phối ưu tiên với độ phức tạp truy vấn $`\mathcal{O}(\log N)`$.
4. **Hàm lợi thế chuẩn hóa $`A(s, a)`$ và hàm giá trị trạng thái $`V(s)`$:** Phản ánh trực quan mức độ an toàn của trạng thái và tiềm năng tương đối của từng hành động.

---

## Evaluation metrics

1. **Phần thưởng tích lũy trung bình (Mean Cumulative Return):**

$$
\bar{R} = \frac{1}{K} \sum_{k=1}^K \sum_{t=0}^{T_k} r_{t}^{(k)}
$$

2. **Mức độ thiên lệch ước lượng (Estimation Bias):** Đo lường chênh lệch giữa giá trị $`Q`$ dự đoán và giá trị thực nghiệm Monte Carlo chiết khấu thu được:

$$
\mathrm{Bias}(Q) = \mathbb{E}\left[ Q(s, a; \theta) - G_t \right]
$$

   *(Vanilla DQN thường có $`\mathrm{Bias}(Q) \gg 0`$, trong khi Double DQN duy trì mức dao động quanh $`0`$)*.
3. **Sai số Temporal Difference trung bình (Mean Absolute TD-Error):**

$$
\overline{|\delta|} = \frac{1}{B} \sum_{i=1}^B |\delta_i|
$$

4. **Tỷ lệ suy hao độ ưu tiên (Priority Loss Variance):** Độ ổn định của việc phân phối mẫu trong cây SumTree qua các chu kỳ cập nhật.

---

## How it works

### 1. Double DQN (DDQN): Hóa giải Maximization Bias

Trong Vanilla DQN, mục tiêu TD được tính bằng:

$$
Y_t^{\mathrm{DQN}} = R_{t+1} + \gamma \max_{a} Q\left( S_{t+1}, a; \theta_t^- \right) = R_{t+1} + \gamma Q\left( S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t^-); \theta_t^- \right)
$$

Toán tử $`\max`$ áp dụng trên các giá trị ước lượng nhiễu dẫn đến bất đẳng thức Jensen: $`\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])`$. Do đó, Vanilla DQN liên tục phóng đại giá trị kỳ vọng của các hành động phụ phụ thuộc vào biến động ngẫu nhiên.

Double DQN giải quyết triệt để vấn đề này bằng cơ chế phân rã quyết định:
- Dùng **mạng chính** ($`\theta_t`$) để **chọn hành động tốt nhất**: $`a^* = \arg\max_a Q(S_{t+1}, a; \theta_t)`$.
- Dùng **mạng mục tiêu** ($`\theta_t^-`$) để **đo lường giá trị của hành động đó**: $`Q(S_{t+1}, a^*; \theta_t^-)`$.

$$
Y_t^{\mathrm{DoubleQ}} = R_{t+1} + \gamma Q\left( S_{t+1}, \arg\max_{a} Q(S_{t+1}, a; \theta_t); \theta_t^- \right)
$$

```
Vanilla DQN:
  Target = R + gamma * max_a Q_target(S', a)
                       [Cùng mạng chọn và đánh giá -> Overestimation]

Double DQN:
  Action* = argmax_a Q_online(S', a)   [Mạng Online chọn hành động]
  Target  = R + gamma * Q_target(S', Action*) [Mạng Target đánh giá giá trị]
```

---

### 2. Dueling DQN: Phân Tách Value Stream và Advantage Stream

Nhiều trạng thái trong thực tế có giá trị trạng thái $`V(s)`$ cao hoặc thấp độc lập với hành động được chọn (ví dụ: chiếc xe đang chạy trên đường thẳng an toàn hay chuẩn bị va vào tường). Việc phải cập nhật giá trị cho toàn bộ các cặp $`(s, a)`$ rời rạc làm chậm quá trình học.

Dueling DQN chia tách tầng ẩn cuối cùng thành 2 dòng tính toán:
1. Nhánh giá trị trạng thái scalar: $`V(s; \theta, \beta) \in \mathbb{R}^1`$.
2. Nhánh giá trị lợi thế vector: $`A(s, a; \theta, \alpha) \in \mathbb{R}^{|\mathcal{A}|}`$.

Nếu chỉ cộng trực tiếp $`Q(s, a) = V(s) + A(s, a)`$, hệ thống gặp vấn đề bất khả định danh (Unidentifiable): vô số cặp $`(V, A)`$ khác nhau cùng tạo ra một giá trị $`Q`$ giống nhau.

Để buộc $`V(s)`$ thực sự phản ánh giá trị trung bình của trạng thái, Dueling DQN áp dụng toán tử trừ trung bình cộng giá trị lợi thế:

$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)
$$

Ưu điểm toán học của phép trừ trung bình thay vì toán tử $\max$: Đảm bảo tính khả vi ổn định liên tục trên toàn bộ không gian tham số và bảo toàn thứ tự ưu tiên của các hành động.

---

### 3. Prioritized Experience Replay (PER): Lấy Mẫu Theo Trọng Số Sai Số TD

Vanilla DQN lấy mẫu ngẫu nhiên đồng đều với xác suất $`P(i) = \frac{1}{N}`$. Tuy nhiên, những chuyển tiếp mang nhiều thông tin mới hoặc đột phá thường xuất hiện ít, khiến mạng học rất chậm.

PER thiết lập xác suất lấy mẫu tỉ lệ thuận với độ lớn sai số TD:

$$
P(i) = \frac{p_i^\alpha}{\sum_{k=1}^N p_k^\alpha}
$$

Trong đó độ ưu tiên được tính từ sai số TD tuyệt đối cộng hằng số làm mịn:

$$
p_i = |\delta_i| + \epsilon_{\mathrm{per}}, \quad \text{với } \delta_i = Y_i - Q(s_i, a_i; \theta)
$$

Tham số $`\alpha \in [0, 1]`$ điều tiết mức độ ưu tiên: $`\alpha = 0`$ trở về phân phối đồng đều, $`\alpha = 1`$ là lấy mẫu tỷ lệ thuận thuần túy.

#### Bù Trừ Thiên Lệch Bằng Importance Sampling (IS)

Việc lấy mẫu ưu tiên làm thay đổi kỳ vọng phân phối dữ liệu huấn luyện, khiến gradient bị lệch và có thể gây phân kỳ. PER khắc phục bằng cách nhân thêm trọng số Importance Sampling $`w_i`$ vào hàm tổn thất:

$$
w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta
$$

Để ổn định gradient, trọng số được chuẩn hóa theo giá trị cực đại trong batch:

$$
\tilde{w}_i = \frac{w_i}{\max_{j} w_j} = \left( \frac{P(i)}{\min_j P(j)} \right)^{-\beta}
$$

Tham số $`\beta`$ bắt đầu từ $`\beta_0 \approx 0.4`$ và tăng tuyến tính đến $`1.0`$ ở giai đoạn cuối huấn luyện.

Hàm tổn thất có trọng số IS:

$$
\mathcal{L}_{\mathrm{PER}}(\theta) = \frac{1}{B} \sum_{i=1}^B \tilde{w}_i \cdot \mathcal{L}_{\mathrm{Huber}}\left( Y_i - Q(s_i, a_i; \theta) \right)
$$

---

### 4. Tổng Hợp Rainbow DQN: 6 Thành Tố Tích Hợp

Rainbow DQN kết hợp 6 cải tiến bổ trợ lẫn nhau:
1. **Double Q-Learning:** Chống thổi phồng giá trị $`Q`$.
2. **Prioritized Replay:** Tăng tần suất học từ các chuyển tiếp có TD-error lớn.
3. **Dueling Networks:** Tách biệt biểu diễn môi trường và hành động.
4. **Multi-step Learning (n-step TD):** Thay thế mục tiêu 1 bước bằng tổng phần thưởng chiết khấu $`n`$ bước:
   
$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1}
$$

5. **Distributional RL (C51):** Thay vì ước lượng giá trị kỳ vọng vô hướng $`Q(s, a)`$, mạng học toàn bộ phân phối xác suất rời rạc của phần thưởng tương lai $`Z(s, a)`$ trên 51 mốc giá trị (atoms).
6. **Noisy Networks for Exploration:** Thay thế cơ chế $`\epsilon`$-greedy ngẫu nhiên thô sơ bằng việc chèn trực tiếp các trọng số tham số hóa ngẫu nhiên $`\mathbf{w} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}`$ vào các lớp tuyến tính của mạng nơ-ron.

---

## Data used

1. **Bộ đệm Replay Buffer dạng SumTree:** Cấu trúc cây phân cấp nhị phân hoàn chỉnh. Các nút lá lưu độ ưu tiên $`p_i^\alpha`$, các nút cha lưu tổng giá trị ưu tiên của hai nút con. Thao tác lấy mẫu prefix sum đạt $`\mathcal{O}(\log N)`$, thao tác cập nhật độ ưu tiên đạt $`\mathcal{O}(\log N)`$.
2. **Batch chuyển tiếp đa chiều:** Mảng tensor chứa $`B`$ mẫu kinh nghiệm kèm theo vector chỉ số cây (tree index) và vector trọng số IS ($`\tilde{w}_i`$).
3. **Môi trường mô phỏng thử nghiệm:** Gymnasium CartPole-v1 (kiểm thử Double vs Vanilla DQN) và LunarLander-v2 (kiểm thử hiệu quả phân tách Dueling và PER).

---

## Design

Dưới đây là mã nguồn Python hoàn chỉnh triển khai cấu trúc cây nhị phân SumTree, Prioritized Experience Replay Buffer, kiến trúc Dueling Q-Network và hàm huấn luyện Double DQN với kỹ thuật bù trừ trọng số IS.

```python
"""Module triển khai các cải tiến cốt lõi của DQN:

- SumTree & Prioritized Experience Replay (PER)
- Kiến trúc Dueling Q-Network
- Thuật toán cập nhật Double DQN với Importance Sampling weights
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. CẤU TRÚC DỮ LIỆU SUMTREE PHỤC VỤ PER O(log N)
# ==============================================================================
class SumTree:
    """Cây nhị phân hoàn chỉnh lưu trữ tổng độ ưu tiên của các mẫu kinh nghiệm.
    
    Các nút lá lưu độ ưu tiên p_i^alpha.
    Các nút cha lưu tổng của các nút con tương ứng.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Cây nhị phân đầy đủ có 2 * capacity - 1 nút
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        # Bộ đệm dữ liệu lưu đối tượng chuyển tiếp tại các nút lá
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Cập nhật ngược hiệu số độ ưu tiên lên nút gốc."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Truy vấn tìm nút lá tương ứng với giá trị tích lũy s."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self) -> float:
        """Trả về tổng độ ưu tiên toàn bộ bộ đệm (giá trị nút gốc)."""
        return float(self.tree[0])

    def add(self, priority: float, data: object) -> None:
        """Thêm một mẫu chuyển tiếp mới vào nút lá và cập nhật cây."""
        tree_idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(tree_idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, priority: float) -> None:
        """Cập nhật độ ưu tiên của một nút lá và lan truyền lên nút gốc."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get_leaf(self, s: float) -> Tuple[int, float, object]:
        """Truy xuất nút lá, giá trị ưu tiên và dữ liệu mẫu với ngưỡng s."""
        leaf_idx = self._retrieve(0, s)
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]


# ==============================================================================
# 2. BỘ NHỚ PRIORITIZED EXPERIENCE REPLAY (PER)
# ==============================================================================
class PrioritizedReplayBuffer:
    """Bộ nhớ Replay Buffer lấy mẫu theo độ lớn sai số TD với Importance Sampling."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-6,
    ) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = eps
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Ghi nhận chuyển tiếp mới với độ ưu tiên khởi tạo cực đại."""
        transition = (state, action, reward, next_state, done)
        priority = (self.max_priority + self.eps) ** self.alpha
        self.tree.add(priority, transition)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Lấy mẫu batch theo phân phối phân đoạn (stratified sampling)."""
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_weights = np.empty((batch_size,), dtype=np.float32)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        total_p = self.tree.total_priority()
        segment = total_p / batch_size

        # Nội suy tuyến tính hệ số beta tiến dần về 1.0
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity :]) / total_p if total_p > 0 else 1.0
        max_weight = (self.tree.size * min_prob) ** (-beta) if min_prob > 0 else 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, p, data = self.tree.get_leaf(s)
            prob = p / total_p
            # Trọng số Importance Sampling chuẩn hóa
            is_weight = (self.tree.size * prob) ** (-beta) if prob > 0 else 1.0
            b_weights[i] = is_weight / (max_weight + 1e-8)
            b_idx[i] = idx

            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            b_idx,
            b_weights,
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Cập nhật độ ưu tiên sau khi mạng tính toán xong sai số TD mới."""
        for idx, td_error in zip(tree_indices, td_errors):
            clipped_error = min(float(abs(td_error)), 100.0)
            priority = (clipped_error + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            if priority > self.max_priority:
                self.max_priority = priority


# ==============================================================================
# 3. MẠNG NƠ-RON DUELING Q-NETWORK
# ==============================================================================
class DuelingQNetwork(nn.Module):
    """Kiến trúc Dueling DQN phân tách luồng Value V(s) và Advantage A(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        # Tầng trích xuất đặc trưng chung
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Nhánh ước lượng giá trị trạng thái vô hướng V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Nhánh ước lượng giá trị lợi thế đa chiều A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Tính Q(s, a) từ V(s) và A(s, a) qua cơ chế trừ trung bình cộng."""
        features = self.feature_extractor(state)
        v = self.value_stream(features)  # Shape: (Batch, 1)
        a = self.advantage_stream(features)  # Shape: (Batch, Action_Dim)

        # Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum(A(s, a')))
        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q


# ==============================================================================
# 4. THUẬT TOÁN DOUBLE DQN VỚI PER LOSS
# ==============================================================================
def train_double_dqn_step(
    online_net: DuelingQNetwork,
    target_net: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: PrioritizedReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> Tuple[float, float]:
    """Thực hiện một bước cập nhật Double DQN với bộ đệm PER."""
    (states, actions, rewards, next_states, dones, tree_indices, is_weights) = buffer.sample(batch_size)

    # Chuyển đổi sang PyTorch Tensors
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
    weights_t = torch.tensor(is_weights, dtype=torch.float32, device=device).unsqueeze(1)

    # 1. Dự đoán Q(s, a) hiện tại của mạng Online
    curr_q = online_net(states_t).gather(1, actions_t)

    # 2. Cơ chế Double DQN: Mạng Online chọn hành động, mạng Target tính giá trị
    with torch.no_grad():
        next_actions = online_net(next_states_t).argmax(dim=-1, keepdim=True)
        target_q_next = target_net(next_states_t).gather(1, next_actions)
        target_q = rewards_t + gamma * target_q_next * (1.0 - dones_t)

    # 3. Tính toán sai số TD và cập nhật lại độ ưu tiên trong SumTree
    td_errors = (curr_q - target_q).detach().squeeze().cpu().numpy()
    buffer.update_priorities(tree_indices, td_errors)

    # 4. Hàm tổn thất Huber có trọng số Importance Sampling
    loss = (weights_t * F.smooth_l1_loss(curr_q, target_q, reduction="none")).mean()

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping chống bùng nổ gradient
    nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item()), float(np.mean(np.abs(td_errors)))


if __name__ == "__main__":
    # Minh họa vận hành tích hợp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 4
    action_dim = 2
    batch_size = 32

    online_net = DuelingQNetwork(state_dim, action_dim).to(device)
    target_net = DuelingQNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)
    buffer = PrioritizedReplayBuffer(capacity=1000)

    # Nạp 100 chuyển tiếp giả lập
    for _ in range(100):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randint(0, action_dim)
        r = np.random.randn()
        s_prime = np.random.randn(state_dim).astype(np.float32)
        d = np.random.rand() > 0.95
        buffer.push(s, a, r, s_prime, d)

    loss_val, mean_td = train_double_dqn_step(
        online_net, target_net, optimizer, buffer, batch_size=batch_size, gamma=0.99, device=device
    )
    print(f"Khởi chạy thành công một bước cập nhật Double Dueling DQN với PER:")
    print(f"- Weighted Smooth L1 Loss: {loss_val:.4f}")
    print(f"- Mean Absolute TD Error: {mean_td:.4f}")
```

---

## Nguồn

- [1] Van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with double Q-learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1), 2094-2100.
- [2] Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016). *Dueling network architectures for deep reinforcement learning*. International Conference on Machine Learning (ICML), PMLR, 48, 1995-2003.
- [3] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized experience replay*. International Conference on Learning Representations (ICLR).
- [4] Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D. (2018). *Rainbow: Combining improvements in deep reinforcement learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 3215-3222.
- [5] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533.
