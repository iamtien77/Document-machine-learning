# Mạng Q Sâu (Deep Q-Network - DQN)

## Purpose

Deep Q-Network (DQN), do nhóm nghiên cứu DeepMind (Mnih et al.) công bố trên tạp chí danh giá *Nature* năm 2015, là **bước ngoặt lịch sử mở ra kỷ nguyên Học tăng cường sâu (Deep Reinforcement Learning - DRL)**.

Trước khi DQN ra đời, việc kết hợp mạng nơ-ron sâu phi tuyến tính với giải thuật Q-Learning luôn thất bại và bị coi là bất khả thi do hiện tượng **"Tam Giác Tử Thần" (The Deadly Triad)** trong lý thuyết RL:
1. **Xấp xỉ hàm (Function Approximation):** Dùng mạng nơ-ron sâu thay cho bảng tra cứu tĩnh.
2. **Cơ chế Bootstrapping:** Cập nhật giá trị ước lượng dựa trên các giá trị ước lượng kế tiếp.
3. **Huấn luyện Off-policy:** Học chính sách tối ưu từ dữ liệu sinh ra bởi chính sách thăm dò.

Sự hội tụ của cả 3 yếu tố này khiến hàm mất mát dễ dàng phát nổ (Divergence) và mạng nơ-ron bị quên thảm họa (Catastrophic Forgetting). DeepMind đã giải quyết triệt để vấn đề này nhờ hai sáng kiến kỹ thuật mang tính cách mạng:
- **Bộ nhớ phát lại kinh nghiệm (Experience Replay Buffer):** Lưu trữ các chuyển dịch và lấy mẫu ngẫu nhiên để phá vỡ tính tương quan chuỗi thời gian, biến luồng dữ liệu tương tác phi dừng thành phân phối độc lập cùng phân phối (i.i.d).
- **Mạng mục tiêu độc lập (Separate Target Network):** Đóng băng định kỳ trọng số của mục tiêu tính toán để triệt tiêu hiện tượng "rượt đuổi mục tiêu di động" (Moving Target Problem).

---

## Input

Hệ thống DQN tiếp nhận các nguồn dữ liệu và cấu hình huấn luyện:

1. **Trạng thái nhiều chiều hoặc liên tục ($`s_t \in \mathbb{R}^d`$):**
   - Không gian liên tục: Vector tọa độ, vận tốc, góc nghiêng ($`d = 4`$ trong CartPole, $`d = 8`$ trong LunarLander).
   - Hình ảnh thị giác (Raw Pixels): Chuỗi 4 khung ảnh xám xếp chồng có kích thước $`84 \times 84 \times 4`$ (trong Atari Games).

2. **Không gian hành động rời rạc ($`\mathcal{A} = \lbrace 0, 1, \dots, |\mathcal{A}| - 1 \rbrace`$):**
   Số lượng hành động hữu hạn.

3. **Bộ đệm Replay Buffer ($`\mathcal{D}`$):**
   Dung lượng chứa $`N`$ bản ghi chuyển dịch gần nhất (thường $`N = 50,000 \to 1,000,000`$).

4. **Siêu tham số tối ưu:**
   - Kích thước mini-batch: $`B \in [32, 128]`$.
   - Chu kỳ cập nhật Target Network: $`C \in [500, 2000]`$ bước.
   - Tốc độ học: $`\alpha \in [10^{-4}, 10^{-3}]`$ (sử dụng AdamW hoặc RMSprop).
   - Lịch trình thăm dò: $`\epsilon`$-greedy phân rã từ $`1.0`$ xuống $`0.01`$.

---

## Output

Mô hình DQN sau khi huấn luyện cung cấp:

1. **Bộ trọng số mạng nơ-ron tối ưu ($`\theta^{\ast}`$):**
   Xấp xỉ hàm $`Q_\theta(s, a) \approx Q^{\ast}(s, a)`$.

2. **Chính sách suy luận tất định thời gian thực:**
   Chỉ với một lần truyền thẳng (Forward Pass), mạng xuất ra vector giá trị của toàn bộ các hành động:

$$
a^{\ast} = \arg\max_{a \in \mathcal{A}} Q_\theta(s, a)
$$

3. **Năng lực xử lý không gian trạng thái vô hạn:** Không còn bị giới hạn bởi bộ nhớ bảng Q-Table truyền thống.

---

## Evaluation metrics

Chỉ số đánh giá độ ổn định và hiệu năng của DQN:

1. **Điểm số Episode Return trung bình trượt:** Thước đo thành công cuối cùng của bài toán (ví dụ: đạt điểm tối đa 500 liên tục trên CartPole-v1, hoặc vượt kỷ lục người chơi trên game Atari Breakout).
2. **Mất mát Bellman trung bình (Mean Bellman Loss):**

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( y_i - Q_\theta(s_i, a_i) \right)^2
$$

3. **Độ lệch ước lượng giá trị Q ($`\mathbb{E}[\max_a Q(s, a)]`$):** Theo dõi mức độ ổn định của hàm giá trị, cảnh báo sớm nếu giá trị Q tăng vọt phi thực tế (Overestimation).

---

## How it works

### 1. Kiến Trúc Hai Mạng Và Luồng Hoạt Động Của DQN

DQN duy trì đồng thời hai mạng nơ-ron có kiến trúc giống hệt nhau:
1. **Online Q-Network (Trọng số $`\theta`$):** Được cập nhật liên tục qua từng bước huấn luyện bằng thuật toán Gradient Descent.
2. **Target Q-Network (Trọng số $`\theta^-`$):** Trọng số được đóng băng và chỉ sao chép từ mạng chính sau mỗi $`C`$ bước thời gian: $`\theta^- \leftarrow \theta`$.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             KIẾN TRÚC HỆ THỐNG DQN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Agent tương tác môi trường: (s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1})          │
│                                │                                            │
│                                ▼                                            │
│                [BỘ NHỚ PHÁT LẠI KINH NGHIỆM (REPLAY BUFFER)]                 │
│                                │                                            │
│                Lấy mẫu ngẫu nhiên Mini-batch (B mẫu)                        │
│                                │                                            │
│         ┌──────────────────────┴──────────────────────┐                     │
│         ▼                                             ▼                     │
│   [ONLINE NETWORK Q_θ]                      [TARGET NETWORK Q_θ^-]          │
│  Tính Q_θ(s_i, a_i)                         Tính max_{a'} Q_θ^-(s'_i, a')   │
│         │                                             │                     │
│         ▼                                             ▼                     │
│    Ước lượng                                    Mục tiêu Target y_i         │
│         │                                             │                     │
│         └──────────────────────┬──────────────────────┘                     │
│                                ▼                                            │
│         Hàm mất mát Bellman: L(θ) = 1/B * ∑ (y_i - Q_θ(s_i, a_i))²          │
│                                │                                            │
│                                ▼                                            │
│                  Lan truyền ngược cập nhật θ                                │
│                                │                                            │
│                  Mỗi C bước: θ^- ← θ (Đồng bộ)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Hai Đột Phá Ổn Định Trọng Yếu Của DQN

#### A. Bộ Nhớ Phát Lại Kinh Nghiệm (Experience Replay Buffer)
- **Vấn đề trong RL chuẩn:** Dữ liệu tương tác thực tế $`s_t, s_{t+1}, s_{t+2}`$ có tương quan chuỗi rất cao (Temporal Correlation). Huấn luyện mạng nơ-ron trên dữ liệu tương quan sẽ làm mạng quên hoàn toàn các bài học trong quá khứ và phá vỡ tính ổn định của gradient.
- **Giải pháp của DQN:** Lưu trữ các chuyển dịch vào hàng đợi vòng tròn $`\mathcal{D} = \lbrace e_1, e_2, \dots, e_N \rbrace`$. Khi huấn luyện, rút ngẫu nhiên không hoàn lại một mini-batch $`B`$ mẫu. Điều này triệt tiêu hoàn toàn tương quan chuỗi, phục hồi tính chất độc lập ngẫu nhiên (i.i.d) của dữ liệu.

#### B. Mạng Mục Tiêu Độc Lập (Separate Target Network)
- **Vấn đề Moving Target:** Trong công thức Q-Learning chuẩn:

$$
y = r + \gamma \max_{a'} Q_\theta(s', a')
$$

  Cả mục tiêu $`y`$ và giá trị dự đoán $`Q_\theta(s, a)`$ đều phụ thuộc vào cùng một bộ trọng số $`\theta`$. Mỗi khi cập nhật $`\theta`$ để tiến gần tới $`y`$, thì chính $`y`$ cũng bị dịch chuyển sang vị trí khác (giống như con lừa đuổi theo củ cà rốt treo trước mũi).
- **Giải pháp của DQN:** Cố định mục tiêu bằng mạng $`\theta^-`$:

$$
y_i = r_i + (1 - d_i) \cdot \gamma \max_{a' \in \mathcal{A}} Q_{\theta^-}(s'_i, a')
$$

  Vì $`\theta^-`$ không thay đổi trong suốt $`C`$ bước, mục tiêu $`y_i`$ là một giá trị tĩnh ổn định, cho phép thuật toán quy hoạch hồi quy hội tụ chuẩn xác.

---

### 3. Hàm Mất Mát Huber (Smooth L1 Loss)

Để chống lại hiện tượng gradient bùng nổ khi gặp các phần thưởng đột biến hoặc ngoại lai, DQN thường sử dụng hàm mất mát Huber thay cho bình phương sai số MSE:

$$
\mathcal{L}_{\mathrm{Huber}}(e) = \begin{cases}
\frac{1}{2} e^2 & \text{nếu } |e| \le 1 \\
|e| - \frac{1}{2} & \text{ngược lại}
\end{cases}
$$

Trong đó $`e = y_i - Q_\theta(s_i, a_i)`$. Hàm này hoạt động như MSE khi sai số nhỏ (mịn màng) và hoạt động như sai số tuyệt đối MAE khi sai số lớn (gradient bị chặn ở $`\pm 1`$).

---

## Data used

Bảng môi trường chuẩn hóa thường dùng để đánh giá DQN:

| Môi trường | Kích thước trạng thái $\mathcal{S}$ | Số hành động $|\mathcal{A}|$ | Mục tiêu giải quyết chuẩn |
| :--- | :--- | :--- | :--- |
| **CartPole-v1** | Liên tục 4 chiều: $[x, \dot{x}, \theta, \dot{\theta}]$ | 2 (Trái, Phải) | Điểm trung bình $\ge 475.0$ qua 100 episodes liên tiếp |
| **LunarLander-v2** | Liên tục 8 chiều: Tọa độ, vận tốc, góc, tiếp đất | 4 (Tắt, Đẩy trái, Đẩy chính, Đẩy phải) | Điểm trung bình $\ge 200.0$ |
| **Atari Breakout** | Hình ảnh $84 \times 84 \times 4$ pixels | 4 (Không làm gì, Bắn bóng, Sang trái, Sang phải) | Vượt qua điểm số trung bình người chơi con người |

---

## Design

### Mã Nguồn Cài Đặt DQN Hoàn Chỉnh (PyTorch / NumPy Standalone Pipeline)

Đoạn mã dưới đây cài đặt hoàn chỉnh tác tử DQN với mạng nơ-ron nhiều tầng (MLP), bộ đệm Replay Buffer, mạng Target Network và vòng lặp huấn luyện chuẩn xác:

```python
import numpy as np
import random
from collections import deque

# ==============================================================================
# 1. BỘ NHỚ PHÁT LẠI KINH NGHIỆM (EXPERIENCE REPLAY BUFFER)
# ==============================================================================
class ReplayBuffer:
    """Bộ đệm kinh nghiệm hàng đợi vòng tròn (Ring Buffer)."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# 2. XẤP XỈ HÀM Q BẰNG MẠNG NƠ-RON TUYẾN TÍNH MỞ RỘNG (MOCK MLP BACKBONE)
# ==============================================================================
class SimpleQNetwork:
    """Mạng xấp xỉ giá trị Q(s, a) thuần Python/NumPy (2 lớp tuyến tính)."""
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        # Trọng số W1: [state_dim, 16], W2: [16, action_dim]
        self.W1 = np.random.randn(state_dim, 16) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, action_dim) * np.sqrt(2.0 / 16)
        self.b2 = np.zeros(action_dim)

    def forward(self, states):
        """Lan truyền xuôi tính toán Q-values."""
        self.z1 = np.dot(states, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU activation
        q_vals = np.dot(self.a1, self.W2) + self.b2
        return q_vals

    def copy_weights_from(self, source_net):
        """Sao chép trọng số từ Online Network sang Target Network."""
        self.W1 = np.copy(source_net.W1)
        self.b1 = np.copy(source_net.b1)
        self.W2 = np.copy(source_net.W2)
        self.b2 = np.copy(source_net.b2)

    def train_step(self, states, actions, targets):
        """Một bước cập nhật Gradient Descent trên mini-batch."""
        batch_size = len(states)
        q_pred = self.forward(states)

        # Đạo hàm sai số MSE đối với output
        grad_out = np.zeros_like(q_pred)
        for i in range(batch_size):
            grad_out[i, actions[i]] = 2.0 * (q_pred[i, actions[i]] - targets[i]) / batch_size

        # Lan truyền ngược (Backprop)
        grad_W2 = np.dot(self.a1.T, grad_out)
        grad_b2 = np.sum(grad_out, axis=0)

        grad_a1 = np.dot(grad_out, self.W2.T)
        grad_z1 = grad_a1 * (self.z1 > 0) # ReLU derivative

        grad_W1 = np.dot(states.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)

        # Cập nhật trọng số
        self.W2 -= self.lr * np.clip(grad_W2, -1.0, 1.0)
        self.b2 -= self.lr * np.clip(grad_b2, -1.0, 1.0)
        self.W1 -= self.lr * np.clip(grad_W1, -1.0, 1.0)
        self.b1 -= self.lr * np.clip(grad_b1, -1.0, 1.0)


# ==============================================================================
# 3. TÁC TỬ DQN HOÀN CHỈNH
# ==============================================================================
class DQNAgent:
    def __init__(self, state_dim=4, action_dim=2, gamma=0.99, batch_size=32, target_update=100):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_counter = 0

        self.memory = ReplayBuffer(capacity=5000)
        self.online_net = SimpleQNetwork(state_dim, action_dim, lr=0.005)
        self.target_net = SimpleQNetwork(state_dim, action_dim)
        self.target_net.copy_weights_from(self.online_net)

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.online_net.forward(np.expand_dims(state, 0))
        return int(np.argmax(q_values[0]))

    def train_iteration(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 1. Tính Q-Target bằng TARGET NETWORK (đóng băng)
        next_q_vals = self.target_net.forward(next_states)
        max_next_q = np.max(next_q_vals, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        # 2. Cập nhật ONLINE NETWORK bằng Gradient Descent
        self.online_net.train_step(states, actions, targets)

        # 3. Đồng bộ định kỳ sang Target Network
        self.step_counter += 1
        if self.step_counter % self.target_update == 0:
            self.target_net.copy_weights_from(self.online_net)


# ==============================================================================
# MÔ PHỎNG HUẤN LUYỆN DQN
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    agent = DQNAgent(state_dim=4, action_dim=2, gamma=0.99, batch_size=16, target_update=50)
    print("=== MÔ PHỎNG QUY TRÌNH HUẤN LUYỆN DEEP Q-NETWORK (DQN) ===")
    print("Khởi tạo Replay Buffer và Target Network độc lập...\n")

    # Nạp dữ liệu mô phỏng vào Replay Buffer
    for _ in range(200):
        s = np.random.randn(4)
        a = np.random.randint(2)
        r = 1.0 if s[2] > 0 else 0.0
        s_next = s + np.random.randn(4) * 0.1
        done = False
        agent.memory.push(s, a, r, s_next, done)
        agent.train_iteration()

    # Kiểm tra suy luận trên một trạng thái mới
    test_state = np.array([0.05, 0.2, -0.1, 0.4], dtype=np.float32)
    q_preds = agent.online_net.forward(np.expand_dims(test_state, 0))[0]
    best_act = agent.select_action(test_state, epsilon=0.0)

    print(f"Trạng thái đầu vào: {test_state}")
    print(f"Giá trị Q dự đoán [Q(s, 0), Q(s, 1)]: [{q_preds[0]:.4f}, {q_preds[1]:.4f}]")
    print(f"Hành động tối ưu suy luận được: Hành động {best_act}")
    print("\n=> Cơ chế Replay Buffer và Target Network bảo đảm gradient hội tụ ổn định.")
```

---

## Nguồn

- [1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). *Playing Atari with deep reinforcement learning*. arXiv preprint arXiv:1312.5602.
- [2] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533.
- [3] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (Chapter 11: Off-policy Methods with Approximation, Section 11.3: The Deadly Triad). MIT Press.

