# Đánh Giá, Khai Thác Và Triển Khai Sau Huấn Luyện Trong Học Tăng Cường

Giai đoạn sau huấn luyện (Post-Training) là bước chuyển tiếp quyết định giữa mô hình học máy trong phòng thí nghiệm và hệ thống vận hành thực tế. Tài liệu này chuẩn hóa toàn bộ quy trình đánh giá tính vững chắc (Robustness), chuyển đổi từ cơ chế dò tìm (Exploration) sang khai thác tất định (Exploitation), thiết lập màng bọc an toàn (Safety Shielding) và phương pháp thu hẹp khoảng cách mô phỏng - thực tế (Sim-to-Real Transfer).

---

## Purpose

1. **Loại bỏ cơ chế thăm dò ngẫu nhiên (Zero Exploration):** Chuyển dịch toàn bộ chính sách hành động từ chế độ học tập phân tán (vốn có $`\epsilon`$-greedy hoặc lấy mẫu Gauss) sang chính sách khai thác tất định (Deterministic Exploitation), tối đa hóa phần thưởng kỳ vọng và triệt tiêu tính bất định không mong muốn.
2. **Đánh giá thống kê chuẩn mực (Rigorous Statistical Evaluation):** Đo lường hiệu năng của tác tử trên tập hợp các seed môi trường độc lập chưa từng xuất hiện trong quá trình huấn luyện, tính toán các chỉ số đuôi rủi ro (Risk-sensitive Metrics) như CVaR nhằm tránh đánh giá thiên lệch dựa trên các giá trị trung bình ảo.
3. **Kiểm thử ứng suất và tính vững chắc (Stress Testing & Robustness):** Đánh giá mức độ suy thoái hiệu năng khi môi trường chịu tác động của dịch chuyển phân phối (Distribution Shift), nhiễu cảm biến, độ trễ truyền thông, hoặc biến thiên các thông số động học.
4. **Màng lọc an toàn và cơ chế phòng vệ (Safety Shielding & Fallback):** Xây dựng lớp rào chắn giám sát độc lập nhằm chặn đứng các hành động có thể đưa hệ thống vào trạng thái nguy hiểm vi phạm ràng buộc vật lý, đồng thời kích hoạt chính sách an toàn dự phòng (Rule-based Fallback Policy).
5. **Đóng gói tối ưu hóa suy luận (Inference Optimization):** Chuyển đổi mô hình từ đồ thị huấn luyện cồng kềnh sang định dạng tối ưu thời gian thực (TorchScript, ONNX, TensorRT) với độ trễ phản hồi cấp mili-giây (sub-millisecond latency).

---

## Input

Quy trình đánh giá và khai thác sau huấn luyện tiếp nhận các thành phần:

1. **Mô hình tác tử tối ưu ($`\pi_\theta^{\ast}`$ hoặc $`Q_\theta^{\ast}`$):** Trọng số mạng nơ-ron đóng băng sau quá trình hội tụ (không còn cập nhật gradient).
2. **Tập cấu hình môi trường kiểm thử (Evaluation Benchmarks):**
   - Danh sách $`K`$ hạt giống ngẫu nhiên độc lập ($`K \ge 30`$).
   - Dải biến thiên thông số vật lý (khối lượng, ma sát, quán tính, độ trễ tín hiệu).
3. **Mô hình nhiễu quan sát ($`\boldsymbol{\epsilon}_{\mathrm{obs}} \sim \mathcal{N}(0, \Sigma)`$):** Các kịch bản giả lập suy hao tín hiệu cảm biến hoặc sai số truyền thông.
4. **Tập ràng buộc an toàn biên (Safety Envelope Constraints $`\mathcal{C}`$):** Giới hạn không gian trạng thái $`\mathcal{S}_{\mathrm{safe}} \subset \mathcal{S}`$ và không gian hành động hợp lệ $`\mathcal{A}_{\mathrm{safe}}(s)`$.
5. **Yêu cầu phi chức năng triển khai:** Ngưỡng trễ tối đa (P99 Latency budget, ví dụ $`< 10\,\mathrm{ms}`$) và giới hạn bộ nhớ phân bổ.

---

## Output

1. **Báo cáo phân phối hiệu năng toàn diện:**
   - Giá trị trung bình (Mean), trung vị (Median), và khoảng phân vị liên phân vị (IQR - Interquartile Range).
   - Giá trị rủi ro có điều kiện tại phân vị 5% ($`\mathrm{CVaR}_{0.05}`$).
2. **Biểu đồ độ nhạy vững chắc (Robustness Degradation Curve):** Đường cong biểu diễn sự sụt giảm phần thưởng khi tăng dần biên độ nhiễu hoặc sai lệch mô hình.
3. **Tỷ lệ vi phạm an toàn (Safety Violation Rate):** Tần suất tiếp cận vùng rủi ro $`\mathcal{S}_{\mathrm{unsafe}}`$.
4. **Artifact triển khai đóng gói hoàn chỉnh:** Tệp mô hình định dạng ONNX/TorchScript đi kèm siêu dữ liệu chuẩn hóa đầu vào/đầu ra và lớp màng an toàn bọc ngoài.

---

## Evaluation metrics

1. **Phần thưởng tích lũy trung bình và khoảng tin cậy (Mean Return with 95% Bootstrap CI):**

$$
\bar{G} = \frac{1}{K} \sum_{k=1}^K G^{(k)}, \quad \mathrm{CI}_{95\%} = \left[ q_{0.025}, q_{0.975} \right]
$$

2. **Giá trị rủi ro có điều kiện (Conditional Value at Risk - $`\mathrm{CVaR}_\alpha`$):** Đo lường kỳ vọng của $`\alpha \times 100\%`$ các quỹ đạo có kết quả tồi tệ nhất, thể hiện rủi ro sập đổ của hệ thống:

$$
\mathrm{CVaR}_\alpha(G) = \mathbb{E}\left[ G \mid G \le \mathrm{VaR}_\alpha(G) \right]
$$

   *(Trong các hệ thống thực tế như xe tự hành hoặc tài chính, $`\mathrm{CVaR}_{0.05}`$ là chỉ số sinh tử quan trọng hơn giá trị trung bình $`\bar{G}`$)*.
3. **Tỷ lệ thành công mục tiêu (Task Success Rate):**

$$
\mathrm{SR} = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\left( \text{Episode } k \text{ đạt trạng thái mục tiêu} \right)
$$

4. **Độ trễ suy luận phân vị thứ 99 (P99 Inference Latency):**

$$
\mathrm{Latency}_{\mathrm{P99}} = \mathrm{Quantile}_{0.99}\left( \lbrace t_{\mathrm{infer}}^{(i)} \rbrace_{i=1}^M \right)
$$

5. **Chỉ số suy hao vững chắc (Robustness Score):**

$$
\rho_{\mathrm{robust}} = \frac{\mathbb{E}_{s \sim \mathcal{E}_{\mathrm{noisy}}}[G]}{\mathbb{E}_{s \sim \mathcal{E}_{\mathrm{clean}}}[G]}
$$

---

## How it works

### 1. Chuyển Đổi Sang Chính Sách Khai Thác Tất Định

Trong giai đoạn huấn luyện, tác tử duy trì tính ngẫu nhiên để khám phá (Exploration). Khi triển khai:
- **Phương pháp Value-based (DQN/DDQN):** Thiết lập vĩnh viễn $`\epsilon = 0`$. Chọn hành động đạt giá trị cao nhất:

$$
a^{\ast} = \arg\max_{a \in \mathcal{A}} Q(s, a; \theta^{\ast})
$$

- **Phương pháp Policy-based (REINFORCE/A2C/PPO):** Thay vì lấy mẫu ngẫu nhiên từ phân phối $`a \sim \pi(a|s)`$:
  - Trong không gian rời rạc: $`a^{\ast} = \arg\max_{a} \pi(a|s; \theta^{\ast})`$.
  - Trong không gian liên tục: Sử dụng trực tiếp vector kỳ vọng $`a^{\ast} = \boldsymbol{\mu}(s; \theta^{\ast})`$, triệt tiêu hoàn toàn vector phương sai $`\boldsymbol{\sigma}(s)`$.

---

### 2. Đánh Giá Đa Seed Và Kiểm Định Thống Kê Chuẩn

Theo nghiên cứu thực nghiệm của Henderson et al. (2018), hiệu năng Deep RL dao động rất lớn giữa các lần khởi tạo ngẫu nhiên. Quy trình chuẩn đòi hỏi:
1. Đóng băng hạt giống môi trường cho tập kiểm thử: $`\mathrm{Seeds} = \lbrace 10001, 10002, \dots, 10000 + K \rbrace`$.
2. Chạy tối thiểu $`K = 30`$ đến $`100`$ tập đánh giá (episodes).
3. Sử dụng Bootstrap Resampling (10,000 lần lặp) để thiết lập khoảng tin cậy 95% không giả định phân phối chuẩn (Non-parametric 95% Confidence Interval).

---

### 3. Kiểm Thử Vững Chắc Và Biến Đổi Phân Phối (Robustness Stress Testing)

Hệ thống triển khai thực tế luôn đối mặt với môi trường biến động so với giả lập huấn luyện:
- **Nhiễu cảm biến (Observation Perturbations):** Chèn nhiễu trắng Gaussian hoặc nhiễu xung (Impulse Noise) vào trạng thái:

$$
\tilde{s}_t = s_t + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(0, \sigma^2 I)
$$

- **Nhiễu truyền động (Action Perturbations):** Hành động thực thi bị trễ $`k`$ bước hoặc suy giảm biên độ do hao mòn cơ khí: $`\tilde{a}_t = (1 - \lambda) a_t + \lambda a_{t-1}`$.
- **Biến thiên thông số vật lý (Domain Shift):** Thay đổi các đặc tính môi trường (ví dụ lực ma sát thay đổi từ 50% đến 200% giá trị danh định) để kiểm tra điểm giới hạn gãy đổ của chính sách.

---

### 4. Lớp Rào Chắn An Toàn (Safety Shielding & Fallback)

Để đảm bảo tuyệt đối không phá hủy phần cứng hay gây tai nạn, chính sách RL được bao bọc bởi một lớp phòng vệ độc lập:

```
                  +-----------------------------------+
                  |        Trạng thái thực s_t        |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
| Chính sách nơ-ron RL  |                       |  Mô hình an toàn      |
| a_RL = pi*(s_t)       |                       |  (Safety Monitor)     |
+-----------+-----------+                       +-----------+-----------+
            |                                               |
            |   Hành động đề xuất                           | Ranh giới an toàn
            +-----------------------+   +-------------------+
                                    |   |
                                    v   v
                        +---------------------------+
                        |  Safety Shield / Projector|
                        +-------------+-------------+
                                      |
                     +----------------+----------------+
                     | An toàn                         | Nguy hiểm
                     v                                 v
         +-----------------------+         +-----------------------+
         | Thực thi a_RL         |         | Kích hoạt Fallback    |
         |                       |         | hoặc chiếu an toàn    |
         +-----------------------+         +-----------------------+
```

Toán tử chiếu an toàn (Safety Projection) giải bài toán tối ưu lồi nhằm tìm hành động an toàn gần nhất với hành động do mạng RL đề xuất:

$$
a_t^{\mathrm{exec}} = \arg\min_{a \in \mathcal{A}_{\mathrm{safe}}(s_t)} \| a - a_t^{\mathrm{RL}} \|_2^2
$$

---

### 5. Thu Hẹp Khoảng Cách Thực Tế (Sim-to-Real Transfer)

Khoảng cách thực tế (Reality Gap) phát sinh từ các chi tiết vật lý mà trình mô phỏng không thể mô hình hóa chính xác 100%. Các kỹ thuật chủ đạo:
- **Domain Randomization (DR):** Trong quá trình huấn luyện, liên tục ngẫu nhiên hóa các thuộc tính môi trường (khối lượng, ánh sáng, ma sát, bề mặt) trên dải rộng để mạng học được các biểu diễn bất biến.
- **System Identification (SysID):** Đo đạc dữ liệu thực tế để hiệu chỉnh chính xác các tham số của bộ mô phỏng số.
- **Residual RL:** Sử dụng một bộ điều khiển cổ điển (PID / MPC) làm nền tảng ổn định, và mạng RL chỉ học phần dư (Residual compensation) để bù trừ phi tuyến tính.

---

## Data used

1. **Tập dữ liệu quỹ đạo kiểm thử (Evaluation Rollout Logs):** Bản ghi chuỗi thời gian chứa $`(s_t, a_t, r_t, \tilde{s}_t, \text{latency}_t)`$ phục vụ phân tích hậu kỳ.
2. **Hồ sơ thông số Benchmark phần cứng:** Dữ liệu đo đạc tài nguyên tính toán (CPU %, GPU VRAM, P50/P90/P99 latency) trên các nền tảng mục tiêu (Edge Devices như Raspberry Pi, NVIDIA Jetson hoặc x86 Server).
3. **Môi trường kiểm định chuẩn:** Gymnasium LunarLander-v2 (mô phỏng hạ cánh tàu vũ trụ với các kịch bản gió đổi chiều và động cơ suy giảm hiệu năng).

---

## Design

Dưới đây là module Python chuẩn công nghiệp thực thi toàn diện quy trình kiểm định sau huấn luyện: đánh giá đa seed thống kê, đo lường P99 latency, kiểm thử độ vững chắc dưới tác động của nhiễu quan sát, và tích hợp màng lọc an toàn (Safety Shield Wrapper).

```python
"""Module kiểm định và triển khai sau huấn luyện cho tác tử Học Tăng Cường:

- Đánh giá thống kê đa Seed (Mean, Bootstrap CI, CVaR)
- Benchmark độ trễ suy luận thời gian thực (P50, P90, P99)
- Stress Testing với nhiễu môi trường (Robustness Analysis)
- Lớp vỏ màng lọc an toàn và chính sách dự phòng (Safety Shield Wrapper)
"""

import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# 1. BỘ ĐÁNH GIÁ HIỆU NĂNG VÀ ĐỘ VỮNG CHẮC (POLICY EVALUATOR)
# ==============================================================================
class PostTrainingEvaluator:
    """Khung đánh giá thống kê toàn diện cho chính sách sau huấn luyện."""

    def __init__(
        self,
        policy_fn: Callable[[np.ndarray], int],
        env_factory: Callable[[int], any],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
            policy_fn: Hàm ánh xạ từ trạng thái sang hành động tất định.
            env_factory: Hàm khởi tạo môi trường nhận seed làm tham số.
            device: Thiết bị tính toán (CPU hoặc GPU).
        """
        self.policy_fn = policy_fn
        self.env_factory = env_factory
        self.device = device

    def evaluate_seeds(
        self, seeds: List[int], max_steps: int = 1000
    ) -> Dict[str, float]:
        """Đánh giá hiệu năng thuần túy (Exploitation) trên danh sách seeds."""
        returns: List[float] = []
        latencies: List[float] = []

        for seed in seeds:
            env = self.env_factory(seed)
            state, _ = env.reset(seed=seed)
            done = False
            truncated = False
            ep_return = 0.0
            steps = 0

            while not (done or truncated) and steps < max_steps:
                # Đo độ trễ suy luận của chính sách
                t0 = time.perf_counter()
                action = self.policy_fn(state)
                latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

                next_state, reward, done, truncated, _ = env.step(action)
                ep_return += reward
                state = next_state
                steps += 1

            returns.append(ep_return)
            env.close()

        returns_arr = np.array(returns, dtype=np.float64)
        latencies_arr = np.array(latencies, dtype=np.float64)

        # Tính toán CVaR ở phân vị 5% (rủi ro đuôi)
        cvar_5 = self._compute_cvar(returns_arr, alpha=0.05)
        # Bootstrap 95% Confidence Interval cho phần thưởng trung bình
        ci_low, ci_high = self._bootstrap_ci(returns_arr, num_bootstraps=5000)

        return {
            "mean_return": float(np.mean(returns_arr)),
            "std_return": float(np.std(returns_arr)),
            "median_return": float(np.median(returns_arr)),
            "cvar_0.05": float(cvar_5),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "p50_latency_ms": float(np.percentile(latencies_arr, 50)),
            "p95_latency_ms": float(np.percentile(latencies_arr, 95)),
            "p99_latency_ms": float(np.percentile(latencies_arr, 99)),
        }

    def stress_test_noise(
        self, seeds: List[int], noise_stds: List[float]
    ) -> Dict[float, float]:
        """Kiểm thử tính vững chắc bằng cách bơm nhiễu Gauss vào quan sát."""
        robustness_profile: Dict[float, float] = {}

        for sigma in noise_stds:
            returns: List[float] = []
            for seed in seeds:
                env = self.env_factory(seed)
                state, _ = env.reset(seed=seed)
                done = False
                truncated = False
                ep_return = 0.0

                while not (done or truncated):
                    # Bơm nhiễu cảm biến
                    noisy_state = state + np.random.normal(0, sigma, size=state.shape)
                    action = self.policy_fn(noisy_state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    ep_return += reward
                    state = next_state

                returns.append(ep_return)
                env.close()

            robustness_profile[sigma] = float(np.mean(returns))

        return robustness_profile

    @staticmethod
    def _compute_cvar(data: np.ndarray, alpha: float = 0.05) -> float:
        """Tính toán Conditional Value at Risk tại ngưỡng alpha."""
        sorted_data = np.sort(data)
        cutoff_idx = max(1, int(np.floor(alpha * len(sorted_data))))
        return float(np.mean(sorted_data[:cutoff_idx]))

    @staticmethod
    def _bootstrap_ci(
        data: np.ndarray, num_bootstraps: int = 5000, ci: float = 0.95
    ) -> Tuple[float, float]:
        """Ước lượng khoảng tin cậy không tham số qua phương pháp Bootstrap."""
        n = len(data)
        boot_means = np.empty(num_bootstraps, dtype=np.float64)
        for i in range(num_bootstraps):
            sample = np.random.choice(data, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        alpha_tail = (1.0 - ci) / 2.0
        low = np.percentile(boot_means, alpha_tail * 100)
        high = np.percentile(boot_means, (1.0 - alpha_tail) * 100)
        return low, high


# ==============================================================================
# 2. MÀNG BỌC AN TOÀN VÀ CƠ CHẾ DỰ PHÒNG (SAFETY SHIELD WRAPPER)
# ==============================================================================
class SafetyShieldWrapper:
    """Lớp rào chắn độc lập giám sát an toàn và kích hoạt Fallback khi cần."""

    def __init__(
        self,
        base_policy: Callable[[np.ndarray], int],
        safety_checker: Callable[[np.ndarray, int], bool],
        fallback_policy: Callable[[np.ndarray], int],
    ) -> None:
        """
        Args:
            base_policy: Chính sách RL đã huấn luyện.
            safety_checker: Hàm xác thực (s, a) có thuộc không gian an toàn.
            fallback_policy: Chính sách chuyên gia dựa trên luật dự phòng.
        """
        self.base_policy = base_policy
        self.safety_checker = safety_checker
        self.fallback_policy = fallback_policy
        self.interventions = 0
        self.total_calls = 0

    def select_action(self, state: np.ndarray) -> int:
        """Lựa chọn hành động với cơ chế kiểm duyệt rủi ro."""
        self.total_calls += 1
        candidate_action = self.base_policy(state)

        # Kiểm tra an toàn trước khi cấp phép thực thi
        if self.safety_checker(state, candidate_action):
            return candidate_action

        # Kích hoạt rào chắn can thiệp
        self.interventions += 1
        safe_action = self.fallback_policy(state)
        return safe_action

    def get_intervention_rate(self) -> float:
        """Tỷ lệ can thiệp của màng bảo vệ an toàn."""
        if self.total_calls == 0:
            return 0.0
        return self.interventions / self.total_calls


# ==============================================================================
# 3. MÔ PHỎNG THỬ NGHIỆM ĐỊNH LƯỢNG
# ==============================================================================
class MockEnvironment:
    """Môi trường giả lập 1D kiểm thử quy trình sau huấn luyện."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.RandomState(seed)
        self.pos = 0.0
        self.velocity = 0.0
        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.pos = self.rng.uniform(-0.5, 0.5)
        self.velocity = 0.0
        self.step_count = 0
        return np.array([self.pos, self.velocity], dtype=np.float32), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Hành động: 0 (Đẩy lùi), 1 (Giữ nguyên), 2 (Đẩy tiến)
        force = (action - 1) * 0.1
        self.velocity += force
        self.pos += self.velocity
        self.step_count += 1

        reward = -abs(self.pos) - 0.01 * (action != 1)
        # Kết thúc nếu chạm đích hoặc vượt ngưỡng an toàn (|pos| > 2.0)
        done = bool(abs(self.pos) < 0.05 and abs(self.velocity) < 0.02)
        truncated = self.step_count >= 100

        return np.array([self.pos, self.velocity], dtype=np.float32), reward, done, truncated, {}

    def close(self) -> None:
        pass


if __name__ == "__main__":
    np.random.seed(42)

    # 1. Giả lập chính sách mạng nơ-ron đóng băng
    def mock_rl_policy(state: np.ndarray) -> int:
        pos, vel = state[0], state[1]
        if pos > 0.1:
            return 0  # Đẩy lùi
        elif pos < -0.1:
            return 2  # Đẩy tiến
        return 1      # Ổn định

    # 2. Định nghĩa hàm an toàn: Ngăn chặn đẩy mạnh khi vận tốc đã quá cao
    def mock_safety_checker(state: np.ndarray, action: int) -> bool:
        velocity = state[1]
        # Không cho phép đẩy tiến thêm nếu vận tốc đã dương lớn
        if velocity > 0.15 and action == 2:
            return False
        # Không cho phép đẩy lùi thêm nếu vận tốc đã âm lớn
        if velocity < -0.15 and action == 0:
            return False
        return True

    # 3. Chính sách dự phòng an toàn (Phanh hãm)
    def mock_fallback_policy(state: np.ndarray) -> int:
        return 1  # Ngắt động cơ giữ nguyên trạng thái

    # Thiết lập màng bảo vệ
    shielded_policy = SafetyShieldWrapper(
        base_policy=mock_rl_policy,
        safety_checker=mock_safety_checker,
        fallback_policy=mock_fallback_policy,
    )

    evaluator = PostTrainingEvaluator(
        policy_fn=shielded_policy.select_action,
        env_factory=lambda s: MockEnvironment(s),
    )

    test_seeds = list(range(10001, 10031))  # 30 seeds độc lập
    print("Bắt đầu đánh giá hiệu năng thống kê trên 30 seeds...")
    results = evaluator.evaluate_seeds(test_seeds)

    print("\nKết Quả Đánh Giá Sau Huấn Luyện:")
    print(f"- Điểm thưởng trung bình (Mean Return): {results['mean_return']:.2f} +/- {results['std_return']:.2f}")
    print(f"- Khoảng tin cậy 95% Bootstrap CI: [{results['ci_95_low']:.2f}, {results['ci_95_high']:.2f}]")
    print(f"- Rủi ro đuôi CVaR 5%: {results['cvar_0.05']:.2f}")
    print(f"- Độ trễ suy luận P99 Latency: {results['p99_latency_ms']:.4f} ms")
    print(f"- Tỷ lệ can thiệp an toàn: {shielded_policy.get_intervention_rate() * 100:.2f}%")

    print("\nKiểm thử ứng suất nhiễu cảm biến (Stress Testing):")
    noise_levels = [0.0, 0.05, 0.1, 0.2]
    stress_results = evaluator.stress_test_noise(test_seeds[:10], noise_levels)
    for sigma, ret in stress_results.items():
        print(f"- Mức nhiễu sigma = {sigma:0.2f} | Phần thưởng trung bình: {ret:.2f}")
```

---

## Nguồn

- [1] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). *Deep reinforcement learning that matters*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 3207-3214.
- [2] Dulac-Arnold, G., Mankowitz, D., & Hester, T. (2019). *Challenges of real-world reinforcement learning*. arXiv preprint arXiv:1904.12901.
- [3] Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). *Domain randomization for transferring deep neural networks from simulation to the real world*. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 23-30.
- [4] Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). *Constrained policy optimization*. International Conference on Machine Learning (ICML), PMLR, 70, 22-31.
- [5] Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., & Topcu, U. (2018). *Safe reinforcement learning via shielding*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 266-273.

