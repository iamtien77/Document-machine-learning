# Kiến Trúc Pha Huấn Luyện Hệ Thống Học Tăng Cường Sản Xuất (Production Training Phase)

Pha huấn luyện (Training Phase) trong Học tăng cường cấp độ sản xuất đòi hỏi một kiến trúc kỹ thuật phần mềm vững chắc vượt xa các vòng lặp đơn luồng trong phòng thí nghiệm. Tài liệu này chuẩn hóa toàn bộ kiến trúc huấn luyện công nghiệp: vận hành môi trường vector hóa đa luồng (Vectorized Environments), chuẩn hóa động quan sát và phần thưởng (Running Normalization), cơ chế lưu trữ điểm kiểm tra định kỳ (Checkpointing), theo dõi số liệu thời gian thực (Telemetry Logging) và màng lọc phòng chống phân kỳ (Explosion Safeguards).

---

## Purpose

1. **Tối đa hóa thông lượng dữ liệu (Throughput & Sample Generation):** Thu hẹp khoảng cách tốc độ giữa CPU (mô phỏng vật lý môi trường) và GPU (tính toán lan truyền tiến/ngược mạng nơ-ron sâu) thông qua kỹ thuật vector hóa môi trường (`AsyncVectorEnv`), nâng thông lượng từ vài trăm lên hàng chục nghìn bước mỗi giây (Steps Per Second - SPS).
2. **Chuẩn hóa động học trực tuyến (Online Running Normalization):** Tự động chuẩn hóa vector quan sát $`s_t`$ và phần thưởng $`r_t`$ theo phân phối trung bình động (Running Mean & Variance) mà không làm rò rỉ dữ liệu tương lai, giữ ổn định gradient trong suốt hàng triệu bước tương tác.
3. **Cơ chế sao lưu phục hồi bền vững (Fault-Tolerant Checkpointing):** Định kỳ đóng gói toàn bộ trạng thái hệ thống (trọng số mô hình, trạng thái optimizer, siêu tham số, bộ nhớ đệm Replay Buffer và trạng thái RNG) cho phép tiếp tục huấn luyện ngay lập tức khi hệ thống gặp sự cố mất điện hoặc ngắt kết nối cụm tính toán.
4. **Phát hiện sớm dấu hiệu phân kỳ (Early Divergence Safeguards):** Tự động ngắt huấn luyện hoặc kích hoạt cơ chế hồi phục nếu phát hiện bùng nổ gradient, sai số Critic tăng đột biến, hoặc giá trị loss xuất hiện $`\mathrm{NaN}`$ / $`\pm\infty`$.
5. **Đảm bảo khả năng tái lập 100% (Full Reproducibility):** Khóa chặt mọi hạt giống ngẫu nhiên (RNG Seeds) của Python, NumPy, PyTorch và từng luồng môi trường worker riêng biệt.

---

## Input

Pha huấn luyện công nghiệp tiếp nhận cấu hình và tài nguyên:

1. **Bộ cấu hình huấn luyện chuẩn hóa (Training Configuration Dataclass):**
   - Tổng số bước tương tác môi trường: $`T_{\mathrm{total}} \in [10^6, 5 \times 10^7]`$.
   - Số lượng môi trường song song: $`N_{\mathrm{workers}} \in [8, 64]`$ (tối ưu theo số lõi CPU vật lý).
   - Tần suất cập nhật mạng chính/mục tiêu: $`f_{\mathrm{train}}`$.
   - Chu kỳ đánh giá kiểm định độc lập (Evaluation Interval): $`I_{\mathrm{eval}} \in [10^4, 10^5]`$ bước.
   - Chu kỳ lưu checkpoint: $`I_{\mathrm{ckpt}} \in [5 \times 10^4, 2 \times 10^5]`$ bước.
2. **Nhà máy khởi tạo môi trường (Environment Factory):** Hàm tạo môi trường tích hợp các Wrapper chuẩn hóa (Observation Normalizer, Reward Clipper, Frame Stacker).
3. **Tài nguyên phần cứng (Compute Resources):** CPU đa lõi, GPU CUDA/MPS, dung lượng RAM/VRAM định mức.
4. **Đường dẫn lưu trữ artifact:** Thư mục checkpoint, cơ sở dữ liệu TensorBoard/WandB, tệp lưu trữ siêu tham số JSON/YAML.

---

## Output

1. **Checkpoint mô hình tối ưu nhất (`best_model.pt`):** Được chọn lọc dựa trên điểm thưởng đánh giá độc lập cao nhất qua các mốc kiểm định.
2. **Checkpoint chu kỳ gần nhất (`latest_checkpoint.pt`):** Lưu trữ toàn bộ trạng thái để khôi phục (Resume capability).
3. **Luồng dữ liệu viễn thám (Telemetry Event Logs):** Dữ liệu chuỗi thời gian ghi nhận SPS, Rolling Return, Value Loss, Policy Entropy, và Gradient Norm.
4. **Báo cáo hiệu suất mẫu (Sample Efficiency Report):** Đánh giá số bước môi trường cần thiết để hệ thống đạt mốc hiệu năng mục tiêu.

---

## Evaluation metrics

1. **Phần thưởng tích lũy trung bình động (Rolling Episode Return):**

$$
\bar{R}_{\mathrm{rolling}} = \frac{1}{\min(E, 100)} \sum_{i=1}^{\min(E, 100)} G_{E - i + 1}
$$

2. **Thông lượng hệ thống (System Throughput - Steps Per Second):**

$$
\mathrm{SPS} = \frac{\Delta T_{\mathrm{env\_steps}}}{\Delta t_{\mathrm{wall\_clock}}}
$$

3. **Chuẩn Gradient trung bình (Mean Gradient Norm):**

$$
\bar{g}_{\mathrm{norm}} = \sqrt{\sum_{j} \left( \frac{\partial \mathcal{L}}{\partial \theta_j} \right)^2}
$$

4. **Tỷ lệ suy giảm thăm dò (Exploration Decay Profile):** Đường cong suy giảm $`\epsilon(t)`$ hoặc Entropy $`\mathcal{H}(t)`$.
5. **Chỉ số bất ổn định Critic (Value Loss Volatility):** Đo lường phương sai của hàm mất mát Critic giữa các batch liên tiếp.

---

## How it works

### 1. Kiến Trúc Phân Tầng Của Pipeline Huấn Luyện Sản Xuất

Hệ thống được tổ chức thành 5 module vận hành độc lập nhưng kết nối chặt chẽ:

```
+-------------------------------------------------------------------------+
|                       WORKER PROCESSES (CPU)                            |
|  Env_1  |  Env_2  |  Env_3  |  Env_4  |  Env_5  |  Env_6  |  Env_7  | Env_8  |
+------------------------------------+------------------------------------+
                                     | (Vectorized Transitions)
                                     v
+-------------------------------------------------------------------------+
|                  RUNNING NORMALIZER & BATCH WRAPPER                     |
|        s_norm = (s - mean) / std,      r_scaled = r / std_return        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   EXPERIENCE BUFFER / ROLLOUT STORAGE                   |
+------------------------------------+------------------------------------+
                                     | (Mini-Batches via PyTorch Dataloader)
                                     v
+-------------------------------------------------------------------------+
|                     DEEP RL TRAINER (GPU ENGINE)                        |
|   Forward Pass -> Compute Loss -> Clip Gradients -> Step Optimizer      |
+------------------------------------+------------------------------------+
                                     |
                                     +-------------------+
                                     |                   |
                                     v                   v
+-----------------------------------------+ +-----------------------------+
|        CHECKPOINT & RECOVERY            | |      TELEMETRY LOGGER       |
| Saves: weights, optimizer, RNG, scaler  | | TensorBoard, SPS, WandB     |
+-----------------------------------------+ +-----------------------------+
```

---

### 2. Thuật Toán Chuẩn Hóa Chạy Welford (Online Running Statistics)

Việc chuẩn hóa quan sát giúp các lớp tuyến tính của mạng nơ-ron không bị bão hòa. Thuật toán Welford tính toán trung bình và phương sai động không cần lưu trữ toàn bộ lịch sử:

$$
\mu_k = \mu_{k-1} + \frac{x_k - \mu_{k-1}}{k}
$$

$$
M_{2, k} = M_{2, k-1} + (x_k - \mu_{k-1})(x_k - \mu_k), \quad \sigma_k^2 = \frac{M_{2, k}}{k}
$$

Quan sát chuẩn hóa nạp vào mạng nơ-ron:

$$
\tilde{s}_t = \mathrm{clip}\left( \frac{s_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon_{\mathrm{norm}}}}, -10.0, 10.0 \right)
$$

---

### 3. Quy Trình Lưu Trữ Điểm Kiểm Tra Toàn Diện (Full State Checkpoint)

Một checkpoint chuẩn sản xuất bắt buộc phải lưu đủ 6 thành phần để đảm bảo tính bất biến khi tiếp tục:
1. `model_state_dict`: Trọng số của tất cả các mạng (Actor, Critic, Target Networks).
2. `optimizer_state_dict`: Các moment bậc 1 và bậc 2 của Adam/RMSprop.
3. `obs_normalizer_state`: Vector kỳ vọng $`\mu`$ và phương sai $`\sigma^2`$ tích lũy của bộ lọc trạng thái.
4. `global_step` và `total_episodes`: Bộ đếm tiến trình toàn cục.
5. `best_eval_return`: Mốc điểm số kỷ lục để so sánh lưu `best_model.pt`.
6. `rng_states`: Trạng thái hạt giống của Python `random`, `np.random` và `torch.random`.

---

## Data used

1. **TensorBoard Event Files:** Cơ sở dữ liệu chuỗi thời gian ghi nhận toàn bộ biến số vận hành (Scalars, Histograms).
2. **Metadata cấu hình:** Tệp JSON lưu trữ siêu tham số, phiên bản thư viện và cấu hình phần cứng tại thời điểm bắt đầu chạy.
3. **Môi trường giả lập chuẩn:** Gymnasium `CartPole-v1` và `LunarLander-v2`.

---

## Design

Dưới đây là module Python cấp sản xuất triển khai lớp `ProductionRLTrainer`, cấu trúc cấu hình `TrainerConfig`, bộ chuẩn hóa trực tuyến `RunningMeanStd` và quy trình lưu checkpoint khép kín.

```python
"""Module kiến trúc pha huấn luyện công nghiệp (Production Training Pipeline) cho RL:

- Online Running Statistics Normalizer (Welford's Algorithm)
- Telemetry Logging & Throughput Monitoring (SPS Engine)
- Fault-Tolerant Checkpoint Manager & Best Model Tracker
- Vòng lặp huấn luyện chuẩn sản xuất có Gradient Clipping & Explosion Safeguards
"""

from dataclasses import asdict, dataclass
import json
import os
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# 1. CẤU HÌNH HUẤN LUYỆN CHUẨN HÓA (TRAINER CONFIG)
# ==============================================================================
@dataclass
class TrainerConfig:
    """Tham số điều khiển toàn diện pha huấn luyện sản xuất."""

    exp_name: str = "rl_production_run"
    total_timesteps: int = 200000
    learning_rate: float = 3e-4
    num_envs: int = 8
    rollout_steps: int = 128
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    eval_interval: int = 20000
    eval_episodes: int = 10
    save_interval: int = 40000
    output_dir: str = "artifacts/training_checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# ==============================================================================
# 2. BỘ CHUẨN HÓA THỐNG KÊ TRỰC TUYẾN (RUNNING MEAN STD)
# ==============================================================================
class RunningMeanStd:
    """Cập nhật trung bình động và phương sai trực tuyến theo giải thuật Welford."""

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Cập nhật phân phối từ một batch dữ liệu mới."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip_range: float = 10.0) -> np.ndarray:
        """Chuẩn hóa dữ liệu đầu vào và cắt ngưỡng ngoại lai."""
        norm_x = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(norm_x, -clip_range, clip_range).astype(np.float32)


# ==============================================================================
# 3. QUẢN LÝ CHECKPOINT VÀ KHÔI PHỤC (CHECKPOINT MANAGER)
# ==============================================================================
class CheckpointManager:
    """Quản lý sao lưu, đánh giá và phục hồi trạng thái hệ thống."""

    def __init__(self, output_dir: str, exp_name: str) -> None:
        self.save_dir = os.path.join(output_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_return = float("-inf")

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        obs_normalizer: RunningMeanStd,
        is_best: bool = False,
    ) -> str:
        """Lưu trữ checkpoint định kỳ hoặc checkpoint tối ưu."""
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "obs_mean": obs_normalizer.mean,
            "obs_var": obs_normalizer.var,
            "obs_count": obs_normalizer.count,
            "best_return": self.best_return,
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
        }

        path = os.path.join(self.save_dir, f"checkpoint_step_{step}.pt")
        torch.save(state, path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(state, best_path)

        return path


# ==============================================================================
# 4. BỘ ĐIỀU PHỐI HUẤN LUYỆN SẢN XUẤT (PRODUCTION RL TRAINER)
# ==============================================================================
class ProductionRLTrainer:
    """Bộ khung điều phối toàn diện quy trình huấn luyện cấp sản xuất."""

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        state_dim: int,
    ) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.state_dim = state_dim

        # Khởi tạo bộ chuẩn hóa và checkpoint
        self.obs_normalizer = RunningMeanStd(shape=(state_dim,))
        self.ckpt_manager = CheckpointManager(config.output_dir, config.exp_name)

        # Lưu cấu hình chạy
        with open(os.path.join(self.ckpt_manager.save_dir, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=4)

        # Trạng thái viễn thám
        self.global_step = 0
        self.start_time = 0.0

    def check_explosion(self, loss_val: float, grad_norm: float) -> bool:
        """Màng lọc phát hiện sớm bất thường và bùng nổ gradient."""
        if np.isnan(loss_val) or np.isinf(loss_val):
            print(f"[CẢNH BÁO NGUY CẤP] Phát hiện Loss là NaN/Inf tại bước {self.global_step}!")
            return True
        if grad_norm > 100.0:
            print(f"[CẢNH BÁO] Gradient Norm vượt ngưỡng an toàn: {grad_norm:.2f}")
        return False

    def train_loop_simulation(self) -> None:
        """Minh họa vòng lặp huấn luyện công nghiệp khép kín."""
        self.start_time = time.time()
        print(f"Bắt đầu quy trình huấn luyện sản xuất: {self.cfg.exp_name}")
        print(f"- Thiết bị tính toán: {self.device}")
        print(f"- Mục tiêu: {self.cfg.total_timesteps:,} bước môi trường")

        step_batch = self.cfg.num_envs * self.cfg.rollout_steps

        while self.global_step < self.cfg.total_timesteps:
            # 1. Giả lập dữ liệu quan sát thu thập được từ các workers
            raw_obs = np.random.randn(step_batch, self.state_dim) * 2.0 + 1.0
            self.obs_normalizer.update(raw_obs)
            norm_obs = self.obs_normalizer.normalize(raw_obs)

            # 2. Giả lập một bước huấn luyện nơ-ron
            obs_t = torch.tensor(norm_obs, dtype=torch.float32, device=self.device)
            # Giả định dummy loss
            dummy_loss = (self.model(obs_t) ** 2).mean()

            self.optimizer.zero_grad()
            dummy_loss.backward()

            # Gradient Clipping bảo vệ mô hình
            grad_norm = float(
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            )
            self.optimizer.step()

            self.global_step += step_batch
            loss_val = float(dummy_loss.item())

            # Kiểm tra an toàn phân kỳ
            if self.check_explosion(loss_val, grad_norm):
                break

            # 3. Tính toán thông lượng (SPS)
            elapsed = time.time() - self.start_time
            sps = int(self.global_step / max(elapsed, 1e-4))

            # 4. Đánh giá và lưu Checkpoint định kỳ
            if self.global_step % self.cfg.eval_interval < step_batch:
                # Giả lập điểm thưởng đánh giá
                eval_ret = float(np.random.normal(200.0, 10.0))
                is_best = eval_ret > self.ckpt_manager.best_return
                if is_best:
                    self.ckpt_manager.best_return = eval_ret

                print(
                    f"Bước {self.global_step:>7,d} | SPS: {sps:>5d} | "
                    f"Loss: {loss_val:.4f} | Eval Return: {eval_ret:.1f} "
                    f"{'(NEW BEST!)' if is_best else ''}"
                )

                if is_best or (self.global_step % self.cfg.save_interval < step_batch):
                    saved_path = self.ckpt_manager.save_checkpoint(
                        self.global_step,
                        self.model,
                        self.optimizer,
                        self.obs_normalizer,
                        is_best=is_best,
                    )
                    print(f" -> Đã lưu checkpoint an toàn tại: {saved_path}")

        print("\n=> Hoàn tất pha huấn luyện sản xuất đạt chuẩn an toàn và thông lượng cao.")


if __name__ == "__main__":
    config = TrainerConfig(
        total_timesteps=60000,
        eval_interval=15000,
        save_interval=30000,
        output_dir="artifacts/test_runs",
    )

    # Mô hình dummy tuyến tính kiểm thử
    dummy_model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=config.learning_rate)

    trainer = ProductionRLTrainer(config, dummy_model, optimizer, state_dim=4)
    trainer.train_loop_simulation()
```

---

## Nguồn

- [1] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). *Deep reinforcement learning that matters*. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 3207-3214.
- [2] Andrychowicz, M., Raichuk, A., Stańczyk, P., Manuelli, M., et al. (2021). *What matters for on-policy deep actor-critic methods? A large-scale study*. International Conference on Learning Representations (ICLR).
- [3] Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). *Stable-Baselines3: Reliable reinforcement learning implementations in Python*. Journal of Machine Learning Research, 22(268), 1-8.
- [4] Welford, B. P. (1962). *Note on a method for calculating corrected sums of squares and products*. Technometrics, 4(3), 419-420.
- [5] Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). *Implementation matters in deep policy gradients: A case study on PPO and TRPO*. International Conference on Learning Representations (ICLR).
