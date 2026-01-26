### Reinforcement Learning on `highway-env` (Final Visual Report)

**Author:** Canberk Yalçın  \\
**Date:** Jan 2026  \\
**Repo:** Clean source + visual results (videos + training curves)

---

### Header & Visual Proof (Evolution)

**Required content:** Each evolution video shows exactly three stages—**sequence** or **side-by-side**:

1. **Untrained Agent:** acting randomly / failing immediately  
2. **Half-Trained Agent:** learning but still making mistakes (e.g. driving longer, then crash)  
3. **Fully Trained Agent:** successfully solving the task  

We **record** these three stages per env, then **stitch** them via `make_evolution_video.py` (see Reproducibility).

#### Merge (`merge-v0`) — Evolution (3 stages)

**Embedded GIF (Untrained → Half-Trained → Fully Trained, sequence):**

<img src="assets/videos/merge-v0_evolution.gif" width="960" alt="Merge — Evolution (Untrained → Half-Trained → Fully Trained)" />

**MP4:** `assets/videos/merge-v0_evolution.mp4` (sequence or side-by-side via `--layout`)

#### Other environments — evolution MP4s

- **Intersection:** `assets/videos/intersection-v1_evolution.mp4`
- **Parking:** `assets/videos/parking-v0_evolution.mp4`
- **Roundabout:** `assets/videos/roundabout-v0_evolution.mp4`

> Convert MP4s to GIF and embed similarly if you want inline evolution for other envs.

---

### Methodology

#### The Math — Custom Reward Function (Highway example)

From `config/highway.yaml`, the environment reward is shaped by:

\[
R_t = R_{\text{collision}} + R_{\text{speed}} + R_{\text{lane\_change}} + R_{\text{right\_lane}}
\]

With the configured weights:
- **Collision penalty:** \(R_{\text{collision}} = -3.0\)
- **High-speed reward:** \(R_{\text{speed}} = 0.6\) within \(v \in [20, 35]\)
- **Lane-change penalty:** \(R_{\text{lane\_change}} = -0.1\)
- **Right-lane reward:** \(R_{\text{right\_lane}} = 0.0\)

#### The Model — Algorithms, Hyperparameters, Network

This repo trains SB3 agents per scenario via `config/*.yaml` and `src/agents/sb3_manager.py`.

- **Algorithms used**
  - **PPO**: `highway-v0`, `merge-v0`, `roundabout-v0`, `racetrack-v0`
  - **DQN**: `intersection-v1`
  - **SAC (+ HER replay buffer)**: `parking-v0`

- **Example hyperparameters (PPO / Highway)**
  - **learning_rate**: `5e-4` (**linear schedule** in code)
  - **n_steps**: `2048`
  - **batch_size**: `64`
  - **n_epochs**: `10`
  - **gamma**: `0.95`
  - **clip_range**: `0.2`
  - **ent_coef**: `0.01`

- **Neural net architecture**
  - PPO/DQN use SB3 **`MlpPolicy`** (MLP)
  - DQN’s `intersection.yaml` explicitly sets `policy_kwargs.net_arch: [256, 256]`

---

### Training Analysis

#### The Graph — Reward vs Episodes (Merge PPO)

![Merge reward curve](assets/tb/merge-v0/rollout__ep_rew_mean__PPO_1.png)

#### The Commentary (Graph Analysis)

- **Early training**: reward is low/unstable because the policy is effectively near-random and collisions/inefficient actions dominate.
- **Mid training**: reward rises as the agent learns stable lane-keeping + merges more consistently (in `merge.yaml`, lane-change is **not** penalized to allow merging).
- **Later training**: the curve stabilizes, suggesting the policy converges and improvements become incremental.

#### Additional labeled curves (Merge PPO)

![Merge episode length](assets/tb/merge-v0/rollout__ep_len_mean__PPO_1.png)
![Merge training loss](assets/tb/merge-v0/train__loss__PPO_1.png)

---

### Challenges & Failures (Narrative)

#### Issue: “Agent keeps changing lanes / unstable driving” (Highway)

- **Symptom**: excessive lane switching (“zig-zag”) increases collision risk and reduces reward consistency.
- **Fix**: introduce a small **lane-change penalty** (`lane_change_reward: -0.1` in `config/highway.yaml`) to discourage unnecessary lane changes while still allowing overtakes.
- **Outcome**: smoother trajectories and more stable reward improvements over training.

---

### Reproducibility (How to run)

**1. Record the three stages** (Untrained, Half-Trained, Fully Trained) per environment:

```bash
pip install -r requirements.txt

# Train (saves untrained, half-trained at 50%, and fully trained)
python3 main.py --env merge --mode train

# Record 3-stage videos: 1_untrained, 2_half_trained, 3_fully_trained → logs/videos/<env>/
python3 main.py --env merge --mode visualize
```

**2. Stitch** them into a single evolution video (sequence or side-by-side):

```bash
# Sequence (uç uca): Untrained → Half-Trained → Fully Trained
python3 scripts/make_evolution_video.py --all

# Side-by-side (3 panels)
python3 scripts/make_evolution_video.py --all --layout side-by-side
```

Output: `assets/videos/<env_id>_evolution.mp4`.

**3. Optional:** Export TensorBoard scalars to PNG, convert MP4→GIF for README:

```bash
python3 scripts/export_tb_report.py --logdir logs/tensorboard --outdir assets
# e.g. ffmpeg -i assets/videos/merge-v0_evolution.mp4 -vf "fps=12,scale=960:-1" ... merge-v0_evolution.gif
```

