# 🚗 Highway Environment — Reinforcement Learning Final Report

**Author:** Canberk Yalçın, Bahar Gencer, Ali Sokullu
---

## 📹 Evolution Videos

### Merge (merge-v0) — PPO
![Video Placeholder](https://github.com/user-attachments/assets/c39dfeb0-09c6-41e3-aedd-9978d4bf9040)

### Intersection (intersection-v1) — DQN
![Video Placeholder](https://github.com/user-attachments/assets/87489532-40b7-43e8-a007-6e1d73ec757a)

### Parking (parking-v0) — SAC+HER
![Video Placeholder](https://github.com/user-attachments/assets/90f96ec3-6e31-4d7e-8e75-82249a405d7a)

### Racetrack (racetrack-v0) — PPO
![Video Placeholder](https://github.com/user-attachments/assets/3cb6acca-7b86-4382-98fd-4ccbf14d3b95)

---

## Methodology

## Merge Reward Function

The reward function is designed to encourage the agent to merge safely into traffic and maintain high speed, without penalizing necessary lane changes.

$$
R_{t} = R_{collision} + R_{speed} + R_{lane\_change}
$$

* **$R_{collision}$**: Terminal penalty for colliding with other vehicles.
* **$R_{speed}$**: Reward for driving at high speed (mapped linearly to the speed range $[20, 30]$ m/s).
* **$R_{lane\_change} = 0$**: **CRITICAL.** No penalty is applied for lane changes to allow the agent to freely merge into the highway traffic.
* **$R_{right\_lane} \approx 0$**: Unlike highway driving, keeping to the rightmost lane is not strictly enforced during the merging phase.

**State:** Kinematics (List of 5 observed vehicles $\times$ 7 features: `[presence, x, y, vx, vy, cos_h, sin_h]`)
**Actions:** DiscreteMetaAction (5 discrete actions: `[LANE_LEFT, LANE_RIGHT, IDLE, FASTER, SLOWER]`)

---

## Intersection Reward Function

The reward function is designed to teach the agent to navigate a busy intersection, negotiate with other drivers, and reach the specific destination **"o1"**.

$$
R_{t} = R_{collision} + R_{arrived} + R_{speed}
$$

* **$R_{collision} = -5$**: Penalty for colliding with other vehicles. (Note: `offroad_terminal` is false, allowing the agent to recover from minor errors if possible).
* **$R_{arrived} = +10$**: Significant reward for successfully exiting the intersection at the correct destination (o1).
* **$R_{speed}$**: Reward mapped linearly to the speed range $[0, 9]$ m/s. Even low speeds are acceptable to encourage patience (waiting for a gap), unlike highway scenarios.

### State & Action Space

* **Observation (State):** `Kinematics` (Flattened Vector)
    * **Capacity:** Observes the **15 closest vehicles** to eliminate blind spots.
    * **Features:** `[presence, x, y, vx, vy, cos_h, sin_h]` (Absolute coordinates).
    * **Intentions:** `observe_intentions: true` is enabled, allowing the agent to infer if other cars are turning or going straight.
    
* **Actions:** `DiscreteMetaAction`
    * Combined **Longitudinal** (Speed) and **Lateral** (Steering) control.
    * **Speeds:** `[0, 4.5, 9]` (Allows full stop for yielding, slow approach, and fast crossing).

### Environment Dynamics
* **Duration:** 25 seconds (Extended to allow "patience" at the intersection).
* **Traffic Density:** High (Spawn probability 0.6, Initial count 10).

---

## Parking Reward Function

The reward function is defined as a weighted $L^1$-norm distance to the goal, plus specific bonuses for successful parking and penalties for collisions.

$$
R(s, a) = - \| s - g \|_{W} + R_{collision} + R_{success}
$$

* **$- \| s - g \|_{W}$**: The dense reward component. It minimizes the weighted error between the current state $s$ and the goal state $g$.
    * **Weights ($W$):** `[1, 0.3, 0.0, 0.0, 0.02, 0.02]` applied to `[x, y, vx, vy, cos_h, sin_h]`.
    * **Interpretation:** The agent prioritizes **longitudinal position ($x$)** heavily, followed by lateral position ($y$). It ignores velocity errors (0.0) during the approach but cares slightly about heading alignment (0.02).
* **$R_{collision} = -5.0$**: Penalty for colliding with obstacles or other parked cars.
* **$R_{success} = +0.12$**: Bonus reward when the agent successfully parks within the tolerance limits.

### State & Action Space

* **Observation:** `KinematicsGoal`
    * **Type:** Dict (Observation + Desired Goal + Achieved Goal).
    * **Features:** `[x, y, vx, vy, cos_h, sin_h]`.
    * **Normalization:** `False` (Raw coordinates in meters).
* **Actions:** `ContinuousAction` (Throttle/Brake + Steering).
* **Technique:** **HER (Hindsight Experience Replay)** is used to replay failed episodes as if the achieved state was the intended goal, drastically speeding up convergence.

## Racetrack Reward Function

The reward function is designed to train a race car driver that prioritizes safety (staying on track) and precision (lane centering) over aggressive maneuvers.

$$
R_{t} = R_{collision} + R_{centering} + R_{lane\_change} + R_{action}
$$

* **$R_{collision} = -1000$**: Terminal penalty for going off-road or crashing. Since `terminate_off_road: true`, the episode ends immediately.
* **$R_{centering}$**: Dense reward for maintaining the ideal racing line (center of the lane).
    * Defined by `lane_centering_reward: 1` and a high sensitivity `lane_centering_cost: 4`.
* **$R_{lane\_change} = -0.05$**: Small penalty to discourage unnecessary weaving or zigzagging on straight roads.
* **$R_{action} = -100$**: High penalty for high-magnitude steering actions. This forces the agent to drive very smoothly and avoid jerky steering inputs.

### State & Action Space

* **Observation:** `OccupancyGrid`
    * **Dimensions:** $12 \times 12$ Grid (Calculated from range $\pm18$ and step $3$).
    * **Features:** 11 channels (`presence`, `on_road`, `x`, `y`, `vx`, `vy`, `cos_h`, `sin_h`, `long_off`, `lat_off`, `ang_off`).
    * **Total Shape:** $(12, 12, 11)$.
* **Actions:** `ContinuousAction`
    * **Lateral:** True (Steering control).
    * **Longitudinal:** False (Speed is fixed/automatic based on `target_speeds`).


---

## Algorithms & Hyperparameters

### Merge - PPO


* **Algorithm (PPO):** Chosen over DQN for its stability in high-risk environments. PPO's clipped objective prevents catastrophic policy updates during collisions, which is crucial for the dynamic physics of merging.
* **Reward Shaping (`lane_change_reward: 0`):** We removed the standard lane-change penalty. Since the task's primary goal is to merge, penalizing lateral movement would create a conflicting objective and cause the agent to hesitate.
* **Discount Factor (`gamma: 0.9`):** Reduced from standard 0.99 to 0.9. Merging is a short-horizon task (30s duration); this forces the agent to prioritize immediate gaps and survival rather than long-term planning.
* **Entropy (`ent_coef: 0.01`):** A small entropy bonus is added to prevent the agent from getting stuck in a local optimum of "staying in the acceleration lane" to avoid risk.


| Param | Value |
|-------|-------|
| Learning Rate | 0.0003 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs| 10 |
| gamma | 0.9 |
| gae_lambda |0.95 |
| clip_range | 0.2 |
| Total Steps | 200,000 |
| Parallel Envs | 8 |

### Intersection - DQN

* **Algorithm (DQN):** Deep Q-Network was selected because intersection navigation is fundamentally a discrete decision-making process (Go, Yield, Stop). DQN is highly effective at mapping these distinct actions to Q-values in environments with clear success/failure states.
* **Observation (`observe_intentions: true`):** **CRITICAL.** We enabled intention observation to make the environment fully observable. Without knowing whether other vehicles plan to turn or go straight, the agent cannot learn to yield correctly, leading to unavoidable collisions.
* **Duration (25s):** Increased from the default to 25 seconds. Intersection safety often requires patience (waiting for a gap). A short duration forces the agent to be aggressive to avoid timeout penalties; extending it allows for safe, defensive driving behaviors.
* **Reward Shaping:** We assigned a high `arrived_reward` (+10) relative to the collision penalty (-5). Since reaching the specific destination ("o1") is a sparse event, a strong positive signal is necessary to drive exploration towards the goal.


| Param | Value |
|-------|-------|
| net_arch | [256, 256] |
| Learning Rate | 0.0005 |
| buffer_size | 50,000 |
| batch_size | 32 |
| gamma | 0.99 |
| learning_starts | 1000 |
| train_freq | 4 |
| gradient_steps | 1 |
| target_update_interval | 1000 |
| exploration_fraction | 0.3 |
| exploration_initial_eps | 1.0 |
| exploration_final_eps | 0.5 |
| Total Steps | 300,000 |

### Parking - SAC+HER

* **Algorithm (SAC - Soft Actor-Critic):** SAC was chosen because parking requires precise, **continuous control** of steering and acceleration. Unlike PPO, SAC is an off-policy algorithm that is highly sample-efficient, making it ideal for learning fine-motor tasks like parking where small adjustments matter.
* **Technique (HER - Hindsight Experience Replay):** **CRITICAL.** Parking is a "goal-conditioned" task with sparse rewards (it is difficult to hit the exact spot by chance). HER allows the agent to learn from failure by treating achieved states as if they were the intended goals, drastically speeding up convergence.
* **Reward Weights (`x:1.0` vs `y:0.3`):** We customized the reward function to prioritize longitudinal alignment ($x$) over lateral offset ($y$). This encourages the agent to enter the parking spot deeply rather than just hovering near the side.
* **Hyperparameter (`learning_starts: 1000`):** Added to ensure the replay buffer is populated with enough random transitions before the gradient updates begin. This stabilizes the initial training phase for the SAC critic networks.


| Param | Value |
|-------|-------|
| net_arch | [256, 256, 256] |
| Learning Rate | 0.004 |
| buffer_size | 500,000 |
| batch_size | 256 |
| gamma | 0.95 |
| tau | 0.05 |
| Total Steps | 100,000 |

### Racetrack - PPO

* **Algorithm (PPO):** Selected for its ability to handle **continuous action spaces** (steering angles) effectively. PPO provides smoother policy updates compared to value-based methods, which is crucial for maintaining control at high speeds.
* **Observation (Occupancy Grid):** Chosen over simple kinematics to give the agent **spatial awareness**. The 12x12 grid allows the agent to "see" upcoming curves and track boundaries, enabling proactive rather than reactive steering.
* **Reward Shaping (Action Penalty -100):** A high penalty on steering magnitude was added to force **smooth driving** and prevent unrealistic oscillating (zig-zag) behaviors.
* **Safety First:** The strict collision penalty (-1000) combined with `terminate_off_road: true` ensures the agent prioritizes staying on track as the absolute prerequisite.

| Param | Value |
|-------|-------|
| net_arch | pi: [256, 256] vf: [256, 256] |
| Learning Rate | 0.0003 |
| n_steps | 1024 |
| batch_size | 64 |
| gamma | 0.99 |
| Total Steps | 300,000 |
|gae_lambda| 0.95 |
|clip_range| 0.2|
|ent_coef|0.0|
---

## Training Analysis

## 📈 Training Analysis: Merge Scenario

### 1. Episode Length (Survival) Analysis
**The Graph:**

<img width="1424" height="700" alt="rollout__ep_len_mean__PPO_1" src="https://github.com/user-attachments/assets/a3f5333d-110d-402a-b581-aab4a228d5f8" />

**The Commentary:**
The graph above illustrates the mean episode length over 200,000 timesteps. In the `merge-v0` environment, a short episode typically indicates an immediate collision.
* **Early Phase (0 - 50k steps):** We observe a sharp incline from ~8 steps to ~15 steps. This indicates that the agent quickly learned the fundamental safety constraints: "Do not crash immediately." The removal of the `lane_change_reward` penalty allowed the agent to explore lateral movements freely without negative feedback.
* **Convergence (125k+ steps):** The curve plateaus around 17 steps. This stability suggests the agent has mastered the merging maneuver and consistently completes the scenario without terminating early due to accidents. The steadiness of the plateau confirms that PPO found a robust policy.

### 2. Value Loss Analysis
**The Graph:**

<img width="1423" height="700" alt="train__value_loss__PPO_1" src="https://github.com/user-attachments/assets/dae74ada-9aa2-4738-9598-167fda1e4f8c" />

**The Commentary:**
The Value Loss graph represents the error between the Critic network's predicted return and the actual return.
* **Rapid Learning:** The loss drops significantly in the first 75,000 steps (from 1.4 to 0.2). This correlates perfectly with the increase in episode length, showing that as the agent survived longer, the Critic became much better at estimating the value of states.
* **Stability:** The loss stabilizes near 0.1 after 125,000 steps, indicating that the training has converged and further training would yield diminishing returns. This validates our choice of 200,000 total timesteps as an optimal stopping point.

---


## 📈 Training Analysis: Intersection Scenario (DQN)

### 1. Reward Mean Analysis (Strategy Shift)
**The Graph:**

<img width="1424" height="700" alt="rollout__ep_rew_mean__DQN_1" src="https://github.com/user-attachments/assets/498dffac-4ffd-45e6-afd8-1e0223edcec7" />


**The Commentary:**
The graph above displays the mean reward over 300,000 timesteps using the DQN algorithm.
* **The "Safety Peak" (0 - 50k steps):** We observe a rapid increase in rewards, peaking around 5.5. During this phase, the agent likely discovered a "conservative" strategy—such as waiting at the intersection or moving very slowly to avoid collisions at all costs. This maximizes the survival time but does not necessarily solve the traffic flow problem efficiently.
* **The Correction (50k - 70k steps):** There is a noticeable drop in rewards. This typically occurs when the exploration rate ($\epsilon$) decays, forcing the agent to rely on its learned Q-values rather than random safe actions. The agent effectively "broke out" of the local optimum of simply waiting and started attempting to navigate the intersection more aggressively.
* **Stability (75k - 300k steps):** The reward stabilizes and oscillates around 4.0. This indicates the agent found a sustainable balance between **risk (crossing)** and **safety (yielding)**. It is no longer "gaming" the system by waiting indefinitely but is actively engaging with the traffic.

### 2. Episode Length Analysis (Efficiency vs. Patience)
**The Graph:**

<img width="1424" height="700" alt="rollout__ep_len_mean__DQN_1" src="https://github.com/user-attachments/assets/5e4a559c-4e20-4492-bb09-5803419ecdef" />


**The Commentary:**
The episode length graph correlates strongly with the reward curve and explains the behavior shift.
* **The "Waiting Game" (0 - 50k steps):** The episode length rises drastically to ~22 steps. This confirms our hypothesis that the agent initially learned to wait for long periods to accumulate "existence rewards" (or avoid negative collision rewards).
* **Efficiency Optimization (70k+ steps):** As the reward stabilized, the average episode length dropped to ~14-15 steps. **This is a positive outcome.** It signifies that the agent learned to cross the intersection more efficiently (faster), reducing the time spent blocking traffic, which is the ultimate goal of an autonomous intersection manager.

---

## 📈 Training Analysis: Parking Scenario (SAC + HER)

### 1. Reward Mean Analysis (The Learning Curve)
**The Graph:**

<img width="1424" height="700" alt="rollout__ep_rew_mean__SAC_1" src="https://github.com/user-attachments/assets/54b1452f-ddc0-498b-ba21-6131c3cbdccb" />

**The Commentary:**
The graph above illustrates the mean reward per episode over the full 400,000 training timesteps.
* **The "Exploration Valley" (0 - 10k steps):** We observe a sharp, deep drop in rewards (reaching nearly -90). This occurs during the `learning_starts` phase. The agent acts randomly, often driving away from the target, accumulating large negative distance penalties. This "dip" is a characteristic signature of distance-based dense rewards during initial exploration.
* **The "HER Effect" (10k - 50k steps):** Following the dip, there is a near-vertical recovery. This demonstrates the power of **Hindsight Experience Replay**. Even though the agent failed in the early episodes, HER relabeled those failures as successes for virtual goals, allowing the agent to learn the vehicle dynamics and steering control incredibly fast.
* **Convergence & Stability (250k - 400k steps):** The reward stabilizes around -5.0 to -6.0. Given that the maximum theoretical reward is close to 0 (perfect overlap), this indicates the agent has converged to an optimal policy, consistently parking the vehicle with high precision.

### 2. Episode Length Analysis (Efficiency)
**The Graph:**

<img width="1424" height="700" alt="rollout__ep_len_mean__SAC_1" src="https://github.com/user-attachments/assets/36ff5bcd-7a80-4724-940e-394eda24d5c3" />


**The Commentary:**
The episode length graph correlates perfectly with the reward curve, showing the agent's efficiency.
* **The Struggle (0 - 20k steps):** Initially, the episode length spikes to ~180 steps. This indicates the agent was struggling to find the parking spot or stabilize the vehicle, often running until the time limit.
* **Optimization Phase (20k - 250k steps):** We see a steady decline in steps. As the agent masters the "parking maneuver" (switching from forward to reverse gear correctly), it minimizes unnecessary movements.
* **Optimal Efficiency (300k+ steps):** The length settles at approximately **20 steps**. This is a critical result. It proves the agent is no longer just "stumbling" into the parking spot; it is driving directly to the coordinates and stopping immediately, achieving the goal with minimum time and fuel consumption.

  
## 📈 Training Analysis: Racetrack Scenario (PPO)

### 1. Episode Length Analysis (Survival & Lane Keeping)
**The Graph:**

<img width="1424" height="700" alt="rollout__ep_len_mean__PPO_1" src="https://github.com/user-attachments/assets/3054a15d-844b-450f-ab21-70e90f5e3de7" />

**The Commentary:**
The graph above illustrates the mean episode length over 300,000 timesteps. In the Racetrack environment where `terminate_off_road: true` is enabled, episode length is a direct proxy for driving capability.
* **The "Crash Phase" (0 - 25k steps):** The episode length starts very low (< 50 steps). This indicates the agent is crashing or going off-road almost immediately after the start. It has not yet correlated the Occupancy Grid features with the track boundaries.
* **The Learning Climb (25k - 200k steps):** We observe a near-linear increase in survival time. This is a strong indicator that the PPO algorithm is effectively optimizing the policy. The agent is learning to steer correctly to stay within the lanes, avoiding the -1000 collision penalty.
* **Mastery & Plateau (200k+ steps):** The curve peaks and stabilizes around **480 steps**. This suggests the agent has mastered the track layout and can consistently drive for the maximum allowed duration without crashing.

### 2. Value Loss Analysis (Critic Stability)
**The Graph:**

<img width="1424" height="700" alt="train__value_loss__PPO_1" src="https://github.com/user-attachments/assets/53c52272-cffd-40de-9d0b-271cbfc52245" />

**The Commentary:**
The Value Loss graph shows the error of the Critic network in estimating expected returns.
* **Initial Shock (0 - 25k steps):** There is a sharp spike in loss (up to ~10.0). This is expected due to the sparse and massive penalty of -1000 for collisions. Initially, the Critic struggles to predict whether a state will lead to a sudden catastrophic end.
* **Convergence (50k - 175k steps):** As the agent learns to stay on track (correlated with the Episode Length graph), the Critic's prediction error drops drastically.
* **Stability (200k+ steps):** The loss converges to near zero (< 1.0). This confirms that the Critic has accurately mapped the state space (Occupancy Grid) to the expected rewards, providing stable gradients for the Actor network.

---

## ⚠️ Challenges & Failures

Reinforcement Learning is notoriously sensitive to hyperparameter tuning and reward shaping. During the development of this project, we encountered several significant challenges that required specific architectural and algorithmic adjustments.

### 1. The "Hesitant Merger" Problem (Merge Scenario)
* **The Failure:** In early PPO training runs, the agent learned a "fear-based" policy. Instead of merging into the highway, it would simply stop at the end of the acceleration lane to avoid the risk of collision, resulting in a local optimum.
* **The Cause:** The default penalty for lane changing ($R_{lane\_change} < 0$) created a conflict. The agent was punished for the very action it was supposed to perform.
* **The Solution:** We set `lane_change_reward: 0`. By removing the penalty for lateral movement, the agent felt free to explore gaps in traffic without negative feedback, leading to successful merging behavior.

### 2. The "Blind Turn" Dilemma (Intersection Scenario)
* **The Failure:** The DQN agent frequently crashed into oncoming vehicles while trying to turn left. It treated all oncoming cars as if they were going straight.
* **The Cause:** The observation space initially lacked *intention* data. The agent could see the position of other cars but not their future path (turn signals).
* **The Solution:** We enabled `observe_intentions: true` in the environment config. This expanded the state space to include the turn signals of other vehicles, allowing the agent to distinguish between safe and unsafe gaps.

### 3. The "Exploration Valley" (Parking Scenario)
* **The Failure:** Using standard SAC, the agent struggled to find the parking spot. For the first 50,000 steps, the rewards remained at the minimum (-100) because the likelihood of randomly driving into a specific parking spot is extremely low (Sparse Reward Problem).
* **The Solution:** We implemented **Hindsight Experience Replay (HER)**. HER allowed the agent to learn from failure by treating the final state of a failed episode as if it were the intended goal. This transformed the sparse reward problem into a dense one, causing a massive spike in learning efficiency (as seen in the training graphs).

### 4. The "Oscillating Driver" (Racetrack Scenario)
* **The Failure:** The agent learned to stay on the road but exhibited violent steering behavior (rapidly zig-zagging left and right).
* **The Cause:** The agent was trying to maximize the `lane_centering_reward` instantly at every step, overcorrecting continuously.
* **The Solution:** We introduced a high penalty for steering magnitude (`action_reward: -100`) and tuned the PPO `clip_range`. This forced the policy to favor smooth, continuous steering actions over jerky movements, resembling a professional racing line.


## 🔄 Reproducibility

### 📦 Prerequisites

- Python 3.11+
- pip or conda
- FFmpeg (optional, for video recording)

### 🚀 Installation

```bash
# Clone repository
git clone https://github.com/canberkylcn/Highway-Environment.git
cd Highway-Environment

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt
```

### 🎯 Usage

#### Training
```bash
python3 main.py --env merge --mode train
python3 main.py --env intersection --mode train
python3 main.py --env parking --mode train
python3 main.py --env racetrack --mode train
```

**Training Time:** ~2-3 hours (8 parallel envs, CPU)

#### Testing
```bash
python3 main.py --env merge --mode test
python3 main.py --env intersection --mode test
python3 main.py --env parking --mode test
python3 main.py --env racetrack --mode test
```

#### Visualization (Generate 3-stage videos)
```bash
python3 main.py --env merge --mode visualize
python3 main.py --env intersection --mode visualize
python3 main.py --env parking --mode visualize
python3 main.py --env racetrack --mode visualize
```

Videos saved to: `logs/videos/{env_id}/`

### 📊 Monitor Training

```bash
tensorboard --logdir logs/tensorboard
```

Open browser to `http://localhost:6006`

### 📈 Export Graphs as PNG

```bash
python scripts/export_tb_report.py --logdir logs/tensorboard --outdir assets
```
#### Create Side-by-Side Evolution Video

Generate a single video showing all 3 stages side-by-side:

```bash
# For all environments
python scripts/make_evolution_video.py --all --layout side-by-side

# For specific environment
python scripts/make_evolution_video.py --env-id merge-v0 --layout sequence
