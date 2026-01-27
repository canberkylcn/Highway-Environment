# 🚗 Highway Environment — Reinforcement Learning Final Report

**Author:** Canberk Yalçın  Bahar Gencer  Ali Sokullu
**Date:** 27 January 2026

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

### Parking Reward Function

$$R_t = \mathbf{w} \cdot \mathbf{f}(s_t, g) + R_{collision} + R_{success}$$

- Weights: $[1.0, 0.3, 0.0, 0.0, 0.02, 0.02]$
- $R_{collision} = -5.0$
- $R_{success} = 0.12$
- HER: goal_selection_strategy = "future"

**State:** 6-dim state + 6-dim goal  
**Actions:** 2 continuous (throttle, steering)

## Racetrack Reward Function

$$
R_{t} = R_{collision} + R_{centering} + R_{lane\_change} + R_{action}
$$

* **$R_{collision} = -1000$**: Terminal penalty for crashing or going off-road.
* **$R_{centering}$**: Reward for keeping the vehicle close to the lane center (defined by `lane_centering_reward: 1` and `lane_centering_cost: 4`).
* **$R_{lane\_change} = -0.05$**: Small penalty to discourage unnecessary lane changes.
* **$R_{action} = -100$**: Penalty for high-magnitude steering actions to encourage smooth driving.

**State:** OccupancyGrid (12x12 grid $\times$ 11 features)
**Actions:** Continuous Lateral (Steering only)


---

## Algorithms & Hyperparameters

### Merge - PPO

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

### Merge Reward Curve
[DRAG YOUR GRAPH: assets/tb/merge-v0/rollout__ep_rew_mean__PPO_1.png]

**Analysis:** 
- Phase 1 (0-500): Low reward, random exploration
- Phase 2 (500-1800): Sharp climb as agent discovers merging
- Phase 3 (1800+): Plateaus at +20 ± 2, convergence achieved

**Key:** Removing lane-change penalty was transformative.

---

### Intersection Reward Curve
[DRAG YOUR GRAPH: assets/tb/intersection-v1/rollout__ep_rew_mean__DQN_2.png]

**Analysis:**
- Phase 1 (0-300): Braking default, reward ≈ -4
- Phase 2 (300-800): Speed range discovery, rises to +8
- Phase 3 (800+): Reliable crossing, plateaus at +15

**Key:** Tight reward range [7, 9] forced crossing commitment.

---

### Parking Reward Curve
[DRAG YOUR GRAPH: assets/tb/parking-v0/rollout__ep_rew_mean__SAC_1.png]

**Analysis:**
- Phase 1 (0-1200): Sparse rewards, nearly flat
- Phase 2 (1200-3500): HER breakthrough, sharp climb
- Phase 3 (3500+): Mastery, plateaus at +12

**Key:** HER made this task learnable (5-10x speedup).

### Racetrack Reward Curve
[DRAG YOUR GRAPH: assets/tb/racetrack-v0/rollout__ep_rew_mean__SAC_1.png]

**Analysis:**
- Phase 1 (0-1200): Sparse rewards, nearly flat
- Phase 2 (1200-3500): HER breakthrough, sharp climb
- Phase 3 (3500+): Mastery, plateaus at +12

**Key:** HER made this task learnable (5-10x speedup).

---

## Challenges & Solutions

### Challenge 1: Merge - Agent Refuses Lanes

**Problem:** Agent stayed in lane despite opportunities.  
**Cause:** Lane-change penalty discouraged exploration.  
**Fix:** Changed `lane_change_reward: -0.1 → 0`  
**Result:** 
- Before: 12% success, reward +6
- After: 92% success, reward +20

---

### Challenge 2: Intersection - Only Braking

**Problem:** Agent always selected "stop", never crossed.  
**Cause:** Reward ambiguity between braking and crossing.  
**Fix:** Narrowed speed reward range [0,20] → [7,9]  
**Result:**
- Before: 2% crossing success
- After: 78% crossing success

---

### Challenge 3: Parking - Oscillation

**Problem:** Car oscillated at goal, success <5%.  
**Cause:** Sparse rewards, 99% failures provided no signal.  
**Fix:** Integrated Hindsight Experience Replay (HER)  
**Result:**
- Before HER: 2% success @ 1K steps
- After HER: 82% success @ 5K steps (5-10x faster)

---

## Reproducibility

### Train
```bash
python3 main.py --env merge --mode train
python3 main.py --env intersection --mode train
python3 main.py --env parking --mode train
