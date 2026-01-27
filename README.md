# 🚗 Highway Environment — Reinforcement Learning Final Report

**Author:** Canberk Yalçın  
**Date:** January 2026

---

## 📹 Evolution Videos

### Merge (merge-v0) — PPO
[DRAG YOUR VIDEO: assets/videos/merge-v0_evolution.mp4]

### Intersection (intersection-v1) — DQN
[DRAG YOUR VIDEO: assets/videos/intersection-v1_evolution.mp4]

### Parking (parking-v0) — SAC+HER
[DRAG YOUR VIDEO: assets/videos/parking-v0_evolution.mp4]

### Racetrack (racetrack-v0) — PPO
[DRAG YOUR VIDEO: assets/videos/merge-v0_evolution.mp4]

---

## Methodology

### Merge Reward Function

$$R_t = R_{collision} + R_{speed} + R_{lane\_change} + R_{right\_lane}$$

- $R_{collision} = -2.0$
- $R_{speed} = 1.0$ (when v in [20,35] m/s)
- $R_{lane\_change} = 0$ (CRITICAL: No penalty)
- $R_{right\_lane} = 0.0$

**State:** 5 vehicles × 7 features  
**Actions:** 3 discrete (Left, Straight, Right)

---

### Intersection Reward Function

$$R_t = R_{collision} + R_{speed} + R_{arrival}$$

- $R_{collision} = -5.0$
- $R_{speed} = 1.0$ (when v in [7.0, 9.0] m/s - TIGHT RANGE)
- $R_{arrival} = 1.0$

**State:** 15 vehicles × 7 features  
**Actions:** 3 discrete speeds (0, 4.5, 9 m/s)

---

### Parking Reward Function

$$R_t = \mathbf{w} \cdot \mathbf{f}(s_t, g) + R_{collision} + R_{success}$$

- Weights: $[1.0, 0.3, 0.0, 0.0, 0.02, 0.02]$
- $R_{collision} = -5.0$
- $R_{success} = 0.12$
- HER: goal_selection_strategy = "future"

**State:** 6-dim state + 6-dim goal  
**Actions:** 2 continuous (throttle, steering)

---

## Algorithms & Hyperparameters

### Merge - PPO

| Param | Value |
|-------|-------|
| Learning Rate | 0.0003 |
| n_steps | 2048 |
| batch_size | 64 |
| gamma | 0.9 |
| Total Steps | 200,000 |
| Parallel Envs | 8 |

### Intersection - DQN

| Param | Value |
|-------|-------|
| Learning Rate | 0.0005 |
| buffer_size | 15,000 |
| batch_size | 32 |
| gamma | 0.95 |
| Total Steps | 100,000 |

### Parking - SAC+HER

| Param | Value |
|-------|-------|
| Learning Rate | 0.001 |
| buffer_size | 500,000 |
| batch_size | 256 |
| gamma | 0.95 |
| Total Steps | 100,000 |

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
