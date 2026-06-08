# ShuttleNet: A Dual-Encoder Transformer for Table Tennis Shot and Rally Prediction

**National Cheng Kung University · AIdea Competition — Table Tennis Tactics & Outcome Prediction · 2026**

---

## Introduction

**Goal**
Predict the next shot's action type and landing point, plus the final rally outcome, from a sequence of table tennis strokes.

**Task Definition**
Given shots 1 … n−1, predict:
1. `actionId` — shot type (19 classes, 40% weight)
2. `pointId` — landing zone (10 classes, 40% weight)
3. `serverGetPoint` — rally winner (binary, 20% weight)

**Scoring Formula**

$$\text{Score} = 0.4 \times \text{Macro-F1}_{act} + 0.4 \times \text{Macro-F1}_{pt} + 0.2 \times \text{AUC}$$

**Competition Result**
> Leaderboard score: **0.3318** | Rank: **168 / 423**

---

## Methods

### 1. Feature Engineering

**13 Categorical Input Features (per stroke):**
`sex`, `handId`, `strengthId`, `spinId`, `pointId`, `actionId`, `positionId`, `strikeId`, `strikeNumber`, `actionType`, `prev_actionType`, `prev_actionId`, `prev_pointId`

**Derived Lag Features:**
- `actionType` — mapped from `actionId` (0=Zero, 1=Serve, 2=Attack, 3=Control, 4=Defensive)
- `prev_actionId`, `prev_actionType`, `prev_pointId` — previous stroke context

**Spatial Coordinate Map:**
Each `pointId` maps to a 2-D coordinate (X: backhand→forehand, Y: short→long), used as an auxiliary regression target. `pointId=0` (miss) is excluded from coordinate loss.

---

### 2. ShuttleNet Architecture

**Dual-Encoder Design**

```
Input: x₁, …, xₙ₋₁  →  13 embeddings (emb_dim=32) → concat (416-dim)
   ├─ proj_shot  → causal_pos → CausalEncoder (3L, Pre-LN) → act_head / pt_head
   └─ proj_rally → bidi_pos   → BidiEncoder   (1L, Pre-LN) → mean-pool → Dropout(0.5) → rly_head
```

**① Causal Transformer Encoder** (3 layers, hidden=192, nhead=4, FFN=768, dropout=0.3)
- Causal mask prevents future leakage for shot-level prediction
- Output at position n−1 → **action head** Linear(192, 19)
- `concat([hidden, act_logits])` → **point head** Linear(211, 10)

**② Bidirectional Transformer Encoder** (1 layer, hidden=192, nhead=4, FFN=768, dropout=0.3)
- No causal mask; sees the full rally sequence
- Random sequence truncation during training to prevent length shortcut
- Mean-pool → extra Dropout(0.5) → **rally head** Linear(192, 1)

---

### 3. Training

**Multi-Task Loss**

$$\mathcal{L} = 0.45 \cdot \mathcal{L}_{act} + 0.45 \cdot \mathcal{L}_{pt} + 0.05 \cdot \mathcal{L}_{rally} + 0.2 \cdot \mathcal{L}_{coord}$$

**Class-Aware Loss Weighting**
- Action: $w = \text{clamp}(1/\text{freq},\ \max=20\times)$, renormalized (41× raw imbalance)
- Point: $w = 1/\sqrt{\text{freq}}$, renormalized

**Focal Cross-Entropy (action head)**

$$\mathcal{L}_{focal} = \alpha \cdot (1 - p_t)^\gamma \cdot \text{CE} \quad \gamma=2.0,\ \alpha=1.0$$

**Data Augmentation (training only)**
- 5% global random token masking
- 20% masking on lag features (`prev_actionType`, `prev_actionId`, `prev_pointId`)

**Regularisation**
- Label smoothing ε=0.1; rally target smoothing: `y * 0.8 + 0.1`
- Gradient clipping = 1.0

**Hyper-parameters**

| Parameter | Value |
|---|---|
| hidden / emb_dim | 192 / 32 |
| layers (causal / bidi) | 3 / 1 |
| nhead | 4 |
| dropout | 0.3 (+ 0.5 on rally head) |
| batch size | 64 |
| optimizer | AdamW, lr=1e-3, wd=1e-4 |
| LR schedule | linear warmup (5 ep) + cosine |
| early stopping | patience=12, val=10% match-level |

---

### 4. Inference

**Serve Mask**
- `pred_strikeNumber = 1` → force `actionId ∈ {15–18}`, mask others to −∞
- Otherwise → mask serves to −∞

**Multi-Seed Ensemble**
Train 3 models (seeds 42 / 123 / 456). Average raw logits across models, then apply Serve Mask and argmax.

**Rally Winner**
`serverGetPoint = int(mean_logit > 0)`

---

## Experiments & Results

### Validation Results (10% match-level split)

| Model | F1_act | F1_pt | AUC | Score |
|---|---|---|---|---|
| ShuttleNet CE (seed 42) | 0.3139 | 0.2354 | 0.9331 | 0.3426 |
| ShuttleNet Focal γ=2.0 (seed 42) | 0.2945 | 0.2373 | 0.9219 | 0.3344 |
| **ShuttleNet 3-Seed Focal Ensemble** | **0.2684** | **0.2273** | **0.9241** | **0.3831** |

### Leaderboard

| Submission | Score | Rank |
|---|---|---|
| Official Baseline (LSTM) | 0.289 | — |
| Single seed CE | 0.3189 | 216 / 380 |
| **3-Seed Focal Ensemble** | **0.3318** | **168 / 423** |

### Key Takeaways
- Dual-encoder prevents future leakage without sacrificing rally prediction quality.
- Class-aware weighting + focal loss addresses severe shot-type imbalance (41× ratio).
- Multi-seed ensemble (+0.013 on leaderboard) is the single largest contributor to score improvement.
