"""
Evaluate the 3-seed focal ensemble on the held-out validation split
(same GroupShuffleSplit as training) and report F1_act, F1_pt, AUC.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
import pandas as pd

from shuttleNet_code import (
    ShuttleNet, RallyDataset,
    add_derived_features, build_sequences,
    CAT_FEATURES, PLAYER_FEATURES,
)

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))
TRAIN_CSV   = os.path.join(BASE, "data", "train.csv")
ENCODER_PKL = os.path.join(BASE, "output", "encoders", "encoder.pkl")
MODEL_PATHS = [
    os.path.join(BASE, "output", "models", "model.pt.focal_seed42"),
    os.path.join(BASE, "output", "models", "model.pt.focal_seed123"),
    os.path.join(BASE, "output", "models", "model.pt.focal_seed456"),
]
# architecture must match the 3-seed pipeline
HIDDEN, LAYERS, DROP = 192, 3, 0.3

# ── load encoder & rebuild val split ───────────────────────────────────────
enc = joblib.load(ENCODER_PKL)
train_df = add_derived_features(
    pd.read_csv(TRAIN_CSV).sort_values(["rally_uid", "strikeNumber"])
)

(Xcat_all, Xply_all,
 yA_all, yP_all, yR_all, L_all, MAXLEN, group_all) = build_sequences(train_df, enc)

act_classes = enc["act_classes"]
pt_classes  = enc["pt_classes"]
act_id2idx  = enc["act_id2idx"]
pt_id2idx   = enc["pt_id2idx"]
n_act = len(act_classes)
n_pt  = len(pt_classes)

yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
yP_all = np.vectorize( pt_id2idx.get)(yP_all, -1)

idx = np.arange(len(Xcat_all))
gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
_, va_idx = next(gss.split(idx, groups=group_all))

val_ds = RallyDataset(
    Xcat_all[va_idx], Xply_all[va_idx],
    yA_all[va_idx],   yP_all[va_idx],
    yR_all[va_idx],   L_all[va_idx],
)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
print(f"Validation set size: {len(val_ds)} rallies")

# ── load models ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_cat_tokens = [len(enc["cats"][c])     + 1 for c in CAT_FEATURES]
num_ply_tokens = [len(enc["ply_cats"][c]) + 1 for c in PLAYER_FEATURES]
n_act_type     = enc.get("n_act_type", 5)

models = []
for p in MODEL_PATHS:
    m = ShuttleNet(
        num_cat_tokens, num_ply_tokens, n_act, n_pt,
        emb_dim=32, hidden=HIDDEN,
        nhead=4, num_layers=LAYERS, dropout=DROP,
        n_act_type=n_act_type,
    ).to(device)
    m.load_state_dict(torch.load(p, map_location=device))
    m.eval()
    models.append(m)
    print(f"Loaded {os.path.basename(p)}")

# ── inference on val set ───────────────────────────────────────────────────
allA, allAp, allP, allPp, allR, allRp = [], [], [], [], [], []

with torch.no_grad():
    for Xc, Xp, yAb, yPb, yRb, Lb in val_loader:
        Xc, Xp, yAb, yPb, yRb, Lb = (t.to(device) for t in (Xc, Xp, yAb, yPb, yRb, Lb))

        # ensemble: average logits
        act_sum = None; pt_sum = None; rly_sum = None
        for m in models:
            la, lp, lr_, _lc, _latype = m(Xc, Xp, Lb)
            act_sum = la if act_sum is None else act_sum + la
            pt_sum  = lp if pt_sum  is None else pt_sum  + lp
            rly_sum = lr_ if rly_sum is None else rly_sum + lr_
        la = act_sum / len(models)
        lp = pt_sum  / len(models)
        lr_ = rly_sum / len(models)

        yA_flat = yAb.view(-1).cpu().numpy()
        yP_flat = yPb.view(-1).cpu().numpy()
        mA = (yA_flat != -1); mP = (yP_flat != -1)
        allA  += yA_flat[mA].tolist()
        allAp += la.argmax(-1).view(-1).cpu().numpy()[mA].tolist()
        allP  += yP_flat[mP].tolist()
        allPp += lp.argmax(-1).view(-1).cpu().numpy()[mP].tolist()
        allR  += yRb.cpu().tolist()
        allRp += torch.sigmoid(lr_).cpu().tolist()

# ── compute metrics ────────────────────────────────────────────────────────
f1_act = f1_score(allA, allAp, average="macro")
f1_pt  = f1_score(allP, allPp, average="macro")
auc    = roc_auc_score(allR, allRp)
final  = 0.4 * f1_act + 0.4 * f1_pt + 0.2 * auc

print()
print("=" * 50)
print("  3-Seed Focal Ensemble — Validation Metrics")
print("=" * 50)
print(f"  F1_act  (Macro F1, shot type) : {f1_act:.4f}")
print(f"  F1_pt   (Macro F1, landing)   : {f1_pt:.4f}")
print(f"  AUC     (ROC-AUC, rally win)  : {auc:.4f}")
print(f"  Final   (competition metric)  : {final:.4f}")
print("=" * 50)
