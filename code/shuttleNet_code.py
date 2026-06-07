import argparse
import copy
import math
import random
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42  # default; override via --seed at CLI
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def _apply_seed(s):
    """Reset all RNGs. Called from __main__ after argparse if --seed differs from default."""
    global SEED
    SEED = int(s)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

CAT_FEATURES    = [
    "sex", "handId", "strengthId", "spinId",
    "pointId", "actionId", "positionId", "strikeId",
    "strikeNumber", "actionType", "prev_actionType",
    "prev_actionId", "prev_pointId"
]
CONT_FEATURES   = []
PLAYER_FEATURES = []  # disabled — likely cause of F1_act regression due to match-based group split

PAD_TOKEN = 0  # padding (masked out)
UNK_TOKEN = 1  # unknown category seen only in test set
# real categories are encoded starting from index 2

# Physical 2-D coords per pointId: X=backhand(0)→forehand(1), Y=short(0)→long(1)
RAW_POINT_COORDS = {
    0: (0.5, -0.5),                        # miss / out-of-bounds (界外)
    1: (1.0, 0.0), 2: (0.5, 0.0), 3: (0.0, 0.0),
    4: (1.0, 0.5), 5: (0.5, 0.5), 6: (0.0, 0.5),
    7: (1.0, 1.0), 8: (0.5, 1.0), 9: (0.0, 1.0),
}


class RallyDataset(Dataset):
    def __init__(self, Xcat, Xply, yA, yP, yR, L):
        self.Xcat = torch.tensor(Xcat, dtype=torch.long)
        self.Xply = torch.tensor(Xply, dtype=torch.long)
        self.yA   = torch.tensor(yA,   dtype=torch.long)
        self.yP   = torch.tensor(yP,   dtype=torch.long)
        self.yR   = torch.tensor(yR,   dtype=torch.float32)
        self.L    = torch.tensor(L,    dtype=torch.long)

    def __len__(self): return self.Xcat.shape[0]

    def __getitem__(self, i):
        return (self.Xcat[i], self.Xply[i],
                self.yA[i], self.yP[i], self.yR[i], self.L[i])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class FocalCrossEntropyLoss(nn.Module):
    """Cross-entropy focal variant for class-imbalance mitigation."""
    def __init__(self, weight=None, gamma=2.0, alpha=1.0,
                 ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        ce_valid = ce[valid]
        pt = torch.exp(-ce_valid)
        loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_valid
        return loss.mean()


class ShuttleNet(nn.Module):
    """
    ShuttleNet with dual-encoder architecture:
    - Causal TransformerEncoder  → next-shot action / point prediction
    - Bidirectional TransformerEncoder + mean pooling → rally outcome prediction

    This removes future leakage for shot-level tasks while letting the rally
    classifier see the full sequence (appropriate because the outcome is only
    labelled after the rally ends).
    """
    def __init__(self, num_cat_tokens, num_ply_tokens,
                 n_act, n_pt, emb_dim=32, hidden=256,
                 nhead=4, num_layers=3, dropout=0.2, n_act_type=5):
        super().__init__()
        # +2 to reserve index 0 (PAD) and index 1 (UNK)
        self.cat_embs = nn.ModuleList([
            nn.Embedding(n + 2, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_cat_tokens
        ])
        self.ply_embs = nn.ModuleList([
            nn.Embedding(n + 2, emb_dim, padding_idx=PAD_TOKEN)
            for n in num_ply_tokens
        ])
        in_dim = (len(num_cat_tokens) + len(num_ply_tokens)) * emb_dim
        self.proj_shot  = nn.Linear(in_dim, hidden)  # for causal (shot-level)
        self.proj_rally = nn.Linear(in_dim, hidden)  # for bidi   (rally-level)

        # --- causal encoder (next-shot heads) ---
        self.causal_pos = PositionalEncoding(hidden, dropout=dropout)
        causal_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.causal_enc = nn.TransformerEncoder(causal_layer, num_layers=num_layers,
                                                enable_nested_tensor=False)

        # --- bidirectional encoder (rally head) ---
        self.bidi_pos = PositionalEncoding(hidden, dropout=dropout)
        bidi_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        bidi_layers = max(1, num_layers // 2)
        self.bidi_enc = nn.TransformerEncoder(bidi_layer, num_layers=bidi_layers,
                                              enable_nested_tensor=False)

        self.drop          = nn.Dropout(dropout)
        # Hierarchical action head kept as auxiliary task only (its logits no longer
        # feed into act_head — raw logits destabilised training).
        self.act_type_head = nn.Linear(hidden, n_act_type)
        self.act_head      = nn.Linear(hidden, n_act)
        self.pt_head       = nn.Linear(hidden + n_act, n_pt)
        self.pt_coord_head = nn.Linear(hidden + n_act, 2)
        self.rly_drop      = nn.Dropout(0.5)   # extra regularisation for rally head
        self.rly_head      = nn.Linear(hidden, 1)

    def forward(self, Xcat, Xply, lengths):
        # sequence masking: randomly replace 5% of tokens with PAD during training
        if self.training:
            Xcat = Xcat.clone()

            # 1. 全域 5% 隨機遮罩
            global_mask = torch.rand(Xcat.shape[:2], device=Xcat.device) < 0.05
            Xcat[global_mask] = PAD_TOKEN

            # 2. 針對最後 3 個滯後特徵 (prev_actionType, prev_actionId, prev_pointId) 實施 20% 遮罩
            lag_mask = torch.rand(Xcat.shape[:2], device=Xcat.device) < 0.20
            Xcat[lag_mask, -3:] = PAD_TOKEN

        cat_e    = torch.cat([e(Xcat[:, :, i]) for i, e in enumerate(self.cat_embs)], dim=-1)
        if self.ply_embs:
            ply_e   = torch.cat([e(Xply[:, :, i]) for i, e in enumerate(self.ply_embs)], dim=-1)
            emb_all = torch.cat([cat_e, ply_e], dim=-1)
        else:
            emb_all = cat_e
        x_base_shot  = self.proj_shot(emb_all)   # causal path input
        x_base_rally = self.proj_rally(emb_all)  # bidi path input

        T = x_base_shot.size(1)

        # --- Causal path (維持原來的長度) ---
        pad_mask_shot = torch.arange(T, device=x_base_shot.device)[None] >= lengths[:, None]

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x_base_shot.device), diagonal=1
        )
        x_causal = self.drop(self.causal_enc(
            self.causal_pos(x_base_shot),
            mask=causal_mask,
            src_key_padding_mask=pad_mask_shot,
            is_causal=True,
        ))

        # --- Bidi path (隨機截斷，防堵長度作弊) ---
        if self.training:
            rand_ratios = torch.rand(lengths.shape, device=lengths.device)
            bidi_lengths = 1 + (rand_ratios * lengths.float()).long()
            bidi_lengths = torch.min(lengths, bidi_lengths)
        else:
            bidi_lengths = lengths

        pad_mask_bidi = torch.arange(T, device=x_base_rally.device)[None] >= bidi_lengths[:, None]

        x_bidi  = self.drop(self.bidi_enc(
            self.bidi_pos(x_base_rally),
            src_key_padding_mask=pad_mask_bidi,
        ))
        valid   = (~pad_mask_bidi).unsqueeze(-1).float()
        rly_h   = (x_bidi * valid).sum(1) / valid.sum(1).clamp(min=1)

        act_type_logits = self.act_type_head(x_causal)
        act_logits   = self.act_head(x_causal)
        cond_input   = torch.cat([x_causal, act_logits], dim=-1)
        pt_logits    = self.pt_head(cond_input)
        coord_logits = self.pt_coord_head(cond_input)
        return (act_logits, pt_logits,
                self.rly_head(self.rly_drop(rly_h)).squeeze(1),
                coord_logits, act_type_logits)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

_ACTION_TYPE_MAP = {
    15: 1, 16: 1, 17: 1, 18: 1,          # Serve
    1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2,  # Attack
    8: 3, 9: 3, 10: 3, 11: 3,            # Control
    12: 4, 13: 4, 14: 4,                  # Defensive
    0: 0,                                 # Zero
}

def add_derived_features(df):
    df = df.copy()
    df["score_diff"]  = df["scoreSelf"] - df["scoreOther"]
    df["is_critical"] = ((df["scoreSelf"] >= 10) | (df["scoreOther"] >= 10)).astype(int)
    df["leading"]     = np.sign(df["scoreSelf"] - df["scoreOther"]).astype(int)
    df["actionType"]  = df["actionId"].map(_ACTION_TYPE_MAP).fillna(0).astype(int)
    df["prev_actionType"] = (df.groupby("rally_uid")["actionType"]
                               .shift(1).fillna(0).astype(int))
    df["prev_actionId"] = (df.groupby("rally_uid")["actionId"]
                             .shift(1).fillna(0).astype(int))
    df["prev_pointId"] = (df.groupby("rally_uid")["pointId"]
                            .shift(1).fillna(0).astype(int))
    return df


def build_encoder(train_df, clip_std):
    cats     = {c: pd.Categorical(train_df[c]).categories for c in CAT_FEATURES}
    ply_cats = {c: pd.Categorical(train_df[c]).categories for c in PLAYER_FEATURES}
    return {"cats": cats, "ply_cats": ply_cats, "clip_std": clip_std}


def encode_cat(df, cats, feature_list):
    if not feature_list:
        return np.empty((len(df), 0), dtype=np.int64)
    cols = []
    for col in feature_list:
        known = cats[col]
        # map values not in known categories to NaN so codes == -1, then assign UNK
        series = df[col].where(df[col].isin(known), other=None)
        codes  = pd.Categorical(series, categories=known).codes
        arr    = np.where(codes == -1, UNK_TOKEN, codes + 2).astype(np.int64)
        cols.append(arr)
    return np.stack(cols, axis=1)


def pad2d(a, m, pad_val=0):
    out = np.full((m, a.shape[1]), pad_val, dtype=a.dtype)
    out[:len(a)] = a
    return out


def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64)
    out[:len(a)] = a
    return out


def build_sequences(df, enc, maxlen=None):
    """Convert raw dataframe into padded arrays ready for the model."""
    Xcat_list, Xply_list = [], []
    yA_list, yP_list, yR_list, L_list, group_list = [], [], [], [], []

    for _, g in df.groupby("rally_uid"):
        if len(g) < 2:
            continue
        Xcat_list.append(encode_cat(g, enc["cats"],     CAT_FEATURES)[:-1])
        Xply_list.append(encode_cat(g, enc["ply_cats"], PLAYER_FEATURES)[:-1])
        yA_list.append(g["actionId"].values[1:].astype(np.int64))
        yP_list.append(g["pointId"].values[1:].astype(np.int64))
        yR_list.append(int(g["serverGetPoint"].iloc[0]))
        L_list.append(len(g) - 1)
        group_val = g["match"].iloc[0] if "match" in g.columns else g["numberGame"].iloc[0]
        group_list.append(group_val)

    if not Xcat_list:
        raise ValueError("No valid rallies found (need length >= 2).")

    ML = maxlen if maxlen is not None else max(L_list)
    Xcat_all  = np.stack([pad2d(s, ML, 0) for s in Xcat_list])
    Xply_all  = np.stack([pad2d(s, ML, 0) for s in Xply_list])
    yA_all    = np.stack([pad1d(s, ML)    for s in yA_list])
    yP_all    = np.stack([pad1d(s, ML)    for s in yP_list])
    yR_all    = np.array(yR_list,  dtype=np.float32)
    L_all     = np.array(L_list,   dtype=np.int64)
    group_all = np.array(group_list)
    return Xcat_all, Xply_all, yA_all, yP_all, yR_all, L_all, ML, group_all


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_scheduler(opt, warmup_epochs, total_epochs):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(args):
    train_df = add_derived_features(pd.read_csv(args.train).sort_values(["rally_uid", "strikeNumber"]))
    test_df  = add_derived_features(pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"]))

    enc = build_encoder(train_df, args.clip_std)

    (Xcat_all, Xply_all,
     yA_all, yP_all, yR_all, L_all, MAXLEN, group_all) = build_sequences(train_df, enc)

    # remap actionId / pointId to dense indices
    act_classes = np.sort(train_df["actionId"].unique()); n_act = len(act_classes)
    pt_classes  = np.sort(train_df["pointId"].unique());  n_pt  = len(pt_classes)
    act_id2idx  = {v: i for i, v in enumerate(act_classes)}
    pt_id2idx   = {v: i for i, v in enumerate(pt_classes)}
    yA_all = np.vectorize(act_id2idx.get)(yA_all, -1)
    yP_all = np.vectorize(pt_id2idx.get)(yP_all,  -1)

    enc.update({"MAXLEN": MAXLEN, "act_classes": act_classes, "pt_classes": pt_classes,
                "act_id2idx": act_id2idx, "pt_id2idx": pt_id2idx})
    joblib.dump(enc, args.encoder_out)
    print(f"Encoder artifact saved to: {args.encoder_out}")

    # --- group-aware split: keep all rallies from the same match in the same fold ---
    idx = np.arange(len(Xcat_all))
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=42)
    tr_idx, va_idx = next(gss.split(idx, groups=group_all))

    def spl(a): return a[tr_idx], a[va_idx]
    Xcat_tr, Xcat_va = spl(Xcat_all)
    Xply_tr, Xply_va = spl(Xply_all)
    yA_tr, yA_va = spl(yA_all); yP_tr, yP_va = spl(yP_all)
    yR_tr, yR_va = spl(yR_all); L_tr,  L_va  = spl(L_all)

    act_counts = np.bincount(yA_tr[yA_tr != -1].ravel(), minlength=n_act) + 1
    pt_counts  = np.bincount(yP_tr[yP_tr != -1].ravel(), minlength=n_pt)  + 1
    # action: clamped inverse (20x) — 19 classes, 41x raw imbalance
    act_w = torch.tensor(1.0 / act_counts.astype(np.float32), dtype=torch.float32)
    act_w = (act_w / act_w.min()).clamp(max=20.0)
    act_w = act_w * (n_act / act_w.sum())
    # point: sqrt inverse — 10 classes, less extreme imbalance; 20x cap hurts F1_pt
    pt_w  = torch.tensor(1.0 / np.sqrt(pt_counts), dtype=torch.float32)
    pt_w  = pt_w * (n_pt / pt_w.sum())

    train_ds = RallyDataset(Xcat_tr, Xply_tr, yA_tr, yP_tr, yR_tr, L_tr)
    val_ds   = RallyDataset(Xcat_va, Xply_va, yA_va, yP_va, yR_va, L_va)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=max(args.batch * 2, 128), shuffle=False, num_workers=0)

    num_cat_tokens = [len(enc["cats"][c])     + 1 for c in CAT_FEATURES]
    num_ply_tokens = [len(enc["ply_cats"][c]) + 1 for c in PLAYER_FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}  |  train={len(train_ds)}  val={len(val_ds)}")

    # actionType lookup for dense action indices (for hierarchical action head)
    n_act_type = 5  # Zero, Serve, Attack, Control, Defensive
    act_type_for_dense = torch.tensor(
        [_ACTION_TYPE_MAP.get(int(act_classes[i]), 0) for i in range(n_act)],
        dtype=torch.long, device=device,
    )
    enc["act_type_for_dense"] = act_type_for_dense.cpu().numpy()
    enc["n_act_type"] = n_act_type
    joblib.dump(enc, args.encoder_out)  # re-save with extra fields

    # coord_map[i] = (X, Y) for pt_classes[i]; used in spatial regression loss
    coord_map = torch.tensor(
        [RAW_POINT_COORDS.get(int(pt_classes[i]), (0.0, 0.0)) for i in range(n_pt)],
        dtype=torch.float32, device=device,
    )
    pt0_dense = int(enc["pt_id2idx"].get(0, -1))  # dense index for pointId=0 (miss)

    if args.use_focal_action:
        ce_action = FocalCrossEntropyLoss(
            ignore_index=-1,
            weight=act_w.to(device),
            label_smoothing=args.label_smoothing,
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
        )
        print(f"Using focal action loss: gamma={args.focal_gamma}, alpha={args.focal_alpha}")
    else:
        ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device), label_smoothing=args.label_smoothing)
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device),  label_smoothing=args.label_smoothing)
    ce_atype  = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)
    bce_rally = nn.BCEWithLogitsLoss()

    # Multi-seed ensemble support: train one model per seed, average softmax at inference.
    # NOTE: when --seeds is empty (default), we do NOT re-seed here — that would change
    # the random state at model-init time and diverge from the original baseline.
    explicit_seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else []
    seeds = explicit_seeds if explicit_seeds else [SEED]
    trained_models = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n===== Training run {seed_idx+1}/{len(seeds)}  (seed={seed}) =====")
        if explicit_seeds:
            # Only re-seed when the user explicitly requested multi-seed; this matches
            # baseline initialization behavior when --seeds is not provided.
            random.seed(seed); np.random.seed(seed)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        model = ShuttleNet(
            num_cat_tokens, num_ply_tokens, n_act, n_pt,
            emb_dim=args.emb, hidden=args.hidden,
            nhead=args.nhead, num_layers=args.layers, dropout=args.drop,
            n_act_type=n_act_type,
        ).to(device)
        opt       = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = get_scheduler(opt, warmup_epochs=args.warmup, total_epochs=args.epochs)

        best_final   = -1.0
        best_weights = None
        patience_cnt = 0

        for ep in range(1, args.epochs + 1):
            model.train(); run_loss = 0.0
            for Xc, Xp, yAb, yPb, yRb, Lb in train_loader:
                Xc, Xp, yAb, yPb, yRb, Lb = (t.to(device) for t in (Xc, Xp, yAb, yPb, yRb, Lb))
                opt.zero_grad()
                la, lp, lr_, lc, latype = model(Xc, Xp, Lb)
                # derive actionType labels from dense action labels
                yAb_flat   = yAb.view(-1)
                yATypeb    = torch.where(yAb_flat == -1,
                                         torch.full_like(yAb_flat, -1),
                                         act_type_for_dense[yAb_flat.clamp(min=0)])
                loss_atype = ce_atype(latype.view(-1, latype.size(-1)), yATypeb)
                loss_act  = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                loss_pt   = ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
                loss_rly  = bce_rally(lr_, yRb * 0.8 + 0.1)
                yp_flat   = yPb.view(-1)
                vmask     = (yp_flat != -1) & (yp_flat != pt0_dense)
                if vmask.any():
                    loss_coord = nn.functional.mse_loss(
                        lc.view(-1, 2)[vmask], coord_map[yp_flat.clamp(min=0)][vmask])
                else:
                    loss_coord = lc.sum() * 0.0
                loss = (0.45 * loss_act + 0.45 * loss_pt +
                        0.05 * loss_rly + args.w_coord * loss_coord +
                        args.w_atype * loss_atype)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run_loss += loss.item() * Xc.size(0)
            scheduler.step()

            model.eval(); val_loss = 0.0
            allA, allAp, allP, allPp, allR, allRp = [], [], [], [], [], []
            with torch.no_grad():
                for Xc, Xp, yAb, yPb, yRb, Lb in val_loader:
                    Xc, Xp, yAb, yPb, yRb, Lb = (t.to(device) for t in (Xc, Xp, yAb, yPb, yRb, Lb))
                    la, lp, lr_, lc, latype = model(Xc, Xp, Lb)
                    yAb_flat   = yAb.view(-1)
                    yATypeb    = torch.where(yAb_flat == -1,
                                             torch.full_like(yAb_flat, -1),
                                             act_type_for_dense[yAb_flat.clamp(min=0)])
                    loss_atype = ce_atype(latype.view(-1, latype.size(-1)), yATypeb)
                    loss_act  = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                    loss_pt   = ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
                    loss_rly  = bce_rally(lr_, yRb * 0.8 + 0.1)
                    yp_flat   = yPb.view(-1)
                    vmask     = (yp_flat != -1) & (yp_flat != pt0_dense)
                    if vmask.any():
                        loss_coord = nn.functional.mse_loss(
                            lc.view(-1, 2)[vmask], coord_map[yp_flat.clamp(min=0)][vmask])
                    else:
                        loss_coord = lc.sum() * 0.0
                    loss = (0.45 * loss_act + 0.45 * loss_pt +
                            0.05 * loss_rly + args.w_coord * loss_coord +
                            args.w_atype * loss_atype)
                    val_loss += loss.item() * Xc.size(0)
                    allR  += yRb.cpu().tolist()
                    allRp += torch.sigmoid(lr_).cpu().tolist()
                    yA_flat = yAb.view(-1).cpu().numpy()
                    yP_flat = yPb.view(-1).cpu().numpy()
                    mA = (yA_flat != -1); mP = (yP_flat != -1)
                    allA  += yA_flat[mA].tolist()
                    allAp += la.argmax(-1).view(-1).cpu().numpy()[mA].tolist()
                    allP  += yP_flat[mP].tolist()
                    allPp += lp.argmax(-1).view(-1).cpu().numpy()[mP].tolist()

            tr_loss = run_loss / len(train_loader.dataset)
            va_loss = val_loss / len(val_loader.dataset)
            try:
                f1A = f1_score(allA, allAp, average="macro") if allA else 0.0
                f1P = f1_score(allP, allPp, average="macro") if allP else 0.0
                auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
            except Exception:
                f1A, f1P, auc = 0.0, 0.0, 0.5

            final  = 0.5 * f1A + 0.4 * f1P + 0.1 * auc
            marker = " ★" if final > best_final else ""
            print(f"[seed {seed} | Epoch {ep:>3}/{args.epochs}] train={tr_loss:.4f} val={va_loss:.4f} "
                  f"F1_act={f1A:.4f} F1_pt={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}{marker}")

            if final > best_final:
                best_final   = final
                best_weights = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"Early stopping at epoch {ep} (no improvement for {args.patience} epochs)")
                    break

        print(f"\n[seed {seed}] Best val Final~{best_final:.4f} — loading best checkpoint")
        model.load_state_dict(best_weights)
        # Per-seed weight file (single-seed run keeps args.model_out path for back-compat)
        ckpt_path = args.model_out if len(seeds) == 1 else f"{args.model_out}.seed{seed}"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model weights saved to: {ckpt_path}")
        trained_models.append(model)

    _run_inference(trained_models, test_df, enc, device, args)


# ---------------------------------------------------------------------------
# Inference (can be run standalone with --mode infer)
# ---------------------------------------------------------------------------

def _run_inference(model_or_list, test_df, enc, device, args):
    """Run inference. Accepts a single model or a list of models for ensembling
    (softmax probabilities are averaged across models).

    Applies two inference-side tricks documented in README:
      - Serve Mask: predicted shot's strikeNumber decides whether serves (15-18)
        are forced or forbidden — this rules out physically impossible predictions.
      - Temperature Sampling: instead of argmax, sample from softmax(logits/temp).
        Reduces mode collapse to common classes, improving macro F1.
    """
    MAXLEN      = enc["MAXLEN"]
    act_classes = enc["act_classes"]
    pt_classes  = enc["pt_classes"]

    # dense indices of serve actions (actionId 15-18) for Serve Mask
    serve_action_ids = {15, 16, 17, 18}
    serve_dense_idx = [i for i, aid in enumerate(act_classes) if int(aid) in serve_action_ids]
    nonserve_dense_idx = [i for i, aid in enumerate(act_classes) if int(aid) not in serve_action_ids]
    serve_dense_t    = torch.tensor(serve_dense_idx,    dtype=torch.long, device=device)
    nonserve_dense_t = torch.tensor(nonserve_dense_idx, dtype=torch.long, device=device)

    temp = max(float(args.temp), 1e-6)
    use_sampling = temp > 0 and args.use_sampling

    models = model_or_list if isinstance(model_or_list, list) else [model_or_list]
    for m in models:
        m.eval()

    pred_rows = []
    with torch.no_grad():
        for rid, g in test_df.groupby("rally_uid"):
            T      = min(len(g), MAXLEN)
            Xcat_g = encode_cat(g, enc["cats"],     CAT_FEATURES)
            Xply_g = encode_cat(g, enc["ply_cats"], PLAYER_FEATURES)

            Xcat_p = np.zeros((MAXLEN, Xcat_g.shape[1]), dtype=np.int64)
            Xply_p = np.zeros((MAXLEN, Xply_g.shape[1]), dtype=np.int64)
            Xcat_p[:T] = Xcat_g[:T]; Xply_p[:T] = Xply_g[:T]

            Xc_t = torch.tensor(Xcat_p[None], dtype=torch.long, device=device)
            Xp_t = torch.tensor(Xply_p[None], dtype=torch.long, device=device)
            L_t  = torch.tensor([max(1, T)],  dtype=torch.long, device=device)

            last_t = L_t.item() - 1
            # average logits across ensembled models (temperature applied later)
            act_logits_avg = None; pt_logits_avg = None; rly_logits_sum = 0.0
            for m in models:
                la, lp, lr_, _lc, _latype = m(Xc_t, Xp_t, L_t)
                al = la[0, last_t]
                pl = lp[0, last_t]
                act_logits_avg = al if act_logits_avg is None else act_logits_avg + al
                pt_logits_avg  = pl if pt_logits_avg  is None else pt_logits_avg  + pl
                rly_logits_sum += float(lr_.item())
            act_logits_avg = act_logits_avg / len(models)
            pt_logits_avg  = pt_logits_avg  / len(models)
            rly_avg = rly_logits_sum / len(models)

            # --- Serve Mask ---
            # predicted shot's strikeNumber = (last shown shot's strikeNumber) + 1
            last_strike_no = int(g["strikeNumber"].iloc[T - 1])
            pred_strike_no = last_strike_no + 1
            if pred_strike_no == 1:
                # force a serve (only allow 15-18)
                act_logits_avg = act_logits_avg.clone()
                if len(nonserve_dense_idx) > 0:
                    act_logits_avg[nonserve_dense_t] = -1e9
            else:
                # forbid serves
                act_logits_avg = act_logits_avg.clone()
                if len(serve_dense_idx) > 0:
                    act_logits_avg[serve_dense_t] = -1e9

            # --- pick action ---
            if use_sampling:
                act_probs = torch.softmax(act_logits_avg / temp, dim=-1)
                act_idx = int(torch.multinomial(act_probs, 1).item())
                pt_probs = torch.softmax(pt_logits_avg / temp, dim=-1)
                pt_idx = int(torch.multinomial(pt_probs, 1).item())
            else:
                act_idx = int(torch.argmax(act_logits_avg).item())
                pt_idx  = int(torch.argmax(pt_logits_avg).item())

            pred_rows.append({
                "rally_uid":      int(rid),
                "serverGetPoint": int(rly_avg > 0),
                "actionId":       int(act_classes[act_idx]),
                "pointId":        int(pt_classes[pt_idx]),
            })

    pred_df = pd.DataFrame(pred_rows).sort_values("rally_uid")
    out = (pd.read_csv(args.sample)
             .drop(columns=["serverGetPoint", "pointId", "actionId"], errors="ignore")
             .merge(pred_df, on="rally_uid", how="left"))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out.to_csv(args.out, index=False)
    print(f"Saved submission to: {args.out}  (ensembled over {len(models)} model(s))")
    print(out.head())


def infer_only(args):
    """Run inference without training. --model_out can be a single path OR
    comma-separated paths (the latter ensembles them via softmax averaging)."""
    enc    = joblib.load(args.encoder_out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_act = len(enc["act_classes"])
    n_pt  = len(enc["pt_classes"])
    num_cat_tokens = [len(enc["cats"][c])     + 1 for c in CAT_FEATURES]
    num_ply_tokens = [len(enc["ply_cats"][c]) + 1 for c in PLAYER_FEATURES]

    paths = [p.strip() for p in args.model_out.split(",") if p.strip()]
    models = []
    for p in paths:
        m = ShuttleNet(
            num_cat_tokens, num_ply_tokens, n_act, n_pt,
            emb_dim=args.emb, hidden=args.hidden,
            nhead=args.nhead, num_layers=args.layers, dropout=args.drop,
            n_act_type=enc.get("n_act_type", 5),
        ).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        models.append(m)
        print(f"loaded {p}")

    test_df = add_derived_features(pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"]))
    _run_inference(models if len(models) > 1 else models[0], test_df, enc, device, args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",       default="train", choices=["train", "infer"],
                    help="'train' to train+infer; 'infer' to run inference only")
    # data paths
    ap.add_argument("--train",       default="data/train.csv")
    ap.add_argument("--test",        default="data/test.csv")
    ap.add_argument("--sample",      default="data/sample_submission.csv")
    ap.add_argument("--out",         default="output/submissions/submission_shuttlenet.csv")
    ap.add_argument("--encoder_out", default="output/encoders/encoder.pkl",
                    help="where to save/load the encoder artifact")
    ap.add_argument("--model_out",   default="output/models/model.pt",
                    help="where to save/load model weights (used in --mode infer)")
    # model architecture
    ap.add_argument("--emb",    type=int,   default=32)
    ap.add_argument("--hidden", type=int,   default=128)
    ap.add_argument("--nhead",  type=int,   default=4)
    ap.add_argument("--layers", type=int,   default=2)
    ap.add_argument("--drop",   type=float, default=0.4)
    # training
    ap.add_argument("--epochs",          type=int,   default=60)
    ap.add_argument("--batch",           type=int,   default=64)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--weight_decay",    type=float, default=1e-4)
    ap.add_argument("--warmup",          type=int,   default=5,
                    help="linear warmup epochs")
    ap.add_argument("--patience",        type=int,   default=10,
                    help="early stopping patience (epochs without improvement)")
    ap.add_argument("--val_size",        type=float, default=0.10)
    ap.add_argument("--clip_std",        type=float, default=5.0,
                    help="clip continuous features at ±clip_std standard deviations")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--use_focal_action", action="store_true",
                    help="Use focal cross-entropy for action head (off by default).")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma (used when --use_focal_action).")
    ap.add_argument("--focal_alpha", type=float, default=1.0,
                    help="Focal loss alpha (used when --use_focal_action).")
    # loss weights (must sum to 1 for the Final metric to be interpretable)
    ap.add_argument("--w_action", type=float, default=0.5)
    ap.add_argument("--w_point",  type=float, default=0.4)
    ap.add_argument("--w_rally",  type=float, default=0.1)
    ap.add_argument("--w_coord",  type=float, default=0.2)
    ap.add_argument("--w_atype",  type=float, default=0.0,
                    help="weight for hierarchical actionType auxiliary loss "
                         "(0 = disabled, default)")
    ap.add_argument("--seeds",    type=str,   default="",
                    help="comma-separated seeds for multi-seed ensemble inference; "
                         "if non-empty, trains one model per seed and averages softmax probs")
    ap.add_argument("--temp",     type=float, default=0.5,
                    help="Temperature for inference sampling (lower = sharper)")
    ap.add_argument("--use_sampling", action="store_true",
                    help="Use temperature sampling at inference (default: argmax). "
                         "Sampling is non-deterministic but reduces mode collapse on rare classes.")

    ap.add_argument("--seed", type=int, default=42,
                    help="single training seed (default 42; matches the existing model.pt)")
    args = ap.parse_args()

    if args.seed != 42:
        _apply_seed(args.seed)

    if args.mode == "train":
        train_model(args)
    else:
        infer_only(args)
