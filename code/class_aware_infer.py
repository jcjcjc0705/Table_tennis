"""
Class-aware inference: instead of pure argmax, assign predictions across rallies
so that the per-class prediction count matches the training distribution. This
directly attacks macro-F1's "predicted-zero-times-for-rare-class" failure mode.

Algorithm (greedy bipartite matching):
  1. Compute target_count[c] = round(N * train_freq[c]) for each class c.
  2. Sort all (rally, class) pairs by softmax probability descending.
  3. Walk down the sorted list: assign each rally to a class if (a) the rally
     is still unassigned and (b) the class is still under its target count.
  4. Any leftover rallies get standard argmax (fallback).

Tunable via `--alpha`:
  alpha = 0  → pure argmax (no class-aware adjustment)
  alpha = 1  → full class-aware (targets ≡ train frequency)
  0 < alpha < 1 → linear interpolation between argmax-counts and train-freq-counts.

Also supports model ensembling (`--model_out` comma-separated paths).
"""
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

from shuttleNet_code import (
    ShuttleNet, add_derived_features, encode_cat,
    CAT_FEATURES, PLAYER_FEATURES,
)


def compute_target_counts(train_freq, n_rallies, argmax_counts, alpha):
    """Blend argmax_counts (alpha=0) with train_freq-based counts (alpha=1)."""
    freq_counts = train_freq * n_rallies
    blended = (1 - alpha) * argmax_counts + alpha * freq_counts
    targets = np.floor(blended).astype(int)
    # distribute the remainder to top remainders
    remainder = n_rallies - targets.sum()
    if remainder > 0:
        # add 1 to the classes with the largest fractional remainders
        frac = blended - targets
        order = np.argsort(-frac)
        for i in range(remainder):
            targets[order[i % len(order)]] += 1
    elif remainder < 0:
        # remove 1 from the classes with the smallest fractional remainders
        frac = blended - targets
        order = np.argsort(frac)
        for i in range(-remainder):
            targets[order[i]] -= 1
            targets[order[i]] = max(0, targets[order[i]])
        # if still not equal (because of clamping), compensate elsewhere
        while targets.sum() != n_rallies:
            diff = n_rallies - targets.sum()
            if diff > 0:
                # add to largest argmax class
                targets[np.argmax(argmax_counts)] += 1
            else:
                # remove from largest class with > 0
                idx = np.argmax(targets)
                if targets[idx] > 0:
                    targets[idx] -= 1
                else:
                    break
    return targets


def adaptive_alpha_from_distribution(probs, train_freq, base_alpha,
                                     deviation_threshold=0.20,
                                     max_boost=0.30):
    """Adjust alpha based on L1 distance between argmax and train distributions."""
    argmax_cls = probs.argmax(axis=1)
    counts = np.bincount(argmax_cls, minlength=len(train_freq)).astype(np.float64)
    pred_freq = counts / max(counts.sum(), 1.0)
    deviation = np.abs(pred_freq - train_freq).sum()

    if deviation <= deviation_threshold:
        return float(base_alpha), float(deviation)

    scale = min(1.0, (deviation - deviation_threshold) / max(deviation_threshold, 1e-9))
    alpha = min(1.0, float(base_alpha) + float(max_boost) * scale)
    return float(alpha), float(deviation)


def class_aware_assign(probs, target_counts):
    """probs: [N, C] softmax probabilities. target_counts: [C] desired count per class.
    Returns: predicted class index for each rally."""
    N, C = probs.shape
    # Sort all (rally, class) pairs by probability descending
    flat_idx = np.argsort(-probs.flatten())
    rallies   = flat_idx // C
    classes   = flat_idx % C

    assigned     = np.full(N, -1, dtype=np.int64)
    class_count  = np.zeros(C, dtype=np.int64)
    remaining_targets = target_counts.copy()

    for r, c in zip(rallies, classes):
        if assigned[r] != -1:
            continue
        if class_count[c] >= remaining_targets[c]:
            continue
        assigned[r] = c
        class_count[c] += 1
        if (assigned != -1).sum() == N:
            break

    # Fallback: any leftover rallies get plain argmax
    leftover = np.where(assigned == -1)[0]
    if len(leftover) > 0:
        for r in leftover:
            assigned[r] = int(np.argmax(probs[r]))

    return assigned


def get_probabilities(models, test_df, enc, device, args):
    """Run inference, return per-rally action and point softmax probabilities
    (averaged across ensemble models), as well as last-strike-number info."""
    MAXLEN = enc["MAXLEN"]
    n_act  = len(enc["act_classes"])
    n_pt   = len(enc["pt_classes"])

    rally_ids = []
    act_probs_all = []
    pt_probs_all = []
    rly_avg_all = []
    last_strike_all = []

    for m in models:
        m.eval()

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

            act_sum = None; pt_sum = None; rly_sum = 0.0
            for m in models:
                la, lp, lr_, _lc, _latype = m(Xc_t, Xp_t, L_t)
                ap = torch.softmax(la[0, last_t], dim=-1).cpu().numpy()
                pp = torch.softmax(lp[0, last_t], dim=-1).cpu().numpy()
                act_sum = ap if act_sum is None else act_sum + ap
                pt_sum  = pp if pt_sum  is None else pt_sum  + pp
                rly_sum += float(lr_.item())
            act_sum /= len(models); pt_sum /= len(models)
            rly_avg = rly_sum / len(models)

            rally_ids.append(int(rid))
            act_probs_all.append(act_sum)
            pt_probs_all.append(pt_sum)
            rly_avg_all.append(rly_avg)
            last_strike_all.append(int(g["strikeNumber"].iloc[T - 1]))

    return (np.array(rally_ids),
            np.array(act_probs_all),
            np.array(pt_probs_all),
            np.array(rly_avg_all),
            np.array(last_strike_all))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = joblib.load(args.encoder_out)

    n_act = len(enc["act_classes"]); n_pt = len(enc["pt_classes"])
    num_cat_tokens = [len(enc["cats"][c])     + 1 for c in CAT_FEATURES]
    num_ply_tokens = [len(enc["ply_cats"][c]) + 1 for c in PLAYER_FEATURES]
    act_classes = enc["act_classes"]
    pt_classes  = enc["pt_classes"]

    # load models
    paths = [p.strip() for p in args.model_out.split(",") if p.strip()]
    models = []
    for p in paths:
        m = ShuttleNet(num_cat_tokens, num_ply_tokens, n_act, n_pt,
                       emb_dim=args.emb, hidden=args.hidden,
                       nhead=args.nhead, num_layers=args.layers,
                       dropout=0.0, n_act_type=enc.get("n_act_type", 5)).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        models.append(m)
        print(f"loaded {p}")

    # compute training frequencies (for target distributions)
    train_df = pd.read_csv(args.train)
    act_freq_raw = np.zeros(n_act)
    pt_freq_raw  = np.zeros(n_pt)
    for orig, dense in enc["act_id2idx"].items():
        act_freq_raw[dense] = (train_df["actionId"] == orig).sum()
    for orig, dense in enc["pt_id2idx"].items():
        pt_freq_raw[dense] = (train_df["pointId"] == orig).sum()
    act_freq = act_freq_raw / act_freq_raw.sum()
    pt_freq  = pt_freq_raw  / pt_freq_raw.sum()

    # inference
    test_df = add_derived_features(pd.read_csv(args.test).sort_values(["rally_uid", "strikeNumber"]))
    rally_ids, act_probs, pt_probs, rly_avg, last_strike = get_probabilities(
        models, test_df, enc, device, args
    )
    N = len(rally_ids)

    # --- Serve Mask: zero out serve probs for predicted_strike > 1 (which is all test rallies) ---
    serve_action_ids = {15, 16, 17, 18}
    serve_dense_idx    = [i for i, aid in enumerate(act_classes) if int(aid) in serve_action_ids]
    nonserve_dense_idx = [i for i, aid in enumerate(act_classes) if int(aid) not in serve_action_ids]

    pred_strike = last_strike + 1
    mask_serve    = pred_strike > 1   # forbid serves
    mask_nonserve = pred_strike == 1  # only allow serves
    for i in range(N):
        if mask_serve[i]:
            act_probs[i, serve_dense_idx] = 0.0
        elif mask_nonserve[i]:
            act_probs[i, nonserve_dense_idx] = 0.0
        # renormalize
        s = act_probs[i].sum()
        if s > 0:
            act_probs[i] /= s

    # --- argmax baseline (for comparison) ---
    argmax_act = act_probs.argmax(axis=1)
    argmax_pt  = pt_probs.argmax(axis=1)

    print(f"\n=== Class distribution analysis ===")
    print(f"Action classes — argmax counts vs train-freq targets:")
    argmax_act_counts = np.bincount(argmax_act, minlength=n_act)
    train_act_targets = (act_freq * N).round().astype(int)
    for c in range(n_act):
        marker = "  Serve" if int(act_classes[c]) in serve_action_ids else ""
        print(f"  class {int(act_classes[c]):>2d}  argmax={argmax_act_counts[c]:>4d}  train-target={train_act_targets[c]:>4d}{marker}")

    print(f"\nPoint classes — argmax counts vs train-freq targets:")
    argmax_pt_counts = np.bincount(argmax_pt, minlength=n_pt)
    train_pt_targets = (pt_freq * N).round().astype(int)
    for c in range(n_pt):
        print(f"  class {int(pt_classes[c]):>2d}  argmax={argmax_pt_counts[c]:>4d}  train-target={train_pt_targets[c]:>4d}")

    # --- Class-aware assignment ---
    # For action: zero out serve targets (Serve Mask makes serves impossible)
    act_freq_masked = act_freq.copy()
    act_freq_masked[serve_dense_idx] = 0
    act_freq_masked = act_freq_masked / act_freq_masked.sum()
    effective_alpha = float(args.alpha)
    if args.adaptive_alpha:
        effective_alpha, act_dev = adaptive_alpha_from_distribution(
            act_probs,
            act_freq_masked,
            base_alpha=args.alpha,
            deviation_threshold=args.alpha_deviation_threshold,
            max_boost=args.alpha_max_boost,
        )
        print(
            f"\nAdaptive alpha enabled: base={args.alpha:.3f}, "
            f"effective={effective_alpha:.3f}, action_L1_deviation={act_dev:.3f}"
        )

    act_target = compute_target_counts(act_freq_masked, N, argmax_act_counts, effective_alpha)

    pt_target = compute_target_counts(pt_freq, N, argmax_pt_counts, effective_alpha)

    print(f"\n=== After class-aware blending (alpha={effective_alpha}) ===")
    print("Action targets:")
    for c in range(n_act):
        if act_target[c] != argmax_act_counts[c]:
            delta = act_target[c] - argmax_act_counts[c]
            sign = "+" if delta >= 0 else ""
            print(f"  class {int(act_classes[c]):>2d}  argmax={argmax_act_counts[c]:>4d}  target={act_target[c]:>4d}  ({sign}{delta})")
    print("Point targets:")
    for c in range(n_pt):
        if pt_target[c] != argmax_pt_counts[c]:
            delta = pt_target[c] - argmax_pt_counts[c]
            sign = "+" if delta >= 0 else ""
            print(f"  class {int(pt_classes[c]):>2d}  argmax={argmax_pt_counts[c]:>4d}  target={pt_target[c]:>4d}  ({sign}{delta})")

    # Run assignment
    act_assigned = class_aware_assign(act_probs, act_target)
    pt_assigned  = class_aware_assign(pt_probs,  pt_target)

    # Build submission
    pred_rows = []
    for i, rid in enumerate(rally_ids):
        pred_rows.append({
            "rally_uid":      int(rid),
            "serverGetPoint": int(rly_avg[i] > 0),
            "actionId":       int(act_classes[act_assigned[i]]),
            "pointId":        int(pt_classes[pt_assigned[i]]),
        })

    pred_df = pd.DataFrame(pred_rows).sort_values("rally_uid")
    out = (pd.read_csv(args.sample)
             .drop(columns=["serverGetPoint", "pointId", "actionId"], errors="ignore")
             .merge(pred_df, on="rally_uid", how="left"))
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]
    out.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}  (alpha={effective_alpha}, models={len(models)})")
    print(out.head())

    # Compare to argmax baseline
    n_diff_act = (np.array([int(act_classes[a]) for a in act_assigned]) !=
                  np.array([int(act_classes[a]) for a in argmax_act])).sum()
    n_diff_pt = (np.array([int(pt_classes[a]) for a in pt_assigned]) !=
                 np.array([int(pt_classes[a]) for a in argmax_pt])).sum()
    print(f"\nDifferences from argmax: action {n_diff_act}/{N} ({n_diff_act/N*100:.1f}%), "
          f"point {n_diff_pt}/{N} ({n_diff_pt/N*100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",       default="data/train.csv")
    ap.add_argument("--test",        default="data/test.csv")
    ap.add_argument("--sample",      default="data/sample_submission.csv")
    ap.add_argument("--encoder_out", default="output/encoders/encoder.pkl")
    ap.add_argument("--model_out",   default="output/models/model.pt")
    ap.add_argument("--out",         default="output/submissions/submission_classaware.csv")
    ap.add_argument("--alpha",       type=float, default=1.0,
                    help="0 = pure argmax, 1 = full train-freq targeting")
    ap.add_argument("--adaptive_alpha", action="store_true",
                    help="Automatically increase alpha when argmax distribution drifts from train frequency.")
    ap.add_argument("--alpha_deviation_threshold", type=float, default=0.20,
                    help="L1 deviation threshold for adaptive alpha activation.")
    ap.add_argument("--alpha_max_boost", type=float, default=0.30,
                    help="Maximum additive boost for adaptive alpha.")
    ap.add_argument("--emb",    type=int, default=32)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--nhead",  type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()
    main(args)
