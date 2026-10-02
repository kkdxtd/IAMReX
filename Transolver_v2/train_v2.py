#!/usr/bin/env python3
"""
Transolver-v2 训练脚本 —— 带完整的过拟合排除机制

相对 `../Transolver/train_transolver_w.py` 的改动
------------------------------------------------
| 项 | 旧版 | 本版 |
| :-- | :-- | :-- |
| 数据划分 | train/val 两分 | **train/val/test 三分**，test 只在最后评估一次 |
| 划分方式 | 按 case 随机 | 按 case 随机 **或按相位卦限留出**（真外推） |
| 停止条件 | 跑满 epochs | **早停**（patience），并恢复最优 val 权重 |
| 正则 | 无 | **weight decay + dropout** |
| 模型 | physicsnemo，778,641 参数固定 | **自包含 TransolverSlim，容量可调**（orig/slim/tiny） |
| 诊断 | 无 | **学习曲线**（`--data-frac`）、**随机标签对照**（`--shuffle-labels`） |
| 外部检验 | 无 | **`--extra-test`** 在从未见过的数据集上评估（不变性验证） |

为什么这些机制是必要的
----------------------
本任务的目标 `w` 是亚格相位的普适函数（见 `../dataset5workspace_shape/DATASET_PLAN.md` §0），
输入空间只是 3 维立方体 `[-0.5,0.5]³`。因此：

* **真正的独立样本是"笼"，不是"笼记录"**。54,000 行只对应 2,000 个独立样本，
  脚本一律按笼计数并在报告里打印 `参数量 / 训练笼数` 之比。
* **按 case 随机划分本质上是 iid 划分**（各 case 只差一个随机亚格偏移），
  val 与 train 同分布，所以"val 不涨"并不足以证明泛化。
  `--split octant` 留出相位立方体的一个卦限完全不训练，才是真外推检验。

用法
----
```bash
# 基线：slim 模型 + 三分划分 + 早停
python3 train_v2.py --data ../dataset5workspace_shape/batch_data_shape_sphere \
                    --preset slim --out out_slim

# 容量对照（回应"参数量 ≫ 样本量"的质疑）
for p in orig slim tiny; do
  python3 train_v2.py --data ... --preset $p --out out_$p
done

# 真外推：留出一个相位卦限
python3 train_v2.py --data ... --preset slim --split octant --out out_octant

# 学习曲线（数据量是否够）
for f in 0.02 0.05 0.1 0.25 0.5 1.0; do
  python3 train_v2.py --data ... --preset slim --data-frac $f --out out_lc_$f
done

# 随机标签对照（容量诊断，Zhang et al. 2017）
python3 train_v2.py --data ... --preset slim --shuffle-labels --out out_shuffle

# 合并多个数据集 + 在其余配置上做不变性检验
python3 train_v2.py --data ../dataset5workspace_shape/batch_data_aniso_k1 \
                    --extra-test ../dataset5workspace_shape/batch_data_rot_45 \
                                 ../dataset5workspace_shape/batch_data_shape_triaxial \
                    --preset slim --out out_inv
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from IAMReX.Transolver_v2.transolver_slim import build_transolver, count_params, PRESETS  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

FEATURES = ["rx_hat", "ry_hat", "rz_hat"]
TARGET = "w"
N_CAGE = 27


# ---------------------------------------------------------------- 数据
def load_cages(data_dirs):
    """读入一个或多个数据集目录 -> X [G,27,3], y [G,27,1], case_ids [G]

    合并多个目录时给 case_id 加目录名前缀，避免不同数据集的 case_0 撞名。
    """
    Xs, ys, cids = [], [], []
    for d in data_dirs:
        f = os.path.join(d, "training_cage.parquet")
        if not os.path.exists(f):
            raise SystemExit(f"未找到训练表: {f}")
        tag = os.path.basename(os.path.normpath(d))
        df = pd.read_parquet(f).sort_values(["case_id", "lp_id"]).reset_index(drop=True)
        n_bad = 0
        for (cid, _lp), g in df.groupby(["case_id", "lp_id"]):
            if len(g) != N_CAGE:          # 截断笼（边界处）直接丢弃，不做 padding
                n_bad += 1
                continue
            Xs.append(g[FEATURES].values.astype(np.float32))
            ys.append(g[[TARGET]].values.astype(np.float32))
            cids.append(f"{tag}/{cid}")
        print(f"[数据] {d}: 笼数={len(df)//N_CAGE}" + (f"（丢弃非27点笼 {n_bad} 个）" if n_bad else ""))
    return np.stack(Xs), np.stack(ys), np.array(cids)


def cage_phase(X):
    """每个笼的 3 维亚格相位：rel_hat = 相位 + 整数偏移，故相位 = mean(rel_hat - round(rel_hat))。"""
    return np.mean(X - np.round(X), axis=1)          # [G, 3]


def split_by_case(case_ids, val_frac, test_frac, seed):
    uniq = sorted(set(case_ids.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n:
        raise SystemExit(f"case 数太少（{n}），无法三分。请合并更多数据集或调小 --val-frac/--test-frac")
    test_c, val_c = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    idx = np.arange(len(case_ids))
    te = idx[[c in test_c for c in case_ids]]
    va = idx[[c in val_c for c in case_ids]]
    tr = idx[[(c not in test_c and c not in val_c) for c in case_ids]]
    return tr, va, te, {"test_cases": sorted(test_c), "val_cases": sorted(val_c)}


def split_by_octant(X, val_frac, seed, holdout_octant=0):
    """留出相位立方体的一个卦限作为 test —— 输入空间里的真外推。

    相位 ∈ [-0.5,0.5]³，按三个分量的符号分成 8 个卦限，整块留出不参与训练。
    """
    ph = cage_phase(X)
    oct_id = ((ph[:, 0] > 0).astype(int) * 4 + (ph[:, 1] > 0).astype(int) * 2
              + (ph[:, 2] > 0).astype(int))
    idx = np.arange(len(X))
    te = idx[oct_id == holdout_octant]
    rest = idx[oct_id != holdout_octant]
    if len(te) == 0:
        raise SystemExit(f"卦限 {holdout_octant} 内没有样本")
    rng = np.random.default_rng(seed)
    rest = rest.copy()
    rng.shuffle(rest)
    n_val = max(1, int(round(len(rest) * val_frac)))
    return rest[n_val:], rest[:n_val], te, {"holdout_octant": int(holdout_octant),
                                            "n_test_cages": int(len(te))}


# ---------------------------------------------------------------- 指标
def evaluate(model, X, y, ym, ys, dev, batch=256):
    """返回反标准化后的回归指标。X/y 为**已标准化**的张量。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            b = X[i:i + batch].to(dev)
            preds.append(model(fx=b, embedding=b).cpu().numpy())
    yp = np.concatenate(preds) * ys + ym                    # [G,27,1] 物理量纲
    yt = y.numpy() * ys + ym
    r = (yp - yt).ravel()
    t = yt.ravel()
    ss = float(np.sum((t - t.mean()) ** 2))
    sw = yp.sum(axis=1).ravel()
    return {
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "r2": 1 - float(np.sum(r ** 2)) / ss if ss > 0 else float("nan"),
        "relL2": float(np.linalg.norm(r) / (np.linalg.norm(t) + 1e-12)),
        "sum_w_mean": float(sw.mean()),
        "sum_w_dev_max": float(np.max(np.abs(sw - 1.0))),
        "n_cages": int(len(yp)),
    }


def make_plots(out, hist, extra=None):
    if not _HAS_MPL:
        return
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(hist["epoch"], hist["train"], label="train")
    plt.plot(hist["epoch"], hist["val"], label="val")
    if hist.get("test"):
        plt.plot(hist["epoch"], hist["test"], "--", label="test (仅绘图，不参与选择)")
    if hist.get("best_epoch"):
        plt.axvline(hist["best_epoch"], color="gray", ls=":", lw=1, label="早停选中")
    plt.xlabel("epoch"); plt.ylabel("loss (标准化 MSE)"); plt.yscale("log")
    plt.title("train / val / test loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "plots", "loss_curve.png"), dpi=130)
    plt.close()


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="Transolver-v2：带过拟合排除机制的训练")
    ap.add_argument("--data", nargs="+", required=True, help="一个或多个数据集目录")
    ap.add_argument("--extra-test", nargs="*", default=[],
                    help="额外的外部测试集目录（从未参与训练，用于不变性验证）")
    ap.add_argument("--out", default="out_v2")

    # 模型容量
    ap.add_argument("--preset", choices=list(PRESETS), default="slim")
    ap.add_argument("--n-hidden", type=int, default=None, help="覆盖预设")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-head", type=int, default=None)
    ap.add_argument("--slice-num", type=int, default=None)
    ap.add_argument("--mlp-ratio", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.0)

    # 划分
    ap.add_argument("--split", choices=["case", "octant"], default="case",
                    help="case=按算例随机(iid)；octant=留出一个相位卦限(真外推)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2, help="仅 --split case 用")
    ap.add_argument("--holdout-octant", type=int, default=0, help="仅 --split octant 用，0..7")

    # 优化
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-cages", type=int, default=128)
    ap.add_argument("--lam-sum", type=float, default=0.0, help="单位分解软约束权重")

    # 早停
    ap.add_argument("--patience", type=int, default=60, help="val 连续多少轮无改善则停；0=关闭")
    ap.add_argument("--min-delta", type=float, default=0.0, help="视为改善的最小相对降幅")

    # 诊断
    ap.add_argument("--data-frac", type=float, default=1.0, help="只用训练集的这一比例（画学习曲线）")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="打乱训练标签（容量诊断：能拟合随机标签=有记忆能力）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. 数据与三分划分 ----
    X, y, case_ids = load_cages(args.data)
    if args.split == "case":
        tr, va, te, split_info = split_by_case(case_ids, args.val_frac, args.test_frac, args.seed)
    else:
        tr, va, te, split_info = split_by_octant(X, args.val_frac, args.seed, args.holdout_octant)

    # 学习曲线：只截取训练集的一部分（val/test 保持不变，保证可比）
    if args.data_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        tr = rng.permutation(tr)[:max(1, int(round(len(tr) * args.data_frac)))]

    print(f"\n[划分] {args.split}  train={len(tr)} 笼  val={len(va)} 笼  test={len(te)} 笼")
    print(f"        {split_info}")

    # ---- 2. 标准化（**只用训练集统计量**，避免信息泄漏）----
    xm = X[tr].reshape(-1, 3).mean(0)
    xs = X[tr].reshape(-1, 3).std(0)
    xs = np.where(xs < 1e-12, 1.0, xs)
    ym = float(y[tr].mean())
    ysd = float(y[tr].std()) or 1.0

    Xn = torch.tensor((X - xm) / xs, dtype=torch.float32)
    yn = torch.tensor((y - ym) / ysd, dtype=torch.float32)

    if args.shuffle_labels:                      # 只打乱训练集标签，val/test 保持真值
        perm = torch.randperm(len(tr))
        yn[tr] = yn[tr][perm]
        print("[诊断] 已打乱训练标签（随机标签对照实验）")

    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xn[tr], yn[tr]),
        batch_size=args.batch_cages, shuffle=True)

    # ---- 3. 模型 ----
    model = build_transolver(
        args.preset, n_hidden=args.n_hidden, n_layers=args.n_layers,
        n_head=args.n_head, slice_num=args.slice_num, mlp_ratio=args.mlp_ratio,
        dropout=args.dropout).to(dev)
    n_par = count_params(model)
    ratio = n_par / max(1, len(tr))
    print(f"[模型] preset={args.preset}  参数量={n_par:,}  训练笼数={len(tr)}  "
          f"参数/笼={ratio:.1f}\n        cfg={model.cfg}\n[设备] {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse = nn.MSELoss()

    # ---- 4. 训练 + 早停 ----
    hist = {"epoch": [], "train": [], "val": [], "test": []}
    best = {"val": float("inf"), "epoch": 0, "state": None}
    stale = 0
    t0 = time.time()

    def _loss_on(idx):
        model.eval()
        tot, nb = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(idx), args.batch_cages):
                j = idx[i:i + args.batch_cages]
                bx, by = Xn[j].to(dev), yn[j].to(dev)
                tot += mse(model(fx=bx, embedding=bx), by).item()
                nb += 1
        return tot / max(1, nb)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for bx, by in ld:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad()
            pred = model(fx=bx, embedding=bx)
            loss = mse(pred, by)
            if args.lam_sum > 0:
                s = (pred * ysd + ym).sum(dim=1)          # 反标准化后按整笼求和
                loss = loss + args.lam_sum * ((s - 1.0) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        v = _loss_on(va)
        # test 每轮也记一次，**仅供画图**；选择权重、早停都只看 val
        t = _loss_on(te)
        hist["epoch"].append(ep); hist["train"].append(tot / max(1, nb))
        hist["val"].append(v); hist["test"].append(t)

        if v < best["val"] * (1.0 - args.min_delta):
            best = {"val": v, "epoch": ep,
                    "state": {k: p.detach().clone() for k, p in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1

        if ep % max(1, args.epochs // 20) == 0 or ep == 1:
            print(f"  ep {ep:4d}  train {tot/max(1,nb):.4e}  val {v:.4e}  test {t:.4e}"
                  f"  (best@{best['epoch']}, stale {stale})")

        if args.patience and stale >= args.patience:
            print(f"[早停] val 连续 {args.patience} 轮无改善，于 ep {ep} 停止；"
                  f"恢复 ep {best['epoch']} 的权重")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    train_sec = time.time() - t0

    # ---- 5. 最终评估：test 到此刻才用于报告 ----
    res = {
        "train": evaluate(model, Xn[tr], yn[tr], ym, ysd, dev),
        "val":   evaluate(model, Xn[va], yn[va], ym, ysd, dev),
        "test":  evaluate(model, Xn[te], yn[te], ym, ysd, dev),
    }
    print(f"\n{'集合':6s} {'笼数':>6s} {'R²':>10s} {'相对L2':>9s} {'Σw最大偏差':>11s}")
    for k in ("train", "val", "test"):
        m = res[k]
        print(f"{k:6s} {m['n_cages']:6d} {m['r2']:10.5f} {m['relL2']*100:8.2f}% {m['sum_w_dev_max']:11.4f}")
    gap = res["train"]["relL2"] and res["val"]["relL2"] / res["train"]["relL2"]
    print(f"\n过拟合诊断: val/train 相对L2 之比 = {gap:.3f}（≈1 表示无过拟合）")
    print(f"           test/train 之比       = {res['test']['relL2']/max(res['train']['relL2'],1e-12):.3f}")

    # ---- 6. 外部测试集（不变性验证）----
    extra = {}
    for d in args.extra_test:
        Xe, ye, _ = load_cages([d])
        Xe_n = torch.tensor((Xe - xm) / xs, dtype=torch.float32)
        ye_n = torch.tensor((ye - ym) / ysd, dtype=torch.float32)
        extra[os.path.basename(os.path.normpath(d))] = evaluate(model, Xe_n, ye_n, ym, ysd, dev)
    if extra:
        print(f"\n[外部测试集 —— 从未参与训练]")
        for k, m in extra.items():
            print(f"  {k:32s} R²={m['r2']:.5f}  相对L2={m['relL2']*100:6.2f}%")
        r2s = [m["r2"] for m in extra.values()] + [res["test"]["r2"]]
        print(f"  → R² 极差 = {max(r2s)-min(r2s):.2e}（越小越说明对形状/旋转/各向异性不敏感）")

    # ---- 7. 落盘 ----
    metrics = {
        "args": vars(args),
        "model_cfg": model.cfg,
        "n_params": n_par,
        "n_train_cages": int(len(tr)),
        "params_per_train_cage": ratio,
        "split": split_info,
        "n_cages": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "best_epoch": best["epoch"],
        "epochs_run": len(hist["epoch"]),
        "train_seconds": train_sec,
        "metrics": res,
        "extra_test": extra,
        "overfit_ratio_val_over_train": gap,
        "history": hist,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg},
               os.path.join(args.out, "model_best.pt"))
    np.savez(os.path.join(args.out, "norm_stats.npz"),
             xm=xm, xs=xs, ym=ym, ys=ysd, features=np.array(FEATURES))
    if not args.no_plots:
        hist["best_epoch"] = best["epoch"]
        make_plots(args.out, hist)
    print(f"\n[产物] -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
