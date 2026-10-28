#!/usr/bin/env python3
"""第 2 段の評価用の結果を、区間つきでまとめる（送信なし。results/<set>/pred_*.jsonl を読むだけ）。

  uv run python analyze_v2.py                 # results/v2test/summary.md
  uv run python analyze_v2.py --audit         # 取り違えの目視確認の表（data/v2test/audit_realword.csv）を作る

- 閾値は README で評価用の前に決めたもの（Jev A は 0.85、ほかは 0.5）
- 95% 区間は文を単位にしたブートストラップ（2,000 回、乱数の種は固定）
- 確率を返す方式は、閾値に依らない比べ方として PR 曲線の下の面積（平均適合率）も出す
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics

HERE = pathlib.Path(__file__).resolve().parent
SET = "v2test"
THRESH = {"jevA": 0.85}
ORDER = ["spell", "lt", "mlm", "jevA", "jevA2", "jevA5", "claude-haiku", "claude-sonnet"]
LABEL = {"spell": "辞書（pyspellchecker）", "lt": "LanguageTool", "mlm": "辞書＋RoBERTa-base（ローカル）",
         "jevA": "Jev A（構想案の聞き方、閾値 0.85）", "jevA2": "Jev A2（この語だけを見て）",
         "jevA5": "Jev A5（調理工程として＋料理名）", "claude-haiku": "Claude Haiku 4.5（API）",
         "claude-sonnet": "Claude Sonnet 5.5（API, effort low）"}
PROB = {"mlm", "jevA", "jevA2", "jevA5"}


def load(name):
    return [json.loads(l) for l in (HERE / name).read_text().splitlines()]


def counts(it, p, th):
    gold = {e["index"]: ("typo" if e["type"].startswith("typo") else e["type"]) for e in it["errors"]}
    flag = {i for i, v in enumerate(p) if v >= th}
    return {"tp": len(flag & gold.keys()), "fp": len(flag - gold.keys()), "fn": len(gold.keys() - flag),
            "rw_hit": sum(1 for i, t in gold.items() if t == "real_word" and i in flag),
            "rw_n": sum(1 for t in gold.values() if t == "real_word"),
            "ty_hit": sum(1 for i, t in gold.items() if t == "typo" and i in flag),
            "ty_n": sum(1 for t in gold.values() if t == "typo"),
            "clean": int(not gold), "alarm": int(not gold and bool(flag))}


def metrics(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
    r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0, "RW": s["rw_hit"] / max(s["rw_n"], 1),
            "TY": s["ty_hit"] / max(s["ty_n"], 1), "FA": s["alarm"] / max(s["clean"], 1)}


def avg_precision(items, preds):
    pts = []
    n_gold = 0
    for it, pr in zip(items, preds):
        gold = {e["index"] for e in it["errors"]}
        n_gold += len(gold)
        pts += [(v, i in gold) for i, v in enumerate(pr["p"])]
    pts.sort(key=lambda x: -x[0])
    tp = 0
    ap = 0.0
    for k, (_, g) in enumerate(pts, 1):
        if g:
            tp += 1
            ap += tp / k
    return ap / n_gold


def ci(vals):
    vals = sorted(vals)
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def fmt(m, lo_hi):
    return f"{m:.0%}（{lo_hi[0]:.0%}〜{lo_hi[1]:.0%}）"


def summary():
    items = load(f"data/{SET}/eval.jsonl")
    rng = random.Random(20261028)
    idx = [rng.randrange(len(items)) for _ in range(2000 * len(items))]
    lines = [f"# 第 2 段 評価用の結果（{SET}、{len(items)} 文、誤り {sum(len(i['errors']) for i in items)} か所）\n",
             "括弧は 95% 区間（文を単位にしたブートストラップ 2,000 回）。閾値は評価用の前に決めたもの。\n",
             "| 方式 | 適合率 | 再現率 | F1 | 取り違えの再現率 | 綴りの再現率 | 正しい文で誤検知 | 平均適合率（PR 曲線） | 応答時間（中央値・95% 点） | 費用（1,000 文） |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for b in ORDER:
        f = HERE / "results" / SET / f"pred_{b}.jsonl"
        if not f.exists():
            lines.append(f"| {LABEL[b]} | 未実施 | | | | | | | | |")
            continue
        by = {r["id"]: r for r in load(f"results/{SET}/pred_{b}.jsonl")}
        preds = [by[it["id"]] for it in items]
        th = THRESH.get(b, 0.5)
        rows = [counts(it, pr["p"], th) for it, pr in zip(items, preds)]
        m = metrics(rows)
        boot = {k: [] for k in m}
        for rep in range(2000):
            mb = metrics([rows[j] for j in idx[rep * len(items):(rep + 1) * len(items)]])
            for k in m:
                boot[k].append(mb[k])
        lat = [pr["latency"] for pr in preds if pr.get("latency") is not None]
        lat_s = f"{statistics.median(lat):.0f} ms・{sorted(lat)[int(0.95 * len(lat)) - 1]:.0f} ms" if lat else "<1 ms"
        if b.startswith("jev"):
            cost = f"${sum(pr.get('tokens', 0) for pr in preds) * 0.042 / 1e6:.3f}"
        elif b.startswith("claude"):
            cost = f"${sum(pr.get('cost', 0) for pr in preds):.2f}"
        else:
            cost = "0"
        ap = f"{avg_precision(items, preds):.2f}" if b in PROB else "-"
        lines.append(f"| {LABEL[b]} | {fmt(m['P'], ci(boot['P']))} | {fmt(m['R'], ci(boot['R']))} | "
                     f"{m['F1']:.2f}（{ci(boot['F1'])[0]:.2f}〜{ci(boot['F1'])[1]:.2f}） | {fmt(m['RW'], ci(boot['RW']))} | "
                     f"{fmt(m['TY'], ci(boot['TY']))} | {fmt(m['FA'], ci(boot['FA']))} | {ap} | {lat_s} | {cost} |")
    out = "\n".join(lines) + "\n"
    (HERE / "results" / SET / "summary.md").write_text(out)
    print(out)


def audit():
    """取り違え 100 か所を無作為に選び、人が「置き換えても正しいままか」を確かめる表を作る。"""
    items = load(f"data/{SET}/eval.jsonl")
    ex = [(it, e) for it in items for e in it["errors"] if e["type"] == "real_word"]
    rng = random.Random(1028)
    pick = rng.sample(ex, 100)
    p = HERE / "data" / SET / "audit_realword.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "id", "dish", "元の語", "置き換えた語", "置き換えた文", "正しいまま？（y=文として正しい／n=誤り）", "メモ"])
        for k, (it, e) in enumerate(pick, 1):
            t = it["text"]
            marked = t[:e["start"]] + "【" + e["word"] + "】" + t[e["end"]:]
            w.writerow([k, it["id"], it["dish"], e["correct"], e["word"], marked, "", ""])
    print(f"{len(pick)} か所 → {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    audit() if a.audit else summary()
