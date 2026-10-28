#!/usr/bin/env python3
"""従来手法の比較相手：辞書＋マスク言語モデル（RoBERTa-base、ローカル・無料）。

  uv run python mlm_baseline.py --set v2dev     # 開発用で閾値を決めて保存する（results/v2dev/mlm_threshold.json）
  uv run python mlm_baseline.py --set v2test    # 開発用で決めた閾値のまま、評価用に当てる

やり方（取り違えを見つける従来の定番。Hirst & Budanitsky 2005 の考え方を言語モデルでやる）:
  - 辞書にない語（小文字で始まるもの）→ 誤り（p = 1）
  - 辞書にある語 → 編集距離 1 の辞書の語（頻度 500 以上、語形変化だけの違いは除く）を候補にし、
    文の中でその語を候補に置き換えたときの「らしさ」（擬似対数尤度：語の各サブワードを 1 つずつ隠して当てる確率の対数の和）を比べる。
    Δ = max(候補のらしさ) − (元の語のらしさ)。Δ が大きいほど、別の語の方が文に合う
  - 開発用で、単語単位の F1 が最も高くなる Δ の閾値 τ を決め、p = sigmoid(Δ − τ) として保存する（probe.py の閾値 0.5 が τ に当たる）

注意：候補の作り方は、評価データの取り違えの作り方（make_data_v2.py）と同じ。正解の語が必ず候補に入るので、この比較相手には有利に出る
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import torch
from spellchecker import SpellChecker
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_data_v2 import match_case, real_word_candidates  # noqa: E402
from probe import tokenize  # noqa: E402

MODEL = "FacebookAI/roberta-base"
spell = SpellChecker()


class PLL:
    def __init__(self):
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL).to(self.dev).eval()
        self.cache: dict[tuple[str, int, int], float] = {}

    @torch.no_grad()
    def scores(self, variants: list[tuple[str, int, int]], batch: int = 64) -> list[float]:
        """variants: (文, 語の開始, 語の終わり)。その語の擬似対数尤度を返す。"""
        jobs = []  # (variant 番号, input_ids, 隠す位置, 正解の id)
        for vi, (text, s, e) in enumerate(variants):
            if (text, s, e) in self.cache:
                continue
            enc = self.tok(text, return_offsets_mapping=True)
            pos = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < e and s < b]
            for k in pos:
                ids = list(enc["input_ids"])
                gold = ids[k]
                ids[k] = self.tok.mask_token_id
                jobs.append((vi, ids, k, gold))
        sums: dict[int, float] = {}
        for i in range(0, len(jobs), batch):
            chunk = jobs[i:i + batch]
            L = max(len(j[1]) for j in chunk)
            ids = torch.full((len(chunk), L), self.tok.pad_token_id)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, (_, x, _, _) in enumerate(chunk):
                ids[r, :len(x)] = torch.tensor(x)
                att[r, :len(x)] = 1
            logits = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).logits
            lp = torch.log_softmax(logits.float(), dim=-1).cpu()
            for r, (vi, _, k, gold) in enumerate(chunk):
                sums[vi] = sums.get(vi, 0.0) + lp[r, k, gold].item()
        for vi, v in sums.items():
            self.cache[variants[vi]] = v
        return [self.cache[v] for v in variants]


def deltas(items: list[dict], pll: PLL) -> list[dict]:
    out = []
    t0 = time.time()
    for n, it in enumerate(items):
        text = it["text"]
        toks = tokenize(text)
        row = []
        for w, s, e in toks:
            if w[0].islower() and spell.unknown([w.lower()]):
                row.append(math.inf)  # 辞書にない語
                continue
            cands = real_word_candidates(w)
            if not cands:
                row.append(-math.inf)
                continue
            variants = [(text, s, e)] + [(text[:s] + match_case(w, c) + text[e:], s, s + len(c)) for c in cands]
            sc = pll.scores(variants)
            row.append(max(sc[1:]) - sc[0])
        out.append({"id": it["id"], "delta": row})
        if n % 50 == 49:
            print(f"  {n + 1}/{len(items)} 文（{time.time() - t0:.0f} 秒）", file=sys.stderr)
    return out


def f1_at(items, rows, tau):
    tp = fp = fn = 0
    for it, r in zip(items, rows):
        gold = {e["index"] for e in it["errors"]}
        flag = {i for i, d in enumerate(r["delta"]) if d >= tau}
        tp += len(flag & gold); fp += len(flag - gold); fn += len(gold - flag)
    p = tp / (tp + fp) if tp + fp else 0
    r_ = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r_ / (p + r_) if p + r_ else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    args = ap.parse_args()
    items = [json.loads(l) for l in (HERE / "data" / args.set / "eval.jsonl").read_text().splitlines()]
    res = HERE / "results" / args.set
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = deltas(items, PLL())
    secs = time.time() - t0
    thr_file = HERE / "results" / "v2dev" / "mlm_threshold.json"
    if args.set == "v2dev":
        finite = sorted({d for r in rows for d in r["delta"] if math.isfinite(d)})
        grid = [finite[int(q * (len(finite) - 1))] for q in [i / 200 for i in range(201)]]
        tau = max(grid, key=lambda t: f1_at(items, rows, t))
        thr_file.write_text(json.dumps({"tau": tau, "f1_dev": f1_at(items, rows, tau)}, indent=1))
        print(f"開発用で決めた閾値 τ = {tau:.3f}（F1 {f1_at(items, rows, tau):.3f}）")
    tau = json.loads(thr_file.read_text())["tau"]

    def sig(d):
        if d == math.inf:
            return 1.0
        if d == -math.inf:
            return 0.0
        return 1 / (1 + math.exp(-(d - tau)))
    with open(res / "pred_mlm.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "p": [sig(d) for d in r["delta"]],
                                "latency": secs * 1000 / len(items)}) + "\n")
    print(f"{len(items)} 文、{secs:.0f} 秒（1 文 {secs * 1000 / len(items):.0f} ms）→ {res / 'pred_mlm.jsonl'}")


if __name__ == "__main__":
    main()
