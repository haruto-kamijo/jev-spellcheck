#!/usr/bin/env python3
"""構想案 §6 の未確認事項：state（文）は質問ごとに課金されるか。

  uv run python billing_probe.py          # dry-run
  uv run python billing_probe.py --run    # 10 リクエスト（約 $0.001）

質問の数（1 / 10）と state の長さ（1〜16 文）を変え、返ってきた入力トークン数を並べる。
state が質問ごとに数えられるなら、増分は「質問の数 × state の長さ」に比例する。1 回だけなら「質問の数」と「state の長さ」の和で増える。
"""
from __future__ import annotations

import argparse
import json

from probe import JEV_MODEL, jev_call

BASE = "Stir in the flour and simmer for 10 minutes."
PAD = "Keep the heat low and stir from time to time so that nothing sticks to the bottom of the pan."


def body(n_sent: int, n_q: int) -> dict:
    state = " ".join([BASE] + [PAD] * (n_sent - 1))
    qs = {f"q{i}": {"type": "noul", "instructions": f'Is word #{i + 1} of the first sentence misspelled or wrong for the context?'}
          for i in range(n_q)}
    return {"model": JEV_MODEL, "state": {"cooking_instruction": state}, "questions": qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    grid = [(s, q) for q in (1, 10) for s in (1, 2, 4, 8, 16)]
    if not a.run:
        print(f"{len(grid)} リクエスト。dry-run。実行するには --run")
        return
    rows = []
    for s, q in grid:
        r = jev_call(body(s, q), "billing")
        rows.append((s, q, r["usage"]["input_tokens"]))
    print("| 文の数 | 質問 1 | 質問 10 | 差（質問 9 個ぶん） |\n|---|---|---|---|")
    by = {(s, q): t for s, q, t in rows}
    for s in (1, 2, 4, 8, 16):
        print(f"| {s} | {by[(s, 1)]} | {by[(s, 10)]} | {by[(s, 10)] - by[(s, 1)]} |")
    print("\n差が文の数に依らずほぼ一定なら、state は 1 回だけ数えられている")


if __name__ == "__main__":
    main()
