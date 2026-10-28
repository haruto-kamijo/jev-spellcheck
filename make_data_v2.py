#!/usr/bin/env python3
"""第 2 段の評価データを作る。第 1 段と違い、文も誤りも手で選ばない。

  uv run python make_data_v2.py     # data/v2dev/eval.jsonl（開発用 300 文）と data/v2test/eval.jsonl（評価用 1,000 文）

材料（どちらも CC BY-SA 4.0。作ったデータも CC BY-SA 4.0 で出す）:
  文    Wikibooks Cookbook の手順（gossminn/wikibooks-cookbook、2024-07-31 取得版）。料理名はレシピの題
  綴り  Wikipedia「Lists of common misspellings/For machines」（人がよく書く誤字と正しい綴りの組）
置き場は $PLAYGROUND_MODELS/datasets/（Playground の共有の置き場。PoC のフォルダには置かない）

作り方:
  - レシピを乱数で開発用と評価用に分け、1 レシピから 1 手順だけ取る（同じレシピの文が両方に入らない）
  - 手順は 6〜30 語。辞書にない小文字の語を含む手順は捨てる（元の文の誤字を正解の「正しい語」にしないため）。
    大文字で始まる辞書にない語（Parmesan など固有名詞）は残す
  - 半分の手順に誤りを 1〜2 か所入れる（1 か所:2 か所 = 2:1）。種類は半々:
      typo       Wikipedia の誤字の組で、その語の誤字があれば使う。無ければ 1 文字の操作（記録上は typo_synthetic）
      real_word  辞書の語のうち、編集距離 1 の別の語に替える（Hirst & Budanitsky 2005 の作り方）。
                 語形変化だけの違い（onion→onions など）と、珍しい語（頻度 500 未満。rewove・leet など）は除く。
                 対象は 3 字以上の内容語
  - typo は、その文の中に Wikipedia の組がある語を先に探し、どれにも無いときだけ 1 文字の操作にする
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import re
import string
from collections import Counter

from spellchecker import SpellChecker

HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(os.environ.get("PLAYGROUND_MODELS", pathlib.Path.home() / "Playground/_shared/models")) / "datasets"
WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")  # 第 1 段と同じ単語分割
SEED = 20261028
N_DEV, N_TEST = 300, 1000
STOP = set("""a an the and or but if then than so to of in on at by for with from into onto over under about as is are was
were be been being it its this that these those there here you your we our they their he she his her them i me my not no
do does did done can could will would should may might must just also very too all any each some more most such only own
same other until while when where which who whom what why how up down out off again further once both few nor""".split())

spell = SpellChecker()
FREQ = spell.word_frequency


def load_steps() -> list[dict]:
    d = json.loads((SRC / "wikibooks-cookbook.mini.json").read_text())
    out = []
    for r in d:
        rd = r["recipe_data"]
        steps = [l["text"].strip() for l in rd["text_lines"]
                 if l["line_type"] == "ol" and (l["section"] or "").lower() in ("procedure", "preparation", "directions", "method", "instructions")]
        if steps:
            out.append({"url": rd["url"], "dish": rd["title"], "steps": steps})
    return out


def clean_enough(text: str) -> bool:
    toks = WORD.findall(text)
    if not 6 <= len(toks) <= 30:
        return False
    unknown = [w for w in toks if w[0].islower() and spell.unknown([w.lower()])]
    return not unknown


def load_misspellings() -> dict[str, list[str]]:
    rev: dict[str, list[str]] = {}
    for line in (SRC / "wiki-misspellings.txt").read_text().splitlines():
        if "->" not in line:
            continue
        bad, good = (s.strip() for s in line.split("->", 1))
        for g in good.split(","):
            g = g.strip()
            if g and " " not in g and " " not in bad and bad.isalpha():
                rev.setdefault(g.lower(), []).append(bad.lower())
    return rev


def inflection_only(a: str, b: str) -> bool:
    a, b = sorted([a, b], key=len)
    return any(b == a + s for s in ("s", "es", "d", "ed", "ing", "er", "r", "y", "ly")) or \
        any(b[: -len(s)] == a[: -1] for s in ("ing", "ed", "es") if b.endswith(s) and len(a) > 2)


def edit1(w: str) -> set[str]:
    letters = string.ascii_lowercase
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return {a + b[1:] for a, b in splits if b} | {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1} | \
        {a + c + b[1:] for a, b in splits if b for c in letters} | {a + c + b for a, b in splits for c in letters}


def real_word_candidates(w: str) -> list[str]:
    lw = w.lower()
    cands = [c for c in edit1(lw) if c != lw and FREQ[c] >= 500 and len(c) >= 2 and not inflection_only(lw, c)]
    return sorted(cands)


def synthetic_typo(w: str, rng: random.Random) -> str | None:
    lw = w.lower()
    for _ in range(30):
        i = rng.randrange(1, len(lw) - 1)
        op = rng.choice(["swap", "drop", "double", "sub"])
        t = {"swap": lw[:i] + lw[i + 1:i + 2] + lw[i] + lw[i + 2:],
             "drop": lw[:i] + lw[i + 1:],
             "double": lw[:i] + lw[i] + lw[i:],
             "sub": lw[:i] + rng.choice("aeioustnrl") + lw[i + 1:]}[op]
        if t != lw and not spell.known([t]):
            return t
    return None


def match_case(src: str, new: str) -> str:
    return new.capitalize() if src[0].isupper() else new


def corrupt(text: str, rng: random.Random, miss: dict[str, list[str]]) -> tuple[str, list[dict]]:
    toks = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]
    content = [i for i, (w, _, _) in enumerate(toks) if len(w) >= 3 and w.lower() not in STOP and "-" not in w and "'" not in w]
    n_err = rng.choice([1, 1, 2])
    picked: dict[int, tuple[str, str]] = {}
    for _ in range(n_err):
        pool = [i for i in content if i not in picked and all(abs(i - j) > 1 for j in picked)]  # 隣り合わせにしない
        rng.shuffle(pool)
        kind = rng.choice(["typo", "real_word"])
        for i in pool:
            w = toks[i][0]
            if kind == "real_word":
                c = real_word_candidates(w)
                if c:
                    picked[i] = ("real_word", match_case(w, rng.choice(c)))
                    break
            else:
                if w.lower() in miss:
                    picked[i] = ("typo", match_case(w, rng.choice(sorted(miss[w.lower()]))))
                    break
        else:
            if kind == "typo":  # Wikipedia の組が文のどの語にも無いときだけ、1 文字の操作
                for i in pool:
                    w = toks[i][0]
                    t = synthetic_typo(w, rng) if len(w) >= 4 else None
                    if t:
                        picked[i] = ("typo_synthetic", match_case(w, t))
                        break
    out = text
    for i in sorted(picked, reverse=True):
        _, s, e = toks[i]
        out = out[:s] + picked[i][1] + out[e:]
    new = list(WORD.finditer(out))
    assert len(new) == len(toks), (text, out)
    errors = [{"index": i, "word": new[i].group(), "correct": toks[i][0], "type": picked[i][0],
               "start": new[i].start(), "end": new[i].end()} for i in sorted(picked)]
    return out, errors


def build(recipes: list[dict], n: int, rng: random.Random, miss, prefix: str) -> list[dict]:
    rows = []
    for rec in recipes:
        ok = [s for s in rec["steps"] if clean_enough(s)]
        if not ok:
            continue
        rows.append({"dish": rec["dish"], "text": rng.choice(ok), "source": rec["url"]})
        if len(rows) == n:
            break
    assert len(rows) == n, f"{prefix}: {len(rows)} < {n}"
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flags)
    out = []
    for k, (row, bad) in enumerate(zip(rows, flags)):
        text, errors = (corrupt(row["text"], rng, miss) if bad else (row["text"], []))
        if bad and not errors:
            raise RuntimeError(f"誤りを入れられなかった: {row['text']}")
        out.append({"id": f"{prefix}{k:04d}", "pair": None, "dish": row["dish"], "text": text, "errors": errors,
                    "source": row["source"]})
    return out


def main():
    rng = random.Random(SEED)
    miss = load_misspellings()
    recipes = load_steps()
    rng.shuffle(recipes)
    cut = len(recipes) * 3 // 10            # 3 割を開発用のレシピに
    sets = {"v2dev": build(recipes[:cut], N_DEV, rng, miss, "d"), "v2test": build(recipes[cut:], N_TEST, rng, miss, "t")}
    for name, rows in sets.items():
        p = HERE / "data" / name / "eval.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        kinds = Counter(e["type"] for r in rows for e in r["errors"])
        print(f"{name}: {len(rows)} 文（誤りあり {sum(bool(r['errors']) for r in rows)}、誤り {sum(kinds.values())} か所 {dict(kinds)}）→ {p}")
        (p.parent / "SOURCE.md").write_text(
            "# 出典とライセンス\n\n"
            "このフォルダのデータは CC BY-SA 4.0。次の 2 つから作った（誤りは機械的に入れた。元の文の作者の誤りではない）。\n\n"
            "- 文：Wikibooks Cookbook（https://en.wikibooks.org/wiki/Cookbook:Table_of_Contents）。各行の `source` が元のページ。"
            "Hugging Face の gossminn/wikibooks-cookbook（2024-07-31 取得版）から\n"
            "- 綴りの誤り：Wikipedia「Lists of common misspellings/For machines」（2026-10-28 取得）\n")


if __name__ == "__main__":
    main()
