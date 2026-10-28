#!/usr/bin/env python3
"""英語の調理工程文の誤字・取り違えを、判断モデル Jev で単語ごとに見つける検証（構想案は 2026-10-17）。

  uv run python probe.py eval                              # dry-run: 送るリクエストの件数・概算費用と、例を1件
  uv run python probe.py eval --run --backend spell,lt     # 無料のものだけ先に
  uv run python probe.py eval --run --backend jevA,jevB    # Jev。cache/ にある分は送らない
  uv run python probe.py detect "Stir in the flower and simmer for 10 minuets." --run

バックエンド（--backend にカンマ区切り）:
  spell   pyspellchecker（辞書にない語を光らせる。ローカル・無料）……………… 下限のベースライン
  lt      LanguageTool 公開 API（api.languagetool.org。無料・1分20回まで）…… 構想案 §8 の比較対象
  jevA    方式 A：単語ごとの Noul ＋ 文単位の Noul を1リクエストで（構想案の推奨）
  jevB    方式 B：「誤っている単語はどれか」を全単語＋none の Choice 1問で
  jevfix  ④ 修正候補：正解の誤り位置を与え、綴りチェッカー＋取り違え辞書の候補から Choice で選ばせる
  claude  Claude（構想案 §8 の LLM 比較）。認証は ANTHROPIC_API_KEY か `ant auth login` のプロファイル
  agent   Claude Code のサブエージェントの答えを採点する（`agent-prep` で束を作り、答えを results/agent_answers.json に置く）

1文 = 1リクエスト（jevA/jevB/claude）。同じ文の正誤ペアを1リクエストに混ぜると答えが漏れるので束ねない
（melody-next-note で確認済み）。LanguageTool は規則ベースで漏れが起きないので 40 文ずつ束ねる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "cache"
DATASET = "v1"                     # --set で切り替える。v1 = 第1段（社内判断用・固定）
RESULTS = HERE / "results" / DATASET
DATA = HERE / "data" / DATASET / "eval.jsonl"
WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")  # 構想案 §4 と同じ。数字・記号は判定対象から外す

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"           # 構想案 §7: 再現のため固定版を指定する
JEV_PRICE = 0.042 / 1e6            # 入力 1 トークンあたり（出力は無料）
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_PRICE = (4.00 / 1e6, 20.00 / 1e6)
# 第 2 段の LLM の比較相手。(モデル, 入力と出力の 1 トークンあたりの定価)
CLAUDE_MODELS = {"claude": (CLAUDE_MODEL, CLAUDE_PRICE),
                 "claude-haiku": ("claude-haiku-4-5", (1.00 / 1e6, 5.00 / 1e6)),
                 "claude-sonnet": ("claude-sonnet-5-5", (2.00 / 1e6, 10.00 / 1e6))}
LT_URL = "https://api.languagetool.org/v2/check"
BANDS = (0.5, 0.85)                # 構想案 §5: 0.5 以上は要確認、0.85 以上は確定


def secret(*parts: str) -> str | None:
    """API キーは ~/.config/<サービス>/<ファイル> から読む。このフォルダには置かない。"""
    p = pathlib.Path.home().joinpath(".config", *parts)
    return p.read_text().strip() if p.exists() else None


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]


def load_items() -> list[dict]:
    if not DATA.exists():
        sys.exit(f"{DATA} が無い。v1 なら `uv run python make_data.py`")
    return [json.loads(line) for line in DATA.read_text().splitlines()]


def cache_path(tag: str, body: dict) -> pathlib.Path:
    h = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE / tag / f"{h}.json"


def http_post(url: str, data: bytes, headers: dict, tag: str) -> dict:
    for attempt in range(5):
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2 ** attempt * (3 if tag == "lt" else 1))
                continue
            sys.exit(f"[{tag}] HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        out["_latency_ms"] = round((time.perf_counter() - t0) * 1000)
        return out
    raise RuntimeError("unreachable")


def cached(tag: str, body: dict, send) -> dict:
    """同じリクエストはキャッシュから返す（同じ問いに二度課金しない）。"""
    path = cache_path(tag, body)
    if path.exists():
        return {**json.loads(path.read_text()), "_cached": True}
    out = send()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


# ---------------------------------------------------------------- Jev のリクエスト（構想案 §3・§4）

def jev_request_A(text: str) -> dict:
    """方式 A。文単位の Noul と、単語ごとの Noul を同じリクエストに入れる。"""
    questions = {"has_error": {"type": "noul", "instructions":
                               "Does this cooking instruction contain a misspelled word or a word that is wrong for the context?"}}
    for i, (w, _, _) in enumerate(tokenize(text)):
        questions[f"w{i}"] = {"type": "noul", "instructions":
                              f'In this cooking instruction, is the word "{w}" (word #{i + 1}) misspelled or wrong for the context?'}
    return {"model": JEV_MODEL, "state": {"cooking_instruction": text}, "questions": questions}


def jev_request_A2(text: str, criteria: bool = False) -> dict:
    """方式 A の言い直し。隣の語が誤っていても、その語自体が正しければ No と答えるよう明示する。
    criteria=True なら NoulCriteria（true/false の説明）も付ける（A3）。"""
    base = jev_request_A(text)
    for i, (w, _, _) in enumerate(tokenize(text)):
        q = {"type": "noul", "instructions":
             f'Look only at word #{i + 1}, "{w}", in this cooking instruction. Is this word itself misspelled, '
             "or a real word that is wrong for the context? Answer about this word only: "
             "if it is correct, answer no even when a nearby word is wrong."}
        if criteria:
            q["criteria"] = {"true": f'"{w}" itself is misspelled or is the wrong word (e.g. "flower" for "flour")',
                             "false": f'"{w}" is spelled correctly and fits the context, regardless of the other words'}
        base["questions"][f"w{i}"] = q
    return base


def jev_request_A3(text: str) -> dict:
    return jev_request_A2(text, criteria=True)


FRAME_Q = ('Read this as one step of a recipe{dish_part}. Look only at word #{n}, "{w}". As part of a cooking step '
           "(an ingredient, a tool, a cooking action, a time or temperature, or an ordinary connecting word), "
           "is this word itself misspelled, or a real word that does not make sense in this step? "
           "Answer about this word only: if it is correct, answer no even when a nearby word is wrong.")


def jev_request_dish(text: str, dish: str, frame: bool) -> dict:
    """料理名を state に入れる版。frame=False は A2 の質問のまま（A2d）、True は調理工程として聞く（A5）。"""
    base = jev_request_A4(text, dish) if frame else jev_request_A2(text)
    base["state"] = {"dish": dish, "cooking_instruction": text}
    return base


def jev_request_A4(text: str, dish: str | None = None) -> dict:
    """調理工程として意味が通るかを聞く。dish を渡すと質問の中でも料理名に触れる（A5 用）。"""
    base = jev_request_A(text)
    part = " for the dish in `dish`" if dish else ""
    for i, (w, _, _) in enumerate(tokenize(text)):
        base["questions"][f"w{i}"] = {"type": "noul", "instructions": FRAME_Q.format(dish_part=part, n=i + 1, w=w)}
    return base


BUILDERS = {
    "A": lambda it: jev_request_A(it["text"]),
    "A2": lambda it: jev_request_A2(it["text"]),
    "A3": lambda it: jev_request_A3(it["text"]),
    "A4": lambda it: jev_request_A4(it["text"]),
    "A2d": lambda it: jev_request_dish(it["text"], it["dish"], frame=False),
    "A5": lambda it: jev_request_dish(it["text"], it["dish"], frame=True),
    "B": lambda it: jev_request_B(it["text"]),
}


def jev_request_B(text: str) -> dict:
    """方式 B。全単語＋none を選択肢にした Choice 1問（上限 255 選択肢）。"""
    toks = tokenize(text)
    criteria = {f"{i + 1}:{w}": f'The word "{w}" (word #{i + 1}) is misspelled or wrong for the context'
                for i, (w, _, _) in enumerate(toks[:254])}
    criteria["none"] = "No word is misspelled or wrong for the context"
    return {"model": JEV_MODEL, "state": {"cooking_instruction": text},
            "questions": {"wrong_word": {"type": "choice", "instructions":
                                         "Which word in this cooking instruction is misspelled or wrong for the context?",
                                         "criteria": criteria}}}


_spell = None


def spellchecker():
    global _spell
    if _spell is None:
        from spellchecker import SpellChecker
        _spell = SpellChecker()
    return _spell


def candidates(word: str) -> list[str]:
    """④ 修正候補。綴りチェッカーの候補＋取り違え辞書の逆引き＋元の語（誤検知の抑制用。構想案 §3.2）。"""
    from make_data import CONFUSIONS
    w = word.lower()
    reverse = [c for c, wrongs in CONFUSIONS.items() if w in wrongs]
    sc = spellchecker()
    found = sorted(sc.candidates(w) or [], key=lambda c: (-sc.word_usage_frequency(c), c))[:8]  # 同点は字の順（集合の順は実行ごとに変わる）
    out = []
    for c in [w, *reverse, *found]:
        if c not in out:
            out.append(c)
    return out[:10]


def jev_request_fix(text: str, errors: list[dict]) -> dict:
    questions = {}
    for e in errors:
        crit = {c: f'"{c}"' + (" (keep the word as written)" if c == e["word"].lower() else "")
                for c in candidates(e["word"])}
        questions[f"fix{e['index']}"] = {"type": "choice", "criteria": crit, "instructions":
                                         f'Word #{e["index"] + 1} "{e["word"]}" in this cooking instruction may be wrong. '
                                         "Which word fits the context?"}
    return {"model": JEV_MODEL, "state": {"cooking_instruction": text}, "questions": questions}


def jev_call(body: dict, tag: str) -> dict:
    def send():
        key = os.environ.get("TYPESAFE_API_KEY") or secret("typesafe", "api_key")
        if not key:
            sys.exit("Jev のキーが無い: ~/.config/typesafe/api_key")
        return http_post(JEV_URL, json.dumps(body).encode(),
                         {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, tag)
    return cached(tag, body, send)


# ---------------------------------------------------------------- 各バックエンド → 単語ごとの「誤りらしさ」

def run_spell(items, args) -> list[dict]:
    sc = spellchecker()
    out = []
    for it in items:
        toks = tokenize(it["text"])
        unknown = sc.unknown([w.lower() for w, _, _ in toks])
        out.append({"id": it["id"], "p": [1.0 if w.lower() in unknown else 0.0 for w, _, _ in toks]})
    return out


def run_lt(items, args) -> list[dict]:
    """40 文を空行区切りで1リクエストにまとめ、返ってきた位置を文ごとの単語に戻す。"""
    out = []
    for k in range(0, len(items), 40):
        chunk = items[k:k + 40]
        joined, starts = "", []
        for it in chunk:
            starts.append(len(joined))
            joined += it["text"] + "\n\n"
        body = {"text": joined, "language": "en-GB"}
        res = cached("lt", body, lambda: http_post(LT_URL, urllib.parse.urlencode(body).encode(),
                                                    {"Content-Type": "application/x-www-form-urlencoded"}, "lt"))
        if not res.get("_cached"):
            time.sleep(3.1)  # 公開 API は1分20回まで
        for it, s0 in zip(chunk, starts):
            toks = tokenize(it["text"])
            p = [0.0] * len(toks)
            for m in res.get("matches", []):
                a, b = m["offset"] - s0, m["offset"] - s0 + m["length"]
                for i, (_, s, e) in enumerate(toks):
                    if s < b and a < e:
                        p[i] = 1.0
            out.append({"id": it["id"], "p": p, "latency": res["_latency_ms"] / len(chunk)})
    return out


def run_jev(items, args, mode: str) -> list[dict]:
    build = BUILDERS[mode]

    def one(it):
        body = build(it)
        r = jev_call(body, f"jev{mode}")
        a = r["answers"]
        toks = tokenize(it["text"])
        if mode != "B":
            p = [a[f"w{i}"]["noul"] for i in range(len(toks))]
            sent = a["has_error"]["noul"]
        else:
            probs = a["wrong_word"]["probabilities"]
            p = [probs.get(f"{i + 1}:{w}", 0.0) for i, (w, _, _) in enumerate(toks)]
            sent = 1 - probs.get("none", 0.0)
        return {"id": it["id"], "p": p, "sent": sent, "latency": r.get("_latency_ms"), "cached": r.get("_cached", False),
                "tokens": r.get("usage", {}).get("input_tokens", 0), "n_questions": len(body["questions"])}

    with ThreadPoolExecutor(args.workers) as ex:
        return list(ex.map(one, items))


CLAUDE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["wrong_words"],
                 "properties": {"wrong_words": {"type": "array", "items": {"type": "integer"}}}}


def claude_body(it: dict, backend: str = "claude") -> dict:
    """Jev A5 と同じ材料（料理名・調理工程として読む・単語の番号）を渡し、誤っている単語の番号を JSON で返させる。"""
    text = it["text"]
    model = CLAUDE_MODELS[backend][0]
    numbered = " ".join(f"[{i + 1}]{w}" for i, (w, _, _) in enumerate(tokenize(text)))
    dish = f"Dish: {it['dish']}\n" if it.get("dish") else ""
    cfg = {"format": {"type": "json_schema", "schema": CLAUDE_SCHEMA}}
    if model != "claude-haiku-4-5":
        cfg["effort"] = "low"  # Haiku 4.5 は effort を受けない
    return {"model": model, "max_tokens": 4000, "output_config": cfg,
            "messages": [{"role": "user", "content":
                          "Read this as one step of a recipe. Find the words that are misspelled, or that are real words "
                          "but do not make sense in this cooking step (e.g. 'flower' for 'flour'). "
                          "Ignore grammar, style and quantities.\n\n"
                          f"{dish}Instruction: {text}\nNumbered words: {numbered}\n\n"
                          "Return the numbers of the wrong words (an empty list if none)."}]}


def run_claude(items, args, backend: str = "claude") -> list[dict]:
    """1 文ずつ、順番に送る（並べると新しいアカウントの上限に当たり、待ち時間が応答時間に混ざるため）。
    認証は `ant auth login --profile poc` のプロファイル。ANTHROPIC_API_KEY が設定されていると、そちらが勝つ。"""
    import anthropic
    os.environ.setdefault("ANTHROPIC_PROFILE", "poc")
    if os.environ.get("ANTHROPIC_API_KEY") is not None:
        print("[claude] 注意: ANTHROPIC_API_KEY が設定されているので、プロファイル poc ではなくそのキーで送る", file=sys.stderr)
    client = anthropic.Anthropic(max_retries=0)
    model, price = CLAUDE_MODELS[backend]

    def one(it):
        body = claude_body(it, backend)

        def send():
            for attempt in range(6):
                t0 = time.perf_counter()
                try:
                    if model == "claude-haiku-4-5":
                        msg = client.messages.create(**body)
                    else:  # 拒否されたときに別モデルで答え直す（調理文で起きることはまず無いが既定で付ける）
                        msg = client.beta.messages.create(**body, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
                except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
                    time.sleep(2 ** attempt * 2)  # 待った時間は応答時間に入れない
                    continue
                latency = round((time.perf_counter() - t0) * 1000)
                if msg.stop_reason == "refusal":
                    sys.exit(f"[{backend}] 拒否: {it['id']}")
                text = next(b.text for b in msg.content if b.type == "text")
                return {"wrong_words": json.loads(text)["wrong_words"], "_latency_ms": latency, "model": msg.model,
                        "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}}
            sys.exit(f"[{backend}] 6 回失敗: {it['id']}")
        r = cached(backend, body, send)
        n = len(tokenize(it["text"]))
        wrong = {i - 1 for i in r["wrong_words"]}
        u = r["usage"]
        return {"id": it["id"], "p": [1.0 if i in wrong else 0.0 for i in range(n)], "latency": r["_latency_ms"],
                "cached": r.get("_cached", False), "tokens_out": u["output_tokens"],
                "cost": u["input_tokens"] * price[0] + u["output_tokens"] * price[1]}

    return [one(it) for it in items]


AGENT_DIR = HERE / "data" / DATASET / "agent"
AGENT_GROUPS = 4


def cmd_agent_prep(args):
    """Claude Code のサブエージェントに渡す文の束を作る。
    同じ文の正誤ペアは別の束に入れ、束の中はシャッフルし、id は中身の分からない番号にする（漏れ対策）。"""
    import random
    rng = random.Random(1017)
    items = load_items()
    groups = [[] for _ in range(AGENT_GROUPS)]
    for it in items:
        g = it["pair"] % AGENT_GROUPS
        if it["errors"]:
            g = (g + AGENT_GROUPS // 2) % AGENT_GROUPS  # 正しい版とは必ず別の束
        groups[g].append(it)
    ids = rng.sample(range(1000, 10000), len(items))
    mapping, k = {}, 0
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    for g, its in enumerate(groups):
        rng.shuffle(its)
        lines = []
        for it in its:
            qid = f"q{ids[k]}"
            k += 1
            mapping[qid] = it["id"]
            numbered = " ".join(f"[{i + 1}]{w}" for i, (w, _, _) in enumerate(tokenize(it["text"])))
            lines.append(f"{qid}\t{it['text']}\t{numbered}")
        (AGENT_DIR / f"batch{g}.txt").write_text("\n".join(lines) + "\n")
    (AGENT_DIR / "mapping.json").write_text(json.dumps(mapping, indent=1))
    print(f"{AGENT_GROUPS} 束（{', '.join(str(len(g)) for g in groups)} 文）→ {AGENT_DIR}")


def run_agent(items, args) -> list[dict]:
    """サブエージェントの答え（results/agent_answers.json: {qid: [単語番号...]}）を採点の形に戻す。"""
    mapping = json.loads((AGENT_DIR / "mapping.json").read_text())
    answers = json.loads((RESULTS / "agent_answers.json").read_text())
    by_item = {mapping[q]: v for q, v in answers.items()}
    missing = [it["id"] for it in items if it["id"] not in by_item]
    if missing:
        sys.exit(f"エージェントの答えが無い文: {len(missing)} 件（{missing[:5]}…）")
    return [{"id": it["id"], "p": [1.0 if i + 1 in set(by_item[it["id"]]) else 0.0
                                   for i in range(len(tokenize(it["text"])))]} for it in items]


def run_mlm(items, args) -> list[dict]:
    """mlm_baseline.py が書いた予測（results/<set>/pred_mlm.jsonl）を読む。"""
    f = RESULTS / "pred_mlm.jsonl"
    if not f.exists():
        sys.exit(f"{f} が無い。先に `uv run python mlm_baseline.py --set {DATASET}`")
    by = {r["id"]: r for r in map(json.loads, f.read_text().splitlines())}
    return [by[it["id"]] for it in items]


# ---------------------------------------------------------------- 採点

def prf(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(items, preds, th: float) -> dict:
    by_id = {p["id"]: p for p in preds}
    tp = fp = fn = 0
    hit = {"typo": [0, 0], "real_word": [0, 0]}
    clean_alarm = clean_n = sent_ok = 0
    for it in items:
        pr = by_id[it["id"]]["p"]
        gold = {e["index"]: ("typo" if e["type"].startswith("typo") else e["type"]) for e in it["errors"]}  # v2 の typo_synthetic も綴りに数える
        flagged = {i for i, v in enumerate(pr) if v >= th}
        tp += len(flagged & gold.keys())
        fp += len(flagged - gold.keys())
        fn += len(gold.keys() - flagged)
        for i, t in gold.items():
            hit[t][0] += i in flagged
            hit[t][1] += 1
        if not gold:
            clean_n += 1
            clean_alarm += bool(flagged)
        sent_ok += bool(flagged) == bool(gold)
    p, r, f = prf(tp, fp, fn)
    return {"P": p, "R": r, "F1": f, "typo_R": hit["typo"][0] / max(hit["typo"][1], 1),
            "real_R": hit["real_word"][0] / max(hit["real_word"][1], 1), "clean_alarm": clean_alarm / max(clean_n, 1),
            "sent_acc": sent_ok / len(items)}


def pct(x: float) -> str:
    return f"{x:.0%}"


def run_jevfix(items, args) -> str:
    bad = [it for it in items if it["errors"]]
    reach = total = right = kept = 0
    lat = []

    def one(it):
        return it, jev_call(jev_request_fix(it["text"], it["errors"]), "jevfix")

    with ThreadPoolExecutor(args.workers) as ex:
        for it, r in ex.map(one, bad):
            lat.append(r.get("_latency_ms"))
            for e in it["errors"]:
                total += 1
                reach += e["correct"].lower() in candidates(e["word"])
                choice = r["answers"][f"fix{e['index']}"]["choice"]
                right += choice == e["correct"].lower()
                kept += choice == e["word"].lower()
    return (f"### ④ 修正候補（jevfix、誤り位置は正解を与える）\n\n"
            f"| 誤り | 候補に正解が入った | Jev が正解を選んだ | 元の語のまま（直さない）を選んだ | 応答時間（中央値） |\n|---|---|---|---|---|\n"
            f"| {total} | {pct(reach / total)} | {pct(right / total)} | {pct(kept / total)} | {statistics.median(lat):.0f} ms |\n")


RUNNERS = {"spell": run_spell, "lt": run_lt, "jevA": lambda i, a: run_jev(i, a, "A"),
           "jevA2": lambda i, a: run_jev(i, a, "A2"), "jevA3": lambda i, a: run_jev(i, a, "A3"),
           "jevA4": lambda i, a: run_jev(i, a, "A4"), "jevA2d": lambda i, a: run_jev(i, a, "A2d"),
           "jevA5": lambda i, a: run_jev(i, a, "A5"),
           "jevB": lambda i, a: run_jev(i, a, "B"), "claude": run_claude, "agent": run_agent,
           "mlm": run_mlm,
           "claude-haiku": lambda i, a: run_claude(i, a, "claude-haiku"),
           "claude-sonnet": lambda i, a: run_claude(i, a, "claude-sonnet")}
LABEL = {"spell": "pyspellchecker（辞書）", "lt": "LanguageTool", "jevA": "Jev 方式 A（単語ごとの Noul）",
         "jevA2": "Jev 方式 A2（言い直し）", "jevA3": "Jev 方式 A3（言い直し＋criteria）",
         "jevA4": "Jev 方式 A4（調理工程として聞く）", "jevA2d": "Jev 方式 A2d（A2＋料理名）",
         "jevA5": "Jev 方式 A5（調理工程として聞く＋料理名）",
         "jevB": "Jev 方式 B（Choice 1問）", "claude": f"Claude（{CLAUDE_MODEL}, effort low）",
         "mlm": "辞書＋RoBERTa-base（ローカル）", "claude-haiku": "Claude Haiku 4.5（API）", "claude-sonnet": "Claude Sonnet 5.5（API, effort low）",
         "agent": "Claude Code サブエージェント（精度のみ）"}
PROB = {"jevA", "jevA2", "jevA3", "jevA4", "jevA2d", "jevA5", "jevB", "mlm"}  # 確率を返すもの。閾値を動かして見る


def plan(items, backends) -> list[str]:
    """dry-run の見積もり。文字数 / 4 をトークン数の目安にする。"""
    lines = []
    for b in backends:
        if b in RUNNERS and b.startswith("jev") or b == "jevfix":
            if b == "jevfix":
                reqs = [jev_request_fix(it["text"], it["errors"]) for it in items if it["errors"]]
            else:
                reqs = [BUILDERS[b[3:]](it) for it in items]
            todo = [r for r in reqs if not cache_path(b, r).exists()]
            tok = sum(len(json.dumps(r)) / 4 for r in todo)
            lines.append(f"  {b:7} {len(reqs)} リクエスト（未送信 {len(todo)}）約 {tok:,.0f} 入力トークン ≒ ${tok * JEV_PRICE:.4f}")
        elif b in CLAUDE_MODELS:
            reqs = [claude_body(it, b) for it in items]
            todo = [r for r in reqs if not cache_path(b, r).exists()]
            price = CLAUDE_MODELS[b][1]
            # 入力 ~250 トークン、出力（思考込み）~300 トークンを目安にする
            lines.append(f"  {b:7} {len(reqs)} リクエスト（未送信 {len(todo)}）≒ ${len(todo) * (250 * price[0] + 300 * price[1]):.2f}（目安）")
        elif b == "lt":
            lines.append(f"  {b:7} {-(-len(items) // 40)} リクエスト（40 文ずつ）無料。公開 API に文面を送る")
        elif b == "mlm":
            lines.append(f"  {b:7} 送信なし（mlm_baseline.py の予測を採点するだけ）")
        elif b == "agent":
            lines.append(f"  {b:7} 送信なし（results/agent_answers.json を採点するだけ）")
        else:
            lines.append(f"  {b:7} ローカル・無料")
    return lines


def cmd_eval(args):
    items = load_items()
    if args.limit:
        items = items[: args.limit * 2]  # 正誤ペアを崩さないように先頭から
    backends = [b for b in args.backend.split(",") if b]
    remote = [b for b in backends if b not in ("spell", "agent")]
    n_err = sum(len(it["errors"]) for it in items)
    print(f"評価データ: {len(items)} 文（誤りあり {sum(bool(it['errors']) for it in items)} 文、誤り {n_err} か所）")
    print("送るもの:")
    print("\n".join(plan(items, backends)))
    if not args.run:
        if "jevA" in backends:
            print("\n例（jevA、1件目の誤りあり文）:")
            print(json.dumps(jev_request_A(items[1]["text"]), ensure_ascii=False, indent=1)[:1500])
        print("\ndry-run。実行するには --run")
        return
    if remote and not args.yes and input("送りますか [y/N] ").lower() != "y":
        sys.exit("中止")

    RESULTS.mkdir(exist_ok=True)
    report = [f"# 結果（{time.strftime('%Y-%m-%d %H:%M')}）\n",
              f"データ: {len(items)} 文（うち誤りあり {sum(bool(it['errors']) for it in items)} 文、誤り {n_err} か所）。"
              f"単語単位の採点。確率を返すものは p ≥ 0.5 で「誤り」とみなす。\n",
              "| 方式 | 適合率 | 再現率 | F1 | 再現率（綴り） | 再現率（取り違え） | 正しい文で誤検知 | 文単位の正答率 | 応答時間（中央値） | 費用 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    sweeps, extras = [], []
    for b in backends:
        if b == "jevfix":
            extras.append(run_jevfix(items, args))
            continue
        preds = RUNNERS[b](items, args)
        (RESULTS / f"pred_{b}.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds))
        s = score(items, preds, 0.5)
        lats = [p["latency"] for p in preds if p.get("latency") is not None]
        lat = f"{statistics.median(lats):.0f} ms" if lats else ("-（50文を約8秒）" if b == "agent" else "<1 ms")
        if b.startswith("jev"):
            cost = f"${sum(p['tokens'] for p in preds) * JEV_PRICE:.4f}"
        elif b in CLAUDE_MODELS:
            cost = f"${sum(p['cost'] for p in preds):.2f}"
        else:
            cost = "契約内" if b == "agent" else "0"
        report.append(f"| {LABEL[b]} | {pct(s['P'])} | {pct(s['R'])} | {s['F1']:.2f} | {pct(s['typo_R'])} | "
                      f"{pct(s['real_R'])} | {pct(s['clean_alarm'])} | {pct(s['sent_acc'])} | {lat} | {cost} |")
        if b in PROB:
            rows = [f"\n### {LABEL[b]}：閾値を動かす（構想案 §5）\n", "| 閾値 | 適合率 | 再現率 | F1 | 正しい文で誤検知 |", "|---|---|---|---|---|"]
            for th in (0.2, 0.3, 0.5, 0.7, 0.85, 0.95):
                t = score(items, preds, th)
                rows.append(f"| {th} | {pct(t['P'])} | {pct(t['R'])} | {t['F1']:.2f} | {pct(t['clean_alarm'])} |")
            # 文単位の Noul / none の確率だけで「誤りあり文」を当てられるか
            sent_acc = sum((p["sent"] >= 0.5) == bool(it["errors"]) for p, it in zip(preds, items)) / len(items) \
                if all("sent" in p for p in preds) else None  # mlm には文単位の確率が無い
            if sent_acc is not None:
                rows.append(f"\n文単位の判定（{'has_error の Noul' if b.startswith('jevA') else '1 − P(none)'} ≥ 0.5）の正答率: {pct(sent_acc)}")
            if b.startswith("jevA"):
                # 構想案 §6 の未確認事項: state が質問ごとに課金されるか。質問数と入力トークンの関係を見る
                xs, ys = [p["n_questions"] for p in preds], [p["tokens"] for p in preds]
                if len(set(xs)) > 1 and any(ys):
                    mx, my = statistics.fmean(xs), statistics.fmean(ys)
                    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
                    rows.append(f"\n入力トークン ≒ {my - slope * mx:.0f} + {slope:.1f} × 質問数"
                                f"（文そのものは平均 {statistics.fmean(len(it['text']) for it in items) / 4:.0f} トークン前後。"
                                "1質問あたりの増分が質問文の長さ＋文の長さに近ければ、state は質問ごとに数えられている）")
            sweeps.append("\n".join(rows))
    out = "\n".join(report) + "\n" + "\n".join(sweeps) + "\n\n" + "\n".join(extras)
    (RESULTS / "report.md").write_text(out)
    print("\n" + out + f"\n→ {RESULTS / 'report.md'}")


def cmd_detect(args):
    """構想案 §4 の detect() と同じ形で返す。"""
    body = jev_request_A(args.text)
    print(f"質問 {len(body['questions'])} 個 ≒ {len(json.dumps(body)) / 4:.0f} 入力トークン")
    if not args.run:
        print(json.dumps(body, ensure_ascii=False, indent=1))
        print("dry-run。実行するには --run")
        return
    r = jev_call(body, "jevA")
    issues = []
    for i, (w, s, e) in enumerate(tokenize(args.text)):
        p = r["answers"][f"w{i}"]["noul"]
        if p >= BANDS[0]:
            issues.append({"word": w, "start": s, "end": e, "p": round(p, 3),
                           "level": "error" if p >= BANDS[1] else "check"})
    print(json.dumps({"has_error": r["answers"]["has_error"]["noul"], "issues": issues,
                      "latency_ms": r.get("_latency_ms"), "usage": r.get("usage")}, ensure_ascii=False, indent=1))
    marked = args.text
    for x in reversed(issues):
        l, rr = ("[[", "]]") if x["level"] == "error" else ("[?", "?]")
        marked = marked[:x["start"]] + l + x["word"] + rr + marked[x["end"]:]
    print(marked + "    （[[確定]] [?要確認?]）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="v1", help="データの版。data/<set>/eval.jsonl を読み、results/<set>/ に書く")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--backend", default="spell,lt,jevA,jevB,jevfix,claude")
    e.add_argument("--run", action="store_true", help="実際に送る（既定は dry-run）")
    e.add_argument("--yes", action="store_true", help="送る前の確認を飛ばす")
    e.add_argument("--limit", type=int, default=0, help="先頭から N 組（2N 文）だけ")
    e.add_argument("--workers", type=int, default=4)
    sub.add_parser("agent-prep", help="サブエージェントに渡す文の束を data/agent/ に作る")
    d = sub.add_parser("detect")
    d.add_argument("text")
    d.add_argument("--run", action="store_true")
    args = ap.parse_args()
    global DATASET, RESULTS, DATA, AGENT_DIR
    DATASET = args.set
    RESULTS = HERE / "results" / DATASET
    DATA = HERE / "data" / DATASET / "eval.jsonl"
    AGENT_DIR = HERE / "data" / DATASET / "agent"
    {"eval": cmd_eval, "detect": cmd_detect, "agent-prep": cmd_agent_prep}[args.cmd](args)


if __name__ == "__main__":
    main()
