#!/usr/bin/env python3
"""評価データを作る（構想案 §8 の第一項目）。

正しい調理工程文 100 文を手で用意し、それぞれから「そのまま」と「誤りを 1〜2 か所混ぜた」版を作る → 200 文。
同じ文の正誤ペアになるので、誤検知（正しい文で何か光るか）と見逃しを同じ土台で比べられる。

  uv run python make_data.py      # data/v1/eval.jsonl を作り直す（乱数の種は固定。結果は毎回同じ）

誤りの種類:
  typo       綴りの誤り。1文字の入れ替え・欠落・重複・置換。辞書にある語になったものは捨てる
  real_word  辞書にある別の語への取り違え（flour→flower、bake→boil など）。綴りチェッカーでは見つからない
"""
from __future__ import annotations

import json
import pathlib
import random
import re

from spellchecker import SpellChecker

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "data" / "v1" / "eval.jsonl"  # 第1段のデータ。固定（作り直しても同じものができる）
SEED = 20261017
WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")  # 構想案 §4 と同じ単語分割

# 正しい調理工程文。ハイフン語・数字と単位・所有格も混ぜてある（構想案 §7「単語分割」）
CLEAN = [
    "Preheat the oven to 180°C and line a baking tray with parchment paper.",
    "Sift the flour and salt into a large bowl.",
    "Stir in the flour and simmer for 10 minutes.",
    "Knead the dough on a lightly floured surface until smooth.",
    "Cover the bowl with a damp towel and let the dough rise for one hour.",
    "Beat the eggs and sugar until pale and fluffy.",
    "Whisk the cream until soft peaks form.",
    "Peel the potatoes and cut them into small cubes.",
    "Grate the cheese and set it aside.",
    "Pour the batter into the prepared pan and smooth the top.",
    "Bake the cake for 35 minutes or until a skewer comes out clean.",
    "Let the cake cool in the tin before turning it out.",
    "Season the steak generously with salt and pepper.",
    "Sear the steak in a hot pan for two minutes on each side.",
    "Rest the meat for five minutes before slicing.",
    "Finely chop the onion and mince the garlic.",
    "Heat the oil in a wok over high heat.",
    "Stir-fry the vegetables for three minutes, tossing constantly.",
    "Add the soy sauce and a pinch of sugar.",
    "Bring a large pot of salted water to the boil.",
    "Cook the pasta until al dente, then drain well.",
    "Reserve a cup of the pasta water for the sauce.",
    "Toss the pasta with the sauce and serve immediately.",
    "Wash the leeks thoroughly to remove any grit.",
    "Slice the leeks and soften them in butter over low heat.",
    "Add the fresh thyme and a bay leaf to the pot.",
    "Simmer the soup gently for twenty minutes.",
    "Blend the soup until smooth and season to taste.",
    "Scrub the mussels and discard any that are open.",
    "Steam the mussels with white wine until the shells open.",
    "Dice the carrots and celery into even pieces.",
    "Roast the vegetables until golden and tender.",
    "Drizzle the salad with olive oil and lemon juice.",
    "Toast the pine nuts in a dry pan until fragrant.",
    "Roll out the pastry to a thickness of three millimetres.",
    "Line the tart tin with the pastry and chill for thirty minutes.",
    "Fill the pastry case with baking beans and bake blind.",
    "Spread the jam evenly over the sponge.",
    "Sprinkle the currants over the dough before rolling it up.",
    "Brush the top with beaten egg for a glossy finish.",
    "Melt the chocolate in a bowl set over simmering water.",
    "Fold the melted chocolate gently into the whipped cream.",
    "Spoon the mousse into glasses and chill until set.",
    "Halve the pears and remove the cores with a spoon.",
    "Poach the pears in red wine and cinnamon.",
    "Serve the dessert with a scoop of vanilla ice cream.",
    "Marinate the chicken thighs overnight in yoghurt and spices.",
    "Thread the chicken onto skewers and grill until charred.",
    "Squeeze the lime over the chicken just before serving.",
    "Soak the dried beans in cold water overnight.",
    "Rinse the rice until the water runs clear.",
    "Bring the rice to the boil, then cover and reduce the heat.",
    "Fluff the rice with a fork and keep it warm.",
    "Crack the eggs into a bowl and whisk with a fork.",
    "Melt a knob of butter in a frying pan.",
    "Pour in the eggs and stir gently until just set.",
    "Fry the bacon until crisp and drain on kitchen paper.",
    "Crumble the bacon over the salad.",
    "Trim the fat from the lamb and score the skin.",
    "Rub the lamb with garlic and rosemary.",
    "Roast the lamb for one hour, basting occasionally.",
    "Deglaze the roasting tin with a splash of stock.",
    "Strain the gravy through a fine sieve.",
    "Stir the cereal into the warm milk and leave to soak.",
    "Top the porridge with sliced banana and honey.",
    "Wash the berries and pat them dry.",
    "Arrange the berries on top of the tart.",
    "Dust the tart with icing sugar before serving.",
    "Combine the minced beef, breadcrumbs and egg in a bowl.",
    "Shape the mixture into small balls with wet hands.",
    "Brown the meatballs in batches so the pan is not crowded.",
    "Add the chopped tomatoes and simmer until thick.",
    "Shred the cabbage as finely as possible.",
    "Mix the cabbage with the dressing and leave for an hour.",
    "Cut the chilli in half and scrape out the seeds.",
    "Pound the chilli, garlic and ginger to a paste.",
    "Fry the paste in a little oil until fragrant.",
    "Stir in the coconut milk and bring to a simmer.",
    "Add the prawns and cook until they turn pink.",
    "Garnish with fresh coriander and a wedge of lime.",
    "Warm the tortillas in a dry pan for a few seconds.",
    "Mash the avocado with lime juice and salt.",
    "Spread the avocado over the toast and add a poached egg.",
    "Grease the loaf tin and dust it with flour.",
    "Mix the yeast with warm water and leave until frothy.",
    "Punch down the dough and shape it into a loaf.",
    "Slash the top of the loaf with a sharp knife.",
    "Bake the bread until it sounds hollow when tapped.",
    "Leave the bread to cool on a wire rack.",
    "Chop the herbs roughly and scatter them over the dish.",
    "Squeeze the excess water from the spinach.",
    "Wilt the spinach in a covered pan with a little butter.",
    "Grill the fish skin-side down until crisp.",
    "Flake the fish and check for any bones.",
    "Stir the risotto constantly, adding the stock a ladle at a time.",
    "Remove the pan from the heat and stir in the parmesan.",
    "Let the risotto rest for two minutes before serving.",
    "Store the cookies in an airtight container for up to a week.",
    "Freeze the leftover sauce in small portions.",
    "Thaw the meat in the fridge overnight before cooking.",
]

# 各工程の料理名（アプリの販売品名に当たる）。CLEAN と同じ順。
# 取り違えの正解をそのまま含む名前（Poached Pears に対する pears→pairs など）は、文字列の一致で当たってしまうので避けた
DISHES = [
    "Chocolate Chip Cookies", "Scones", "Beef Stew", "Bread Rolls", "Bread Rolls",
    "Sponge Cake", "Strawberry Shortcake", "Potato Salad", "Macaroni and Cheese", "Brownies",
    "Banana Cake", "Banana Cake", "Beef Plate with Fries", "Beef Plate with Fries", "Roast Beef Sandwich",
    "Spaghetti Bolognese", "Vegetable Stir-Fry", "Vegetable Stir-Fry", "Teriyaki Chicken", "Carbonara",
    "Carbonara", "Carbonara", "Carbonara", "Winter Vegetable Soup", "Winter Vegetable Soup",
    "Winter Vegetable Soup", "Winter Vegetable Soup", "Pumpkin Soup", "Seafood Pot", "Seafood Pot",
    "Minestrone", "Grilled Vegetable Platter", "Greek Salad", "Pesto Pasta", "Apple Tart",
    "Apple Tart", "Lemon Tart", "Victoria Sponge", "Cinnamon Rolls", "Sausage Rolls",
    "Chocolate Mousse", "Chocolate Mousse", "Chocolate Mousse", "Red Wine Fruit Compote", "Red Wine Fruit Compote",
    "Apple Crumble", "Tandoori Chicken", "Yakitori", "Thai Grilled Chicken", "Chili con Carne",
    "Curry Plate", "Curry Plate", "Curry Plate", "Omelette", "Scrambled Eggs on Toast",
    "Scrambled Eggs on Toast", "BLT Sandwich", "Cobb Salad", "Roast Lamb", "Roast Lamb",
    "Roast Lamb", "Roast Lamb with Gravy", "Roast Lamb with Gravy", "Overnight Oats", "Breakfast Bowl",
    "Fruit Tart", "Fruit Tart", "Fruit Tart", "Meatball Sub", "Meatball Sub",
    "Meatball Sub", "Meatball Sub", "Coleslaw", "Coleslaw", "Thai Green Curry",
    "Thai Green Curry", "Thai Green Curry", "Thai Green Curry", "Thai Green Curry", "Thai Green Curry",
    "Tacos", "Tacos", "Avocado Toast", "Banana Bread", "White Bread",
    "White Bread", "White Bread", "White Bread", "White Bread", "Herb Roasted Chicken",
    "Quiche", "Quiche", "Grilled Salmon", "Fish Cakes", "Mushroom Risotto",
    "Mushroom Risotto", "Mushroom Risotto", "Chocolate Chip Cookies", "Bolognese", "Beef Stew",
]
assert len(DISHES) == len(CLEAN)

# 正しい語 → 取り違えやすい実在語（real-word error）。構想案 §1 の flower / boil を含む
CONFUSIONS = {
    "flour": ["flower"], "floured": ["flowered"], "steak": ["stake"], "thyme": ["time"],
    "leeks": ["leaks"], "dough": ["doe"], "pour": ["pore", "poor"], "peel": ["peal"],
    "grate": ["great"], "beat": ["beet"], "currants": ["currents"], "mussels": ["muscles"],
    "cereal": ["serial"], "dessert": ["desert"], "pears": ["pairs"], "pan": ["pen"],
    "sift": ["shift"], "bake": ["boil"], "oven": ["over"], "knead": ["need"],
    "dice": ["dies"], "simmer": ["summer"], "meat": ["meet"], "wok": ["walk"],
    "berries": ["buries"], "sauce": ["source"], "fry": ["fly"], "chop": ["shop"],
    "drain": ["brain"], "roll": ["role"], "whole": ["hole"], "tray": ["try"],
    "chilli": ["chilly"], "minutes": ["minuets"], "cool": ["coal"], "stir": ["steer"],
    "serve": ["swerve"], "rest": ["rust"], "heat": ["hate"], "salt": ["salty"],
    "rice": ["race"], "lime": ["line"], "cubes": ["cubs"], "seeds": ["seats"],
    "mince": ["mints"], "dust": ["dusk"],
}

spell = SpellChecker()


def make_typo(word: str, rng: random.Random) -> str | None:
    """1文字の入れ替え・欠落・重複・置換。辞書にある語になったら捨てる。"""
    w = word.lower()
    for _ in range(30):
        i = rng.randrange(1, len(w) - 1)  # 先頭の文字はいじらない（人の誤字に近づける）
        op = rng.choice(["swap", "drop", "double", "sub"])
        if op == "swap":
            t = w[:i] + w[i + 1] + w[i] + w[i + 2:] if i + 1 < len(w) else None
        elif op == "drop":
            t = w[:i] + w[i + 1:]
        elif op == "double":
            t = w[:i] + w[i] + w[i:]
        else:
            t = w[:i] + rng.choice("aeioustnrl") + w[i + 1:]
        if t and t != w and not spell.known([t]):
            return word[0] + t[1:] if word[0].isupper() else t
    return None


def corrupt(text: str, rng: random.Random) -> tuple[str, list[dict]]:
    toks = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]
    n_err = rng.choice([1, 1, 2])
    real = [i for i, (w, _, _) in enumerate(toks) if w.lower() in CONFUSIONS]
    typo = [i for i, (w, _, _) in enumerate(toks) if len(w) >= 4 and "-" not in w and w.lower() not in CONFUSIONS]
    picked: dict[int, tuple[str, str]] = {}
    for _ in range(n_err):
        # 半々で real_word / typo。候補が無い方はもう一方に回す
        kind = "real_word" if (real and (rng.random() < 0.5 or not typo)) else "typo"
        pool = [i for i in (real if kind == "real_word" else typo) if i not in picked]
        if not pool:
            continue
        i = rng.choice(pool)
        w = toks[i][0]
        if kind == "real_word":
            new = rng.choice(CONFUSIONS[w.lower()])
            new = new.capitalize() if w[0].isupper() else new
        else:
            new = make_typo(w, rng)
            if new is None:
                continue
        picked[i] = (kind, new)
    # 後ろから置き換えて文字位置を保つ。最後に位置を数え直す
    out = text
    for i in sorted(picked, reverse=True):
        _, s, e = toks[i]
        out = out[:s] + picked[i][1] + out[e:]
    new_toks = list(WORD.finditer(out))
    errors = [{"index": i, "word": new_toks[i].group(), "correct": toks[i][0], "type": picked[i][0],
               "start": new_toks[i].start(), "end": new_toks[i].end()} for i in sorted(picked)]
    return out, errors


def main():
    rng = random.Random(SEED)
    rows = []
    for k, text in enumerate(CLEAN):
        rows.append({"id": f"s{k:03d}c", "pair": k, "dish": DISHES[k], "text": text, "errors": []})
        bad, errors = corrupt(text, rng)
        assert errors, text
        rows.append({"id": f"s{k:03d}e", "pair": k, "dish": DISHES[k], "text": bad, "errors": errors})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_err = sum(len(r["errors"]) for r in rows)
    kinds = {t: sum(e["type"] == t for r in rows for e in r["errors"]) for t in ("typo", "real_word")}
    print(f"{len(rows)} 文（誤りあり {sum(bool(r['errors']) for r in rows)} 文、誤り {n_err} か所 {kinds}）→ {OUT}")


if __name__ == "__main__":
    main()
