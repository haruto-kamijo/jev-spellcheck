# jev-spellcheck

英語のレシピ（調理工程）の文から、綴りの誤りと**実在する語の取り違え**（flour → flower など）を、判断モデル [Jev](https://typesafe.ai)（TypeSafe AI）で単語の位置まで見つける検証。辞書・LanguageTool・辞書＋RoBERTa・Claude（Haiku 4.5 / Sonnet 5.5）と比べた。

*Detecting misspellings and real-word errors (e.g. "flower" for "flour") in English recipe steps with TypeSafe's Jev decision model, compared with a dictionary, LanguageTool, a dictionary + RoBERTa baseline and Claude Haiku 4.5 / Sonnet 5.5.*

- 記事：<!-- TODO: Zenn の URL -->
- コードは MIT、データ（`data/v2*/`）は CC BY-SA 4.0

## 結果（評価用 1,000 文・誤り 672 か所）

括弧は 95% 区間（文を単位にしたブートストラップ）。全体は `results/v2test/summary.md`。

| 方式 | F1 | 取り違えの再現率 | 正しい文で誤検知 | 1 文あたり（中央値） | 費用（1,000 文） |
|---|---|---|---|---|---|
| 辞書（pyspellchecker） | 0.67（0.64〜0.70） | 0% | 2% | 1 ms 未満 | 0 |
| LanguageTool | 0.64（0.60〜0.67） | 8% | 10% | 79 ms | 0 |
| 辞書＋RoBERTa-base | 0.92（0.90〜0.94） | 74% | 1% | 633 ms | 0（ローカル） |
| Jev A5（調理工程として＋料理名） | 0.94（0.92〜0.95） | 88% | 2% | 186 ms | $0.08 |
| Claude Haiku 4.5 | 0.91（0.89〜0.92） | 95% | 13% | 1,060 ms | $0.40 |
| Claude Sonnet 5.5 | 0.95（0.94〜0.96） | 97% | 6% | 1,152 ms | $1.05 |

Jev の課金では、文（state）は質問の数に関係なく 1 回分しか数えられない（`results/billing.md`）。

## データ

- `data/v2dev/`（開発用 300 文）・`data/v2test/`（評価用 1,000 文）：[Wikibooks Cookbook](https://en.wikibooks.org/wiki/Cookbook:Table_of_Contents) の手順に、乱数で誤りを入れたもの。綴りは [Wikipedia の誤字の表](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines)、取り違えは 1 文字違いの辞書の語。出典は各フォルダの `SOURCE.md`
- 聞き方と閾値は開発用で決め、評価用は最後に 1 回だけ測った
- 取り違えのうち 100 か所を無作為に選んで判定した結果：`data/v2test/audit_realword_judged.csv`（判定は Claude Fable 5.1。人ではない）
- `data/v1/`：最初に試した、手作りの 200 文（結果は `results/v1/`）。作り手と評価する側が同じなので、数字は甘く出ている

## 再現のしかた

```bash
uv sync

# 1. 材料を置く（どちらも CC BY-SA 4.0）
export PLAYGROUND_MODELS="$HOME/.cache/jev-spellcheck"   # 材料の置き場（リポジトリの外）
mkdir -p "$PLAYGROUND_MODELS/datasets"
curl -L -o "$PLAYGROUND_MODELS/datasets/wikibooks-cookbook.mini.json" \
  https://huggingface.co/datasets/gossminn/wikibooks-cookbook/resolve/main/recipes_parsed.mini.json
curl -L -o "$PLAYGROUND_MODELS/datasets/wiki-misspellings.txt" \
  "https://en.wikipedia.org/w/index.php?title=Wikipedia:Lists_of_common_misspellings/For_machines&action=raw"
uv run python make_data_v2.py        # data/v2dev・data/v2test を作る（同梱のものと同じになる。Wikipedia の表が更新されていると少し変わる）

# 2. 送るものを確かめる（既定は dry-run。件数と見積もりだけ出す）
uv run python probe.py --set v2test eval --backend spell,lt,jevA5,claude-haiku

# 3. 実際に送る
uv run python probe.py --set v2test eval --run --backend spell,lt,jevA,jevA2,jevA5
uv run python mlm_baseline.py --set v2dev      # RoBERTa の閾値を開発用で決める（ローカル。約 500MB を落とす）
uv run python mlm_baseline.py --set v2test
uv run python probe.py --set v2test eval --run --backend claude-haiku,claude-sonnet
uv run python billing_probe.py --run           # 課金の確かめ（10 リクエスト）

# 4. 区間つきでまとめる（送信なし）
uv run python analyze_v2.py
```

- **Jev**：`TYPESAFE_API_KEY` か `~/.config/typesafe/api_key`。モデルは `jev-1.13.0`（固定版）
- **Claude**：`ANTHROPIC_API_KEY`、または `ant auth login --profile poc` のプロファイル
- **LanguageTool**：公開 API（無料、1 分 20 回まで）に文面を送る
- 応答はリクエストの中身ごとに `cache/` に保存し、同じものは二度送らない
- 費用の目安（評価用 1,000 文）：Jev 3 方式で約 $0.16、Claude Haiku＋Sonnet で約 $1.5

## ファイル

| ファイル | 中身 |
|---|---|
| `probe.py` | 各方式で判定し、採点する（`--set` でデータを選ぶ。既定は dry-run） |
| `make_data_v2.py` | 開発用・評価用のデータを作る |
| `mlm_baseline.py` | 辞書＋RoBERTa の比較相手 |
| `analyze_v2.py` | 区間つきの集計（`--audit` で取り違えの確認用の表を作る） |
| `billing_probe.py` | state の課金が 1 回か質問ごとかを確かめる |
| `make_data.py` | 最初の手作りデータ（v1） |

## 限界

- 誤りは機械的に入れたもので、人が実際に書く取り違えの分布とは違う（取り違えは 1 文字違いの辞書の語に限る）
- 辞書＋RoBERTa は、候補の作り方が誤りの作り方と同じなので、有利に出ている
- 取り違えの確認は人ではなく LLM が行った。比べる相手にも Claude が入っている
- LLM のプロンプトは 1 種類だけ
- Jev の結果は `jev-1.13.0`（2026-10 時点）のもの

## 開示

TypeSafe AI とは関係がない。Jev の利用規約（Master Customer Agreement 2026-09-23 版ほか）に、ベンチマーク結果の公開を禁じる条項がないことを 2026-10-28 に確かめた。評価データの作成・コード・集計は AI（Claude）と行った。
