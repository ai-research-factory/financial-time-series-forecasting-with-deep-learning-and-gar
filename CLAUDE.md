# Financial Time-series Forecasting with Deep Learning and GARCH models

## Project ID
proj_7d1c0d6f

## Taxonomy
StatArb, ResidualFactors

## Current Cycle
5

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Financial time series are characterized by non-linear patterns and time-varying volatility (volatility clustering). Traditional econometric models often fail to capture the complex, non-linear dynamics of asset returns, while standard deep learning models may not explicitly account for volatility regimes. This paper addresses this gap by proposing a hybrid model. The goal is to improve financial time series forecasting by combining the strengths of two distinct modeling paradigms: a Long Short-Term Memory (LSTM) network to capture non-linear patterns in the conditional mean (expected return), and a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model to capture the volatility clustering observed in the residuals of the LSTM model. This two-step approach aims to produce more accurate forecasts of both returns and their associated risk.

### Datasets
BTC-USD daily data from yfinance

### Targets
Conditional Mean (Expected Return) of the next period.
- Conditional Volatility of the next period.

### Model
The model is a hybrid, two-stage process. First, an LSTM network is trained to forecast the one-step-ahead conditional mean of a financial time series (e.g., daily log returns). Second, the residuals from this LSTM forecast (actual return - predicted return) are extracted. A GARCH(1,1) model is then fitted to these residuals to forecast their conditional volatility. The final output for any given time step is a pair of forecasts: the expected return from the LSTM and the expected volatility from the GARCH model.

### Training
The model is trained using a walk-forward validation approach. For each window, the LSTM is trained on the training portion of the data to predict returns. The residuals from this training are then used to fit the GARCH model. On the test set, the model makes one-step-ahead predictions. The training window then slides forward to include the next data point, and the process is repeated. The paper implies using standard optimizers like Adam for the LSTM and Maximum Likelihood Estimation for the GARCH model.

### Evaluation
The primary evaluation involves a walk-forward backtest. The LSTM's forecast accuracy is measured by Mean Squared Error (MSE) against actual returns. The GARCH model's volatility forecast is evaluated by its ability to explain the variance of the LSTM residuals. A trading strategy is simulated based on the forecasts (e.g., go long if predicted return > k * predicted volatility). The performance of this strategy is evaluated using metrics like Sharpe Ratio, Calmar Ratio, and Maximum Drawdown, both with and without transaction costs.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## Preflight チェック（実装開始前に必ず実施）

**Phase の実装コードを書く前に**、以下のチェックを実施し結果を `reports/cycle_5/preflight.md` に保存すること。

### 1. データ境界表
以下の表を埋めて、未来データ混入がないことを確認:

```markdown
| 項目 | 値 |
|---|---|
| データ取得終了日 | YYYY-MM-DD (今日以前であること) |
| Train期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Validation期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Test期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| 重複なし確認 | Yes / No |
| 未来日付なし確認 | Yes / No |
```

### 2. Feature timestamp 契約
- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes / No
- Scaler / Imputer は train データのみで fit しているか？ → Yes / No
- Centered rolling window を使用していないか？ → Yes / No (使用していたら修正)

### 3. Paper spec 差分表
論文の主要パラメータと現在の実装を比較:

```markdown
| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | (論文の記述) | (実装の値) | Yes/No |
| ルックバック期間 | (論文の記述) | (実装の値) | Yes/No |
| リバランス頻度 | (論文の記述) | (実装の値) | Yes/No |
| 特徴量 | (論文の記述) | (実装の値) | Yes/No |
| コストモデル | (論文の記述) | (実装の値) | Yes/No |
```

**preflight.md が作成されるまで、Phase の実装コードに進まないこと。**

## ★ 今回のタスク (Cycle 5)


### Phase 5: ハイパーパラメータ最適化（近傍探索） [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: 論文で示唆される標準的な値の周辺でLSTMとGARCHのハイパーパラメータを最適化する。

**具体的な作業指示**:
1. `scripts/optimize_params.py`を作成します。 2. `optuna`ライブラリを使用し、最適化スタディを作成します。 3. LSTMの探索空間: `n_layers` (1-2), `hidden_units` (16, 32, 64), `dropout` (0.1-0.3)。 4. GARCHの探索空間: `p` (1), `q` (1) は固定とし、分布モデルを `normal` と `t` で試します。 5. 目的関数は、単一の訓練・検証スプリット（全データの最初の80%）におけるNetシャープレシオとします。 6. 最適化の結果（最良パラメータとトライアル履歴）を`reports/cycle_5/optimization_results.json`と`reports/cycle_5/optimization_history.png`に保存します。論文既定値（Phase 1の値）での結果も記録し比較します。

**期待される出力ファイル**:
- scripts/optimize_params.py
- reports/cycle_5/optimization_results.json
- reports/cycle_5/optimization_history.png

**受入基準 (これを全て満たすまで完了としない)**:
- 基準1: `optimization_results.json`に最良のハイパーパラメータセットと対応するシャープレシオが記録されている。
- 基準2: Optunaの`plot_optimization_history`による可視化がPNGファイルとして保存されている。
- 基準3: 探索はテストデータを一切使用せず、訓練・検証データのみで行われる。




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない


## スコア推移
Cycle 1: 45% → Cycle 2: 55% → Cycle 3: 45% → Cycle 4: 45%
改善速度: -5.0%/cycle ⚠ 停滞気味 — アプローチの転換を検討





## レビューからのフィードバック
### レビュー改善指示
1. [object Object]
2. [object Object]
3. [object Object]
### マネージャー指示 (次のアクション)
1. 【REPLAN: ブロッカー修正】
primaryBlocker: ハードコードされたボラティリティ閾値によるリスク調整戦略の性能劣化

上記のブロッカーを最優先で解決してください。

完了条件: ボラティリティ閾値を10パーセンタイルから90パーセンタイルまで10刻みで変化させた際のSharpe Ratio、累積リターン、最大ドローダウンを記録した感度分析レポート（例: `reports/sensitivity_analysis_vol_threshold.csv`）が生成される。
2. 【最優先】`src/trading_rules.py`内のハードコードされた`vol_percentile: float = 40.0`を、外部から設定可能なパラメータに変更する。mainスクリプトや設定ファイル（例: `config.yaml`）から値を渡せるようにリファクタリングする。
3. 【重要】変更したボラティリティ閾値パラメータを用いて感度分析を実施するスクリプト（例: `scripts/run_sensitivity_analysis.py`）を作成する。閾値を10%から90%まで10%刻みで変化させ、各々のバックテスト結果（Sharpe Ratio, 累積リターン, 最大ドローダウン）を`reports/sensitivity_analysis_vol_threshold.csv`に出力する。
4. 【推奨】`preflight.md`を作成し、プロジェクトのテンプレートに沿って必須項目を埋める。今回のサイクルで得られた知見（ハードコードされたパラメータの問題点、リスク調整戦略の失敗）を明確に記載する。


## 全体Phase計画 (参考)

✓ Phase 1: コアモデル実装（LSTM+GARCH） — LSTMで平均を予測し、その残差をGARCHでモデル化するハイブリッドモデルの基本構造を実装する。
✓ Phase 2: 実データパイプライン構築 — yfinanceからBTC-USDの日足データを取得し、モデルが使用できる形式に前処理するパイプラインを構築する。
✓ Phase 3: ウォークフォワード評価フレームワーク — モデルの性能を時系列に沿って頑健に評価するためのウォークフォワード検証を実装する。
✓ Phase 4: 取引戦略とコストモデルの実装 — モデルの予測に基づいて簡単な取引戦略をバックテストし、取引コストを考慮した純パフォーマンスを評価する。
→ Phase 5: ハイパーパラメータ最適化（近傍探索） — 論文で示唆される標準的な値の周辺でLSTMとGARCHのハイパーパラメータを最適化する。
  Phase 6: ロバスト性検証 — ウォークフォワードの分割数を増やし、異なる市場環境でのモデルの安定性を評価する。
  Phase 7: 代替ボラティリティモデルとの比較 — GARCHの有効性を検証するため、より単純なボラティリティモデル（移動標準偏差）との性能を比較する。
  Phase 8: 残差分析 — モデルの仮定が満たされているか確認するため、LSTMとGARCHの残差を統計的に分析する。
  Phase 9: 統合レポート生成 — すべてのフェーズで生成された結果を集約し、包括的なテクニカルレポートを生成する。
  Phase 10: エグゼクティブサマリーと仕上げ — 非技術者向けの要約を作成し、コードの品質を確保してプロジェクトを完了する。


## ベースライン比較（必須）

戦略の評価には、以下のベースラインとの比較が**必須**。metrics.json の `customMetrics` にベースライン結果を含めること。

| ベースライン | 実装方法 | 意味 |
|---|---|---|
| **1/N (Equal Weight)** | 全資産に均等配分、月次リバランス | 最低限のベンチマーク |
| **Vol-Targeted 1/N** | 1/N にボラティリティターゲティング (σ_target=10%) を適用 | リスク調整後の公平な比較 |
| **Simple Momentum** | 12ヶ月リターン上位50%にロング | モメンタム系論文の場合の自然な比較対象 |

```python
# metrics.json に含めるベースライン比較
"customMetrics": {
  "baseline_1n_sharpe": 0.5,
  "baseline_1n_return": 0.05,
  "baseline_1n_drawdown": -0.15,
  "baseline_voltarget_sharpe": 0.6,
  "baseline_momentum_sharpe": 0.4,
  "strategy_vs_1n_sharpe_diff": 0.1,
  "strategy_vs_1n_return_diff": 0.02,
  "strategy_vs_1n_drawdown_diff": -0.05,
  "strategy_vs_1n_turnover_ratio": 3.2,
  "strategy_vs_1n_cost_sensitivity": "論文戦略はコスト10bpsで1/Nに劣後"
}
```

「敗北」の場合、**どの指標で負けたか** (return / sharpe / drawdown / turnover / cost) を technical_findings.md に明記すること。

## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項

### データ・特徴量の禁止パターン（具体的）
- `scaler.fit(full_data)` してから split → **禁止**。`scaler.fit(train_data)` のみ
- `df.rolling(window=N, center=True)` → **禁止**。`center=False` (デフォルト) を使用
- データの `end_date` が今日以降 → **禁止**。`end_date` を明示的に過去に設定
- `merge` で未来のタイムスタンプを持つ行が特徴量に混入 → **禁止**
- ラベル生成後に特徴量を合わせる（ラベルの存在を前提に特徴量を選択）→ **禁止**

### 評価・報告の禁止パターン
- コストなしのgross PnLだけで判断しない
- テストセットでハイパーパラメータを調整しない
- 時系列データにランダムなtrain/test splitを使わない
- README に metrics.json と異なる数値を手書きしない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_5/preflight.md` — Preflight チェック結果（必須、実装前に作成）
- `reports/cycle_5/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_5/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ（Single Source of Truth）
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。

### レポート生成ルール（重要: 数値の一貫性）
- **`metrics.json` が全ての数値の唯一のソース (Single Source of Truth)**
- README や technical_findings に書く数値は **必ず metrics.json から引用** すること
- **手打ちの数値は禁止**。metrics.json に含まれない数値を README に書かない
- technical_findings.md で数値に言及する場合も metrics.json の値を参照
- README.md の Results セクションは metrics.json を読み込んで生成すること

### テスト必須
- `tests/test_data_integrity.py` のテストを実装状況に応じて有効化すること
- 新しいデータ処理や特徴量生成を追加したら、対応する leakage テストも追加
- `pytest tests/` が全パスしない場合、サイクルを完了としない

### その他の出力
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
