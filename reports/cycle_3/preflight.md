# Preflight Check — Cycle 3 (Phase 3: Walk-Forward Evaluation)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-30 (今日以前であること) ✓ |
| Train期間 | Expanding window: earliest split starts ~2016-03-31 |
| Validation期間 | N/A (walk-forward OOS serves as validation) |
| Test期間 | Walk-forward OOS windows across full dataset |
| 重複なし確認 | Yes |
| 未来日付なし確認 | Yes |

Walk-forward with 5 splits (expanding window): each split's train end < test start, with gap=1 to prevent leakage.

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes** (log returns use Close[t]/Close[t-1], LSTM lookback uses past window only)
- Scaler / Imputer は train データのみで fit しているか？ → **Yes** (scaler.fit on train split only, transform on test)
- Centered rolling window を使用していないか？ → **Yes** (not used)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | BTC-USD daily | BTC-USD daily | Yes |
| ルックバック期間 | ~60 days (LSTM sequence) | 60 days | Yes |
| リバランス頻度 | Daily (1-step ahead) | Daily | Yes |
| 特徴量 | Log returns | Log returns | Yes |
| コストモデル | 10bps fee + 5bps slippage | 10bps fee + 5bps slippage | Yes |
| LSTM hidden units | 50 | 50 | Yes |
| LSTM layers | 1 | 1 | Yes |
| GARCH model | GARCH(1,1) | GARCH(1,1) | Yes |
| Walk-forward splits | 5 | 5 | Yes |
| Training approach | Walk-forward expanding | Walk-forward expanding | Yes |
