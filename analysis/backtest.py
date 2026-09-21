"""
事件驅動回測系統
================
核心原則：回測必須測試「真正在交易的那套規則」，不能用另一套簡化模型
代替，否則勝率/盈虧比回答的是錯的問題（詳見專案自我檢討記錄）。

做法：在歷史每一根K線上，只用「當時已知」的資料（df.iloc[:i+1]，
擴張視窗）重新跑一次跟即時分析完全相同的訊號管線
（detect_all_patterns → analyze_market_structure → analyze_volume →
find_support_resistance → generate_signals），得到當時真正會顯示的
BUY/SELL/NEUTRAL 判斷與 ATR 止損/目標價，然後逐根走訪後續K線，
用當時設定的止損/目標價判斷這筆交易的勝負——不使用任何之後才知道的
資訊（支撐阻力、ATR、評分全部只用截至當根的資料重新計算）。

已知假設（會直接影響結果，非憑空捏造）：
  - MIN_WARMUP：至少 60 根才開始評估訊號（EMA50/ATR14/20根支撐阻力
    在資料太短時只是退化的預設值，太早進場會用到不可靠的指標）
  - MAX_HOLD_BARS：單筆交易最多持有 20 根，逾期用當根收盤強制平倉
    （避免無限期持倉吃掉所有後續資料，也更貼近實際交易紀律）
  - require_good_rrr=True（預設）：即時畫面會顯示「⚠️不建議入場」的
    情況（too_close / rrr_poor），回測也視同觀望、不進場——否則測的是
    「無視系統自己的風險警告硬幹」，不是這套系統實際會建議使用者做的事
  - 同一根K線內止損與目標同時被觸及時，保守假設「止損先到」
    （OHLC 資料無法得知盤中先後順序，寧可低估勝率不高估）
  - 固定風險部位（每筆冒 1% 權益），用 R 倍數換算權益曲線，
    而非任意的「半倉」係數
  - 資料集最後一筆仍未平倉的交易，用最後一根實際收盤價結算，
    標記為 end_of_data（用的是真實成交價，不是預測值）
"""
import numpy as np
import pandas as pd

from analysis.pattern_detector    import detect_all_patterns
from analysis.market_structure    import analyze_market_structure
from analysis.volume_analysis     import analyze_volume
from analysis.support_resistance  import find_support_resistance
from analysis.signals             import generate_signals

MIN_WARMUP         = 60
MAX_HOLD_BARS       = 20
RISK_PER_TRADE_PCT  = 1.0   # 固定風險部位模型：每筆交易冒 1% 權益


def _empty_result(reason: str = "", require_good_rrr: bool = True) -> dict:
    return {
        "win_rate": 0, "profit_factor": 0, "max_dd": 0,
        "total_trades": 0, "net_return": 0, "equity_curve": [100],
        "avg_r_multiple": 0, "avg_hold_bars": 0,
        "warmup_bars": MIN_WARMUP, "insufficient_reason": reason,
        "require_good_rrr": require_good_rrr,
        "trades": [],
    }


def _resolve_trade_setup(sub_df: pd.DataFrame, require_good_rrr: bool = True):
    """
    對截至目前為止的資料重新跑一次真正的訊號管線，
    回傳 (primary, entry_price, stop_loss, target) 或 (None, ...)。
    這一步跟即時分析用的是完全同一套函式，唯一差別是只餵「當時已知」的資料。

    require_good_rrr=True（預設）：若當時的 trade_setup 本身標記
    too_close 或 rrr_poor（即即時畫面會顯示「⚠️不建議入場」的那些情況），
    視同觀望、不產生交易——否則回測等於在測「無視系統自己的風險警告
    硬幹」，答的又是另一個錯的問題，跟系統實際會建議使用者做的事脫鉤。
    """
    patterns      = detect_all_patterns(sub_df)
    market_struct = analyze_market_structure(sub_df)
    vol           = analyze_volume(sub_df)
    sr            = find_support_resistance(sub_df)
    sig           = generate_signals(sub_df, patterns, market_struct, vol, sr)

    primary = sig.get("primary")
    if primary not in ("BUY", "SELL"):
        return None, None, None, None

    trade = sig.get("trade_setup", {})
    if require_good_rrr and (trade.get("too_close") or trade.get("rrr_poor")):
        return None, None, None, None

    entry_price = float(sub_df["Close"].iloc[-1])
    stop_loss   = trade.get("stop_loss")
    # generate_signals 內部用 target = key_resistance(BUY) / key_support(SELL)
    # 算風報比，但沒有把 target 直接存進 trade_setup，這裡用同一套邏輯還原，
    # 確保回測用的目標價跟即時畫面顯示的關鍵阻力/支撐完全一致。
    target = (trade.get("key_resistance") if primary == "BUY"
              else trade.get("key_support"))

    if stop_loss is None or target is None:
        return None, None, None, None
    return primary, entry_price, stop_loss, target


def run_backtest(df: pd.DataFrame, require_good_rrr: bool = True) -> dict:
    n = len(df)
    if n < MIN_WARMUP + 10:
        return _empty_result(f"資料僅 {n} 根，至少需要 {MIN_WARMUP + 10} 根"
                              f"（{MIN_WARMUP} 根暖機 + 10 根以上供交易發生）",
                              require_good_rrr=require_good_rrr)

    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    trades   = []
    position = None   # {'direction','entry_idx','entry_price','stop_loss','target'}

    for i in range(MIN_WARMUP, n):
        if position is None:
            # 空手中：用「截至當根」的資料重新跑一次真正的訊號邏輯
            sub_df = df.iloc[: i + 1]
            primary, entry_price, stop_loss, target = _resolve_trade_setup(
                sub_df, require_good_rrr=require_good_rrr
            )
            if primary is not None:
                position = {
                    "direction":   primary,
                    "entry_idx":   i,
                    "entry_price": entry_price,
                    "stop_loss":   stop_loss,
                    "target":      target,
                }
            continue

        # 持倉中：用「下一根」的真實高低點判斷有沒有觸及止損/目標
        # （進場價是 bar i 的收盤，所以最早從 bar i+1 開始才可能出場，
        #  這一根本身不會拿來判斷出場，避免用進場當根的走勢反推）
        if i <= position["entry_idx"]:
            continue

        bar_high, bar_low = highs[i], lows[i]
        hit_stop = hit_target = False
        if position["direction"] == "BUY":
            hit_stop   = bar_low  <= position["stop_loss"]
            hit_target = bar_high >= position["target"]
        else:
            hit_stop   = bar_high >= position["stop_loss"]
            hit_target = bar_low  <= position["target"]

        exit_reason = exit_price = None
        if hit_stop:
            # 同根K線兩者皆觸及時，保守假設止損先到（無法得知盤中順序）
            exit_reason, exit_price = "stop", position["stop_loss"]
        elif hit_target:
            exit_reason, exit_price = "target", position["target"]
        elif (i - position["entry_idx"]) >= MAX_HOLD_BARS:
            exit_reason, exit_price = "timeout", float(closes[i])
        elif i == n - 1:
            exit_reason, exit_price = "end_of_data", float(closes[i])

        if exit_reason:
            entry_price = position["entry_price"]
            if position["direction"] == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100

            risk_pct    = abs(entry_price - position["stop_loss"]) / entry_price * 100
            r_multiple  = pnl_pct / risk_pct if risk_pct > 0 else 0

            trades.append({
                "entry_idx":    position["entry_idx"],
                "exit_idx":     i,
                "direction":    position["direction"],
                "entry_price":  entry_price,
                "exit_price":   exit_price,
                "exit_reason":  exit_reason,
                "pnl_pct":      pnl_pct,
                "r_multiple":   r_multiple,
                "hold_bars":    i - position["entry_idx"],
            })
            position = None

    if len(trades) < 3:
        result = _empty_result(f"僅產生 {len(trades)} 筆完整交易，樣本數不足以計算"
                                f"可靠的勝率/盈虧比（建議至少3筆，理想上10筆以上）",
                                require_good_rrr=require_good_rrr)
        result["total_trades"] = len(trades)
        result["trades"] = trades
        return result

    # ── 用固定風險部位模型（每筆冒1%權益）把 R 倍數換算成權益曲線 ──────────
    equity     = [100.0]
    max_equity = 100.0
    max_dd     = 0.0
    for t in trades:
        equity_change = t["r_multiple"] * RISK_PER_TRADE_PCT
        equity.append(equity[-1] * (1 + equity_change / 100))
        max_equity = max(max_equity, equity[-1])
        dd = (max_equity - equity[-1]) / max_equity * 100 if max_equity > 0 else 0
        max_dd = max(max_dd, dd)

    r_values = [t["r_multiple"] for t in trades]
    wins     = [r for r in r_values if r > 0]
    losses   = [r for r in r_values if r <= 0]

    win_rate = len(wins) / len(trades) * 100
    pos_sum  = sum(wins)
    neg_sum  = abs(sum(losses))
    profit_factor = (pos_sum / neg_sum) if neg_sum > 0 else (pos_sum if pos_sum > 0 else 0)

    net_return    = equity[-1] - 100
    avg_r         = float(np.mean(r_values))
    avg_hold_bars = float(np.mean([t["hold_bars"] for t in trades]))

    return {
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "max_dd":         max_dd,
        "total_trades":   len(trades),
        "net_return":     net_return,
        "equity_curve":   equity,
        "avg_r_multiple": avg_r,
        "avg_hold_bars":  avg_hold_bars,
        "warmup_bars":    MIN_WARMUP,
        "insufficient_reason": "",
        "require_good_rrr": require_good_rrr,
        "trades":         trades,
    }
