"""
市場結構分析 - 修正版
核心修正：
1. window 根據 bar 數量自適應（360根不能用 window=5）
2. 最小振幅過濾：擺動點之間必須 > ATR，排除噪音小波
3. 雙時間框架：長期擺動點判斷大趨勢 / 近期30根判斷局部結構，分開不互相污染
"""
import numpy as np
import pandas as pd


def _ema(data, period):
    k = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema


def find_swing_points(df: pd.DataFrame, window: int) -> dict:
    """找擺動高低點，並用最小振幅過濾噪音"""
    highs = df['High'].values
    lows  = df['Low'].values
    n     = len(df)

    # ATR 用來過濾太小的擺動（小於 0.5 ATR 的不算）
    atr_arr = (df['High'] - df['Low']).rolling(14).mean().bfill().values
    atr_mean = np.mean(atr_arr[-20:]) if n >= 20 else atr_arr[-1]

    sh, sl = [], []
    for i in range(window, n - window):
        local_max = max(highs[i-window:i+window+1])
        local_min = min(lows[i-window:i+window+1])

        if highs[i] == local_max:
            # 過濾：這個高點必須比左右 window 根的低點高出至少 0.5 ATR
            surrounding_lows = min(lows[i-window:i+window+1])
            if highs[i] - surrounding_lows >= atr_mean * 0.5:
                sh.append((i, highs[i]))

        if lows[i] == local_min:
            surrounding_highs = max(highs[i-window:i+window+1])
            if surrounding_highs - lows[i] >= atr_mean * 0.5:
                sl.append((i, lows[i]))

    return {"swing_highs": sh, "swing_lows": sl}


def analyze_market_structure(df: pd.DataFrame) -> dict:
    n       = len(df)
    closes  = df['Close'].values
    highs   = df['High'].values
    lows    = df['Low'].values
    current = closes[-1]

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)

    # ── EMA 斜率（全局趨勢判斷）─────────────────────────────────────────────
    ema50_slope = (ema50[-1] - ema50[-10]) / (ema50[-10] + 1e-9) * 100
    ema20_slope = (ema20[-1] - ema20[-5])  / (ema20[-5]  + 1e-9) * 100
    global_bull = ema50_slope >  0.5
    global_bear = ema50_slope < -0.5
    local_bull  = ema20_slope >  0.3
    local_bear  = ema20_slope < -0.3

    # ── 自適應 window ────────────────────────────────────────────────────────
    # 規則：window = max(5, n // 15)
    # 120根 → window=8，360根 → window=24，500根 → window=33
    window_global = max(5, n // 15)
    window_local  = max(3, min(8, n // 30))   # 近期結構用更小窗口，捕捉最新反彈

    # ── 長期擺動點（判斷大趨勢結構）────────────────────────────────────────
    swings_global = find_swing_points(df, window_global)
    sh_global = swings_global['swing_highs'][-6:]
    sl_global = swings_global['swing_lows'][-6:]

    # ── 近期擺動點（判斷局部/最新結構，只看最近 60 根）────────────────────
    local_bars = min(60, n)
    df_local   = df.iloc[-local_bars:]
    swings_local = find_swing_points(df_local, window_local)

    # 索引換算回全局（+offset）
    offset = n - local_bars
    sh_local = [(i + offset, v) for i, v in swings_local['swing_highs'][-4:]]
    sl_local = [(i + offset, v) for i, v in swings_local['swing_lows'][-4:]]

    # ── 選擇哪套擺動點做 HH/HL 判斷 ──────────────────────────────────────────
    # 優先用局部擺動點（更能反映最新市場方向）
    # 若局部點不足則 fallback 到全局點
    sh_use = sh_local if len(sh_local) >= 2 else sh_global
    sl_use = sl_local if len(sl_local) >= 2 else sl_global

    # 圖表顯示用（取全局點，讓圖上標記更清晰）
    sh_display = sh_global[-8:]
    sl_display = sl_global[-8:]

    if len(sh_use) < 2 or len(sl_use) < 2:
        return _default_struct(ema20, ema50, current, highs, lows, n,
                               sh_display, sl_display,
                               global_bull, global_bear, local_bull,
                               ema20_slope, ema50_slope)

    # ── HH / HL / LH / LL 判斷 ───────────────────────────────────────────────
    hh = sh_use[-1][1] > sh_use[-2][1]
    hl = sl_use[-1][1] > sl_use[-2][1]
    lh = sh_use[-1][1] < sh_use[-2][1]
    ll = sl_use[-1][1] < sl_use[-2][1]

    # ── 大趨勢結構（用全局擺動點再判斷一次）──────────────────────────────────
    macro_hh = macro_hl = macro_lh = macro_ll = False
    if len(sh_global) >= 2 and len(sl_global) >= 2:
        macro_hh = sh_global[-1][1] > sh_global[-2][1]
        macro_hl = sl_global[-1][1] > sl_global[-2][1]
        macro_lh = sh_global[-1][1] < sh_global[-2][1]
        macro_ll = sl_global[-1][1] < sl_global[-2][1]

    # ── 即時結構檢查（補償擺動點的確認延遲）────────────────────────────────
    # 擺動點必須等右側 window 根K線才算「確認」，所以最新幾根的急跌/急漲，
    # 在確認之前完全不會出現在 HH/HL/LH/LL 裡。實測：真正的頂部要在價格已
    # 回落 5% 以上才被確認成新的 Higher High，結果「頂部被確認」的那一刻反而
    # 被解讀成「創新高=多頭趨勢」，標籤隨下跌越走越看多，強度還給到 85。
    # 因此在擺動點結構之上疊加兩個「不需要等確認」的即時證據：
    #   1. 收盤價是否已跌破/突破「最近一個已確認的 HL / LH」
    #      （上升結構失守 = SMC 的 CHoCH，定義上趨勢結構已被破壞）
    #   2. 價格是否同時跌破/站上 EMA20 與 EMA50（趨勢濾網，兩條都失守才算）
    last_hl_price  = sl_use[-1][1]
    last_lh_price  = sh_use[-1][1]
    below_both_ema = current < ema20[-1] and current < ema50[-1]
    above_both_ema = current > ema20[-1] and current > ema50[-1]
    hl_broken      = current < last_hl_price
    lh_broken      = current > last_lh_price
    structure_state = "intact"   # intact | pullback | broken

    # ── 趨勢判斷（局部結構 + EMA 大趨勢 + 即時結構檢查）──────────────────────
    if hh and hl:
        if hl_broken:
            # 擺動點仍顯示 HH+HL，但收盤已跌破最近的 Higher Low → 上升結構失守
            trend = "結構轉空"
            sub   = (f"擺動結構曾為 HH+HL，但收盤 ${current:.2f} 已跌破前一個 "
                     f"Higher Low（${last_hl_price:.2f}），上升結構失守（CHoCH）")
            swing_desc    = "HH ▲ + HL ▲（HL 已跌破）"
            strength_base = 35
            structure_state = "broken"
        elif global_bull:
            if below_both_ema:
                # 結構未破，但價格已同時失守 EMA20 與 EMA50：多頭承壓，不是健康趨勢
                trend = "多頭回調"
                sub   = (f"擺動結構仍為 HH+HL，但價格已跌破 EMA20 與 EMA50，多頭承壓；"
                         f"關鍵看前一個 Higher Low（${last_hl_price:.2f}）能否守住")
                swing_desc    = "HH ▲ + HL ▲（回調承壓）"
                strength_base = 55
                structure_state = "pullback"
            else:
                trend = "多頭趨勢"
                sub   = "Higher High + Higher Low，主趨勢多頭延續"
                swing_desc    = "HH ▲ + HL ▲"
                strength_base = 80
        else:
            trend = "局部多頭反彈"
            sub   = "近期出現 HH + HL 結構，大趨勢仍偏空，注意上方阻力"
            swing_desc    = "HH ▲ + HL ▲"
            strength_base = 62

    elif lh and ll:
        if lh_broken:
            # 鏡像：擺動點仍顯示 LH+LL，但收盤已突破最近的 Lower High → 下降結構失守
            trend = "結構轉多"
            sub   = (f"擺動結構曾為 LH+LL，但收盤 ${current:.2f} 已突破前一個 "
                     f"Lower High（${last_lh_price:.2f}），下降結構失守（CHoCH）")
            swing_desc    = "LH ▼ + LL ▼（LH 已突破）"
            strength_base = 65
            structure_state = "broken"
        elif local_bull:
            # 再確認：如果局部點顯示 LH+LL，但 EMA20 已明顯回升，降低空頭權重
            trend = "局部反彈中"
            sub   = "擺動結構仍為 LH+LL，但 EMA20 開始上翹，局部動能轉強"
            swing_desc    = "LH ▼ + LL ▼（EMA20 回升）"
            strength_base = 38
            structure_state = "pullback"
        elif above_both_ema:
            # 鏡像於「多頭回調」：價格已同時站上 EMA20 與 EMA50，空頭承壓
            trend = "局部反彈中"
            sub   = (f"擺動結構仍為 LH+LL，但價格已站上 EMA20 與 EMA50，空頭承壓；"
                     f"關鍵看前一個 Lower High（${last_lh_price:.2f}）能否壓住")
            swing_desc    = "LH ▼ + LL ▼（站上均線）"
            strength_base = 38
            structure_state = "pullback"
        else:
            trend = "空頭趨勢"
            sub   = "Lower High + Lower Low，空頭結構完整"
            swing_desc    = "LH ▼ + LL ▼"
            strength_base = 20

    elif lh and hl:
        trend = "橫盤收斂"
        sub   = "Lower High + Higher Low，三角收斂蓄勢"
        swing_desc    = "LH ▼ + HL ▲（收斂）"
        strength_base = 50

    elif hh and ll:
        trend = "趨勢發散"
        sub   = "Higher High + Lower Low，多空均有力，震盪擴大"
        swing_desc    = "HH ▲ + LL ▼（發散）"
        strength_base = 45

    else:
        trend = "橫盤整理"
        sub   = "無明確擺動結構，區間震盪"
        swing_desc    = "無明確 HH/HL/LH/LL"
        strength_base = 45

    # ── 趨勢反轉偵測 ─────────────────────────────────────────────────────────
    # reversal_bias 是給評分系統用的「明確方向欄位」，取代過去 signals.py 靠
    # 「文字裡有沒有『多頭』二字」來判斷方向的做法——那會讓「頂部反轉」這種
    # 看空警告（文字裡也有『多頭』）反過來幫買方加分。
    #   'bull' = 看多反轉   'bear' = 看空反轉   'bull_weak' = 弱證據（不計分）
    #   '' = 無，或已由「結構轉空/結構轉多」標籤本身計分（避免重複計分）
    reversal_signal = ""
    reversal_bias   = ""
    if hh and hl and hl_broken:
        reversal_signal = (f"⚠️ 收盤 ${current:.2f} 已跌破前一個 Higher Low"
                           f"（${last_hl_price:.2f}），上升結構失守（CHoCH）")
    elif lh and ll and lh_broken:
        reversal_signal = (f"⚠️ 收盤 ${current:.2f} 已突破前一個 Lower High"
                           f"（${last_lh_price:.2f}），下降結構失守（CHoCH）")
    elif global_bear and hh and hl:
        reversal_signal = "⚠️ 大趨勢空頭中出現局部多頭結構（HH+HL），疑似底部反轉初期"
        reversal_bias   = "bull"
    elif global_bear and local_bull and not (lh and ll):
        reversal_signal = "⚠️ EMA20 開始回升，局部動能轉多，空頭結構可能弱化"
        reversal_bias   = "bull_weak"
    elif global_bull and lh and ll:
        reversal_signal = "⚠️ 大趨勢多頭中出現局部空頭結構（LH+LL），疑似頂部反轉初期"
        reversal_bias   = "bear"

    # ── 趨勢強度 ─────────────────────────────────────────────────────────────
    strength       = strength_base
    above_ema20    = current > ema20[-1]
    above_ema50    = current > ema50[-1]
    ema_aligned    = ema20[-1] > ema50[-1]

    # 「多頭回調」的價格定義上就在兩條 EMA 之下，不再給 EMA 加分
    if "多頭" in trend and "回調" not in trend:
        if above_ema20:  strength = min(strength + 8,  100)
        if above_ema50:  strength = min(strength + 7,  100)
        if ema_aligned:  strength = min(strength + 5,  100)
    elif "空頭" in trend:
        if not above_ema20: strength = max(strength - 8, 0)
        if not above_ema50: strength = max(strength - 7, 0)
        if not ema_aligned: strength = max(strength - 5, 0)

    # ── 結構突破 ──────────────────────────────────────────────────────────────
    recent_high = max(highs[-20:])
    recent_low  = min(lows[-20:])
    if current > recent_high * 0.999:
        struct_break = "突破阻力 ↑"
    elif current < recent_low * 1.001:
        struct_break = "跌破支撐 ↓"
    else:
        struct_break = "區間內整理"

    # ── 市場狀態 ──────────────────────────────────────────────────────────────
    atr     = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
    vol_std = float(df['Close'].rolling(20).std().iloc[-1])
    if atr > vol_std * 1.6:
        market_state = "高波動擴張"
    elif atr < vol_std * 0.6:
        market_state = "低波動蓄勢"
    else:
        market_state = "正常波動"

    return {
        "trend":           trend,
        "sub_trend":       sub,
        "swing_desc":      swing_desc,
        "trend_strength":  int(strength),
        "market_state":    market_state,
        "structure_break": struct_break,
        "reversal_signal": reversal_signal,
        "swing_highs":     sh_display,
        "swing_lows":      sl_display,
        "ema20":           ema20,
        "ema50":           ema50,
        "global_bull":     global_bull,
        "global_bear":     global_bear,
        "local_bull":      local_bull,
        "hh": hh, "hl": hl, "lh": lh, "ll": ll,
        "macro_hh": macro_hh, "macro_hl": macro_hl,
        "macro_lh": macro_lh, "macro_ll": macro_ll,
        "above_ema20":     above_ema20,
        "above_ema50":     above_ema50,
        "ema_aligned":     ema_aligned,
        "ema20_slope":     ema20_slope,
        "ema50_slope":     ema50_slope,
        "recent_high":     recent_high,
        "recent_low":      recent_low,
    }


def _default_struct(ema20, ema50, current, highs, lows, n,
                    sh, sl, global_bull, global_bear, local_bull,
                    ema20_slope, ema50_slope):
    return {
        "trend": "橫盤整理", "sub_trend": "數據不足",
        "swing_desc": "N/A", "trend_strength": 45,
        "market_state": "觀望", "structure_break": "無",
        "reversal_signal": "", "swing_highs": sh, "swing_lows": sl,
        "ema20": ema20, "ema50": ema50,
        "global_bull": global_bull, "global_bear": global_bear,
        "local_bull": local_bull,
        "hh": False, "hl": False, "lh": False, "ll": False,
        "macro_hh": False, "macro_hl": False, "macro_lh": False, "macro_ll": False,
        "above_ema20": current > ema20[-1], "above_ema50": current > ema50[-1],
        "ema_aligned": ema20[-1] > ema50[-1],
        "ema20_slope": ema20_slope, "ema50_slope": ema50_slope,
        "recent_high": max(highs[-20:]) if n >= 20 else highs[-1],
        "recent_low":  min(lows[-20:])  if n >= 20 else lows[-1],
    }
