"""Data fetching: curl_cffi + yfinance 多策略，含完整執行日誌"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

INTERVAL_PERIOD_MAP = {
    "1m":  "5d",
    "5m":  "60d",
    "15m": "60d",
    "30m": "60d",
    "1h":  "730d",
    "1d":  "5y",
    "1wk": "10y",
}

INTRADAY_INTERVALS = {"1m", "5m", "15m", "30m", "1h"}

_YF_INTERVAL = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '60m', '1d': '1d', '1wk': '1wk',
}


def _concat_keep_attrs(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """pd.concat 會丟失 attrs，這裡手動保留（否則診斷資訊會消失）"""
    saved = dict(base.attrs)
    out   = pd.concat([base, add])
    out   = out[~out.index.duplicated(keep='last')].sort_index()
    out.attrs.update(saved)
    return out


def _fetch_via_curl(ticker: str, interval: str, lookback_days: int,
                    log: list) -> pd.DataFrame | None:
    """
    用 curl_cffi 模擬瀏覽器直抓 Yahoo Finance chart API。
    query1 / query2 兩個 host 都試，失敗原因寫入 log。
    """
    try:
        from curl_cffi import requests as curl_requests
    except Exception as e:
        log.append(f"curl:未安裝({type(e).__name__})")
        return None

    now      = datetime.utcnow()
    end_ts   = int((now + timedelta(days=2)).timestamp())
    start_ts = int((now - timedelta(days=lookback_days)).timestamp())
    yf_iv    = _YF_INTERVAL.get(interval, '1d')

    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://finance.yahoo.com/quote/{ticker}",
    }

    data = None
    for host in ("query1", "query2"):
        url = (
            f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval={yf_iv}&period1={start_ts}&period2={end_ts}"
            f"&includePrePost=false&events=div%2Csplit"
        )
        try:
            resp = curl_requests.get(url, impersonate="chrome124",
                                     timeout=15, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                break
            log.append(f"curl:{host}={resp.status_code}")
        except Exception as e:
            log.append(f"curl:{host}={type(e).__name__}")

    if data is None:
        return None

    try:
        result = data.get('chart', {}).get('result') or []
        if not result:
            log.append("curl:無result")
            return None

        r          = result[0]
        timestamps = r.get('timestamp') or []
        if not timestamps:
            log.append("curl:無timestamp")
            return None

        quote    = (r.get('indicators', {}).get('quote') or [{}])[0]
        adj_list = r.get('indicators', {}).get('adjclose') or []
        adj_arr  = adj_list[0].get('adjclose', []) if adj_list else []

        n       = len(timestamps)
        opens   = quote.get('open')   or [None] * n
        highs   = quote.get('high')   or [None] * n
        lows    = quote.get('low')    or [None] * n
        closes  = quote.get('close')  or [None] * n
        volumes = quote.get('volume') or [None] * n

        rows, idxs = [], []
        for i, ts in enumerate(timestamps):
            o, h, l = opens[i], highs[i], lows[i]
            c, v    = closes[i], volumes[i]
            if c is None:
                continue
            # OHL 缺值用 Close 補
            o = c if o is None else o
            h = c if h is None else h
            l = c if l is None else l
            if adj_arr and i < len(adj_arr) and adj_arr[i] and c:
                ratio = adj_arr[i] / c
                o, h, l, c = o * ratio, h * ratio, l * ratio, adj_arr[i]
            rows.append({'Open': o, 'High': h, 'Low': l,
                         'Close': c, 'Volume': v or 0})
            idxs.append(ts)

        if not rows:
            log.append("curl:全部NaN")
            return None

        idx = pd.to_datetime(idxs, unit='s', utc=True).tz_convert('America/New_York')
        df  = pd.DataFrame(rows, index=idx)

        if interval in ('1d', '1wk'):
            df.index = df.index.normalize().tz_localize(None)
            df = df[~df.index.duplicated(keep='last')]

        if len(df) < 5:
            log.append(f"curl:僅{len(df)}根")
            return None

        log.append(f"curl:ok({len(df)}根,末{str(df.index[-1])[:10]})")
        return df

    except Exception as e:
        log.append(f"curl:解析失敗({type(e).__name__})")
        return None


def fetch_ohlcv(ticker: str, interval: str, bar_count: int = 120) -> pd.DataFrame | None:
    """
    多策略抓取 OHLCV。每一步的結果都寫入 df.attrs['log']，
    方便在 UI 診斷面板看到究竟哪一步失敗。
    """
    log = []

    now    = datetime.utcnow()
    end_dt = now + timedelta(days=3)

    LOOKBACK_DAYS = {
        "1m": 7, "5m": 60, "15m": 60, "30m": 60,
        "1h": 730, "1d": 1825, "1wk": 3650,
    }
    lookback = LOOKBACK_DAYS.get(interval, 365)
    start_dt = now - timedelta(days=lookback)
    s_str    = start_dt.strftime('%Y-%m-%d')
    e_str    = end_dt.strftime('%Y-%m-%d')

    meta = {}   # 收集診斷資訊，最後一次寫回 attrs

    def _clean(raw):
        """清洗：只要求 Close 有值，OHL 缺值用 Close 補"""
        if raw is None or len(raw) < 10:
            return None
        meta['raw_latest'] = str(raw.index[-1])[:10]

        cols = [c for c in ["Open", "High", "Low", "Close"] if c in raw.columns]
        nan_rows = raw[raw[cols].isnull().any(axis=1)]
        if len(nan_rows) > 0:
            meta['filtered_nan'] = [str(d)[:10] for d in nan_rows.index[-3:]]

        out = raw.dropna(subset=["Close"]).copy()
        for c in ("Open", "High", "Low"):
            if c in out.columns:
                out[c] = out[c].fillna(out["Close"])

        if interval in INTRADAY_INTERVALS:
            out = _filter_trading_hours(out, interval)

        if "Volume" in out.columns:
            zero = out[out["Volume"] <= 0]
            if len(zero) > 0:
                meta['filtered_zero_vol'] = [str(d)[:10] for d in zero.index[-3:]]
            out = out[out["Volume"] > 0]

        return out if len(out) >= 10 else None

    tk = yf.Ticker(ticker)
    df = None
    strategy = "none"

    # ── 策略0：curl_cffi 直抓 Yahoo API ───────────────────────────────────
    try:
        cand = _fetch_via_curl(ticker, interval, lookback, log)
        cand = _clean(cand)
        if cand is not None:
            df, strategy = cand, "curl_cffi"
    except Exception as e:
        log.append(f"curl:例外({type(e).__name__})")

    # ── 策略1：yfinance start/end + auto_adjust=False ─────────────────────
    if df is None:
        try:
            cand = _clean(tk.history(start=s_str, end=e_str, interval=interval,
                                     auto_adjust=False, actions=False))
            if cand is not None:
                df, strategy = cand, "yf_se_noadj"
                log.append(f"yf_se_noadj:ok({len(df)}根,末{str(df.index[-1])[:10]})")
            else:
                log.append("yf_se_noadj:空")
        except Exception as e:
            log.append(f"yf_se_noadj:{type(e).__name__}")

    # ── 策略2：yfinance start/end + auto_adjust=True ──────────────────────
    if df is None:
        try:
            cand = _clean(tk.history(start=s_str, end=e_str, interval=interval,
                                     auto_adjust=True, actions=False))
            if cand is not None:
                df, strategy = cand, "yf_se_adj"
                log.append(f"yf_se_adj:ok({len(df)}根)")
            else:
                log.append("yf_se_adj:空")
        except Exception as e:
            log.append(f"yf_se_adj:{type(e).__name__}")

    # ── 策略3：yfinance period fallback ───────────────────────────────────
    if df is None:
        try:
            period = INTERVAL_PERIOD_MAP.get(interval, "1y")
            cand = _clean(tk.history(period=period, interval=interval,
                                     auto_adjust=True))
            if cand is not None:
                df, strategy = cand, "yf_period"
                log.append(f"yf_period:ok({len(df)}根)")
            else:
                log.append("yf_period:空")
        except Exception as e:
            log.append(f"yf_period:{type(e).__name__}")

    if df is None:
        return None

    df = df.tail(bar_count).copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)

    # ── 新鮮度補充：日線落後時，用分鐘線重建缺失的交易日 ────────────────────
    if interval == '1d':
        df = _patch_missing_days(df, tk, ticker, log)

    df.attrs.update(meta)
    df.attrs['strategy']   = strategy
    df.attrs['log']        = log
    df.attrs['fetch_time'] = datetime.utcnow().strftime('%H:%M UTC')
    return df


def _last_completed_session(now_et=None):
    """
    回傳最後一個「已收盤」的美股交易日（美東時間）。
    盤前/盤中不把今天算成已完成，避免用未收盤資料當成完整日線。
    now_et 可注入以便測試。
    """
    if now_et is None:
        now_et = pd.Timestamp.utcnow().tz_convert('America/New_York')
    d = now_et.date()
    # 週末，或今天尚未收盤（ET 16:00 前）→ 往前找
    if now_et.weekday() >= 5 or now_et.hour < 16:
        d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d = d - timedelta(days=1)
    return d


def _patch_missing_days(df: pd.DataFrame, tk, ticker: str,
                        log: list, now_et=None) -> pd.DataFrame:
    """
    日線落後時的補救：只用「真實成交」重建缺失交易日。

    核心原則（對應資料準確性目標）：
      1. 絕不製造合成K線 —— 沒有真實成交量就不補，寧可顯示落後也不顯示假資料
      2. 盤前不把今天當成缺失日（今天根本還沒開始交易）
      3. 盤中可補當日「未完成」K線，但必須有真實成交量，並標記為 partial
    """
    try:
        if now_et is None:
            now_et = pd.Timestamp.utcnow().tz_convert('America/New_York')
        today_et    = now_et.date()
        last_closed = _last_completed_session(now_et)
        last        = df.index[-1].date()

        # 盤中（今天已開盤但未收盤）才允許補當日未完成K線
        market_open_now = (now_et.weekday() < 5
                           and (now_et.hour > 9 or (now_et.hour == 9 and now_et.minute >= 30))
                           and now_et.hour < 16)
        end_day = today_et if market_open_now else last_closed

        missing = []
        d = last + timedelta(days=1)
        while d <= end_day:
            if d.weekday() < 5:
                missing.append(d)
            d += timedelta(days=1)

        if not missing:
            log.append(f"patch:無需補充(末{last},已收盤日{last_closed})")
            return df

        log.append(f"patch:缺{len(missing)}日({missing[0]}~{missing[-1]})")

        # ── 方法A：5分鐘數據按日重建（要求真實成交量）──────────────────────
        try:
            intra = tk.history(period='5d', interval='5m',
                               auto_adjust=False, actions=False)
            if intra is None or len(intra) == 0:
                log.append("5m:空")
            else:
                intra = intra.dropna(subset=['Close'])
                iidx  = pd.to_datetime(intra.index)
                if getattr(iidx, 'tz', None) is not None:
                    iidx = iidx.tz_convert('America/New_York')
                days = np.array([t.date() for t in iidx])

                rebuilt_idx, rebuilt_rows, skipped = [], [], []
                for md in missing:
                    sel = intra[days == md]
                    if len(sel) == 0:
                        continue
                    vol_sum = float(sel['Volume'].sum())
                    if vol_sum <= 0:
                        # 有K線但無成交（盤前掛單/資料佔位）→ 不補，避免製造假K線
                        skipped.append(str(md))
                        continue
                    rebuilt_idx.append(pd.Timestamp(md))
                    rebuilt_rows.append({
                        'Open':   float(sel['Open'].iloc[0]),
                        'High':   float(sel['High'].max()),
                        'Low':    float(sel['Low'].min()),
                        'Close':  float(sel['Close'].iloc[-1]),
                        'Volume': vol_sum,
                    })

                if rebuilt_rows:
                    add = pd.DataFrame(rebuilt_rows, index=rebuilt_idx)
                    df  = _concat_keep_attrs(df, add)
                    log.append(f"5m:重建{len(rebuilt_rows)}根"
                               f"@{rebuilt_rows[-1]['Close']:.2f}")
                    if rebuilt_idx[-1].date() == today_et and market_open_now:
                        df.attrs['partial_last_bar'] = str(today_et)
                        log.append(f"5m:末根為盤中未完成({today_et})")
                if skipped:
                    log.append(f"5m:零成交跳過({','.join(skipped)})")
                if not rebuilt_rows and not skipped:
                    avail = sorted(set(days))[-3:]
                    log.append(f"5m:無對應日(有{','.join(str(a) for a in avail)})")
        except Exception as e:
            log.append(f"5m:{type(e).__name__}")

        # ── 方法B：yf.download（同樣要求真實成交量）─────────────────────────
        if df.index[-1].date() < missing[-1]:
            try:
                dl = yf.download(ticker, period='5d', interval='1d',
                                 auto_adjust=False, progress=False)
                if dl is None or len(dl) == 0:
                    log.append("dl:空")
                else:
                    if isinstance(dl.columns, pd.MultiIndex):
                        dl.columns = dl.columns.get_level_values(0)
                    dl = dl.dropna(subset=['Close'])
                    for c in ('Open', 'High', 'Low'):
                        if c in dl.columns:
                            dl[c] = dl[c].fillna(dl['Close'])
                    dl.index = pd.to_datetime(dl.index)
                    if getattr(dl.index, 'tz', None) is not None:
                        dl.index = dl.index.tz_localize(None)
                    add = dl[dl.index > df.index[-1]]
                    # 只收有真實成交量的K線
                    if 'Volume' in add.columns:
                        before = len(add)
                        add = add[add['Volume'].fillna(0) > 0]
                        if before > len(add):
                            log.append(f"dl:濾掉{before-len(add)}根零成交")
                    if len(add) > 0:
                        df = _concat_keep_attrs(df, add)
                        log.append(f"dl:補{len(add)}根")
                    else:
                        log.append(f"dl:無新增(末{str(dl.index[-1])[:10]})")
            except Exception as e:
                log.append(f"dl:{type(e).__name__}")

        # 方法C（fast_info 合成K線）已移除：
        # 它會產生 OHLC 全等、Volume=0 的假K線，污染量比/ATR/型態/訊號強度。
        if df.index[-1].date() < missing[-1]:
            log.append(f"patch:仍落後至{df.index[-1].date()}(拒絕製造合成K線)")

    except Exception as e:
        log.append(f"patch:例外({type(e).__name__})")

    # ── 最後防線：任何零成交量K線都不得進入分析（patch 繞過了 _clean）────────
    if 'Volume' in df.columns:
        bad = df[df['Volume'].fillna(0) <= 0]
        if len(bad) > 0:
            df = df[df['Volume'].fillna(0) > 0]
            log.append(f"guard:剔除{len(bad)}根零成交K線")

    return df


def _filter_trading_hours(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    過濾非交易時段 K 線。
    美股正式盤：09:30-16:00 ET（America/New_York）
    """
    try:
        idx = df.index

        if idx.tz is None:
            idx_et = idx.tz_localize("America/New_York")
        else:
            idx_et = idx.tz_convert("America/New_York")

        hour_et   = idx_et.hour
        minute_et = idx_et.minute
        time_min  = hour_et * 60 + minute_et

        is_weekday = idx_et.dayofweek <= 4
        is_market  = (time_min >= 570) & (time_min < 960)
        mask = is_weekday & is_market

        filtered = df[mask.to_numpy() if hasattr(mask, 'to_numpy') else mask]

        if len(filtered) < max(5, len(df) // 4):
            return df

        return filtered

    except Exception:
        return df
