# -*- coding: utf-8 -*-
"""
fomc_sp500_5d_overlay_fred.py

Берёт ставки ФРС и S&P 500 с FRED, сопоставляет даты, считает доходность за 5 торговых дней
и строит график.
"""

import os
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

# -------------------------
# Настройки
# -------------------------
START_YEAR = 2005
END_YEAR = 2023
OUTDIR = "out"
FIG_DPI = 220
FIGSIZE = (14, 7)
FRED_SERIES_UPPER = "DFEDTARU"
FRED_SERIES_LOWER = "DFEDTARL"
FRED_SPX = "SP500"  # S&P 500 index from FRED

os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# 1) Получаем данные ФРС с FRED
# -------------------------
def fetch_fomc_decisions(start_year=START_YEAR, end_year=END_YEAR):
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    upper = pdr.DataReader(FRED_SERIES_UPPER, "fred", start, end)
    lower = pdr.DataReader(FRED_SERIES_LOWER, "fred", start, end)

    df = pd.concat([upper, lower], axis=1).dropna()
    df.columns = ["upper", "lower"]

    df["ffr_mid"] = df[["upper", "lower"]].mean(axis=1).round(2)
    df = df[df["ffr_mid"].diff().fillna(0) != 0].reset_index()
    df.rename(columns={"DATE": "decision_date"}, inplace=True)

    df["delta_bps"] = df["ffr_mid"].diff().apply(lambda x: None if pd.isna(x) else round(x * 100))

    def classify(bps):
        if bps is None:
            return "n/a"
        if bps > 0:
            return "hike"
        if bps < 0:
            return "cut"
        return "unch"

    df["move"] = df["delta_bps"].apply(classify)
    df["ffr_text"] = df["lower"].astype(str) + "% – " + df["upper"].astype(str) + "%"

    return df

# -------------------------
# 2) Загружаем S&P 500 через FRED
# -------------------------
def download_spx(start="2005-01-01", end="2023-12-31"):
    df = pdr.DataReader(FRED_SPX, "fred", start, end)
    df = df.rename_axis("Date").reset_index().rename(columns={FRED_SPX: "Close"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna()
    return df

# -------------------------
# 3) Сопоставление дат и расчёт +5d доходности
# -------------------------
def map_decisions_to_trading_days(decisions: pd.DataFrame, prices: pd.DataFrame, forward_days=5):
    px = prices.set_index("Date").sort_index()
    out = []
    for _, r in decisions.iterrows():
        dec_date = pd.Timestamp(r["decision_date"])
        window = pd.date_range(dec_date, dec_date + pd.Timedelta(days=3), freq="B")
        sub = px.reindex(window)["Close"].dropna()
        if sub.empty:
            next_idx = px.index[px.index >= dec_date]
            if len(next_idx) == 0:
                continue
            trade0 = next_idx[0]
        else:
            trade0 = sub.index[0]
        all_trading = px.index[px.index >= trade0]
        if len(all_trading) < forward_days + 1:
            continue
        pos = all_trading.get_indexer([trade0])[0]
        tradeN = all_trading[pos + forward_days]
        p0 = float(px.loc[trade0, "Close"])
        pN = float(px.loc[tradeN, "Close"])
        ret_pct = (pN / p0 - 1.0) * 100.0
        out.append({
            "decision_date": r["decision_date"].date(),
            "trade0": trade0.date(),
            "tradeN": tradeN.date(),
            "close0": p0,
            "closeN": pN,
            "retN_pct": ret_pct,
            "ffr_text": r["ffr_text"],
            "ffr_mid": r["ffr_mid"],
            "delta_bps": r["delta_bps"],
            "move": r["move"],
        })
    return pd.DataFrame(out)

# -------------------------
# 4) Построение графика
# -------------------------
def plot_results(prices: pd.DataFrame, mapped: pd.DataFrame, outpath=os.path.join(OUTDIR, "fomc_spx_overlay_2005_2023.png")):
    # Используем стандартный стиль Matplotlib вместо seaborn
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(prices["Date"], prices["Close"], linewidth=1.3, label="S&P 500 (FRED)")

    colors = {"hike": "#2ca02c", "cut": "#d62728", "unch": "#1f77b4", "n/a": "#7f7f7f"}

    for _, r in mapped.iterrows():
        t0 = pd.Timestamp(r["trade0"])
        tN = pd.Timestamp(r["tradeN"])
        c = colors.get(r["move"], "#7f7f7f")
        ax.axvline(t0, color=c, linewidth=0.9, alpha=0.65)
        ax.scatter([t0], [r["close0"]], s=18, color=c, edgecolor="k", zorder=5)
        ax.scatter([tN], [r["closeN"]], s=28, color=c, marker="D", edgecolor="k", zorder=6)
        ax.plot([t0, tN], [r["close0"], r["closeN"]], linestyle="--", linewidth=0.9, color=c, alpha=0.7)
        x_label = t0 + timedelta(days=3)
        try:
            idx = (prices["Date"] - x_label).abs().idxmin()
            y_ref = prices.loc[idx, "Close"]
            label = f"{r['retN_pct']:+.2f}%"
            ax.text(x_label, y_ref, label, fontsize=8, weight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, boxstyle="round,pad=0.2"))
        except Exception:
            pass

    ax.set_title("S&P 500: FOMC Rate Decisions (2005–2023) и изменение за 5 торговых дней", fontsize=14, weight="bold")
    ax.set_xlabel("Год", fontsize=12)
    ax.set_ylabel("Уровень индекса (Close)", fontsize=12)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=colors["hike"], lw=2, label="hike (повышение)"),
        Line2D([0], [0], color=colors["cut"], lw=2, label="cut (понижение)"),
        Line2D([0], [0], color=colors["unch"], lw=2, label="unch (без изм.)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(outpath, dpi=FIG_DPI)
    plt.close(fig)

# -------------------------
# Main
# -------------------------
def main():
    print("1) Получаем решения ФРС (FRED)...")
    decisions = fetch_fomc_decisions()
    decisions.to_csv(os.path.join(OUTDIR, "fomc_decisions_2005_2023.csv"), index=False)
    print(f"  Найдено записей: {len(decisions)}")

    print("2) Загружаем котировки S&P 500 (FRED)...")
    prices = download_spx(start="2005-01-01", end="2023-12-31")
    prices.to_csv(os.path.join(OUTDIR, "spx_2005_2023_daily.csv"), index=False)
    print(f"  Загружено {len(prices)} торговых строк")

    print("3) Сопоставляем решения с торговыми днями и считаем +5d...")
    mapped = map_decisions_to_trading_days(decisions, prices, forward_days=5)
    mapped.to_csv(os.path.join(OUTDIR, "fomc_overlay_spx_2005_2023.csv"), index=False)
    print(f"  Записей с +5d: {len(mapped)}")

    print("4) Рисуем график и сохраняем PNG...")
    plot_results(prices, mapped, outpath=os.path.join(OUTDIR, "fomc_spx_overlay_2005_2023.png"))
    print("Готово. Файлы в папке", OUTDIR)

if __name__ == "__main__":
    main()


