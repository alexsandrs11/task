# -*- coding: utf-8 -*-
"""
fomc_sp500_plotly_final.py

S&P 500 и ставки ФРС с FRED, интерактивный график с цветовой кодировкой hike/cut/unch,
подсказками +5d доходности и вертикальными линиями на датах решений.
"""

import os
from datetime import timedelta
import pandas as pd
from pandas_datareader import data as pdr
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# -------------------------
# Настройки
# -------------------------
START_YEAR = 2005
END_YEAR = 2023
OUTDIR = "out"
FRED_SERIES_UPPER = "DFEDTARU"
FRED_SERIES_LOWER = "DFEDTARL"
FRED_SPX = "SP500"
os.makedirs(OUTDIR, exist_ok=True)

COLORS_MOVE = {"hike": "green", "cut": "red", "unch": "blue", "n/a": "gray"}

# -------------------------
# 1) Данные ФРС
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
    df["ffr_text"] = df["lower"].astype(str) + "% – " + df["upper"].astype(str) + "%"

    def classify(bps):
        if bps is None:
            return "n/a"
        if bps > 0:
            return "hike"
        if bps < 0:
            return "cut"
        return "unch"

    df["move"] = df["delta_bps"].apply(classify)
    return df

# -------------------------
# 2) Данные S&P 500
# -------------------------
def download_spx(start="2005-01-01", end="2023-12-31"):
    df = pdr.DataReader(FRED_SPX, "fred", start, end)
    df = df.rename_axis("Date").reset_index().rename(columns={FRED_SPX: "Close"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna()
    return df

# -------------------------
# 3) Сопоставление дат и +5d доходности
# -------------------------
def map_decisions_to_trading_days(decisions, prices, forward_days=5):
    px = prices.set_index("Date").sort_index()
    out = []
    for _, r in decisions.iterrows():
        dec_date = pd.Timestamp(r["decision_date"])
        window = pd.date_range(dec_date, dec_date + timedelta(days=3), freq="B")
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
# 4) Построение интерактивного графика
# -------------------------
def plot_interactive_colored(prices, decisions, mapped, filename=os.path.join(OUTDIR, "fomc_spx_interactive_final.html")):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("S&P 500 (FRED)", "FFR Midpoint (%)")
    )

    # S&P 500 линия
    fig.add_trace(go.Scatter(x=prices["Date"], y=prices["Close"], name="S&P 500", line=dict(color='blue')), row=1, col=1)

    # +5d доходность с подсказкой и вертикальные линии решений
    for _, r in mapped.iterrows():
        # пунктирная линия +5d доходности
        fig.add_trace(go.Scatter(
            x=[r["trade0"], r["tradeN"]],
            y=[r["close0"], r["closeN"]],
            mode="markers+lines",
            marker=dict(color=COLORS_MOVE.get(r["move"], "gray"), size=6),
            line=dict(dash='dash', color=COLORS_MOVE.get(r["move"], "gray")),
            hovertemplate="Move: " + r['move'] + "<br>Return +5d: " + f"{r['retN_pct']:+.2f}%" +
                          "<br>Date: %{x}<br>Close: %{y}<extra></extra>",
            showlegend=False
        ), row=1, col=1)

        # вертикальная линия на дате решения
        fig.add_trace(go.Scatter(
            x=[r["trade0"], r["trade0"]],
            y=[prices["Close"].min(), prices["Close"].max()],
            mode="lines",
            line=dict(color=COLORS_MOVE.get(r["move"], "gray"), dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

    # FFR с цветовой кодировкой move
    for move_type, color in COLORS_MOVE.items():
        sub_df = decisions[decisions["move"] == move_type]
        if not sub_df.empty:
            fig.add_trace(go.Scatter(
                x=sub_df["decision_date"], y=sub_df["ffr_mid"],
                mode="lines+markers",
                marker=dict(size=8, color=color),
                line=dict(color=color),
                name=move_type
            ), row=2, col=1)

    fig.update_layout(
        height=700,
        width=1200,
        title_text="S&P 500 и решения ФРС (FRED) с цветовой кодировкой hike/cut/unch и +5d подсказками",
        hovermode="x unified"
    )

    fig.write_html(filename)
    print(f"Интерактивный график сохранён: {filename}")

# -------------------------
# Main
# -------------------------
def main():
    print("1) Получаем решения ФРС (FRED)...")
    decisions = fetch_fomc_decisions()
    print(f"  Найдено записей: {len(decisions)}")

    print("2) Загружаем котировки S&P 500 (FRED)...")
    prices = download_spx()
    print(f"  Загружено {len(prices)} торговых строк")

    print("3) Сопоставляем решения с торговыми днями и считаем +5d...")
    mapped = map_decisions_to_trading_days(decisions, prices)
    print(f"  Записей с +5d: {len(mapped)}")

    print("4) Строим финальный интерактивный график...")
    plot_interactive_colored(prices, decisions, mapped)

if __name__ == "__main__":
    main()
