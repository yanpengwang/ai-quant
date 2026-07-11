#!/usr/bin/env python3
"""
TASK4 — 海龟交易策略（Turtle Trading Strategy）
================================================
完整实现：Donchian 通道 + ATR + Unit 仓位管理 + 金字塔加仓 + 回测
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os
from pathlib import Path

# ── 中文字体设置 ─────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent  # AI-Quant/
DATA_DIR = BASE_DIR / "TASK3" / "data" / "standard"
OUT_DIR = Path(__file__).parent  # TASK4/
CHART_DIR = OUT_DIR / "TASK4_report_charts"

STOCKS = {
    "002594.SZ": {"name": "比亚迪 (A股)", "market": "CN", "currency": "CNY"},
    "600900.SH": {"name": "长江电力 (A股)", "market": "CN", "currency": "CNY"},
    "688981.SH": {"name": "中芯国际 (A股)", "market": "CN", "currency": "CNY"},
    "688099.SH": {"name": "晶晨股份 (A股)", "market": "CN", "currency": "CNY"},
    "00981.HK":  {"name": "中芯国际 (H股)", "market": "HK", "currency": "HKD"},
    "01211.HK":  {"name": "比亚迪 (H股)", "market": "HK", "currency": "HKD"},
}

def load_data(symbol):
    """从 TASK3 加载前复权数据"""
    path = DATA_DIR / symbol / "1d.parquet"
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df

def calc_donchian(df, channel_period=20, exit_period=10):
    """计算 Donchian 通道"""
    df = df.copy()
    df["upper_channel"] = df["high"].rolling(channel_period).max()
    df["lower_channel"] = df["low"].rolling(channel_period).min()
    df["mid_channel"] = (df["upper_channel"] + df["lower_channel"]) / 2
    df["exit_lower"] = df["low"].rolling(exit_period).min()
    return df

def calc_atr(df, period=14):
    """计算 ATR"""
    df = df.copy()
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = abs(df["high"] - df["prev_close"])
    df["tr3"] = abs(df["low"] - df["prev_close"])
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr"] = df["tr"].rolling(period).mean()
    return df

def calc_unit(df, account_value=100000, risk_pct=0.01):
    """计算 Unit（仓位单位）"""
    df = df.copy()
    risk_amount = account_value * risk_pct
    df["unit"] = np.floor(risk_amount / df["atr"]).fillna(0).astype(int)
    df["unit"] = df["unit"].clip(lower=0)
    return df

def generate_signals(df, channel_period=20, exit_period=10, atr_period=14,
                     account_value=100000, risk_pct=0.01, enable_pyramiding=True,
                     max_units=4, stop_multiplier=2.0):
    """
    生成海龟策略信号 + 回测

    Returns:
        df: DataFrame with signals, positions, and metrics
        trades: list of trade records
    """
    df = calc_donchian(df, channel_period, exit_period)
    df = calc_atr(df, atr_period)
    df = calc_unit(df, account_value, risk_pct)

    # 初始化状态
    position = 0       # 当前 Unit 数量
    entry_price = 0    # 首次入场价
    last_add_price = 0 # 最近一次加仓价
    units_held = 0     # 持有 Unit 数
    cash = account_value
    shares = 0
    portfolio_values = []
    trades = []

    df["signal"] = 0         # 1=买入, -1=卖出, 0=持有
    df["position"] = 0
    df["entry_price"] = np.nan
    df["exit_reason"] = ""
    df["units_held"] = 0
    df["stop_price"] = np.nan

    warmup = max(channel_period, atr_period)

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        price = row["close"]
        high = row["high"]
        low = row["low"]

        upper_prev = df.iloc[i-1]["upper_channel"]
        exit_lower_prev = df.iloc[i-1]["exit_lower"]
        atr_val = row["atr"]
        unit_size = row["unit"]

        if pd.isna(upper_prev) or pd.isna(atr_val) or unit_size <= 0:
            portfolio_values.append(cash)
            continue

        # ── 已持仓状态 ──
        if position > 0 and units_held > 0:
            # 检查 ATR 止损
            stop_price = entry_price - stop_multiplier * atr_val
            df.at[df.index[i], "stop_price"] = stop_price

            exit_triggered = False
            exit_reason = ""

            # 规则五：止损立即执行（盘中触发立即平仓）
            if row["low"] <= stop_price:
                exit_price = min(row["open"], stop_price)
                exit_triggered = True
                exit_reason = "stop"
            # 通道退出（收盘价跌破 exit_lower）
            elif price < exit_lower_prev:
                exit_price = price
                exit_triggered = True
                exit_reason = "channel"

            # 金字塔加仓
            if enable_pyramiding and not exit_triggered and units_held < max_units:
                target_price = entry_price + 0.5 * atr_val * units_held
                if price >= target_price:
                    add_shares = int(account_value * risk_pct / atr_val)
                    if add_shares > 0:
                        cost = add_shares * price
                        if cash >= cost:
                            shares += add_shares
                            cash -= cost
                            units_held += 1
                            last_add_price = price
                            df.at[df.index[i], "signal"] = 1
                            trades.append({
                                "date": row["trade_date"], "action": "add",
                                "price": price, "shares": add_shares,
                                "units": units_held, "reason": "pyramid"
                            })

            if exit_triggered:
                cash += shares * exit_price
                trades.append({
                    "date": row["trade_date"], "action": "exit",
                    "price": exit_price, "shares": shares,
                    "units": units_held, "reason": exit_reason,
                    "entry_price": entry_price, "pnl_pct": (exit_price / entry_price - 1) * 100
                })
                df.at[df.index[i], "signal"] = -1
                df.at[df.index[i], "exit_reason"] = exit_reason
                shares = 0
                position = 0
                units_held = 0
                entry_price = 0
                last_add_price = 0

        # ── 空仓状态 ──
        elif position == 0:
            # 入场信号：价格突破上轨
            if price > upper_prev:
                # 计算 Unit
                unit_size_now = int(account_value * risk_pct / atr_val) if atr_val > 0 else 0
                if unit_size_now > 0:
                    buy_shares = unit_size_now
                    cost = buy_shares * price
                    if cash >= cost:
                        shares = buy_shares
                        cash -= cost
                        position = 1
                        units_held = 1
                        entry_price = price
                        last_add_price = price
                        df.at[df.index[i], "signal"] = 1
                        df.at[df.index[i], "entry_price"] = price
                        df.at[df.index[i], "stop_price"] = price - stop_multiplier * atr_val
                        trades.append({
                            "date": row["trade_date"], "action": "entry",
                            "price": price, "shares": buy_shares,
                            "units": 1, "reason": "breakout"
                        })

        df.at[df.index[i], "position"] = position
        df.at[df.index[i], "units_held"] = units_held
        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)

    # 在 warmup 之前的行填充 NaN
    padded = [np.nan] * warmup + portfolio_values[len(portfolio_values) - (len(df) - warmup):]
    df["portfolio_value"] = padded
    df["portfolio_value"] = df["portfolio_value"].fillna(account_value)

    return df, trades

def calc_metrics(df, trades, account_value=100000, risk_free=0.02):
    """计算回测指标"""
    df = df.dropna(subset=["portfolio_value"])

    if len(df) == 0:
        return {}

    # 累计回报
    final_value = df["portfolio_value"].iloc[-1]
    cumulative_return = (final_value - account_value) / account_value * 100

    # 年化收益率
    days = len(df)
    annualized_return = ((1 + cumulative_return/100) ** (252/days) - 1) * 100

    # 最大回撤
    peak = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - peak) / peak * 100
    max_drawdown = drawdown.min()

    # 夏普比率
    daily_returns = df["portfolio_value"].pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() * 252 - risk_free) / (daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0

    # 买入持有基准
    buy_hold_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    # 交易统计
    exit_trades = [t for t in trades if t["action"] == "exit"]
    wins = [t for t in exit_trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in exit_trades if t.get("pnl_pct", 0) <= 0]

    win_rate = len(wins) / len(exit_trades) * 100 if exit_trades else 0
    total_profit = sum(t.get("pnl_pct", 0) for t in wins)
    total_loss = abs(sum(t.get("pnl_pct", 0) for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf") if total_profit > 0 else 0

    avg_hold_days = 0
    if exit_trades:
        entry_actions = [t for t in trades if t["action"] == "entry"]
        exit_dates = [t["date"] for t in exit_trades]
        entry_dates = [t["date"] for t in entry_actions]
        if len(entry_dates) == len(exit_dates):
            avg_hold_days = np.mean([
                (e - s).days for s, e in zip(entry_dates, exit_dates)
            ])

    avg_units = np.mean([t.get("units", 0) for t in trades]) if trades else 0

    return {
        "累计回报率 (%)": round(cumulative_return, 2),
        "年化收益率 (%)": round(annualized_return, 2),
        "最大回撤 (%)": round(max_drawdown, 2),
        "夏普比率": round(sharpe, 4),
        "胜率 (%)": round(win_rate, 2),
        "盈亏比": round(profit_factor, 4),
        "总交易次数": len(exit_trades),
        "平均持仓天数": round(avg_hold_days, 1),
        "平均 Unit 数": round(avg_units, 2),
        "买入持有收益 (%)": round(buy_hold_return, 2),
    }


def plot_charts(df, symbol, info, chart_dir, channel_period=20, exit_period=10):
    """生成三张策略图表"""
    name = info["name"]
    safe_name = symbol.replace(".", "_")

    # ── 图1: 股价与 Donchian 通道 ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"图1 {symbol} {name} 海龟策略信号\n"
                 f"(通道周期={channel_period}天, 退出周期={exit_period}天)",
                 fontsize=14, fontweight="bold")

    # 只取最近 3 年数据绘图
    df_plot = df.dropna(subset=["upper_channel"]).copy()
    if len(df_plot) > 756:
        df_plot = df_plot.iloc[-756:]

    dates = df_plot["trade_date"]

    ax1.plot(dates, df_plot["close"], color="black", linewidth=0.8, label="收盘价", zorder=1)
    ax1.plot(dates, df_plot["upper_channel"], color="red", linestyle="--", linewidth=0.8,
             label=f"上轨 ({channel_period}日)", alpha=0.7)
    ax1.plot(dates, df_plot["lower_channel"], color="green", linestyle="--", linewidth=0.8,
             label=f"下轨 ({channel_period}日)", alpha=0.7)
    ax1.plot(dates, df_plot["mid_channel"], color="gray", linestyle=":", linewidth=0.5,
             label="中轨", alpha=0.5)

    # 买入信号
    buys = df_plot[df_plot["signal"] == 1]
    if len(buys) > 0:
        ax1.scatter(buys["trade_date"], buys["close"], marker="^", color="red",
                    s=80, zorder=5, label=f"买入 ({len(buys)})")

    # 卖出信号
    sells = df_plot[df_plot["signal"] == -1]
    if len(sells) > 0:
        stops = sells[sells["exit_reason"] == "stop"]
        channels = sells[sells["exit_reason"] == "channel"]
        if len(channels) > 0:
            ax1.scatter(channels["trade_date"], channels["close"], marker="v",
                        color="green", s=80, zorder=5,
                        label=f"通道退出 ({len(channels)})")
        if len(stops) > 0:
            ax1.scatter(stops["trade_date"], stops["close"], marker="D",
                        color="blue", s=80, zorder=5,
                        label=f"止损 ({len(stops)})")

    ax1.set_ylabel("价格")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # ATR 副图
    ax2.fill_between(dates, 0, df_plot["atr"], color="blue", alpha=0.3, label="ATR")
    ax2.plot(dates, df_plot["atr"], color="blue", linewidth=0.8)
    ax2.set_ylabel("ATR")
    ax2.set_xlabel("日期")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = chart_dir / f"chart1_strategy_{safe_name}.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()

    # ── 图2: 净值曲线 ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"图2 {symbol} {name} 策略净值 vs 买入持有", fontsize=14, fontweight="bold")

    df_valid = df.dropna(subset=["portfolio_value"])
    dates = df_valid["trade_date"]

    nav = df_valid["portfolio_value"] / 100000
    bh = df_valid["close"] / df_valid["close"].iloc[0]

    ax1.plot(dates, nav, color="red", linewidth=1.2, label="海龟策略")
    ax1.plot(dates, bh, color="gray", linestyle="--", linewidth=0.8, label="买入持有")
    ax1.axhline(y=1, color="black", linestyle=":", linewidth=0.5)
    ax1.set_ylabel("净值 (初始=1)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 回撤
    peak = df_valid["portfolio_value"].cummax()
    dd = (df_valid["portfolio_value"] - peak) / peak * 100
    ax2.fill_between(dates, dd, 0, color="red", alpha=0.3, label="回撤")
    ax2.plot(dates, dd, color="red", linewidth=0.5)
    ax2.set_ylabel("回撤 (%)")
    ax2.set_xlabel("日期")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = chart_dir / f"chart2_nav_{safe_name}.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()

    return path1, path2


def run_experiments():
    """多场景对比实验：6 股票 × 3 通道周期"""
    combinations = [
        {"channel": 10, "exit": 10, "label": "短周期 (10,10)"},
        {"channel": 20, "exit": 10, "label": "系统一 (20,10)"},
        {"channel": 55, "exit": 20, "label": "系统二 (55,20)"},
    ]

    all_results = []

    for symbol, info in STOCKS.items():
        print(f"\n{'='*60}")
        print(f"回测: {symbol} {info['name']}")
        print(f"{'='*60}")

        df = load_data(symbol)

        for combo in combinations:
            channel = combo["channel"]
            exit_p = combo["exit"]
            label = combo["label"]

            try:
                df_sig, trades = generate_signals(
                    df, channel_period=channel, exit_period=exit_p,
                    atr_period=14, account_value=100000, risk_pct=0.01,
                    enable_pyramiding=True, max_units=4
                )
                metrics = calc_metrics(df_sig, trades)

                row = {
                    "股票": f"{symbol} {info['name']}",
                    "symbol": symbol,
                    "通道周期": channel,
                    "退出周期": exit_p,
                    "策略标签": label,
                    **metrics
                }
                all_results.append(row)

                print(f"  {label:20s} | 累计回报: {metrics['累计回报率 (%)']:8.2f}% | "
                      f"夏普: {metrics['夏普比率']:6.2f} | "
                      f"MDD: {metrics['最大回撤 (%)']:6.2f}% | "
                      f"胜率: {metrics['胜率 (%)']:5.1f}% | "
                      f"交易: {metrics['总交易次数']}次")

                # 只对默认参数(20,10)生成图表
                if channel == 20 and exit_p == 10:
                    plot_charts(df_sig, symbol, info, CHART_DIR, channel, exit_p)
            except Exception as e:
                print(f"  {label:20s} | ERROR: {e}")

    # 保存对比矩阵
    matrix_df = pd.DataFrame(all_results)
    matrix_path = OUT_DIR / "backtest_matrix.csv"
    matrix_df.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    print(f"\n对比矩阵已保存: {matrix_path}")

    # 汇总 JSON
    summary = {}
    for symbol in STOCKS:
        rows = [r for r in all_results if r["symbol"] == symbol]
        if rows:
            best = max(rows, key=lambda r: r["累计回报率 (%)"])
            summary[symbol] = {
                "name": STOCKS[symbol]["name"],
                "best_strategy": best["策略标签"],
                "best_return": best["累计回报率 (%)"],
                "best_sharpe": best["夏普比率"],
                "best_mdd": best["最大回撤 (%)"],
                "all_results": rows
            }

    with open(OUT_DIR / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"汇总 JSON 已保存: {OUT_DIR / 'backtest_summary.json'}")

    return matrix_df, summary


# ── 入口 ──
if __name__ == "__main__":
    os.makedirs(CHART_DIR, exist_ok=True)
    matrix_df, summary = run_experiments()
    print("\n✅ TASK4 海龟策略回测完成！")
