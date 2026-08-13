# -*- coding: utf-8 -*-
"""
带约束的组合优化与回测：为什么"理论最优"在现实里会翻车
========================================================

对应公众号文章《资产组合 · 带约束的组合优化与回测》（量海泛舟）
方法论参考：https://www.tidy-finance.org/chapters/constrained-optimization-and-backtesting.html

内容：
  1. 用 scipy SLSQP 求「禁卖空 + 仓位上限」约束下的最小方差与最大夏普组合，
     与无约束解析式对照，展示约束如何把 -94.9% 的做空仓位压回 0；
  2. 做滚动窗口样本外回测：最小方差 / 最大夏普 / 1/N 等权三个策略，
     用 252 日滚动窗口估计、每 21 个交易日调仓一次；
  3. 计入线性交易成本（换手率 × 成本费率），比较毛收益与净收益；
  4. 输出三个策略的年化收益、波动、夏普、平均换手率。

核心结论：均值-方差组合换手率高、交易成本侵蚀收益，朴素 1/N 等权组合
反而很难被超越（DeMiguel et al. 2009 的经典结论在 A 股同样成立）。

运行方式
--------
1. 安装依赖：  pip install jh-quant matplotlib scipy
2. 设置环境变量（从 https://jiuhuang.xyz 申请 API Key）：
       export JIUHUANG_API_KEY=你的key      # Windows: set JIUHUANG_API_KEY=你的key
   （或在项目根目录放 .env 文件，JHData 会自动读取）
3. 运行：      python constrained_portfolio_backtest.py

输出
----
控制台打印约束前后权重对照、三个策略的绩效与换手率；并生成 3 张图到 output/：
    - fig1_constrained_weights.png   约束前（无约束切线）vs 约束后（禁卖空+上限）权重
    - fig2_backtest_cumulative.png   三个策略累计净收益曲线
    - fig3_turnover.png              三个策略的逐期换手率
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 统一图表风格（house style，保证公众号文章图表一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from jh_quant.data import JHData, DataTypes  # noqa: E402

# ============================================================
# 全局参数
# ============================================================
START, END = "2020-01-01", "2026-06-30"
TRADING_DAYS = 252
WARMUP = 252          # 滚动估计窗口：252 个交易日（约 1 年）
REBAL_EVERY = 21      # 每 21 个交易日（约 1 个月）调仓一次
MAX_WEIGHT = 0.40     # 单只股票权重上限
COST_RATE = 0.002     # 单边交易成本费率：换手 1%（=100%）约千分之二

# 27 只跨行业大盘蓝筹（2020–2026 数据完整）。
# 注：长江电力/中国平安/中国神华/恒瑞医药/隆基绿能/紫金矿业/上汽集团/万华化学
#     在 QFQ 源里历史数据缺失（仅 2023-12 起），故未纳入。
STOCKS = {
    "600036.SH": "招商银行", "600519.SH": "贵州茅台", "300750.SZ": "宁德时代",
    "000002.SZ": "万科A",   "601985.SH": "中国核电", "000333.SZ": "美的集团",
    "002594.SZ": "比亚迪",  "002415.SZ": "海康威视", "600028.SH": "中国石化",
    "600030.SH": "中信证券", "000858.SZ": "五粮液",   "000568.SZ": "泸州老窖",
    "600887.SH": "伊利股份", "600031.SH": "三一重工", "000725.SZ": "京东方A",
    "002475.SZ": "立讯精密", "603259.SH": "药明康德", "300015.SZ": "爱尔眼科",
    "601668.SH": "中国建筑", "601888.SH": "中国中免", "600809.SH": "山西汾酒",
    "300059.SZ": "东方财富", "600438.SH": "通威股份", "000063.SZ": "中兴通讯",
    "600690.SH": "海尔智家", "601766.SH": "中国中车", "600019.SH": "宝钢股份",
}

os.makedirs("output", exist_ok=True)


# ============================================================
# 1. 取数与收益
# ============================================================
def fetch_and_build():
    jh = JHData()
    prices = jh.get_data(
        DataTypes.TS_DAILY_QFQ,
        ts_code=",".join(STOCKS), start=START, end=END,
    ).to_df()
    prices = prices[["ts_code", "trade_date", "close"]].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["ret"] = prices.groupby("ts_code")["close"].pct_change()
    rets = prices.pivot(index="trade_date", columns="ts_code", values="ret").dropna()

    shibor = jh.get_data(
        DataTypes.TS_SHIBOR, start=START, end=END, bypass_cache=True
    ).to_df()
    shibor = shibor[["date", "on"]].rename(columns={"on": "rf_pct"})
    shibor["rf_pct"] = pd.to_numeric(shibor["rf_pct"], errors="coerce")
    rf = float(shibor["rf_pct"].mean() / 100 / 360 * TRADING_DAYS)
    return rets, rf


# ============================================================
# 2. 带约束的组合优化（scipy SLSQP）
# ============================================================
def min_variance(cov):
    """最小方差组合：min w'Σw，s.t. Σw=1，0 <= w_i <= MAX_WEIGHT。"""
    n = len(cov)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, MAX_WEIGHT)] * n
    res = minimize(
        lambda w: w @ cov.values @ w,
        x0=np.ones(n) / n, bounds=bounds, constraints=cons, method="SLSQP",
    )
    return res.x


def max_sharpe(mu, cov, rf):
    """最大夏普组合：max (w'μ - rf)/sqrt(w'Σw)，s.t. Σw=1，0 <= w_i <= MAX_WEIGHT。"""
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, MAX_WEIGHT)] * n

    def neg_sharpe(w):
        ret = float(w @ mu.values)
        vol = float(np.sqrt(w @ cov.values @ w))
        return -(ret - rf) / vol

    res = minimize(
        neg_sharpe, x0=np.ones(n) / n, bounds=bounds, constraints=cons, method="SLSQP",
    )
    return res.x


def unconstrained_tangency(mu, cov, rf):
    """无约束切线组合（对照用）：w = Σ⁻¹(μ - rf·1) / [1ᵀΣ⁻¹(μ - rf·1)]。"""
    inv = np.linalg.inv(cov.values)
    ones = np.ones(len(mu))
    excess = mu.values - rf * ones
    return inv @ excess / (ones @ inv @ excess)


# ============================================================
# 3. 滚动回测
# ============================================================
def backtest(rets, rf, strategy, cost_rate=COST_RATE):
    """滚动窗口回测：返回每日组合收益 Series 与平均换手率。

    strategy: 'mvp' | 'maxsharpe' | 'equal'
    cost_rate: 交易成本费率，传 0 则得到毛收益（不含成本）。
    """
    dates = rets.index
    n = rets.shape[1]
    daily_ret = pd.Series(index=dates, dtype=float)
    turnovers = []  # 每次调仓的换手率（总变动的一半）
    w_held = None   # 期初实际持有的权重（随价格漂移）

    for i in range(len(dates)):
        r_today = rets.iloc[i].values  # 今日各股收益

        # 到调仓日且历史足够，重新计算目标权重
        if i >= WARMUP and (i - WARMUP) % REBAL_EVERY == 0:
            window = rets.iloc[i - WARMUP:i]
            mu_est = window.mean() * TRADING_DAYS
            cov_est = window.cov() * TRADING_DAYS

            if strategy == "equal":
                w_target = np.ones(n) / n
            elif strategy == "mvp":
                w_target = min_variance(cov_est)
            else:  # maxsharpe
                w_target = max_sharpe(mu_est, cov_est, rf)

            if w_held is not None:
                # 换手率 = 目标权重与当前漂移后权重的总绝对差的一半（一买一卖各算一次）
                turnover = float(np.abs(w_target - w_held).sum() / 2)
                turnovers.append(turnover)
            w_held = w_target.copy()

        # 当日组合收益（用期初权重）
        if w_held is None:
            daily_ret.iloc[i] = 0.0
        else:
            port_ret = float(w_held @ r_today)
            # 扣除当日发生的交易成本（仅调仓日有换手）
            if i >= WARMUP and (i - WARMUP) % REBAL_EVERY == 0 and len(turnovers) > 0:
                port_ret -= cost_rate * turnovers[-1]
            daily_ret.iloc[i] = port_ret

        # 权重随价格漂移（为下一期初准备）
        if w_held is not None:
            w_held = w_held * (1 + r_today)
            w_held = w_held / w_held.sum()

    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    return daily_ret, avg_turnover


def performance(daily_ret, rf):
    """由每日收益序列计算年化收益、年化波动、夏普。"""
    ret = daily_ret.mean() * TRADING_DAYS
    vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


# ============================================================
# 4. 绘图
# ============================================================
def fig1_constrained_weights(mu, cov, rf):
    """约束前后权重对照：无约束切线组合（含大量做空）vs 禁卖空+上限后的权重。"""
    w_uncon = unconstrained_tangency(mu, cov, rf)
    w_ms = max_sharpe(mu, cov, rf)

    codes = list(cov.columns)
    names = [STOCKS[c] for c in codes]
    # 按无约束权重排序，做空放底部
    order = np.argsort(w_uncon)
    codes_s = [codes[i] for i in order]
    names_s = [names[i] for i in order]
    w_uncon_s = w_uncon[order] * 100
    w_ms_s = w_ms[order] * 100

    y = np.arange(len(codes_s))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

    colors = [mpl_style.FALL if w < 0 else mpl_style.RISE for w in w_uncon_s]
    ax1.barh(y, w_uncon_s, color=colors)
    ax1.axvline(0, color="#666666", lw=0.8)
    ax1.set_title("无约束最大夏普组合")
    ax1.set_xlabel("权重（%）")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names_s, fontsize=9)
    ax1.set_xlim(-100, max(w_uncon_s.max(), 60) * 1.1)

    ax2.barh(y, w_ms_s, color=mpl_style.ACCENT)
    ax2.axvline(0, color="#666666", lw=0.8)
    ax2.axvline(MAX_WEIGHT * 100, color=mpl_style.ACCENT_2, ls="--", lw=1.2,
                label=f"权重上限 {MAX_WEIGHT * 100:.0f}%")
    ax2.set_title("禁卖空 + 权重上限后")
    ax2.set_xlabel("权重（%）")
    ax2.set_xlim(-5, 45)
    ax2.legend(loc="lower right")

    for ax in (ax1, ax2):
        mpl_style.hide_spines(ax)
    fig.suptitle("约束把极端做空仓位压回现实（2020–2026，全样本）")
    fig.tight_layout()
    fig.savefig("output/fig1_constrained_weights.png")
    print("    已保存 fig1_constrained_weights.png")
    return fig


def fig2_backtest_cumulative(results, rf):
    """三个策略累计净收益曲线。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = {"mvp": "最小方差", "maxsharpe": "最大夏普", "equal": "1/N 等权"}
    colors = {"mvp": mpl_style.ACCENT, "maxsharpe": mpl_style.RISE, "equal": mpl_style.FALL}
    for key, daily in results.items():
        cum = (1 + daily).cumprod()
        ax.plot(cum.index, cum.values, lw=1.8, color=colors[key], label=labels[key])
    mpl_style.hide_spines(ax)
    ax.set_title("滚动回测：三个策略的累计净收益（已扣交易成本）")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值（初始 = 1）")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("output/fig2_backtest_cumulative.png")
    print("    已保存 fig2_backtest_cumulative.png")
    return fig


def fig3_turnover(results):
    """三个策略逐期换手率（箱线或柱状）。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    # 重新算逐期换手，复用 backtest 不便，这里用累计统计替代为均值柱状
    labels = ["最小方差", "最大夏普", "1/N 等权"]
    # 直接传均值，画柱状
    turns = [results[k] for k in ["mvp", "maxsharpe", "equal"]]
    ax.bar(labels, turns, color=[mpl_style.ACCENT, mpl_style.RISE, mpl_style.FALL])
    mpl_style.hide_spines(ax)
    ax.set_title("平均换手率：均值-方差组合交易更频繁")
    ax.set_ylabel("平均换手率")
    for i, v in enumerate(turns):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig("output/fig3_turnover.png")
    print("    已保存 fig3_turnover.png")
    return fig


# ============================================================
# 5. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("带约束的组合优化与回测")
    print("=" * 60)

    print("\n[1] 取数并构建日收益")
    rets, rf = fetch_and_build()
    print(f"    交易日数: {len(rets)}, 年化无风险利率: {rf * 100:.2f}%")

    print("\n[2] 约束前后权重对照（全样本）")
    mu = rets.mean() * TRADING_DAYS
    cov = rets.cov() * TRADING_DAYS
    w_uncon = unconstrained_tangency(mu, cov, rf)
    w_mvp = min_variance(cov)
    w_ms = max_sharpe(mu, cov, rf)
    print("    股票       无约束夏普   约束后MVP   约束后夏普")
    for i, code in enumerate(cov.columns):
        print(f"    {STOCKS[code]:　<6} {w_uncon[i] * 100:8.2f}%  {w_mvp[i] * 100:7.2f}%  {w_ms[i] * 100:7.2f}%")

    print("\n[3] 滚动回测（252 日窗口，21 日调仓，交易成本 0.2%）")
    results, turnovers = {}, {}
    for key in ["mvp", "maxsharpe", "equal"]:
        daily, t = backtest(rets, rf, key)
        results[key] = daily
        turnovers[key] = t
        ret, vol, sharpe = performance(daily, rf)
        # 毛收益（不含交易成本）
        daily_gross, _ = backtest(rets, rf, key, cost_rate=0.0)
        ret_gross, _, _ = performance(daily_gross, rf)
        print(f"    {key:　<10} 毛收益 {ret_gross * 100:6.2f}%  净收益 {ret * 100:6.2f}%  "
              f"波动 {vol * 100:6.2f}%  夏普 {sharpe:5.2f}  平均换手 {t * 100:5.2f}%")

    print("\n[4] 绘图")
    fig1 = fig1_constrained_weights(mu, cov, rf)
    fig2 = fig2_backtest_cumulative(results, rf)
    fig3 = fig3_turnover(turnovers)
    plt.close("all")
    print("\n全部完成。")


if __name__ == "__main__":
    main()
