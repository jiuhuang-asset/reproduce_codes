# -*- coding: utf-8 -*-
"""
Beta 估计（A股）：从三只股票到全市场截面
==========================================

对应公众号文章《资产定价入门 · Beta 是什么，用 A 股数据算给你看》（量海泛舟）
方法论参考：https://www.tidy-finance.org/chapters/beta-estimation.html

内容分两部分：
  Part 1 — 三只代表性 A 股（日频）：招商银行 / 贵州茅台 / 宁德时代
          用市场模型 R_i - R_f = alpha + beta*(R_m - R_f) + e 做 OLS，
          输出 beta / alpha / R^2，并画散点回归、beta 对比、滚动 beta 三张图。
  Part 2 — 全 A 股（月频）：对每一只股票跑同样的市场模型回归，
          得到全市场 Beta 分布、行业 Beta 排序、Beta 与平均收益的关系三张图。

市场基准 = 沪深300（000300.SH）。
无风险利率：日频用 SHIBOR 隔夜（on），月频用 SHIBOR 1 个月期（1m）。

运行方式
--------
1. 安装依赖：  pip install jh-quant matplotlib
2. 设置环境变量（从 https://jiuhuang.xyz 申请 API Key）：
       export JIUHUANG_API_KEY=你的key      # Windows: set JIUHUANG_API_KEY=你的key
   （或在项目根目录放 .env 文件，JHData 会自动读取）
3. 运行：      python beta_estimation_a_share.py

输出
----
控制台打印三只股票的 beta / alpha / R^2，以及全市场 Beta 描述统计与行业 Beta 表，
并生成 6 张图到 output/ 子目录：
  Part 1:
    - fig1_scatter_regression.png   三只股票日收益散点 + 回归线
    - fig2_beta_compare.png         三只股票 beta 对比柱状图
    - fig3_rolling_beta.png         60 交易日滚动 beta 时序
  Part 2:
    - fig1_beta_distribution.png    全市场 Beta 分布直方图
    - fig2_beta_by_industry.png     行业 Beta 中位数（前 12 + 后 8）
    - fig3_beta_vs_return.png       Beta 与平均月收益（散点 + 分桶均值）
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style，保证公众号文章图表一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================
# Part 1：三只代表性股票（日频）
# ============================================================

# 数据期间：2020 至 2026 年中（约 6.5 年，约 1600 个交易日）
STOCK_START, STOCK_END = "2020-01-01", "2026-06-30"

# 三只代表性股票：代码 -> 简称
STOCKS = {
    "600036.SH": "招商银行",
    "600519.SH": "贵州茅台",
    "300750.SZ": "宁德时代",
}


def fetch_stock_data():
    """用 jh_quant 拉取：个股前复权日线 + 沪深300 + SHIBOR 无风险利率。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()  # 自动从环境变量 JIUHUANG_API_KEY 读取

    # 1) 三只股票前复权日线（close 已复权，可直接算收益）
    stock_prices = jh.get_data(
        DataTypes.TS_DAILY_QFQ,
        ts_code=",".join(STOCKS.keys()),
        start=STOCK_START,
        end=STOCK_END,
    ).to_df()
    stock_prices = stock_prices[["ts_code", "trade_date", "close"]]

    # 2) 沪深300 指数日线（TS 源，沪深300 的 ts_code 是 000300.SH）
    market = jh.get_data(
        DataTypes.TS_INDEX_DAILY,
        ts_code="000300.SH",
        start=STOCK_START,
        end=STOCK_END,
    ).to_df()
    market = market[["trade_date", "close"]].rename(
        columns={"close": "mkt_close"}
    )

    # 3) SHIBOR 隔夜利率（无风险利率，TS_SHIBOR 用 on 列，日化 /360）
    #    数据量小，bypass_cache 直接拉取：TS_SHIBOR 服务端 DDL 列名尚未同步为 1m/1w
    shibor = jh.get_data(
        DataTypes.TS_SHIBOR, start=STOCK_START, end=STOCK_END, bypass_cache=True
    ).to_df()
    shibor = shibor[["date", "on"]].rename(columns={"on": "rf_pct"})
    shibor["rf_pct"] = pd.to_numeric(shibor["rf_pct"], errors="coerce")  # 远程返回字符串，转数值

    return stock_prices, market, shibor


def prepare_stock_returns(stock_prices, market, shibor):
    """算收益 + 超额收益（减无风险利率），按交易日对齐。"""
    # 个股日收益：前复权 close 的百分比变化
    stock_prices = stock_prices.sort_values(["ts_code", "trade_date"])
    stock_prices["ret"] = (
        stock_prices.groupby("ts_code")["close"].pct_change() * 100  # 换算成百分数
    )
    # 沪深300 日收益
    market = market.sort_values("trade_date")
    market["mkt_ret"] = market["mkt_close"].pct_change() * 100

    # SHIBOR 按交易日对齐（隔夜利率年化百分比 -> 日度小数，再对齐后向前填充）
    shibor["date"] = pd.to_datetime(shibor["date"])
    shibor = shibor.dropna(subset=["rf_pct"]).set_index("date").sort_index()
    rf_daily = shibor["rf_pct"] / 100 / 360  # 年化隔夜利率 -> 日收益

    out = []
    for code, name in STOCKS.items():
        s = stock_prices[stock_prices["ts_code"] == code].copy()
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        s = s.set_index("trade_date")
        m = market.set_index(pd.to_datetime(market["trade_date"]))[["mkt_ret"]]
        df = s[["ret"]].join(m, how="inner")  # 只保留共同交易日
        df["rf"] = rf_daily.reindex(df.index).ffill().fillna(0.0)
        df["stock_excess"] = df["ret"] - df["rf"]   # R_i - R_f
        df["mkt_excess"] = df["mkt_ret"] - df["rf"]  # R_m - R_f
        df["symbol"] = code
        df["name"] = name
        out.append(df.reset_index())
    return pd.concat(out, ignore_index=True)


def ols_beta(y, x):
    """手写最小二乘：回归 y = alpha + beta * x。返回 (beta, alpha, r2, t_beta, n)。"""
    X = np.column_stack([np.ones_like(x), x])
    beta_vec, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = beta_vec[0], beta_vec[1]
    resid = y - X @ beta_vec
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = len(y)
    # beta 的标准误与 t 统计量
    dof = max(n - 2, 1)
    sigma2 = ss_res / dof
    sxx = float(((x - x.mean()) ** 2).sum())
    se_beta = np.sqrt(sigma2 / sxx) if sxx > 0 else np.nan
    t_beta = beta / se_beta if se_beta else np.nan
    return beta, alpha, r2, t_beta, n


def rolling_beta(df, window=60):
    """60 个交易日滚动窗口的 beta 时序（收盘对齐，作为 beta 随时间变化的直观演示）。"""
    x = df["mkt_excess"].values
    y = df["stock_excess"].values
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        out[i - 1] = ols_beta(y[i - window:i], x[i - window:i])[0]
    return out


def part1_single_stock():
    """三只代表性股票的 Beta 估计 + 3 张图。"""
    print(">>> [Part 1] 拉取三只股票数据（日频）...")
    stock_prices, market, shibor = fetch_stock_data()
    print(f"    个股行数: {len(stock_prices)}, 指数行数: {len(market)}, SHIBOR 行数: {len(shibor)}")

    df = prepare_stock_returns(stock_prices, market, shibor)
    print(f"    合并后行数: {len(df)}（三只股票 × {df['trade_date'].nunique()} 个交易日）")

    print("    全样本 OLS 回归（手写最小二乘）...")
    results = []
    for code, name in STOCKS.items():
        sub = df[df["symbol"] == code].dropna()
        beta, alpha, r2, t_beta, n = ols_beta(sub["stock_excess"], sub["mkt_excess"])
        results.append(
            {"name": name, "code": code, "beta": beta, "alpha": alpha,
             "r2": r2, "t_beta": t_beta, "n": n}
        )
    res = pd.DataFrame(results).set_index("name")
    print(res.round(3).to_string())

    # 对照：jh_quant 内置的 calculate_exposures（同一份数据，应得到几乎一样的 beta）
    print("    对照：jh_quant.factors.calculate_exposures ...")
    try:
        from jh_quant.factors import calculate_exposures
        stock_ret = df[["symbol", "trade_date", "stock_excess"]].rename(
            columns={"symbol": "symbol", "trade_date": "date", "stock_excess": "return"}
        ).dropna()
        factor_ret = (
            df[["trade_date", "mkt_excess"]].drop_duplicates("trade_date")
            .set_index("trade_date")[["mkt_excess"]].rename(columns={"mkt_excess": "mkt"})
        )
        exposure = calculate_exposures(stock_ret, factor_ret, period="D", lookback=252)
        print("      jh_quant 内置 beta（全样本）:")
        for code, name in STOCKS.items():
            b = exposure[exposure["symbol"] == code]["mkt"].mean()
            print(f"        {name}: {b:.3f}")
    except Exception as e:  # 内置接口或环境缺失时不影响主流程
        print(f"      [跳过] calculate_exposures 不可用: {e}")

    # fig1：散点 + 回归线（三子图纵向排布，手机端单屏阅读更友好）
    # 纵向布局：x 轴语义一致 → 共用 x 标签（只放最底一列）；每只股票各自收益轴 → y 标签各子图独立
    fig1, axes = plt.subplots(3, 1, figsize=(7.5, 14), sharex=True)
    for ax, (code, name) in zip(axes, STOCKS.items()):
        sub = df[df["symbol"] == code].dropna()
        ax.scatter(sub["mkt_excess"], sub["stock_excess"], s=8, alpha=0.45,
                   color=mpl_style.COLOR_CYCLE[0], label="日收益")
        beta, alpha, r2, _, _ = ols_beta(sub["stock_excess"], sub["mkt_excess"])
        xline = np.linspace(sub["mkt_excess"].min(), sub["mkt_excess"].max(), 50)
        ax.plot(xline, alpha + beta * xline, color=mpl_style.RISE, lw=2.2,
                label=f"回归线 β={beta:.2f}")
        ax.axhline(0, color="#B2BABB", lw=0.8, ls="--")
        ax.axvline(0, color="#B2BABB", lw=0.8, ls="--")
        ax.set_title(f"{name}\nβ={beta:.2f}  R²={r2:.2f}", fontsize=13)
        ax.set_ylabel("个股日收益 (%)")
        ax.set_ylim(-20, 20)  # 三张子图统一 y 轴范围，斜率（Beta）才能直观对比
        ax.legend(loc="upper left", fontsize=10)
        mpl_style.hide_spines(ax)
    axes[-1].set_xlabel("沪深300 日收益 (%)")  # 三个子图共用的 x 标签
    fig1.suptitle("三只股票 vs 沪深300：日收益散点与市场模型回归线", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("output/fig1_scatter_regression.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_scatter_regression.png")

    # fig2：beta 对比柱状图
    fig2, ax = plt.subplots(figsize=(8, 5))
    names = [r["name"] for r in results]
    betas = [r["beta"] for r in results]
    colors = [mpl_style.FALL if b < 1 else mpl_style.RISE for b in betas]
    bars = ax.bar(names, betas, color=colors, width=0.5)
    ax.axhline(1.0, color="#7F8C8D", lw=1.2, ls="--", label="β = 1（与大盘同步）")
    for bar, b in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{b:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Beta")
    ax.set_title("三只股票的 CAPM 市场 Beta（2020–2026）", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(betas) * 1.25)
    ax.legend(fontsize=10)
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_beta_compare.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_beta_compare.png")

    # fig3：滚动 beta 时序
    fig3, ax = plt.subplots(figsize=(11, 5))
    for code, name in STOCKS.items():
        sub = df[df["symbol"] == code].dropna().sort_values("trade_date")
        rb = rolling_beta(sub, window=60)
        ax.plot(sub["trade_date"], rb, lw=1.6, label=f"{name}")
    ax.set_ylabel("60 日滚动 Beta")
    ax.set_title("三只股票的 60 个交易日滚动 Beta（2020–2026）", fontsize=14, fontweight="bold")
    ax.axhline(1.0, color="#7F8C8D", lw=1, ls="--")
    ax.legend(fontsize=10, ncol=3)
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("output/fig3_rolling_beta.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_rolling_beta.png")

    return df, res


# ============================================================
# Part 2：全市场截面（月频）
# ============================================================

# 样本期（2014-12 多拉一个月供 pct_change 用）
MKT_START, MKT_END = "2014-12-01", "2026-06-30"
MIN_OBS = 60          # 单只股票最少样本月数
MAX_ABS_RET = 1.0     # 月度收益绝对值上限（100%），剔除极端


def fetch_market_data():
    """用 jh_quant 拉取：全市场月度前复权、沪深300 指数、SHIBOR、行业。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()

    # 1) 全市场月度前复权行情（2014-12 起，供 pct_change 计算 2015-05 之后收益）
    monthly = jh.get_data(DataTypes.TS_MONTHLY_QFQ, start=MKT_START, end=MKT_END).to_df()
    monthly = monthly[["trade_date", "ts_code", "close"]]

    # 2) 沪深300 指数日线（市场组合收益，TS 源 ts_code=000300.SH）
    idx = jh.get_data(DataTypes.TS_INDEX_DAILY,
                      ts_code="000300.SH", start="2015-01-01", end="2026-06-30").to_df()
    idx = idx[["trade_date", "close"]].rename(columns={"trade_date": "date"})

    # 3) SHIBOR 无风险利率（TS_SHIBOR，月度用 1 个月期列 1m）
    #    数据量小，bypass_cache 直接拉取：TS_SHIBOR 服务端 DDL 列名尚未同步为 1m/1w
    shibor = jh.get_data(DataTypes.TS_SHIBOR,
                         start="2015-01-01", end="2026-06-30",
                         bypass_cache=True).to_df()
    shibor = shibor[["date", "1m"]]

    # 4) 股票基本信息（行业分类）
    basic = jh.get_data(DataTypes.TS_STOCK_BASIC).to_df()
    basic = basic[["ts_code", "industry"]]

    return monthly, idx, shibor, basic


def build_panel(monthly, idx, shibor):
    """把个股 / 市场 / 无风险利率按月度对齐成面板，返回超额收益。"""
    # 个股月度收益
    px = monthly.copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px["ym"] = px["trade_date"].dt.to_period("M")
    px = px.sort_values(["ts_code", "trade_date"])
    px["ret"] = px.groupby("ts_code")["close"].pct_change()
    px = px.dropna(subset=["ret"])
    px = px[np.abs(px["ret"]) < MAX_ABS_RET]  # 剔除极端收益

    # 市场月收益：指数日线 -> 每月最后一个交易日 close -> pct_change
    idx["ym"] = pd.to_datetime(idx["date"]).dt.to_period("M")
    mkt = idx.sort_values("date").groupby("ym")["close"].last().astype(float)
    mkt_ret = mkt.pct_change().rename("mkt_ret")

    # 无风险利率：SHIBOR 1个月期 -> 每月最后一个交易日 1m -> 月化（/100/12）
    shibor["ym"] = pd.to_datetime(shibor["date"]).dt.to_period("M")
    rf = shibor.sort_values("date").groupby("ym")["1m"].last().astype(float) / 100 / 12
    rf = rf.rename("rf")

    # 对齐：个股收益按 ym 合并市场收益与无风险利率
    panel = px.merge(mkt_ret, on="ym", how="left").merge(rf, on="ym", how="left")
    panel["excess_ret"] = panel["ret"] - panel["rf"]
    panel["excess_mkt"] = panel["mkt_ret"] - panel["rf"]
    panel = panel.dropna(subset=["excess_ret", "excess_mkt"])
    return panel


def estimate_beta(g):
    """对单只股票回归市场模型，返回 beta / alpha / r2 / 样本数。"""
    y = g["excess_ret"].values
    x = g["excess_mkt"].values
    if len(y) < MIN_OBS:
        return pd.Series({"beta": np.nan, "alpha": np.nan, "r2": np.nan, "n": len(y)})
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    r2 = 1.0 - resid.var() / y.var()
    return pd.Series({"beta": beta, "alpha": alpha, "r2": r2, "n": len(y)})


def part2_cross_section():
    """全市场 Beta 截面分析 + 3 张图。"""
    print(">>> [Part 2] 拉取全市场数据（月频）...")
    monthly, idx, shibor, basic = fetch_market_data()
    print(f"    行情: {len(monthly)} 行, 指数: {len(idx)} 行, SHIBOR: {len(shibor)} 行")

    panel = build_panel(monthly, idx, shibor)
    print(f"    面板: {len(panel)} 行, 月度 {panel['ym'].nunique()} 个月, 股票 {panel['ts_code'].nunique()} 只")

    est = panel.groupby("ts_code").apply(estimate_beta, include_groups=False)
    est = est.dropna(subset=["beta"]).reset_index()
    print(f"    有效股票（样本 >= {MIN_OBS} 个月）: {len(est)} 只")

    # 描述统计
    desc = est["beta"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print("\n    Beta 描述统计：")
    print(desc.round(3).to_string())

    # 行业 Beta（合并行业分类，取行业中位数）
    est = est.merge(basic, on="ts_code", how="left")
    est["industry"] = est["industry"].fillna("其他")
    ind = (est.groupby("industry")["beta"]
             .agg(["median", "count"])
             .sort_values("median", ascending=False))
    print("\n    行业 Beta（按中位数降序）：")
    print(ind.round(3).to_string())

    beta = est["beta"].dropna()

    # 图1：Beta 分布直方图
    fig1, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(beta, bins=60, color=mpl_style.COLOR_CYCLE[0], alpha=0.85,
            edgecolor="white", linewidth=0.4)
    med = beta.median()
    ax.axvline(med, color=mpl_style.RISE, lw=1.8, ls="--",
               label=f"中位数 {med:.2f}")
    ax.axvline(1.0, color="#7F8C8D", lw=1.2, ls=":", label="Beta = 1（与大盘同步）")
    ax.set_xlim(-1.0, 3.0)
    ax.set_xlabel("Beta")
    ax.set_ylabel("股票数量")
    ax.set_title("全 A 股月度 Beta 分布（2015-05 ~ 2026-06）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("output/fig1_beta_distribution.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_beta_distribution.png")

    # 图2：行业平均 Beta（前 12 + 后 8，按中位数排序）
    top = ind.head(12)
    bottom = ind.tail(8)
    sel = pd.concat([top, bottom]).sort_values("median")
    fig2, ax = plt.subplots(figsize=(10, 7))
    colors = [mpl_style.ACCENT if v < 1 else mpl_style.RISE for v in sel["median"]]
    ax.barh(sel.index, sel["median"], color=colors, height=0.65)
    for y, v in enumerate(sel["median"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(1.0, color="#7F8C8D", lw=1.2, ls="--")
    ax.set_xlabel("行业 Beta 中位数")
    ax.set_ylabel("行业")
    ax.set_title("各行业 Beta 中位数（前 12 + 后 8）", fontsize=14, fontweight="bold")
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_beta_by_industry.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_beta_by_industry.png")

    # 图3：Beta 与平均月收益（散点 + 分桶均值线）
    est["mean_ret"] = est["ts_code"].map(panel.groupby("ts_code")["ret"].mean())
    fig3, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(est["beta"], est["mean_ret"] * 100, s=8, alpha=0.35,
               color=mpl_style.COLOR_CYCLE[0])
    # 分桶：每 0.2 一个桶，桶内均值连线
    bins = np.arange(-0.5, 3.0, 0.2)
    labels = (bins[:-1] + bins[1:]) / 2
    bucket = pd.cut(est["beta"], bins=bins, labels=labels)
    grp = est.groupby(bucket, observed=False)["mean_ret"].mean().dropna()
    # 供文章「结果三」引用的数字：Beta 与平均月收益的相关性 + 分桶均值
    corr_br = est["beta"].corr(est["mean_ret"])
    print(f"\n    Beta 与平均月收益的相关系数: {corr_br:.2f}")
    print("    分桶均值（Beta 区间 -> 平均月收益 %）：")
    print((grp * 100).round(2).to_string())
    ax.plot(grp.index.astype(float), grp.values * 100,
            color=mpl_style.RISE, lw=2.2, marker="o", ms=4, label="分桶均值")
    ax.set_xlabel("Beta")
    ax.set_ylabel("平均月收益 (%)")
    ax.set_title("Beta 与平均月收益（2015-05 ~ 2026-06）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 3.0)
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("output/fig3_beta_vs_return.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_beta_vs_return.png")

    return panel, est


def main():
    os.makedirs("output", exist_ok=True)
    part1_single_stock()
    print()
    part2_cross_section()
    print("\n完成。6 张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
