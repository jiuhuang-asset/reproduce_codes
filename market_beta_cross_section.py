# -*- coding: utf-8 -*-
"""
全市场 Beta 截面分析（A股月度实证）
====================================

对应公众号文章《全市场的 Beta 长什么样》（量海泛舟 · 资产定价系列第二篇）
方法论参考：https://www.tidy-finance.org/chapters/beta-estimation.html

做什么
------
对全 A 股每一只股票，用月度数据跑市场模型回归：
    R_i - R_f = alpha + beta x (R_m - R_f) + epsilon
得到每只股票的 Beta，然后看三件事：
  1. Beta 在全市场的分布长什么样（fig1）
  2. 哪些行业平均 Beta 高 / 低（fig2 + 行业表）
  3. 高 Beta 股票是不是真的伴随更高收益（fig3，CAPM 的核心预测）

关键设计
--------
- 月度频率：样本 2015-05 ~ 2024-12
- 市场组合：沪深300 指数（TS 源）；无风险利率：SHIBOR 1个月期（月末值，月度化）
- 最少观测：单只股票 < 60 个月样本则剔除（Beta 估计不可靠）
- 极端收益：|月度收益| > 100% 剔除（数据噪音，避免单点主导回归）

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python market_beta_cross_section.py

输出
----
控制台打印 Beta 描述统计与行业 Beta 表，
并生成 3 张图：
  - fig1_beta_distribution.png  Beta 分布直方图
  - fig2_beta_by_industry.png   行业平均 Beta（按中位数排序，纵向条形）
  - fig3_beta_vs_return.png     Beta 与平均月收益散点（含分桶均值线）
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 样本期（2014-12 多拉一个月供 pct_change 用）
START, END = "2014-12-01", "2024-12-31"
MIN_OBS = 60          # 单只股票最少样本月数
MAX_ABS_RET = 1.0     # 月度收益绝对值上限（100%），剔除极端


def fetch_data():
    """用 jh_quant 拉取：全市场月度前复权、沪深300 指数、SHIBOR、行业。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()

    # 1) 全市场月度前复权行情（2014-12 起，供 pct_change 计算 2015-05 之后收益）
    monthly = jh.get_data(DataTypes.TS_MONTHLY_QFQ, start=START, end=END).to_df()
    monthly = monthly[["trade_date", "ts_code", "close"]]

    # 2) 沪深300 指数日线（市场组合收益，TS 源 ts_code=000300.SH）
    idx = jh.get_data(DataTypes.TS_INDEX_DAILY,
                      ts_code="000300.SH", start="2015-01-01", end="2024-12-31").to_df()
    idx = idx[["trade_date", "close"]].rename(columns={"trade_date": "date"})

    # 3) SHIBOR 无风险利率（TS_SHIBOR，月度用 1 个月期列 1m）
    #    数据量小，bypass_cache 直接拉取：TS_SHIBOR 服务端 DDL 列名尚未同步为 1m/1w
    shibor = jh.get_data(DataTypes.TS_SHIBOR,
                         start="2015-01-01", end="2024-12-31",
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


def main():
    os.makedirs("output", exist_ok=True)
    print(">>> 1/4 拉取数据（jh_quant）...")
    monthly, idx, shibor, basic = fetch_data()
    print(f"    行情: {len(monthly)} 行, 指数: {len(idx)} 行, SHIBOR: {len(shibor)} 行")

    print(">>> 2/4 构建月度面板 + 超额收益 ...")
    panel = build_panel(monthly, idx, shibor)
    print(f"    面板: {len(panel)} 行, 月度 {panel['ym'].nunique()} 个月, 股票 {panel['ts_code'].nunique()} 只")

    print(">>> 3/4 逐股回归市场模型 ...")
    est = panel.groupby("ts_code").apply(estimate_beta, include_groups=False)
    est = est.dropna(subset=["beta"])
    est = est.reset_index()
    print(f"    有效股票（样本 >= {MIN_OBS} 个月）: {len(est)} 只")

    # 描述统计
    desc = est["beta"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print("\nBeta 描述统计：")
    print(desc.round(3).to_string())

    # 行业 Beta（合并行业分类，取行业中位数）
    est = est.merge(basic, on="ts_code", how="left")
    est["industry"] = est["industry"].fillna("其他")
    ind = (est.groupby("industry")["beta"]
             .agg(["median", "count"])
             .sort_values("median", ascending=False))
    print("\n行业 Beta（按中位数降序）：")
    print(ind.round(3).to_string())

    print(">>> 4/4 绘图（mpl_style 统一风格）...")
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
    ax.set_title("全 A 股月度 Beta 分布（2015-05 ~ 2024-12）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("output/fig1_beta_distribution.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_beta_distribution.png")

    # 图2：行业平均 Beta（前 12 + 后 8，按中位数排序）
    top = ind.head(12)
    bottom = ind.tail(8)
    sel = pd.concat([top, bottom])
    sel = sel.sort_values("median")
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
    est["mean_ret"] = np.nan
    mean_ret_map = panel.groupby("ts_code")["ret"].mean()
    est["mean_ret"] = est["ts_code"].map(mean_ret_map)
    fig3, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(est["beta"], est["mean_ret"] * 100, s=8, alpha=0.35,
               color=mpl_style.COLOR_CYCLE[0])
    # 分桶：每 0.2 一个桶，桶内均值连线
    bins = np.arange(-0.5, 3.0, 0.2)
    labels = (bins[:-1] + bins[1:]) / 2
    bucket = pd.cut(est["beta"], bins=bins, labels=labels)
    grp = est.groupby(bucket, observed=False)["mean_ret"].mean()
    grp = grp.dropna()
    ax.plot(grp.index.astype(float), grp.values * 100,
            color=mpl_style.RISE, lw=2.2, marker="o", ms=4, label="分桶均值")
    ax.set_xlabel("Beta")
    ax.set_ylabel("平均月收益 (%)")
    ax.set_title("Beta 与平均月收益（2015-05 ~ 2024-12）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 3.0)
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("output/fig3_beta_vs_return.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_beta_vs_return.png")

    print("\n完成。三张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
