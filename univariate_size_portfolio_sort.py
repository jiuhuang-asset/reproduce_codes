# -*- coding: utf-8 -*-
"""
单变量投资组合排序：按市值十分位分组（A股月度实证）
=====================================================

对应公众号文章《投资组合排序 · 小盘股真的跑赢大盘股吗》（量海泛舟 · 资产定价系列第二篇）
方法论参考：https://www.tidy-finance.org/chapters/univariate-portfolio-sorts.html

做什么
------
把全 A 股按「总市值」每月排序，分成十分位组合 D1（最小盘）… D10（最大盘），
按市值加权计算每个组合下个月的收益，并构造多空组合 LS = D1 - D10（小盘减大盘，
这就是 Fama-French SMB 因子的雏形）。

关键设计
--------
- 月度再平衡：每个月末重新按市值分组
- 滞后一期：用 t 月末的市值分组，持有 t+1 月的收益（避免前视偏差）
- 市值加权：组合内收益按各股市值加权，权重 clip 到 >= 0
- 剔除次新股：上市不足 12 个月的股票剔除（避免新股炒作噪音）
- Newey-West 调整 t 值（月度 lag=3）检验多空组合平均收益是否显著非零

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python univariate_size_portfolio_sort.py

输出
----
控制台打印十分位平均月收益表、多空组合 LS 均值与 Newey-West t 值，
并生成 3 张图：
  - fig1_decile_returns.png  十分位平均月收益柱状图
  - fig2_cumulative.png      D1/D5/D10/LS 累计净值曲线
  - fig3_cap_share.png       每组总市值占全市场比例
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 收益样本期 2015-01 ~ 2024-12；市值多拉一个月（2014-12）用于滞后一期
RET_START, RET_END = "2015-01-01", "2024-12-31"
CAP_START = "2014-12-01"  # 需要比收益早一个月

# 十分位切点
DECILE_BPS = np.linspace(0.1, 0.9, 9).tolist()


def fetch_data():
    """用 jh_quant 拉取：交易日历、月度市值、月度前复权行情、上市日期。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()

    # 0) 上市日期（用于剔除上市不足 12 个月的次新股，避免新股炒作噪音）
    basic = jh.get_data(DataTypes.TS_STOCK_BASIC).to_df()
    basic = basic[["ts_code", "list_date"]].dropna()
    basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d")

    # 1) 交易日历 -> 月末交易日网格（start/end 对该表无效，拉全量后本地筛）
    cal = jh.get_data(DataTypes.TS_TRADE_CAL).to_df()
    cal = cal[cal["is_open"] == 1].copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    cal["ym"] = cal["cal_date"].dt.to_period("M")
    month_end = (
        cal.groupby("ym")["cal_date"].max().dt.date.reset_index(name="month_end")
    )
    month_end = month_end[(month_end["month_end"] >= pd.Timestamp("2014-12-01").date())
                          & (month_end["month_end"] <= pd.Timestamp("2024-12-31").date())]

    # 2) 月度市值表（各股当月最后交易日不同 -> 按 股票+月 取 last）
    mb = jh.get_data(DataTypes.TS_MONTHLY_BASIC, start=CAP_START, end=RET_END).to_df()
    mb = mb[["trade_date", "ts_code", "total_mv"]].dropna(subset=["total_mv"])
    mb["trade_date"] = pd.to_datetime(mb["trade_date"])
    mb["ym"] = mb["trade_date"].dt.to_period("M")
    # 每 (股票, 月) 保留当月最后一次市值观测
    mcap = mb.sort_values("trade_date").groupby(["ts_code", "ym"], as_index=False).agg(
        mktcap=("total_mv", "last")
    )
    mcap = mcap[mcap["mktcap"] > 0]

    # 3) 月度前复权行情（每月一个统一月末日期）-> 月度收益
    mpx = jh.get_data(DataTypes.TS_MONTHLY_QFQ, start=CAP_START, end=RET_END).to_df()
    mpx = mpx[["trade_date", "ts_code", "close"]]
    mpx["trade_date"] = pd.to_datetime(mpx["trade_date"])
    mpx = mpx[mpx["close"] > 0]
    mpx = mpx.sort_values(["ts_code", "trade_date"])
    mpx["ret"] = mpx.groupby("ts_code")["close"].pct_change()

    return month_end, mcap, mpx, basic


def vw_return(group):
    """市值加权收益：Σ(ret × mktcap) / Σ(mktcap)，权重 clip 到 >= 0。"""
    valid = group[["ret", "mktcap"]].notna().all(axis=1)
    if valid.sum() == 0:
        return np.nan
    w = group.loc[valid, "mktcap"].clip(lower=0)
    r = group.loc[valid, "ret"]
    if w.sum() == 0:
        return r.mean()
    return (r * w).sum() / w.sum()


def assign_decile(s):
    """横截面十分位：返回 1(最小盘) … 10(最大盘)。缺失市值给 NaN。"""
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    clean = s.dropna()
    if clean.empty:
        return out
    q = clean.quantile(DECILE_BPS)
    # D1 = 市值最小, D10 = 市值最大
    labels = pd.cut(clean, bins=[-np.inf, *q.values, np.inf],
                    labels=False, duplicates="drop") + 1
    out.loc[labels.index] = labels
    return out


def newey_west_t(values, mean, lag=3):
    """Newey-West 调整后的样本均值 t 值（Bartlett 权重，月度惯例 lag=3）。"""
    values = values[np.isfinite(values)]
    n = len(values)
    if n <= 1:
        return np.nan
    resid = values - mean
    q = np.sum(resid ** 2)
    for j in range(1, min(lag, n - 1) + 1):
        gamma_j = np.sum(resid[j:] * resid[:-j])
        weight = 1.0 - j / (lag + 1)
        q += 2 * weight * gamma_j
    se = np.sqrt(q) / n if q >= 0 else np.nan
    return mean / se if se and se > 0 else 0.0


def main():
    os.makedirs("output", exist_ok=True)
    print(">>> 1/4 拉取数据（jh_quant 月度表）...")
    month_end, mcap, mpx, basic = fetch_data()
    print(f"    月末网格: {len(month_end)} 个月, 市值表: {len(mcap)} 行, 行情: {len(mpx)} 行")

    print(">>> 2/4 构建面板 + 滞后一期 + 剔除次新股 ...")
    # 收益面板：只保留样本期内的月度收益
    ret = mpx[(mpx["trade_date"] >= RET_START)].dropna(subset=["ret"]).copy()
    ret["ym"] = ret["trade_date"].dt.to_period("M")
    ret["mktcap"] = np.nan

    # 市值滞后一期：t 月末市值 -> 对应 t 月末分组，持有 t+1 收益
    # 做法：把每行收益的「分组月」= 上一月，市值取上一月末
    ret = ret.merge(
        mcap.rename(columns={"ym": "ym_prev", "mktcap": "mktcap_prev"}),
        left_on=["ts_code", ret["ym"] - 1],
        right_on=["ts_code", "ym_prev"],
        how="left",
        suffixes=("", "_cap"),
    )
    ret["mktcap"] = ret["mktcap_prev"].astype(float)
    ret = ret.dropna(subset=["mktcap"])
    ret = ret[["trade_date", "ym", "ts_code", "ret", "mktcap"]]

    # 剔除上市不足 12 个月的次新股（避免新股上市初期炒作对结果的干扰）
    ret = ret.merge(basic, on="ts_code", how="left")
    ret = ret[ret["trade_date"] - ret["list_date"] >= pd.Timedelta(days=365)]
    ret = ret.drop(columns=["list_date"])
    print(f"    面板行数(有市值+有收益+剔次新): {len(ret)}")

    print(">>> 3/4 十分位分组 + 市值加权组合收益 ...")
    ret["decile"] = ret.groupby("ym")["mktcap"].transform(assign_decile)
    ret = ret.dropna(subset=["decile"])

    rows = []
    for ym, grp in ret.groupby("ym"):
        if grp.empty:
            continue
        d_ret = {d: vw_return(sub) for d, sub in grp.groupby("decile")}
        # LS = D1(最小盘) - D10(最大盘)
        d_ret["LS"] = d_ret.get(1, np.nan) - d_ret.get(10, np.nan)
        rows.append({"ym": ym, **{f"D{d}": d_ret.get(d, np.nan) for d in range(1, 11)},
                     "LS": d_ret["LS"]})
    panel = pd.DataFrame(rows).set_index("ym").sort_index()

    # 每组平均月收益（%）
    summary = panel.drop(columns=["LS"]).mean() * 100
    ls_mean = panel["LS"].mean() * 100
    ls_t = newey_west_t(panel["LS"].dropna().values, panel["LS"].mean())

    decile_summary = pd.DataFrame({
        "组合": [f"D{d}" for d in range(1, 11)],
        "平均月收益(%)": summary.values.round(3),
    })
    print("\n十分位平均月收益：")
    print(decile_summary.to_string(index=False))
    print(f"\n多空组合 LS = D1 - D10：")
    print(f"  平均月收益 = {ls_mean:.3f}%  |  Newey-West t 值 = {ls_t:.2f}")

    # 每组平均股票数 + 总市值占比
    cnt = ret.groupby(["ym", "decile"]).size().groupby("decile").mean()
    print("\n每组平均股票数：", {int(d): int(c) for d, c in cnt.items()})
    cap_share = (
        ret.groupby(["ym", "decile"])["mktcap"].sum()
        .groupby("decile").mean()
    )
    cap_share = cap_share / ret.groupby(["ym"])["mktcap"].sum().mean()
    print("每组平均总市值占比：", {int(d): f"{v*100:.1f}%" for d, v in cap_share.items()})

    print(">>> 4/4 绘图（mpl_style 统一风格）...")
    # 图1：十分位平均月收益
    fig1, ax = plt.subplots(figsize=(9, 5))
    means = summary.values
    colors = [mpl_style.FALL if v < 0 else mpl_style.RISE for v in means]
    bars = ax.bar([f"D{i}" for i in range(1, 11)], means, color=colors, width=0.65)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0, color="#7F8C8D", lw=1)
    ax.set_ylabel("平均月收益 (%)")
    ax.set_xlabel("组合（D1 = 最小盘 → D10 = 最大盘）")
    ax.set_title("全 A 股按市值十分位的平均月收益（2015–2024）", fontsize=14, fontweight="bold")
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("output/fig1_decile_returns.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_decile_returns.png")

    # 图2：累计净值（复利）
    fig2, ax = plt.subplots(figsize=(11, 5.5))
    for name in ["D1", "D5", "D10", "LS"]:
        s = (panel[name] + 1).cumprod()
        ax.plot(s.index.astype(str), s.values, lw=1.8,
                label=f"{name}" + ("" if name != "LS" else "（小盘减大盘）"))
    ax.set_ylabel("累计净值（起始 = 1）")
    ax.set_xlabel("月份")
    ax.set_title("不同市值组合的累计净值（2015–2024）", fontsize=14, fontweight="bold")
    ax.axhline(1.0, color="#7F8C8D", lw=1, ls="--")
    ax.legend(fontsize=10, ncol=2)
    # x 轴每隔 12 个月标一次年份
    ticks = panel.index[::12]
    ax.set_xticks(range(0, len(panel), 12))
    ax.set_xticklabels([str(t)[:4] for t in ticks], rotation=45)
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_cumulative.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_cumulative.png")

    # 图3：每组总市值占全市场比例（说明市值加权 vs 等权为什么差这么多）
    fig3, ax = plt.subplots(figsize=(9, 4.5))
    shares = cap_share.reindex(range(1, 11)).fillna(0) * 100
    colors = [mpl_style.ACCENT if i in (1, 10) else mpl_style.COLOR_CYCLE[6]
              for i in range(1, 11)]
    bars = ax.bar([f"D{i}" for i in range(1, 11)], shares.values, color=colors, width=0.65)
    for bar, v in zip(bars, shares.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("占全市场总市值 (%)")
    ax.set_xlabel("组合（D1 = 最小盘 → D10 = 最大盘）")
    ax.set_title("每个十分位组合的总市值占比（2015–2024）", fontsize=14, fontweight="bold")
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("output/fig3_cap_share.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_cap_share.png")

    print("\n完成。三张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
