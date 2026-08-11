# -*- coding: utf-8 -*-
"""
规模溢价的 p-hacking 检验（A股月度实证）
=========================================

对应公众号文章《规模排序与 p-hacking》（量海泛舟 · 资产定价系列第四篇）
方法论参考：https://www.tidy-finance.org/chapters/size-sorts-and-p-hacking.html

做什么
------
上一篇发现小盘-大盘多空组合（LS = 小盘减大盘）月赚 2.01%。但研究者有大量
"自由选择"：组合分成几组、用等权还是市值加权、样本期取多长、股票池怎么定，
每个选择都会改变结果。本文把 4 个维度排列组合成 54 种规格，逐一计算 LS 溢价，
看结果对设计选择有多敏感——这就是 p-hacking 的温和版：选择多了，你总能
找到"看起来很棒"的结果。

规格网格（4 个研究者自由维度）
--------------------------------
- 组合数：2 / 5 / 10
- 加权：等权（EW）/ 市值加权（VW）
- 样本期：2015-2024 / 2015-2019 / 2020-2024
- 股票池：全 A / 主板 / 剔除金融

共 3 x 2 x 3 x 3 = 54 种规格。

关键设计（与系列第三篇一致，属硬规则而非自由选择）
----------------------------------------------------
- 月度再平衡：t 月末市值分组，持有 t+1 月收益（滞后一期，避免前视偏差）
- 剔除上市不足 12 个月的次新股
- 收益面板构建一次，54 种规格只做分组/聚合

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python size_premium_p_hacking.py

输出
----
控制台打印全部 54 种规格的 LS 平均月收益与极值规格明细，
并生成 3 张图：
  - fig1_spec_landscape.png   全部规格 LS 收益条形图（按值排序）
  - fig2_driver_sensitivity.png 按 4 个维度分组的箱线图
  - fig3_best_vs_robust.png   最激进 / 最保守规格的累计净值对比
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd

# 统一图表风格（house style）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RET_START, RET_END = "2015-01-01", "2024-12-31"
CAP_START = "2014-12-01"  # 多拉一个月，供滞后一期用

# 金融行业（剔除金融维度用）
FINANCE_INDUSTRIES = {"银行", "保险", "证券", "多元金融"}

# 规格网格
N_PORTS = [2, 5, 10]
WEIGHTINGS = ["ew", "vw"]
PERIODS = [("2015", "2024"), ("2015", "2019"), ("2020", "2024")]
POOLS = ["all", "mainboard", "no_finance"]


def fetch_data():
    """用 jh_quant 拉取：月度市值、月度前复权行情、上市日期、行业。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()

    # 上市日期 + 行业（剔除次新股 / 金融 / 判断主板用）
    basic = jh.get_data(DataTypes.TS_STOCK_BASIC).to_df()
    basic = basic[["ts_code", "list_date", "industry"]].dropna(subset=["list_date"])
    basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d")
    basic["industry"] = basic["industry"].fillna("其他")

    # 月度市值表（各股当月最后交易日不同 -> 按 股票+月 取 last）
    mb = jh.get_data(DataTypes.TS_MONTHLY_BASIC, start=CAP_START, end=RET_END).to_df()
    mb = mb[["trade_date", "ts_code", "total_mv"]].dropna(subset=["total_mv"])
    mb["trade_date"] = pd.to_datetime(mb["trade_date"])
    mb["ym"] = mb["trade_date"].dt.to_period("M")
    mcap = mb.sort_values("trade_date").groupby(["ts_code", "ym"], as_index=False).agg(
        mktcap=("total_mv", "last")
    )
    mcap = mcap[mcap["mktcap"] > 0]

    # 月度前复权行情（每月统一月末日期）-> 月度收益
    mpx = jh.get_data(DataTypes.TS_MONTHLY_QFQ, start=CAP_START, end=RET_END).to_df()
    mpx = mpx[["trade_date", "ts_code", "close"]]
    mpx["trade_date"] = pd.to_datetime(mpx["trade_date"])
    mpx = mpx[mpx["close"] > 0]
    mpx = mpx.sort_values(["ts_code", "trade_date"])
    mpx["ret"] = mpx.groupby("ts_code")["close"].pct_change()

    return basic, mcap, mpx


def build_panel(basic, mcap, mpx):
    """构建月度面板：收益 + 滞后一期市值 + 剔次新，一次搞定，54 规格共用。"""
    # 收益面板：只保留样本期内的月度收益
    ret = mpx[(mpx["trade_date"] >= RET_START)].dropna(subset=["ret"]).copy()
    ret["ym"] = ret["trade_date"].dt.to_period("M")

    # 市值滞后一期：用 t-1 月末市值分组（对应 t 月末时点可用信息）
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

    # 合并行业信息，并生成股票池标记
    ret = ret.merge(basic, on="ts_code", how="left")
    ret = ret[ret["trade_date"] - ret["list_date"] >= pd.Timedelta(days=365)]
    ret = ret.drop(columns=["list_date"])

    # 主板标记：排除创业板(30x) / 科创板(688)
    ret["is_mainboard"] = ret["ts_code"].str.startswith(
        ("600", "601", "603", "605", "000", "001", "002", "003"))
    # 金融标记
    ret["is_finance"] = ret["industry"].isin(FINANCE_INDUSTRIES)
    return ret


def assign_groups(s, n):
    """横截面等分组：返回 1(最小盘) ... n(最大盘)。缺失市值给 NaN。"""
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    clean = s.dropna()
    if clean.empty:
        return out
    q = clean.quantile(np.linspace(1 / n, (n - 1) / n, n - 1))
    labels = pd.cut(clean, bins=[-np.inf, *q.values, np.inf],
                    labels=False, duplicates="drop") + 1
    out.loc[labels.index] = labels
    return out


def group_returns(grp, weighting):
    """组合内收益：等权均值 或 市值加权。返回 (组, 收益) 序列。"""
    if weighting == "vw":
        w = grp["mktcap"].clip(lower=0)
        r = (grp["ret"] * w).sum() / w.sum() if w.sum() > 0 else grp["ret"].mean()
    else:
        r = grp["ret"].mean()
    return r


def compute_ls(panel, n_port, weighting, period, pool):
    """按一组规格计算每月 LS（组1 小盘 - 组n 大盘）收益序列。"""
    start, end = period
    sub = panel[(panel["ym"].astype(str) >= start) & (panel["ym"].astype(str) <= end)]
    if pool == "mainboard":
        sub = sub[sub["is_mainboard"]]
    elif pool == "no_finance":
        sub = sub[~sub["is_finance"]]

    sub = sub.copy()
    sub["group"] = sub.groupby("ym")["mktcap"].transform(lambda s: assign_groups(s, n_port))
    sub = sub.dropna(subset=["group"])

    ls_series = []
    for ym, grp in sub.groupby("ym"):
        if grp.empty:
            continue
        r = grp.groupby("group").apply(lambda g: group_returns(g, weighting))
        if 1 in r.index and n_port in r.index:
            ls_series.append((ym, r[1] - r[n_port]))
    out = pd.Series(dict(ls_series))
    return out.dropna()


def main():
    os.makedirs("output", exist_ok=True)
    print(">>> 1/3 拉取数据并构建月度面板（滞后一期 + 剔次新）...")
    basic, mcap, mpx = fetch_data()
    panel = build_panel(basic, mcap, mpx)
    print(f"    面板: {len(panel)} 行, 股票 {panel['ts_code'].nunique()} 只")

    print(">>> 2/3 跑 54 种规格 ...")
    rows = []
    for n_port, w, period, pool in itertools.product(N_PORTS, WEIGHTINGS, PERIODS, POOLS):
        ls = compute_ls(panel, n_port, w, period, pool)
        rows.append({
            "组合数": n_port, "加权": "等权" if w == "ew" else "市值加权",
            "样本期": f"{period[0]}-{period[1]}", "股票池": pool,
            "规模溢价%": ls.mean() * 100,
            "月数": len(ls),
        })
    specs = pd.DataFrame(rows)

    print(f"\n全部 {len(specs)} 种规格的规模溢价（小盘减大盘，%/月）：")
    print(specs.to_string(index=False))
    print(f"\n统计：均值 {specs['规模溢价%'].mean():.2f}% | "
          f"最小 {specs['规模溢价%'].min():.2f}% | "
          f"最大 {specs['规模溢价%'].max():.2f}% | "
          f"中位数 {specs['规模溢价%'].median():.2f}%")
    print(f"正值规格占比: {(specs['规模溢价%'] > 0).mean() * 100:.0f}%")

    print("\n最保守（最低）：")
    print(specs.loc[specs["规模溢价%"].idxmin()].to_string())
    print("\n最激进（最高）：")
    print(specs.loc[specs["规模溢价%"].idxmax()].to_string())

    # 找出最激进 / 最保守规格，供图 3 画累计净值。
    # 必须限定在 全样本期 2015-2024 内选，否则两条曲线时间窗不同、x 轴错开不可比。
    full = specs[specs["样本期"] == "2015-2024"]
    best = full.loc[full["规模溢价%"].idxmax()]
    worst = full.loc[full["规模溢价%"].idxmin()]

    def series_of(spec_row):
        return compute_ls(panel, int(spec_row["组合数"]),
                          "ew" if spec_row["加权"] == "等权" else "vw",
                          (spec_row["样本期"][:4], spec_row["样本期"][5:]),
                          spec_row["股票池"])
    best_series = series_of(best).sort_index()
    worst_series = series_of(worst).sort_index()

    print(">>> 3/3 绘图（mpl_style 统一风格）...")
    ls_vals = specs.sort_values("规模溢价%")

    # 图1：全部规格 LS 收益条形图（按值排序）
    fig1, ax = plt.subplots(figsize=(11, 6))
    colors = [mpl_style.ACCENT if v <= specs["规模溢价%"].median()
              else mpl_style.RISE for v in ls_vals["规模溢价%"]]
    ax.bar(range(len(ls_vals)), ls_vals["规模溢价%"], color=colors, width=0.9)
    ax.axhline(0, color="#7F8C8D", lw=1)
    ax.axhline(specs["规模溢价%"].median(), color="#7F8C8D", lw=1.2, ls="--",
               label=f"中位数 {specs['规模溢价%'].median():.2f}%")
    ax.set_xlabel("规格编号（按规模溢价升序）")
    ax.set_ylabel("规模溢价（小盘−大盘，%/月）")
    ax.set_title("54 种设计规格下的规模溢价（A 股 2015-2024）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xticks([])
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("output/fig1_spec_landscape.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_spec_landscape.png")

    # 图2：4 个维度的边际敏感性（箱线图）
    dims = [
        ("组合数", [2, 5, 10], specs),
        ("加权", ["等权", "市值加权"], specs),
        ("样本期", ["2015-2024", "2015-2019", "2020-2024"], specs),
        ("股票池", ["all", "mainboard", "no_finance"], specs),
    ]
    fig2, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, (dim, cats, _) in zip(axes, dims):
        data = [specs.loc[specs[dim] == c, "规模溢价%"].values for c in cats]
        bp = ax.boxplot(data, tick_labels=[str(c) for c in cats], patch_artist=True)
        for patch, c in zip(bp["boxes"], cats):
            patch.set_facecolor(mpl_style.COLOR_CYCLE[0] if c == cats[0]
                                else mpl_style.COLOR_CYCLE[3])
            patch.set_alpha(0.6)
        ax.axhline(0, color="#7F8C8D", lw=0.8, ls=":")
        ax.set_title(dim, fontsize=12)
        ax.set_xlabel("取值")
        ax.set_ylabel("规模溢价（%/月）" if dim == "组合数" else "")
        mpl_style.hide_spines(ax)
    fig2.suptitle("哪个因素影响最大？", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig("output/fig2_driver_sensitivity.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_driver_sensitivity.png")

    # 图3：最激进 vs 最保守规格的累计净值
    fig3, ax = plt.subplots(figsize=(11, 5.5))
    label_best = (f"最激进: {best['组合数']}组/等权/{best['样本期']}/{best['股票池']} "
                  f"({best['规模溢价%']:.2f}%/月)")
    label_worst = (f"最保守: {worst['组合数']}组/市值加权/{worst['样本期']}/{worst['股票池']} "
                   f"({worst['规模溢价%']:.2f}%/月)")
    ax.plot(best_series.index.astype(str), (best_series + 1).cumprod(),
            lw=1.8, color=mpl_style.RISE, label=label_best)
    ax.plot(worst_series.index.astype(str), (worst_series + 1).cumprod(),
            lw=1.8, color=mpl_style.FALL, label=label_worst)
    ax.axhline(1.0, color="#7F8C8D", lw=1, ls="--")
    ax.set_ylabel("累计净值（起始 = 1）")
    ax.set_xlabel("月份")
    ax.set_title("同一个小盘-大盘溢价，两种设计差多远", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ticks = best_series.index[::12]
    ax.set_xticks(range(0, len(best_series), 12))
    ax.set_xticklabels([str(t)[:4] for t in ticks], rotation=45)
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("output/fig3_best_vs_robust.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_best_vs_robust.png")

    print("\n完成。三张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
