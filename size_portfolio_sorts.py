# -*- coding: utf-8 -*-
"""
投资组合排序：市值十分位 + 规模溢价的 p-hacking 检验（A股月度实证）
====================================================================

对应公众号文章《投资组合排序 · 小盘股真的跑赢大盘股吗？》（量海泛舟）
方法论参考：
  - https://www.tidy-finance.org/chapters/univariate-portfolio-sorts.html
  - https://www.tidy-finance.org/chapters/size-sorts-and-p-hacking.html

内容分两部分：
  Part 1 — 单变量市值十分位排序
      把全 A 股按「总市值」每月分成十分位 D1（最小盘）… D10（最大盘），
      按市值加权算每个组合下个月的收益，构造多空组合 LS = D1 - D10，
      用 Newey-West t 检验小盘溢价是否显著。
  Part 2 — 规模溢价的 p-hacking 检验
      研究者有 4 个可自由选择的维度（组合数 / 加权 / 样本期 / 股票池），
      排列组合成 3x2x3x3 = 54 种规格，逐一算 LS 溢价，
      看结果对设计选择有多敏感。

关键设计（两部分的硬规则一致）
--------------------------------
- 月度再平衡：t 月末市值分组，持有 t+1 月收益（滞后一期，避免前视偏差）
- 剔除上市不足 12 个月的次新股
- 收益面板只构建一次，Part 1 / Part 2 共用同一份面板

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python size_portfolio_sorts.py

输出
----
控制台打印十分位平均月收益、LS 多空组合均值与 NW t 值、54 种规格的规模溢价，
并生成 6 张图到 output/ 子目录：
  Part 1:
    - fig1_decile_returns.png      十分位平均月收益柱状图
    - fig2_cumulative.png          D1/D5/D10/LS 累计净值曲线
    - fig3_cap_share.png           每组总市值占全市场比例
  Part 2:
    - fig4_spec_landscape.png      54 种规格的规模溢价条形图（按值排序）
    - fig5_driver_sensitivity.png  4 个维度的边际敏感性箱线图
    - fig6_best_vs_robust.png      最激进 / 最保守规格的累计净值对比
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd

# 统一图表风格（house style，保证公众号文章图表一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RET_START, RET_END = "2015-01-01", "2024-12-31"
CAP_START = "2014-12-01"  # 多拉一个月，供滞后一期用

# 金融行业（Part 2 的「剔除金融」股票池维度用）
FINANCE_INDUSTRIES = {"银行", "保险", "证券", "多元金融"}

# Part 2 的规格网格
N_PORTS = [2, 5, 10]
WEIGHTINGS = ["ew", "vw"]
PERIODS = [("2015", "2024"), ("2015", "2019"), ("2020", "2024")]
POOLS = ["all", "mainboard", "no_finance"]


def fetch_data():
    """用 jh_quant 拉取：上市日期+行业、月度市值、月度前复权行情。"""
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
    """构建月度面板：收益 + 滞后一期市值 + 剔次新，一次搞定，两部分共用。"""
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


def assign_groups(s, n):
    """横截面等分 n 组：返回 1(最小盘) … n(最大盘)。缺失市值给 NaN。"""
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
        return vw_return(grp)
    return grp["ret"].mean()


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


# ============================================================
# Part 1：单变量市值十分位排序
# ============================================================

def part1_decile_sort(panel):
    """按市值十分位分组，算组合收益 + LS 多空 + 3 张图。"""
    print(">>> [Part 1] 十分位分组 + 市值加权组合收益 ...")
    panel = panel.copy()
    panel["decile"] = panel.groupby("ym")["mktcap"].transform(lambda s: assign_groups(s, 10))
    panel = panel.dropna(subset=["decile"])

    rows = []
    for ym, grp in panel.groupby("ym"):
        if grp.empty:
            continue
        d_ret = {d: vw_return(sub) for d, sub in grp.groupby("decile")}
        d_ret["LS"] = d_ret.get(1, np.nan) - d_ret.get(10, np.nan)
        rows.append({"ym": ym, **{f"D{d}": d_ret.get(d, np.nan) for d in range(1, 11)},
                     "LS": d_ret["LS"]})
    decile_panel = pd.DataFrame(rows).set_index("ym").sort_index()

    summary = decile_panel.drop(columns=["LS"]).mean() * 100
    ls_mean = decile_panel["LS"].mean() * 100
    ls_t = newey_west_t(decile_panel["LS"].dropna().values, decile_panel["LS"].mean())

    decile_summary = pd.DataFrame({
        "组合": [f"D{d}" for d in range(1, 11)],
        "平均月收益(%)": summary.values.round(3),
    })
    print("\n十分位平均月收益：")
    print(decile_summary.to_string(index=False))
    print(f"\n多空组合 LS = D1 - D10：")
    print(f"  平均月收益 = {ls_mean:.3f}%  |  Newey-West t 值 = {ls_t:.2f}")

    # 每组平均总市值占比
    cap_share = (
        panel.groupby(["ym", "decile"])["mktcap"].sum()
        .groupby("decile").mean()
    )
    cap_share = cap_share / panel.groupby(["ym"])["mktcap"].sum().mean()

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
        s = (decile_panel[name] + 1).cumprod()
        ax.plot(s.index.astype(str), s.values, lw=1.8,
                label=f"{name}" + ("" if name != "LS" else "（小盘减大盘）"))
    ax.set_ylabel("累计净值（起始 = 1）")
    ax.set_xlabel("月份")
    ax.set_title("不同市值组合的累计净值（2015–2024）", fontsize=14, fontweight="bold")
    ax.axhline(1.0, color="#7F8C8D", lw=1, ls="--")
    ax.legend(fontsize=10, ncol=2)
    ticks = decile_panel.index[::12]
    ax.set_xticks(range(0, len(decile_panel), 12))
    ax.set_xticklabels([str(t)[:4] for t in ticks], rotation=45)
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_cumulative.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_cumulative.png")

    # 图3：每组总市值占全市场比例
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

    return decile_panel


# ============================================================
# Part 2：规模溢价的 p-hacking 检验
# ============================================================

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
    return pd.Series(dict(ls_series)).dropna()


def part2_p_hacking(panel):
    """54 种规格网格逐一算 LS 溢价 + 3 张图。"""
    print(">>> [Part 2] 跑 54 种规格 ...")
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

    # 最激进 / 最保守规格（限定全样本期，保证两条累计净值曲线时间窗可比）
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

    # 图4：全部规格 LS 收益条形图（按值排序）
    ls_vals = specs.sort_values("规模溢价%")
    fig4, ax = plt.subplots(figsize=(11, 6))
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
    fig4.tight_layout()
    fig4.savefig("output/fig4_spec_landscape.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig4_spec_landscape.png")

    # 图5：4 个维度的边际敏感性（箱线图）
    dims = [
        ("组合数", [2, 5, 10]),
        ("加权", ["等权", "市值加权"]),
        ("样本期", ["2015-2024", "2015-2019", "2020-2024"]),
        ("股票池", ["all", "mainboard", "no_finance"]),
    ]
    fig5, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, (dim, cats) in zip(axes, dims):
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
    fig5.suptitle("哪个因素影响最大？", fontsize=14, fontweight="bold")
    fig5.tight_layout(rect=[0, 0, 1, 0.95])
    fig5.savefig("output/fig5_driver_sensitivity.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig5_driver_sensitivity.png")

    # 图6：最激进 vs 最保守规格的累计净值
    fig6, ax = plt.subplots(figsize=(11, 5.5))
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
    fig6.tight_layout()
    fig6.savefig("output/fig6_best_vs_robust.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig6_best_vs_robust.png")

    return specs


def main():
    os.makedirs("output", exist_ok=True)
    print(">>> 拉取数据并构建月度面板（滞后一期 + 剔次新）...")
    basic, mcap, mpx = fetch_data()
    panel = build_panel(basic, mcap, mpx)
    print(f"    面板: {len(panel)} 行, 股票 {panel['ts_code'].nunique()} 只, "
          f"月度 {panel['ym'].nunique()} 期")

    part1_decile_sort(panel)
    print()
    part2_p_hacking(panel)
    print("\n完成。6 张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
