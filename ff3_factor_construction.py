# -*- coding: utf-8 -*-
"""
Fama-French 三因子（MKT / SMB / HML）构造（A股月度实证）
=======================================================

对应公众号文章《FF3 三因子复刻 · 上：从排序到因子》（量海泛舟 · 资产定价系列第六篇）
方法论参考：https://www.tidy-finance.org/chapters/replicating-fama-and-french-factors.html

做什么
------
把前面学的两个排序工具——市值排序（第三篇）、BM 排序（第五篇）——组合成
Fama-French 三因子的标准构造（CLASSIC 口径，与 Fama & French 1993 一致）：

  1. 每个月末按「市值」中位数分成 2 组（S 小盘 / B 大盘）
  2. 每个月末按「账面市值比 BM」30%/70% 分位分成 3 组（L 成长 / M / H 价值）
  3. 独立排序交叉成 2 x 3 = 6 个组合，组合内按市值加权算月度收益
  4. SMB = 小盘 3 组等权平均 - 大盘 3 组等权平均
     HML = 高BM 2 组等权平均 - 低BM 2 组等权平均
  5. MKT = 全市场市值加权收益 - 无风险利率（SHIBOR 1M）

关键设计（延续系列惯例）
------------------------
- 月度再平衡：t 月末的市值/BM 分组，持有 t+1 月收益（滞后一期，避免前视偏差）
- BM 用月末市净率 PB 的倒数（TS_MONTHLY_BASIC 的 pb 字段）——A 股无 Compustat
  账面值，这是通行代理口径，与官方（真实账面权益）存在差异，文章会点明
- 剔除上市不足 12 个月的次新股
- 组合内市值加权，组合间等权平均（SMB/HML 的定义就是等权）

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python ff3_factor_construction.py

输出
----
控制台打印 2x3 组合平均月收益矩阵与三因子统计表（均值/标准差/Newey-West t），
并生成 2 张图：
  - fig1_ff3_cumulative.png   三因子累计净值曲线
  - fig2_ff3_portfolios.png   2x3 组合平均月收益热力图（SMB/HML 的来源）
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RET_START, RET_END = "2015-01-01", "2024-12-31"
CAP_START = "2014-12-01"  # 多拉一个月，供滞后一期用

# 2x3 排序的分位断点
SIZE_BREAKS = [0.5]            # 市值中位数二分
BM_BREAKS = [0.3, 0.7]         # BM 30%/70% 三分


def fetch_data():
    """用 jh_quant 拉取：上市日期、月度市值+PB、月度前复权行情、SHIBOR。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()

    # 上市日期（剔除次新股）
    basic = jh.get_data(DataTypes.TS_STOCK_BASIC).to_df()
    basic = basic[["ts_code", "list_date"]].dropna()
    basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d")

    # 月度基本面表：total_mv(市值) 与 pb(市净率)，BM = 1/PB
    mb = jh.get_data(DataTypes.TS_MONTHLY_BASIC, start=CAP_START, end=RET_END).to_df()
    mb = mb[["trade_date", "ts_code", "total_mv", "pb"]].dropna(subset=["total_mv"])
    mb["trade_date"] = pd.to_datetime(mb["trade_date"])
    mb["ym"] = mb["trade_date"].dt.to_period("M")
    mb = mb.sort_values("trade_date").groupby(["ts_code", "ym"], as_index=False).agg(
        mktcap=("total_mv", "last"),
        pb=("pb", "last"),
    )
    mb = mb[mb["mktcap"] > 0]

    # 月度前复权行情 -> 月度收益
    mpx = jh.get_data(DataTypes.TS_MONTHLY_QFQ, start=CAP_START, end=RET_END).to_df()
    mpx = mpx[["trade_date", "ts_code", "close"]]
    mpx["trade_date"] = pd.to_datetime(mpx["trade_date"])
    mpx = mpx[mpx["close"] > 0]
    mpx = mpx.sort_values(["ts_code", "trade_date"])
    mpx["ret"] = mpx.groupby("ts_code")["close"].pct_change()

    # 无风险利率：TS_SHIBOR 1M（月末值，月化；akshare SHIBOR 已停更）
    shibor = jh.get_data(DataTypes.TS_SHIBOR,
                         start="2015-01-01", end="2024-12-31").to_df()
    shibor = shibor[["date", "f_1m"]]
    shibor["date"] = pd.to_datetime(shibor["date"])
    shibor["ym"] = shibor["date"].dt.to_period("M")
    rf = shibor.sort_values("date").groupby("ym")["f_1m"].last().astype(float) / 100 / 12
    rf = rf.rename("rf")

    return basic, mb, mpx, rf


def build_panel(basic, mb, mpx):
    """构建月度面板：收益 + 滞后一期市值/BM + 剔次新。"""
    ret = mpx[(mpx["trade_date"] >= RET_START)].dropna(subset=["ret"]).copy()
    ret["ym"] = ret["trade_date"].dt.to_period("M")

    # 市值 / BM 滞后一期（t-1 月末已知 -> t 月收益）
    mb["bm"] = 1.0 / mb["pb"]  # 账面市值比 = 1 / 市净率
    ret = ret.merge(
        mb[["ts_code", "ym", "mktcap", "bm"]].rename(columns={"ym": "ym_prev"}),
        left_on=["ts_code", ret["ym"] - 1], right_on=["ts_code", "ym_prev"],
        how="left", suffixes=("", "_cap"),
    )
    ret = ret.dropna(subset=["mktcap", "bm"])
    ret = ret[["trade_date", "ym", "ts_code", "ret", "mktcap", "bm"]]

    ret = ret.merge(basic, on="ts_code", how="left")
    ret = ret[ret["trade_date"] - ret["list_date"] >= pd.Timedelta(days=365)]
    ret = ret.drop(columns=["list_date"])
    return ret


def vw_return(group):
    """市值加权收益：Σ(ret × mktcap) / Σ(mktcap)，权重 clip 到 >= 0。"""
    w = group["mktcap"].clip(lower=0)
    return (group["ret"] * w).sum() / w.sum() if w.sum() > 0 else group["ret"].mean()


def assign_size2(s):
    """横截面中位数二分：返回 0(小盘) / 1(大盘)，缺失给 NaN。"""
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    clean = s.dropna()
    if clean.empty:
        return out
    med = clean.median()
    out.loc[clean.index] = (clean > med).astype(float)
    return out


def assign_bm3(s):
    """横截面 BM 三分位：返回 1(成长) / 2(中) / 3(价值)，缺失给 NaN。"""
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    clean = s.dropna()
    if clean.empty:
        return out
    q = clean.quantile(BM_BREAKS)
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
    basic, mb, mpx, rf = fetch_data()
    print(f"    月度基本面: {len(mb)} 行, 行情: {len(mpx)} 行, SHIBOR: {len(rf)} 期")

    print(">>> 2/4 构建面板 + 2x3 独立排序 ...")
    panel = build_panel(basic, mb, mpx)
    print(f"    面板: {len(panel)} 行, 股票 {panel['ts_code'].nunique()} 只, "
          f"月度 {panel['ym'].nunique()} 期")

    # 每月横截面：市值二分(0小/1大) × BM 三分(1成长..3价值)
    panel["size_g"] = panel.groupby("ym")["mktcap"].transform(assign_size2)
    panel["bm_g"] = panel.groupby("ym")["bm"].transform(assign_bm3)
    panel = panel.dropna(subset=["size_g", "bm_g"])

    # 每月算 6 个组合的市值加权收益，以及全市场市值加权收益
    monthly, mkt_map = {}, {}
    for ym, grp in panel.groupby("ym"):
        mat = np.full((2, 3), np.nan)  # 行=size 0小/1大, 列=bm 1成长..3价值
        for (sg, bg), sub in grp.groupby(["size_g", "bm_g"]):
            mat[int(sg), int(bg) - 1] = vw_return(sub)
        monthly[ym] = mat
        mkt_map[ym] = vw_return(grp)

    ym_index = sorted(monthly.keys())
    arr = np.stack([monthly[y] for y in ym_index])  # (期, 2, 3)

    # 三因子收益序列
    smb = arr[:, 0, :].mean(axis=1) - arr[:, 1, :].mean(axis=1)   # 小盘层 - 大盘层
    hml = arr[:, :, 2].mean(axis=1) - arr[:, :, 0].mean(axis=1)   # 高BM层 - 低BM层
    mkt_ex = pd.Series([mkt_map[y] for y in ym_index], index=ym_index) - rf.reindex(ym_index)
    factors = pd.DataFrame({
        "mkt": mkt_ex.values, "smb": smb, "hml": hml,
    }, index=ym_index)

    # 2x3 组合平均月收益矩阵（%）
    mean_mat = np.nanmean(arr, axis=0) * 100
    print("\n2x3 组合平均月收益（%，行=市值 小/大, 列=BM 成长->价值）：")
    print(pd.DataFrame(mean_mat, index=["S 小盘", "B 大盘"],
                       columns=["L 成长", "M", "H 价值"]).round(2).to_string())

    print("\n三因子月度统计（2015-2024，120 个月）：")
    stats = []
    for name, cn in [("MKT", "mkt"), ("SMB", "smb"), ("HML", "hml")]:
        s = factors[cn].dropna()
        mean = s.mean() * 100
        std = s.std(ddof=1) * 100
        t = newey_west_t(s.values, s.mean())
        stats.append({"因子": name, "均值(%/月)": round(mean, 2),
                      "标准差(%/月)": round(std, 2), "NW t 值": round(t, 2)})
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    print(">>> 3/4 绘图（mpl_style 统一风格）...")
    # 图1：三因子累计净值
    fig1, ax = plt.subplots(figsize=(11, 5.5))
    labels = {
        "mkt": "MKT 市场因子", "smb": "SMB 规模因子（小−大）",
        "hml": "HML 价值因子（高BM−低BM）",
    }
    colors = {"mkt": mpl_style.ACCENT, "smb": mpl_style.RISE, "hml": mpl_style.FALL}
    for col in ["mkt", "smb", "hml"]:
        s = (factors[col] + 1).cumprod()
        ax.plot([str(y) for y in ym_index], s.values, lw=1.8, color=colors[col],
                label=labels[col])
    ax.axhline(1.0, color="#7F8C8D", lw=1, ls="--")
    ax.set_ylabel("累计净值（起始 = 1）")
    ax.set_xlabel("月份")
    ax.set_title("Fama-French 三因子累计净值（A 股 2015–2024）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, len(ym_index), 12))
    ax.set_xticklabels([str(y)[:4] for y in ym_index[::12]], rotation=45)
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("output/fig1_ff3_cumulative.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_ff3_cumulative.png")

    # 图2：2x3 组合平均月收益热力图（SMB/HML 的来源）
    fig2, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(mean_mat, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=3.0)
    for i in range(2):
        for j in range(3):
            v = mean_mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if abs(v - np.nanmean(mean_mat)) > 0.8 else "black",
                    fontsize=11)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["L\n成长", "M", "H\n价值"])
    ax.set_yticks(range(2))
    ax.set_yticklabels(["S 小盘", "B 大盘"])
    ax.set_xlabel("账面市值比（低 BM → 高 BM）")
    ax.set_ylabel("市值")
    ax.set_title("6 个组合平均月收益（%）：SMB 与 HML 的来源", fontsize=14, fontweight="bold")
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_ff3_portfolios.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_ff3_portfolios.png")

    print("\n完成。两张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
