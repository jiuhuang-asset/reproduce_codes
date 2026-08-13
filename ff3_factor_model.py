# -*- coding: utf-8 -*-
"""
Fama-French 三因子模型：从二维排序到构造、验证与归因（A股月度实证）
====================================================================

对应公众号文章《价值因子与三因子模型》（量海泛舟）
方法论参考：
  - https://www.tidy-finance.org/chapters/value-and-bivariate-sorts.html
  - https://www.tidy-finance.org/chapters/replicating-fama-and-french-factors.html

内容分三部分：
  Part 1 — 价值因子与二维排序（5x5）
      把全 A 股按「市值」和「账面市值比 BM=1/PB」同时分成 5x5=25 个组合，
      看价值效应与规模效应是否同时存在（市值加权）。
  Part 2 — FF3 三因子构造（2x3，CLASSIC 口径）
      市值中位数二分 × BM 30%/70% 三分，构造 SMB / HML / MKT 三条因子序列。
  Part 3 — 复刻质量验证 + 个股归因
      (a) 手写因子与 jh_quant 库内置 FF3 做一元回归对比，检验复刻是否可靠；
      (b) 用 FF3 给个股做时序归因回归，看 alpha 与因子载荷。

关键设计（三部分一致）
----------------------
- 月度再平衡：t 月末的市值/BM 分组，持有 t+1 月收益（滞后一期，避免前视偏差）
- BM 用月末市净率 PB 的倒数（TS_MONTHLY_BASIC 的 pb 字段）——A 股无 Compustat
  账面值，这是通行代理口径，与官方（真实账面权益）存在差异
- 剔除上市不足 12 个月的次新股
- 组合内市值加权；SMB/HML 定义为组合间等权平均

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python ff3_factor_model.py

输出
----
控制台打印 5x5 平均月收益矩阵、价值/市值溢价、2x3 组合矩阵、三因子统计表、
复刻对比回归表、3 只股票归因表，并生成 7 张图到 output/ 子目录：
  Part 1:
    - fig1_bm_matrix.png          5x5 平均月收益热力图
    - fig2_vmg_cumulative.png     价值多空(VMG) vs 市值多空(SMB)累计净值
    - fig3_cap_share_matrix.png   5x5 总市值占比热力图
  Part 2:
    - fig4_ff3_portfolios.png     2x3 组合平均月收益热力图
    - fig5_ff3_cumulative.png     三因子累计净值曲线
  Part 3:
    - fig6_replication_compare.png 手写 vs 库版 SMB/HML 散点（45 度线）
    - fig7_ff3_regression.png      3 只股票 alpha 对比条形图
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style，保证公众号文章图表一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RET_START, RET_END = "2015-01-01", "2026-06-30"
CAP_START = "2014-12-01"  # 多拉一个月，供滞后一期用

# 2x3 排序的分位断点（CLASSIC 口径）
SIZE_BREAKS = [0.5]            # 市值中位数二分
BM_BREAKS = [0.3, 0.7]         # BM 30%/70% 三分

# 用于个股归因的 3 只代表性股票（含名字方便展示）
STOCKS = [
    ("000001.SZ", "平安银行"),
    ("600519.SH", "贵州茅台"),
    ("300750.SZ", "宁德时代"),
]


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

    # 无风险利率：TS_SHIBOR 1M（月末值，月化；bypass_cache 绕开服务端 DDL 列名不一致）
    shibor = jh.get_data(DataTypes.TS_SHIBOR,
                         start="2015-01-01", end="2026-06-30",
                         bypass_cache=True).to_df()
    shibor = shibor[["date", "1m"]]
    shibor["date"] = pd.to_datetime(shibor["date"])
    shibor["ym"] = shibor["date"].dt.to_period("M")
    rf = shibor.sort_values("date").groupby("ym")["1m"].last().astype(float) / 100 / 12
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


def assign_quintile(s):
    """横截面五分位：返回 1(最小/最便宜) … 5(最大/最贵)。"""
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    clean = s.dropna()
    if clean.empty:
        return out
    q = clean.quantile(np.linspace(0.2, 0.8, 4))
    labels = pd.cut(clean, bins=[-np.inf, *q.values, np.inf],
                    labels=False, duplicates="drop") + 1
    out.loc[labels.index] = labels
    return out


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


def construct_ff3(panel, rf):
    """2x3 独立排序构造 SMB/HML/MKT（CLASSIC 口径）。返回 (factors, ym_index, arr)。"""
    panel = panel.dropna(subset=["mktcap", "bm"]).copy()
    panel["size_g"] = panel.groupby("ym")["mktcap"].transform(assign_size2)
    panel["bm_g"] = panel.groupby("ym")["bm"].transform(assign_bm3)
    panel = panel.dropna(subset=["size_g", "bm_g"])

    monthly, mkt_map = {}, {}
    for ym, grp in panel.groupby("ym"):
        mat = np.full((2, 3), np.nan)  # 行=size 0小/1大, 列=bm 1成长..3价值
        for (sg, bg), sub in grp.groupby(["size_g", "bm_g"]):
            mat[int(sg), int(bg) - 1] = vw_return(sub)
        monthly[ym] = mat
        mkt_map[ym] = vw_return(grp)

    ym_index = sorted(monthly.keys())
    arr = np.stack([monthly[y] for y in ym_index])  # (期, 2, 3)

    smb = arr[:, 0, :].mean(axis=1) - arr[:, 1, :].mean(axis=1)   # 小盘层 - 大盘层
    hml = arr[:, :, 2].mean(axis=1) - arr[:, :, 0].mean(axis=1)   # 高BM层 - 低BM层
    mkt_ex = pd.Series([mkt_map[y] for y in ym_index], index=ym_index) - rf.reindex(ym_index)
    factors = pd.DataFrame({"mkt": mkt_ex.values, "smb": smb, "hml": hml}, index=ym_index)
    return factors, ym_index, arr


def ols_beta(x, y):
    """一元回归 y = a + b·x，返回 (b, a, r2, n)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) < 5:
        return np.nan, np.nan, np.nan, len(y)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1.0 - resid.var() / y.var()
    return b, a, r2, len(y)


def ols_pvalue(x, y):
    """一元回归斜率/截距的 t 值（普通 OLS 标准误）。返回 (t_intercept, t_slope)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta[0] / se[0], beta[1] / se[1]


def ff3_regression(ret_excess, factors):
    """个股 FF3 时序回归：ret_excess ~ mkt + smb + hml。返回 [alpha, bmkt, bsmb, bhml]。"""
    ret_excess = ret_excess.rename("ret").to_frame()
    df = pd.concat([ret_excess, factors[["mkt", "smb", "hml"]]], axis=1).dropna()
    y = df["ret"].values
    X = np.column_stack([np.ones(len(df)), df["mkt"].values, df["smb"].values, df["hml"].values])
    if len(df) < 10:
        return np.full(4, np.nan)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta  # [alpha, bmkt, bsmb, bhml]


# ============================================================
# Part 1：价值因子与二维排序（5x5）
# ============================================================

def part1_bivariate(panel):
    """5x5 市值×BM 二维排序 + 3 张图。"""
    print(">>> [Part 1] 5x5 二维排序（市值 × 账面市值比）...")
    panel = panel.copy()
    panel["size_q"] = panel.groupby("ym")["mktcap"].transform(assign_quintile)
    panel["bm_q"] = panel.groupby("ym")["bm"].transform(assign_quintile)
    panel = panel.dropna(subset=["size_q", "bm_q"])
    # size_q=5 大盘 / size_q=1 小盘；bm_q=5 最便宜(价值) / bm_q=1 最贵(成长)

    monthly = {}
    for ym, grp in panel.groupby("ym"):
        if grp.empty:
            continue
        mat = np.full((5, 5), np.nan)
        for (sq, bq), sub in grp.groupby(["size_q", "bm_q"]):
            w = sub["mktcap"].clip(lower=0)
            mat[int(sq) - 1, int(bq) - 1] = (sub["ret"] * w).sum() / w.sum() if w.sum() > 0 else sub["ret"].mean()
        monthly[ym] = mat
    arr = np.stack(list(monthly.values()))  # (月, 5, 5)，行=size_q 1..5, 列=bm_q 1..5

    mean_mat = np.nanmean(arr, axis=0) * 100
    print("\n5x5 平均月收益矩阵（%，行=size 小->大, 列=bm 成长->价值）：")
    print(pd.DataFrame(mean_mat, index=[f"S{i}" for i in range(1, 6)],
                       columns=[f"B{i}" for i in range(1, 6)]).round(2).to_string())

    # 价值溢价 VMG = 高BM层 - 低BM层（每层 5 个 size 组合等权平均）
    layer_ret = np.nanmean(arr, axis=1)  # (月, bm_q 1..5)
    vmg = layer_ret[:, 4] - layer_ret[:, 0]
    vmg_mean = vmg.mean() * 100
    vmg_t = vmg.mean() / (vmg.std(ddof=1) / np.sqrt(len(vmg)))
    print(f"\n价值溢价 VMG = 价值层 - 成长层：")
    print(f"  平均月收益 {vmg_mean:.2f}% | t 值 {vmg_t:.2f}（样本 {len(vmg)} 个月）")

    # 市值溢价 SMB = 小盘层 - 大盘层（每层 5 个 BM 组合等权平均）
    size_layer = np.nanmean(arr, axis=2)  # (月, size_q 1..5)
    smb = size_layer[:, 0] - size_layer[:, 4]
    smb_mean = smb.mean() * 100
    print(f"  同样本期市值溢价 SMB = {smb_mean:.2f}%")

    # 总市值占比矩阵（%）
    cap_sum = panel.groupby(["size_q", "bm_q"])["mktcap"].sum()
    cap_share = cap_sum / cap_sum.sum() * 100
    cap_mat = np.zeros((5, 5))
    for (sq, bq), v in cap_share.items():
        cap_mat[int(sq) - 1, int(bq) - 1] = v
    print("\n5x5 总市值占比矩阵（%，行=size 小->大, 列=bm 成长->价值）：")
    print(pd.DataFrame(cap_mat, index=[f"S{i}" for i in range(1, 6)],
                       columns=[f"B{i}" for i in range(1, 6)]).round(1).to_string())

    # 图1：5x5 平均月收益热力图
    fig1, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(mean_mat, cmap="RdBu_r", vmin=-1.5, vmax=3.0, aspect="auto")
    for i in range(5):
        for j in range(5):
            v = mean_mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if abs(v - np.nanmean(mean_mat)) > 0.8 else "black", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels(["B1\n成长", "B2", "B3", "B4", "B5\n价值"])
    ax.set_yticks(range(5)); ax.set_yticklabels(["S1\n小盘", "S2", "S3", "S4", "S5\n大盘"])
    ax.set_xlabel("账面市值比（B1 成长 → B5 价值）")
    ax.set_ylabel("市值（S1 小盘 → S5 大盘）")
    ax.set_title("5×5 二维排序的平均月收益（% ，2015-2026）", fontsize=14, fontweight="bold")
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("output/fig1_bm_matrix.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_bm_matrix.png")

    # 图2：价值多空(VMG) vs 市值多空(SMB) 累计净值
    vmg_s = pd.Series(vmg, index=sorted(monthly.keys())).sort_index()
    smb_s = pd.Series(smb, index=sorted(monthly.keys())).sort_index()
    fig2, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(vmg_s.index.astype(str), (vmg_s + 1).cumprod(), lw=1.8,
            color=mpl_style.COLOR_CYCLE[1], label=f"价值溢价 VMG（月 {vmg_mean:.2f}%）")
    ax.plot(smb_s.index.astype(str), (smb_s + 1).cumprod(), lw=1.8,
            color=mpl_style.COLOR_CYCLE[0], label=f"市值溢价 SMB（月 {smb_mean:.2f}%）")
    ax.axhline(1.0, color="#7F8C8D", lw=1, ls="--")
    ax.set_ylabel("累计净值（起始 = 1）")
    ax.set_xlabel("月份")
    ax.set_title("控制另一个维度后：价值溢价 vs 市值溢价", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ticks = vmg_s.index[::12]
    ax.set_xticks(range(0, len(vmg_s), 12))
    ax.set_xticklabels([str(t)[:4] for t in ticks], rotation=45)
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_vmg_cumulative.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_vmg_cumulative.png")

    # 图3：5x5 总市值占比热力图
    fig3, ax = plt.subplots(figsize=(8, 6.5))
    im3 = ax.imshow(cap_mat, cmap="YlOrBr", aspect="auto")
    for i in range(5):
        for j in range(5):
            v = cap_mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 20 else "black", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels(["B1\n成长", "B2", "B3", "B4", "B5\n价值"])
    ax.set_yticks(range(5)); ax.set_yticklabels(["S1\n小盘", "S2", "S3", "S4", "S5\n大盘"])
    ax.set_xlabel("账面市值比（B1 成长 → B5 价值）")
    ax.set_ylabel("市值（S1 小盘 → S5 大盘）")
    ax.set_title("5×5 组合占总市值比例（%）", fontsize=14, fontweight="bold")
    fig3.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("output/fig3_cap_share_matrix.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_cap_share_matrix.png")


# ============================================================
# Part 2：FF3 三因子构造（2x3）
# ============================================================

def part2_construct(factors, ym_index, arr):
    """打印 2x3 组合矩阵与三因子统计表 + 2 张图。"""
    print(">>> [Part 2] FF3 三因子统计 ...")
    mean_mat = np.nanmean(arr, axis=0) * 100
    print("\n2x3 组合平均月收益（%，行=市值 小/大, 列=BM 成长->价值）：")
    print(pd.DataFrame(mean_mat, index=["S 小盘", "B 大盘"],
                       columns=["L 成长", "M", "H 价值"]).round(2).to_string())

    print("\n三因子月度统计（2015-2026）：")
    stats = []
    for name, cn in [("MKT", "mkt"), ("SMB", "smb"), ("HML", "hml")]:
        s = factors[cn].dropna()
        mean = s.mean() * 100
        std = s.std(ddof=1) * 100
        t = newey_west_t(s.values, s.mean())
        stats.append({"因子": name, "均值(%/月)": round(mean, 2),
                      "标准差(%/月)": round(std, 2), "NW t 值": round(t, 2)})
    print(pd.DataFrame(stats).to_string(index=False))

    # 图4：2x3 组合平均月收益热力图（SMB/HML 的来源，先看它）
    fig4, ax = plt.subplots(figsize=(8, 4.6))
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
    fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    mpl_style.hide_spines(ax)
    fig4.tight_layout()
    fig4.savefig("output/fig4_ff3_portfolios.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig4_ff3_portfolios.png")

    # 图5：三因子累计净值
    fig5, ax = plt.subplots(figsize=(11, 5.5))
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
    ax.set_title("Fama-French 三因子累计净值（A 股 2015–2026）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, len(ym_index), 12))
    ax.set_xticklabels([str(y)[:4] for y in ym_index[::12]], rotation=45)
    mpl_style.hide_spines(ax)
    fig5.tight_layout()
    fig5.savefig("output/fig5_ff3_cumulative.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig5_ff3_cumulative.png")


# ============================================================
# Part 3：复刻质量验证 + 个股归因
# ============================================================

def part3_validate(panel, rf, factors, ym_index):
    """手写 vs 库版对比 + 个股归因 + 2 张图。"""
    print(">>> [Part 3] 复刻质量验证 + 个股归因 ...")
    # 库版 FF3（jh_quant.factors，CLASSIC）
    from jh_quant.factors import load_ts_factor_inputs, calculate_factor_returns

    inputs = load_ts_factor_inputs(
        start_date=RET_START, end_date=RET_END, period="M", lag_features=True,
    )
    lib = calculate_factor_returns("ff3", **inputs, method="classic", n_jobs=1,
                                   use_polars=True, verbose=False)
    lib.index = pd.PeriodIndex(lib.index, freq="M")
    lib = lib.reindex(ym_index)
    print(f"    库版: {len(lib)} 期")

    # 对齐后对比 SMB/HML（MKT 因库版为等权口径，不参与对比）
    cmp = pd.DataFrame({
        "smb_hand": factors["smb"], "smb_lib": lib["smb"],
        "hml_hand": factors["hml"], "hml_lib": lib["hml"],
    }).dropna()

    print("\n复刻对比（手写版 = 复刻, 库版 = 参考）：")
    rows = []
    for cn in ["smb", "hml"]:
        x = cmp[f"{cn}_lib"].values
        y = cmp[f"{cn}_hand"].values
        b, a, r2, n = ols_beta(x, y)
        t_int, t_slp = ols_pvalue(x, y)
        corr = np.corrcoef(x, y)[0, 1]
        rows.append({"因子": cn.upper(), "斜率b": round(b, 3), "R²": round(r2, 3),
                     "相关系数": round(corr, 3), "截距a(%/月)": round(a * 100, 3),
                     "截距t": round(t_int, 2), "样本月": n})
    cmp_df = pd.DataFrame(rows)
    print(cmp_df.to_string(index=False))

    # 个股 FF3 归因
    ret_panel = panel[["ym", "ts_code", "ret"]]
    attr_rows = []
    for ts_code, name in STOCKS:
        r = ret_panel[ret_panel["ts_code"] == ts_code].set_index("ym")["ret"]
        excess = r - rf.reindex(r.index)
        alpha, bmkt, bsmb, bhml = ff3_regression(excess, factors)
        attr_rows.append({
            "股票": name, "代码": ts_code, "Alpha(%/月)": round(alpha * 100, 3),
            "b_MKT": round(bmkt, 2), "b_SMB": round(bsmb, 2), "b_HML": round(bhml, 2),
            "样本月": len(excess.dropna()),
        })
    attr_df = pd.DataFrame(attr_rows)
    print("\n个股 FF3 归因（3 只代表性股票）：")
    print(attr_df.to_string(index=False))

    # 图6：手写 vs 库版 SMB/HML 散点 + 45 度线（手机端优先：两张纵向堆叠）
    fig6, axes = plt.subplots(2, 1, figsize=(8, 12))
    for ax, cn, label in zip(axes, ["smb", "hml"], ["SMB", "HML"]):
        x = cmp[f"{cn}_lib"].values * 100
        y = cmp[f"{cn}_hand"].values * 100
        ax.scatter(x, y, s=14, alpha=0.5, color=mpl_style.ACCENT)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, ls="--", color="#7F8C8D", lw=1, label="45° 线（完全一致）")
        ax.set_xlabel(f"jh_quant 库版 {label}（%/月）")
        ax.set_ylabel(f"手写版 {label}（%/月）")
        ax.set_title(f"{label} 复刻对比", fontsize=13)
        ax.legend(fontsize=9)
        mpl_style.hide_spines(ax)
    fig6.suptitle("手写 FF3 与 jh_quant 库版对比（A 股 2015–2026）", fontsize=14, fontweight="bold")
    fig6.tight_layout(rect=[0, 0, 1, 0.95])
    fig6.savefig("output/fig6_replication_compare.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig6_replication_compare.png")

    # 图7：3 只股票 alpha 对比
    fig7, ax = plt.subplots(figsize=(9, 4.6))
    names = [row["股票"] for row in attr_rows]
    alphas = [row["Alpha(%/月)"] for row in attr_rows]
    colors = [mpl_style.RISE if a > 0 else mpl_style.FALL for a in alphas]
    bars = ax.bar(names, alphas, color=colors, width=0.5)
    for bar, a in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width() / 2, a + (0.02 if a >= 0 else -0.02),
                f"{a:.2f}", ha="center", fontsize=10)
    ax.axhline(0, color="#7F8C8D", lw=1)
    ax.set_ylabel("Alpha（%/月）")
    ax.set_xlabel("股票")
    ax.set_title("三只 A 股的 FF3 回归 Alpha（>0 跑赢因子模型）", fontsize=14, fontweight="bold")
    mpl_style.hide_spines(ax)
    fig7.tight_layout()
    fig7.savefig("output/fig7_ff3_regression.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig7_ff3_regression.png")


def main():
    os.makedirs("output", exist_ok=True)
    print(">>> 拉取数据（jh_quant 月度表）...")
    basic, mb, mpx, rf = fetch_data()
    print(f"    月度基本面: {len(mb)} 行, 行情: {len(mpx)} 行, SHIBOR: {len(rf)} 期")

    panel = build_panel(basic, mb, mpx)
    print(f"    面板: {len(panel)} 行, 股票 {panel['ts_code'].nunique()} 只, "
          f"月度 {panel['ym'].nunique()} 期")

    factors, ym_index, arr = construct_ff3(panel, rf)

    part1_bivariate(panel)
    print()
    part2_construct(factors, ym_index, arr)
    print()
    part3_validate(panel, rf, factors, ym_index)
    print("\n完成。7 张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
