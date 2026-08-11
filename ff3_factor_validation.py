# -*- coding: utf-8 -*-
"""
FF3 三因子：复刻质量验证 + 个股归因（A股月度实证）
====================================================

对应公众号文章《FF3 三因子复刻 · 下：验证与应用》（量海泛舟 · 资产定价系列第七篇）
方法论参考：https://www.tidy-finance.org/chapters/replicating-fama-and-french-factors.html

做什么
------
上一篇我们用自己写的代码构造了 MKT/SMB/HML（ff3_factor_construction.py）。
这篇做两件事：

  1. 复刻质量验证：把手写的 SMB/HML，与 jh_quant 库内置的经典 FF3 输出
     做一元回归对比（y = a + b·x）。如果 b 接近 1、R² 接近 1、截距 a 不显著，
     说明两条独立实现"对得上"，复刻是可靠的。这正是 tidy-finance 官方复刻
     的验证套路（官方 vs 我写的）。
  2. 个股归因：用 FF3 做时间序列回归
        R_i - R_f = alpha + b_MKT·MKT + b_SMB·SMB + b_HML·HML + eps
     看 3 只代表性 A 股的 alpha（跑赢/跑输三因子模型的程度）与因子载荷。

关键设计
--------
- 手写版复用上一篇逻辑（月度再平衡、滞后一期、2x3 独立排序、市值加权、剔次新）
- 库版用 jh_quant.factors 的 load_ts_factor_inputs + calculate_factor_returns
  (method="classic")，口径与 Fama-French 1993 一致
- 对比聚焦 SMB/HML（两版口径一致）；MKT 因库版为等权口径，不参与对比
- 个股回归用 6 篇构造的手写因子 + SHIBOR 无风险利率

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python ff3_factor_validation.py

输出
----
控制台打印 SMB/HML 复刻对比回归表、3 只股票 FF3 归因表，
并生成 2 张图：
  - fig1_replication_compare.png  手写 vs 库版 SMB/HML 散点（45 度线）
  - fig2_ff3_regression.png       3 只股票 alpha 对比条形图
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 复用第六篇的手写构造函数（同仓库，保证口径一致）
from ff3_factor_construction import (  # noqa: E402
    RET_START, RET_END, fetch_data, build_panel,
    vw_return, assign_size2, assign_bm3, newey_west_t,
)

# 用于个股归因的 3 只代表性股票（含名字方便展示）
STOCKS = [
    ("000001.SZ", "平安银行"),
    ("600519.SH", "贵州茅台"),
    ("300750.SZ", "宁德时代"),
]


def hand_ff3(panel, rf):
    """手写版 FF3：2x3 独立排序 + 市值加权（复用第六篇逻辑）。返回 (factors, ym_index)。"""
    panel = panel.dropna(subset=["mktcap", "bm"]).copy()
    panel["size_g"] = panel.groupby("ym")["mktcap"].transform(assign_size2)
    panel["bm_g"] = panel.groupby("ym")["bm"].transform(assign_bm3)
    panel = panel.dropna(subset=["size_g", "bm_g"])

    monthly, mkt_map = {}, {}
    for ym, grp in panel.groupby("ym"):
        mat = np.full((2, 3), np.nan)
        for (sg, bg), sub in grp.groupby(["size_g", "bm_g"]):
            mat[int(sg), int(bg) - 1] = vw_return(sub)
        monthly[ym] = mat
        mkt_map[ym] = vw_return(grp)

    ym_index = sorted(monthly.keys())
    arr = np.stack([monthly[y] for y in ym_index])
    smb = arr[:, 0, :].mean(axis=1) - arr[:, 1, :].mean(axis=1)
    hml = arr[:, :, 2].mean(axis=1) - arr[:, :, 0].mean(axis=1)
    mkt_ex = pd.Series([mkt_map[y] for y in ym_index], index=ym_index) - rf.reindex(ym_index)
    factors = pd.DataFrame({"mkt": mkt_ex.values, "smb": smb, "hml": hml}, index=ym_index)
    return factors, ym_index


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
    """个股 FF3 时序回归：ret_excess ~ mkt + smb + hml。返回 (alpha, bmkt, bsmb, bhml, r2)。"""
    ret_excess = ret_excess.rename("ret").to_frame()
    df = pd.concat([ret_excess, factors[["mkt", "smb", "hml"]]], axis=1).dropna()
    y = df["ret"].values
    X = np.column_stack([np.ones(len(df)), df["mkt"].values, df["smb"].values, df["hml"].values])
    if len(df) < 10:
        return np.full(5, np.nan)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - resid.var() / y.var()
    return beta  # [alpha, bmkt, bsmb, bhml] + r2


def main():
    os.makedirs("output", exist_ok=True)
    print(">>> 1/5 拉取数据（jh_quant 月度表）...")
    basic, mb, mpx, rf = fetch_data()
    print(f"    月度基本面: {len(mb)} 行, 行情: {len(mpx)} 行, SHIBOR: {len(rf)} 期")

    print(">>> 2/5 手写版 FF3 因子 ...")
    panel = build_panel(basic, mb, mpx)
    hand, ym_index = hand_ff3(panel, rf)
    print(f"    手写版: {len(hand)} 期")

    print(">>> 3/5 库版 FF3 因子（jh_quant.factors，CLASSIC）...")
    from jh_quant.factors import load_ts_factor_inputs, calculate_factor_returns

    inputs = load_ts_factor_inputs(
        start_date=RET_START, end_date=RET_END, period="M", lag_features=True,
    )
    lib = calculate_factor_returns("ff3", **inputs, method="classic", n_jobs=1,
                                   use_polars=True, verbose=False)
    lib.index = pd.PeriodIndex(lib.index, freq="M")
    lib = lib.reindex(ym_index)
    print(f"    库版: {len(lib)} 期")

    # 对齐后对比 SMB/HML
    cmp = pd.DataFrame({
        "smb_hand": hand["smb"], "smb_lib": lib["smb"],
        "hml_hand": hand["hml"], "hml_lib": lib["hml"],
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

    print(">>> 4/5 个股 FF3 归因（3 只代表性股票）...")
    # 构造个股超额收益（与手写因子同月对齐）
    ret_panel = panel[["ym", "ts_code", "ret"]]
    attr_rows = []
    for ts_code, name in STOCKS:
        r = ret_panel[ret_panel["ts_code"] == ts_code].set_index("ym")["ret"]
        excess = r - rf.reindex(r.index)
        alpha, bmkt, bsmb, bhml = ff3_regression(excess, hand)
        attr_rows.append({
            "股票": name, "代码": ts_code, "Alpha(%/月)": round(alpha * 100, 3),
            "b_MKT": round(bmkt, 2), "b_SMB": round(bsmb, 2), "b_HML": round(bhml, 2),
            "样本月": len(excess.dropna()),
        })
    attr_df = pd.DataFrame(attr_rows)
    print(attr_df.to_string(index=False))

    print(">>> 5/5 绘图（mpl_style 统一风格）...")
    # 图1：手写 vs 库版 SMB/HML 散点 + 45 度线
    fig1, axes = plt.subplots(1, 2, figsize=(13, 5.2))
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
    fig1.suptitle("手写 FF3 与 jh_quant 库版对比（A 股 2015–2024）", fontsize=14, fontweight="bold")
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig("output/fig1_replication_compare.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_replication_compare.png")

    # 图2：3 只股票 alpha 对比
    fig2, ax = plt.subplots(figsize=(9, 4.6))
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
    fig2.tight_layout()
    fig2.savefig("output/fig2_ff3_regression.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_ff3_regression.png")

    print("\n完成。两张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
