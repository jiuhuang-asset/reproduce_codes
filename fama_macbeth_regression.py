# -*- coding: utf-8 -*-
"""
Fama-MacBeth 回归：检验哪些特征真的被定价（A股月度实证）
========================================================

对应公众号文章《因子真的被定价了吗？Fama-MacBeth 回归》（量海泛舟）
方法论参考：https://www.tidy-finance.org/chapters/fama-macbeth-regressions.html

做什么
------
前几篇文章把 FF3 因子造出来、验证了，但还有一个根本问题没回答：
这些风险特征（市场 beta、市值、账面市值比）真的"被定价"了吗？收益真的随它们
系统性变化吗？

Fama-MacBeth (1973) 两步法就是干这个的：

  Step 0 估计每只股票的月度特征（全部滞后一期，避免前视）：
     - beta：滚动 36 个月的 CAPM 市场 beta（窗口截止到 t-1）
     - log_mktcap：对数市值
     - bm：账面市值比（1/PB，与前文同口径）
  Step 1 每个月做一次横截面回归：
     R_i,t - R_f,t = a_t + lambda * 特征_i,t + eps_i,t
     得到每个月的"特征风险溢价" lambda_t（每个特征单独回归，FM 1973 原意）
  Step 2 把每个 lambda 在时间序列上平均，检验均值是否显著不为零
     （普通 t + Newey-West t，月度惯例 lag=3）：
     lambda = mu + u,  H0: mu = 0

  lambda 显著不为零 = 该特征被市场定价（收益随该特征系统性变化）。

结果（2015-2026 A股，诚实版）
------------------------------
- Beta：从未被定价（各规格、各窗口 |NW t| 都 < 1）
- log 市值：全样本显著为负（NW t ≈ -2.3）——小盘溢价真实存在；
  但同一窗口（2017 起）下显著性退潮，说明溢价主要来自 2015-2017 小盘行情
- BM：方向始终为正（价值溢价），但统计上不显著（NW t ≈ 0.7）

关键设计（延续系列惯例）
------------------------
- 特征全部滞后 / 用 t-1 及以前数据：beta 滚动窗口截止 t-1，市值/BM 取 t-1 月末值
- 特征先在每月横截面内做 1%/99% winsorize，再 Z-Score 标准化，
  这样 lambda 跨特征可比（= 该特征每增加 1 个标准差对应的月收益）
- 剔次新、市值加权、月度再平衡，与前文完全一致
- Newey-West 标准误修正 lambda 的序列自相关

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python fama_macbeth_regression.py

输出
----
控制台打印单特征风险溢价表（各自最全样本）、同一窗口稳健性表、三特征联合回归表，
并生成 2 张图：
  - fig1_risk_premia.png    单特征风险溢价柱状图（NW 误差棒 + 显著性标注）
  - fig2_rolling_lambda.png 每月横截面 lambda 时序（可见 beta 检验起点更晚）
最后附一段 FactorSelector 演示：jh_quant.backtest.FactorSelector 把 FM 验证
（validate_factor(method="fama_macbeth")）嵌进因子选股，用 mean_lambda 当权重。
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from ff3_factor_model import (  # noqa: E402
    fetch_data, build_panel,
)

LOOKBACK = 36    # 滚动估计 beta 的窗口长度（月）
MIN_OBS = 24     # 窗口内最少有效观测月数
NW_LAG = 3       # Newey-West 滞后阶数（月度惯例）
MIN_STOCKS = 50  # 每月横截面回归最少股票数
WINSOR = 0.01    # 横截面 winsorize 分位


def estimate_rolling_beta(ret_piv, mkt_ex):
    """滚动 36 月 CAPM beta。

    先用 shift(1) 把收益整体滞后一期再做滚动窗口，因此 beta_t 只用到
    t-1 及以前的数据：beta_t = cov(ret, mkt)[t-36..t-1] / var(mkt)[t-36..t-1]。
    """
    y = ret_piv.shift(1)              # 股票收益滞后一期
    xs = mkt_ex.shift(1)              # 市场超额收益滞后一期

    # 把市场序列广播成与 ret_piv 同形的 DataFrame，避免 Series/DataFrame 错位对齐
    ones = pd.DataFrame(1.0, index=y.index, columns=y.columns)
    S_x = ones.mul(xs, axis=0)
    S_x2 = ones.mul(xs.pow(2), axis=0)

    S_y = y.rolling(LOOKBACK, min_periods=MIN_OBS).sum()
    S_xy = y.mul(xs, axis=0).rolling(LOOKBACK, min_periods=MIN_OBS).sum()
    S_x = S_x.rolling(LOOKBACK, min_periods=MIN_OBS).sum()
    S_x2 = S_x2.rolling(LOOKBACK, min_periods=MIN_OBS).sum()
    n = y.notna().rolling(LOOKBACK, min_periods=MIN_OBS).sum()

    cov = S_xy - S_y.mul(S_x, axis=0).div(n)
    var = S_x2 - S_x.pow(2).div(n)
    return cov.div(var.replace(0, np.nan))


def nw_se(values, lag=NW_LAG):
    """Newey-West 样本均值的标准误（Bartlett 权重，月度惯例 lag=3）。"""
    values = values[np.isfinite(values)]
    n = len(values)
    resid = values - values.mean()
    q = np.sum(resid ** 2)
    for j in range(1, min(lag, n - 1) + 1):
        gamma_j = np.sum(resid[j:] * resid[:-j])
        q += 2 * (1 - j / (lag + 1)) * gamma_j
    return np.sqrt(q) / n if q >= 0 else np.nan


def cross_sectional_lambdas(chars, months, feat):
    """Step 1：对指定月份逐月做横截面回归，返回每月 lambda（单特征或多特征）。"""
    rows = []
    for t in months:
        y = ret_piv.loc[t] - rf.get(t, 0.0)   # 个股超额收益
        X = pd.DataFrame({c: feat[c].loc[t] for c in chars})
        d = pd.concat([y.rename("y"), X], axis=1)
        d = d[np.isfinite(d).all(axis=1)]
        if len(d) < MIN_STOCKS:
            continue
        # 横截面 winsorize + Z-Score 标准化（lambda 跨特征可比）
        d[chars] = d[chars].apply(lambda c: c.clip(c.quantile(WINSOR), c.quantile(1 - WINSOR)))
        Xs = d[chars].apply(lambda c: (c - c.mean()) / c.std(ddof=1))
        A = np.column_stack([np.ones(len(d)), Xs.values])
        coef, *_ = np.linalg.lstsq(A, d["y"].values, rcond=None)
        rows.append({"ym": t, **dict(zip(chars, coef[1:]))})
    return pd.DataFrame(rows).set_index("ym")


def summarize_lambda(lam, label):
    """Step 2：单个 lambda 序列的均值 / 普通 t / Newey-West t。"""
    s = lam.dropna()
    n, mean = len(s), s.mean()
    t_ols = mean / (s.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    se_nw = nw_se(s.values)
    t_nw = mean / se_nw if se_nw and se_nw > 0 else 0.0
    sig = "**" if abs(t_nw) > 2.58 else ("*" if abs(t_nw) > 1.96 else
                                         ("·" if abs(t_nw) > 1.64 else ""))
    return {"特征": label, "λ(%/月)": mean * 100, "普通t": t_ols, "NW t": t_nw,
            "显著性": sig, "期数": n}


def print_table(df):
    out = df.copy()
    for col in ["λ(%/月)", "普通t", "NW t"]:
        out[col] = out[col].round(2)
    print(out.to_string(index=False))


def part_factor_selector():
    """jh_quant 的 FactorSelector：把 Fama-MacBeth 验证嵌进因子选股。

    FactorSelector 内部先对每个因子跑一遍 FM 回归，用得到的 mean_lambda 作为
    该因子的权重（显著因子取 mean_lambda，不显著因子降权），再按
    score = Σ w_j × exposure 给股票打分，取 top_n / bottom_n。这里演示 FF3。
    """
    print(">>> [附加] jh_quant.backtest.FactorSelector（FM 验证驱动的因子选股）...")
    try:
        from jh_quant.backtest import FactorSelector
        from jh_quant.factors import FactorType

        jh_data = __import__("jh_quant.data", fromlist=["JHData"]).JHData()
        selector = FactorSelector(jh_data=jh_data)
        res = selector.select(
            factor=FactorType.FF3,
            start="2015-01-01", end="2026-06-30",
            top_n=50, bottom_n=50,
            period="M", factor_alpha=0.10, test_window=36,
            verbose=False,
        )
        print("    FM 验证结果（每因子：mean_lambda / NW t / NW p / 是否显著）：")
        print(res.fm_result.to_dataframe().round(4).to_string())
        print("\n    归一化后的因子权重（显著因子 = |mean_lambda|，不显著因子降权）：")
        print("    ", {k: round(v, 4) for k, v in res.weights.items()})
        print(f"\n    选股结果：top {len(res.top_selections)} 只 / bottom {len(res.bottom_selections)} 只")
        print("    top 前 5:", res.top_selections[:5])
    except Exception as e:  # 因子数据不可用或依赖缺失时不影响主流程
        print(f"    [跳过] FactorSelector 演示不可用: {e}")


def main():
    global ret_piv, rf
    os.makedirs("output", exist_ok=True)
    print(">>> 1/5 拉取数据（jh_quant 月度表）...")
    basic, mb, mpx, rf = fetch_data()
    panel = build_panel(basic, mb, mpx)
    ym_index = sorted(panel["ym"].unique())
    print(f"    面板: {len(panel)} 行, 股票 {panel['ts_code'].nunique()} 只, "
          f"月度 {len(ym_index)} 期")

    print(">>> 2/5 透视表 + 市场超额收益 ...")
    mcap_piv = panel.pivot_table(index="ym", columns="ts_code", values="mktcap").reindex(ym_index)
    ret_piv = panel.pivot_table(index="ym", columns="ts_code", values="ret").reindex(ym_index)
    bm_piv = panel.pivot_table(index="ym", columns="ts_code", values="bm").reindex(ym_index)
    w = mcap_piv.clip(lower=0)
    mkt_raw = (ret_piv * w).sum(axis=1) / w.sum(axis=1).replace(0, np.nan)
    mkt_ex = mkt_raw - rf.reindex(ym_index)   # 全市场市值加权超额收益

    print(">>> 3/5 Step 0：滚动 36 月 CAPM beta ...")
    beta_piv = estimate_rolling_beta(ret_piv, mkt_ex)
    beta_months = beta_piv.notna().any(axis=1)
    beta_months = beta_months[beta_months].index
    print(f"    beta 可用期数: {len(beta_months)} ({beta_months.min()} 起)")

    print(">>> 4/5 Step 1+2：单特征 Fama-MacBeth 回归 ...")
    feat = {"beta": beta_piv, "log_mktcap": np.log(mcap_piv), "bm": bm_piv}
    meta = {"beta": "Beta", "log_mktcap": "log市值", "bm": "BM"}
    month_sets = {"beta": beta_months, "log_mktcap": ym_index, "bm": ym_index}

    # 主表：各自最全样本（beta 因需滚动窗口，起点更晚）
    main_lam, main_rows = {}, []
    for c in ["beta", "log_mktcap", "bm"]:
        main_lam[c] = cross_sectional_lambdas([c], month_sets[c], feat)[c]
        main_rows.append(summarize_lambda(main_lam[c], meta[c]))
    print("\n[表1] 单特征风险溢价（各自最全样本；λ = 特征每 +1 个标准差对应的月收益）：")
    print_table(pd.DataFrame(main_rows))

    # 稳健性表：统一用 beta 可用窗口（2017-02 起），看显著性是否稳健
    rob_rows = []
    for c in ["beta", "log_mktcap", "bm"]:
        s = cross_sectional_lambdas([c], beta_months, feat)[c]
        rob_rows.append(summarize_lambda(s, meta[c]))
    print("\n[表2] 同一窗口稳健性（2017-02 起 113 期，三个特征对等比较）：")
    print_table(pd.DataFrame(rob_rows))

    # 补充：三特征联合横截面回归（同 113 期）
    j = cross_sectional_lambdas(["beta", "log_mktcap", "bm"], beta_months, feat)
    print("\n[表3] 三特征联合回归（同 113 期；观察多重共线性对显著性的稀释）：")
    print_table(pd.DataFrame([summarize_lambda(j[c], meta[c]) for c in ["beta", "log_mktcap", "bm"]]))

    print(">>> 5/5 绘图（mpl_style 统一风格）...")
    # 图1：单特征风险溢价 + NW 误差棒 + 显著性（主表）
    fig1, ax = plt.subplots(figsize=(9, 5))
    labels = [r["特征"] for r in main_rows]
    vals = [r["λ(%/月)"] for r in main_rows]
    tvals = [r["NW t"] for r in main_rows]
    errs = [1.96 * abs(v) / max(abs(t), 1e-9) for v, t in zip(vals, tvals)]  # 1.96*se_nw
    colors = []
    for t in tvals:
        if abs(t) > 1.96:
            colors.append(mpl_style.RISE if t > 0 else mpl_style.FALL)
        else:
            colors.append("#7F8C8D")
    bars = ax.bar(labels, vals, yerr=errs, color=colors, width=0.5, capsize=6,
                  error_kw={"lw": 1.2, "color": "#2C3E50"})
    for bar, r in zip(bars, main_rows):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if bar.get_height() >= 0 else -0.02),
                r["显著性"], ha="center", va="bottom", fontsize=14)
    ax.axhline(0, color="#7F8C8D", lw=1)
    ax.set_ylabel("风险溢价 λ（%/月，每 1 个标准差）")
    ax.set_xlabel("特征")
    ax.set_title("Fama-MacBeth：A股 2015–2026 哪些特征被定价？",
                 fontsize=14, fontweight="bold")
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("output/fig1_risk_premia.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_risk_premia.png")

    # 图2：每月 lambda 时序（单特征，各自最全样本）
    fig2, ax = plt.subplots(figsize=(12, 4.8))
    for c, label, color in [
        ("beta", "Beta λ", mpl_style.ACCENT),
        ("log_mktcap", "log市值 λ", mpl_style.FALL),
        ("bm", "BM λ", mpl_style.RISE),
    ]:
        s = main_lam[c].dropna() * 100
        ax.plot([str(y) for y in s.index], s.values, lw=1.0, color=color, label=label)
    ax.axhline(0, color="#7F8C8D", lw=1, ls="--")
    ax.set_ylabel("λ（%/月）")
    ax.set_xlabel("月份")
    ax.set_title("每月横截面回归得到的特征风险溢价 λ", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=3)
    ax.set_xticks(range(0, len(ym_index), 12))
    ax.set_xticklabels([str(y)[:4] for y in ym_index[::12]], rotation=45)
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_rolling_lambda.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_rolling_lambda.png")

    print("\n完成。两张图与本文对应，可插入公众号文章。")
    print()
    part_factor_selector()


if __name__ == "__main__":
    main()
