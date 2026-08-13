# -*- coding: utf-8 -*-
"""
均值-方差组合（马科维茨）入门：从两只股票到有效前沿
======================================================

对应公众号文章《资产组合入门 · 马科维茨的均值方差理论》（量海泛舟）
方法论参考：https://www.tidy-finance.org/chapters/parametric-portfolio-policies.html
           https://www.tidy-finance.org/chapters/constrained-optimization-and-backtesting.html

内容：
  1. 用 6 只跨行业 A 股蓝筹（日频，2020–2024）算期望收益、波动率、协方差矩阵；
  2. 画两资产分散化示意（相关性如何改变组合风险）；
  3. 画 6 只股票的风险-收益散点；
  4. 用解析式画有效前沿，标出最小方差组合与最大夏普（切线）组合；
  5. 手写 numpy 结果与 Riskfolio-Lib 库版结果对照，验证两者一致。

市场数据用 jh_quant 的 JHData 接口。
无风险利率：SHIBOR 隔夜（on），日化 on/100/360，再年化。

运行方式
--------
1. 安装依赖：  pip install jh-quant matplotlib riskfolio-lib
2. 设置环境变量（从 https://jiuhuang.xyz 申请 API Key）：
       export JIUHUANG_API_KEY=你的key      # Windows: set JIUHUANG_API_KEY=你的key
   （或在项目根目录放 .env 文件，JHData 会自动读取）
3. 运行：      python mean_variance_portfolio.py

输出
----
控制台打印 6 只股票的年化收益/波动、协方差相关性、MVP 与切线组合权重、
手写 vs Riskfolio 对照表；并生成 3 张图到 output/ 子目录：
    - fig1_diversification.png    两资产分散化示意（相关性 = +1 / 0 / -1）
    - fig2_risk_return_scatter.png 6 只股票年化风险-收益散点
    - fig3_efficient_frontier.png 有效前沿 + 最小方差组合 + 最大夏普组合 + 资本市场线
"""

import os
import sys
import numpy as np
import pandas as pd

# 统一图表风格（house style，保证公众号文章图表一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from jh_quant.data import JHData, DataTypes  # noqa: E402

# ============================================================
# 全局参数
# ============================================================
START, END = "2020-01-01", "2024-12-31"
TRADING_DAYS = 252  # 一年约 252 个交易日，用于年化

# 6 只跨行业蓝筹：代码 -> 简称
# 说明：长江电力(600900.SH)/中国平安(601318.SH)在 QFQ 源里历史数据缺失（仅 2023-12 起），
#       故用同样有完整 5 年数据、且同属公用/消费板块的中国核电、美的集团替代。
STOCKS = {
    "600036.SH": "招商银行",
    "600519.SH": "贵州茅台",
    "300750.SZ": "宁德时代",
    "000002.SZ": "万科A",
    "601985.SH": "中国核电",
    "000333.SZ": "美的集团",
}

os.makedirs("output", exist_ok=True)


# ============================================================
# 1. 取数：个股前复权日线 + SHIBOR 无风险利率
# ============================================================
def fetch_data():
    jh = JHData()

    # 1) 6 只股票前复权日线
    prices = jh.get_data(
        DataTypes.TS_DAILY_QFQ,
        ts_code=",".join(STOCKS),
        start=START,
        end=END,
    ).to_df()
    prices = prices[["ts_code", "trade_date", "close"]].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])

    # 2) SHIBOR 隔夜利率（无风险利率，TS_SHIBOR 用 on 列；远程返回字符串，转数值）
    shibor = jh.get_data(
        DataTypes.TS_SHIBOR, start=START, end=END, bypass_cache=True
    ).to_df()
    shibor = shibor[["date", "on"]].rename(columns={"on": "rf_pct"})
    shibor["rf_pct"] = pd.to_numeric(shibor["rf_pct"], errors="coerce")
    shibor["date"] = pd.to_datetime(shibor["date"])

    return prices, shibor


def build_returns(prices, shibor):
    """构造 6 只股票的日收益面板 + 年化无风险利率。"""
    prices = prices.sort_values(["ts_code", "trade_date"])
    # 每只股票按交易日算简单日收益（小数，非百分数）
    prices["ret"] = prices.groupby("ts_code")["close"].pct_change()

    # 宽表：行 = 交易日，列 = 股票代码，值 = 日收益
    rets = prices.pivot(index="trade_date", columns="ts_code", values="ret")
    rets = rets.dropna()  # 去掉首行 NaN 及任何缺失日

    # 年化无风险利率：SHIBOR 隔夜（年化百分比） -> 日收益 -> 年化
    shibor = shibor.dropna(subset=["rf_pct"]).set_index("date").sort_index()
    rf_daily = shibor["rf_pct"] / 100 / 360  # 年化隔夜利率 -> 日收益（小数）
    rf_annual = rf_daily.mean() * TRADING_DAYS  # 年化（小数）

    return rets, float(rf_annual)


def annualize(rets):
    """由日收益面板计算年化期望收益、年化波动率、年化协方差/相关矩阵。"""
    mu = rets.mean() * TRADING_DAYS          # 年化期望收益（小数）
    sigma = rets.std() * np.sqrt(TRADING_DAYS)  # 年化波动率（小数）
    cov = rets.cov() * TRADING_DAYS          # 年化协方差矩阵
    corr = rets.corr()                       # 相关系数矩阵（无量纲，无需年化）
    return mu, sigma, cov, corr


# ============================================================
# 2. 有效前沿解析式
# ============================================================
def frontier_parameters(mu, cov):
    """计算有效前沿超双曲线参数 a、b、c。

    对给定目标收益 μ0，前沿组合方差 σ0² = (a·μ0² - 2b·μ0 + c) / (a·c - b²)。
    其中 a = 1ᵀΣ⁻¹1，b = 1ᵀΣ⁻¹μ，c = μᵀΣ⁻¹μ。
    """
    inv = np.linalg.inv(cov.values)
    ones = np.ones(len(mu))
    mu_v = mu.values
    a = float(ones @ inv @ ones)
    b = float(ones @ inv @ mu_v)
    c = float(mu_v @ inv @ mu_v)
    return a, b, c


def min_variance_portfolio(mu, cov):
    """最小方差组合：w = Σ⁻¹1 / (1ᵀΣ⁻¹1)，及其收益与波动。"""
    inv = np.linalg.inv(cov.values)
    ones = np.ones(len(mu))
    w = inv @ ones / (ones @ inv @ ones)
    mu_mvp = float(w @ mu.values)
    sigma_mvp = float(np.sqrt(w @ cov.values @ w))
    return w, mu_mvp, sigma_mvp


def tangency_portfolio(mu, cov, rf):
    """最大夏普（切线）组合：w = Σ⁻¹(μ - rf·1) / [1ᵀΣ⁻¹(μ - rf·1)]。"""
    inv = np.linalg.inv(cov.values)
    ones = np.ones(len(mu))
    excess = mu.values - rf * ones
    w = inv @ excess / (ones @ inv @ excess)
    mu_tan = float(w @ mu.values)
    sigma_tan = float(np.sqrt(w @ cov.values @ w))
    sharpe = (mu_tan - rf) / sigma_tan
    return w, mu_tan, sigma_tan, sharpe


# ============================================================
# 3. Riskfolio-Lib 对照（可选依赖，缺失则跳过）
# ============================================================
def riskfolio_long_only(rets, rf):
    """用 Riskfolio-Lib 计算「禁卖空」的最小方差组合（MVP）。

    Riskfolio 7.x 里 l=0 表示长仓约束（禁止做空）。本组合的权重恰好全为正，
    因此与手写的无约束 MVP 完全一致——这也直观说明：最小方差组合通常无需做空。
    （最大夏普组合在禁卖空下的求解留到第二篇的约束优化。）
    返回权重 Series。
    """
    import contextlib
    import riskfolio as rp

    port = rp.Portfolio(returns=rets)
    port.assets_stats(method_mu="hist", method_cov="hist")

    with open(os.devnull, "w") as fnull, \
            contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        w = port.optimization(model="Classic", rm="MV", obj="MinRisk", rf=rf, l=0, hist=True)
    s = w.iloc[:, 0]
    s.index = [STOCKS.get(str(i), str(i)) for i in s.index]
    return s


# ============================================================
# 4. 绘图
# ============================================================
def fig1_diversification(mu, sigma, corr):
    """两资产分散化示意：用贵州茅台 + 宁德时代，展示相关性对组合风险的影响。"""
    name_a, name_b = "贵州茅台", "宁德时代"
    code_a, code_b = "600519.SH", "300750.SZ"
    mu_a, mu_b = float(mu[code_a]), float(mu[code_b])
    sg_a, sg_b = float(sigma[code_a]), float(sigma[code_b])

    fig, ax = plt.subplots(figsize=(9, 6))
    w = np.linspace(0, 1, 101)  # 组合中 A 的权重，0 -> 1

    # 组合收益与风险：mu_p = w*mu_a + (1-w)*mu_b
    # 组合方差：sg_p² = w²sg_a² + (1-w)²sg_b² + 2w(1-w)ρ sg_a sg_b
    mu_p = w * mu_a + (1 - w) * mu_b
    for rho, label, ls in [(1.0, "相关系数 = +1", "-"),
                           (0.0, "相关系数 = 0", "--"),
                           (-1.0, "相关系数 = -1", "-.")]:
        var_p = (w ** 2 * sg_a ** 2 + (1 - w) ** 2 * sg_b ** 2
                 + 2 * w * (1 - w) * rho * sg_a * sg_b)
        sg_p = np.sqrt(np.maximum(var_p, 0.0))
        ax.plot(sg_p * 100, mu_p * 100, ls, lw=1.8, label=label)

    # 实际组合（用真实相关系数）
    rho_real = float(corr.loc[code_a, code_b])
    var_real = (w ** 2 * sg_a ** 2 + (1 - w) ** 2 * sg_b ** 2
                + 2 * w * (1 - w) * rho_real * sg_a * sg_b)
    sg_real = np.sqrt(var_real)
    ax.plot(sg_real * 100, mu_p * 100, "-", lw=2.8, color=mpl_style.ACCENT,
            label=f"{name_a}+{name_b} 实际组合（ρ={rho_real:.2f}）")

    # 端点
    ax.scatter([sg_a * 100, sg_b * 100], [mu_a * 100, mu_b * 100],
               color=mpl_style.RISE, zorder=5)
    ax.annotate(name_a, (sg_a * 100, mu_a * 100),
                textcoords="offset points", xytext=(6, 10))
    ax.annotate(name_b, (sg_b * 100, mu_b * 100),
                textcoords="offset points", xytext=(6, -12))

    mpl_style.hide_spines(ax)
    ax.set_title("分散化的秘密：相关性越低，组合风险越低")
    ax.set_xlabel("组合年化波动率（%）")
    ax.set_ylabel("组合年化收益率（%）")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("output/fig1_diversification.png")
    print("    已保存 fig1_diversification.png")
    return fig


def fig2_risk_return_scatter(mu, sigma):
    """6 只股票的年化风险-收益散点。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    for code, name in STOCKS.items():
        ax.scatter(sigma[code] * 100, mu[code] * 100,
                   color=mpl_style.COLOR_CYCLE[list(STOCKS).index(code) % len(mpl_style.COLOR_CYCLE)],
                   s=70, zorder=3)
        ax.annotate(name, (sigma[code] * 100, mu[code] * 100),
                    textcoords="offset points", xytext=(8, 4))
    mpl_style.hide_spines(ax)
    ax.set_title("6 只股票：年化风险 vs 年化收益（2020–2024）")
    ax.set_xlabel("年化波动率（%，即风险）")
    ax.set_ylabel("年化收益率（%）")
    fig.tight_layout()
    fig.savefig("output/fig2_risk_return_scatter.png")
    print("    已保存 fig2_risk_return_scatter.png")
    return fig


def fig3_efficient_frontier(mu, sigma, cov, rf, w_mvp, mu_mvp, sg_mvp,
                            w_tan, mu_tan, sg_tan):
    """有效前沿 + 最小方差组合 + 最大夏普组合 + 资本市场线。"""
    a, b, c = frontier_parameters(mu, cov)

    # 前沿：从 MVP 收益到略高于「最高个股收益」与「切线组合收益」的较大者
    mu_min = mu_mvp
    mu_max = max(float(mu.max()), mu_tan) * 1.05
    mu_grid = np.linspace(mu_min, mu_max, 200)
    var_grid = (a * mu_grid ** 2 - 2 * b * mu_grid + c) / (a * c - b ** 2)
    sg_grid = np.sqrt(np.maximum(var_grid, 0.0))

    fig, ax = plt.subplots(figsize=(9, 6))

    # 有效前沿
    ax.plot(sg_grid * 100, mu_grid * 100, "-", lw=2.6,
            color=mpl_style.ACCENT, label="有效前沿")

    # 个股散点（灰）
    for code in STOCKS:
        ax.scatter(sigma[code] * 100, mu[code] * 100,
                   color="#95A5A6", s=50, zorder=3)
    for code, name in STOCKS.items():
        ax.annotate(name, (sigma[code] * 100, mu[code] * 100),
                    textcoords="offset points", xytext=(6, 2), fontsize=9)

    # 最小方差组合
    ax.scatter(sg_mvp * 100, mu_mvp * 100, marker="*", s=320,
               color=mpl_style.RISE, zorder=4, label="最小方差组合")
    # 最大夏普（切线）组合
    ax.scatter(sg_tan * 100, mu_tan * 100, marker="*", s=320,
               color=mpl_style.FALL, zorder=4, label="最大夏普组合")

    # 资本市场线：从无风险利率到切线组合
    rf_pct = rf * 100
    ax.plot([0, sg_tan * 100], [rf_pct, mu_tan * 100], "--", lw=1.8,
            color=mpl_style.ACCENT_2, label="资本市场线（CML）")
    ax.scatter([0], [rf_pct], marker="o", s=60, color=mpl_style.ACCENT_2, zorder=4)
    ax.annotate("无风险利率", (0, rf_pct), textcoords="offset points",
                xytext=(8, -4), fontsize=9, color=mpl_style.ACCENT_2)

    mpl_style.hide_spines(ax)
    ax.set_title("有效前沿：给定风险下收益最高、给定收益下风险最低")
    ax.set_xlabel("年化波动率（%）")
    ax.set_ylabel("年化收益率（%）")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("output/fig3_efficient_frontier.png")
    print("    已保存 fig3_efficient_frontier.png")
    return fig


# ============================================================
# 5. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("均值-方差组合入门（马科维茨）")
    print("=" * 60)

    print("\n[1] 取数：6 只跨行业蓝筹日线 + SHIBOR 无风险利率")
    prices, shibor = fetch_data()
    print(f"    个股行数: {len(prices)}, SHIBOR 行数: {len(shibor)}")

    print("\n[2] 构造日收益面板并年化")
    rets, rf = build_returns(prices, shibor)
    print(f"    交易日数: {len(rets)}, 股票数: {rets.shape[1]}")
    print(f"    年化无风险利率（SHIBOR 隔夜）: {rf * 100:.2f}%")

    mu, sigma, cov, corr = annualize(rets)
    names = [STOCKS[c] for c in rets.columns]

    print("\n    年化收益 / 波动率：")
    for code in rets.columns:
        print(f"      {STOCKS[code]:　<6} 收益 {mu[code] * 100:6.2f}%  波动 {sigma[code] * 100:6.2f}%")

    print("\n    相关系数矩阵：")
    corr_disp = corr.copy()
    corr_disp.index = names
    corr_disp.columns = names
    print(corr_disp.round(2).to_string())

    print("\n[3] 有效前沿解析式 + 两个关键组合")
    w_mvp, mu_mvp, sg_mvp = min_variance_portfolio(mu, cov)
    w_tan, mu_tan, sg_tan, sharpe = tangency_portfolio(mu, cov, rf)

    print(f"    最小方差组合: 收益 {mu_mvp * 100:.2f}%, 波动 {sg_mvp * 100:.2f}%")
    print(f"    最大夏普组合: 收益 {mu_tan * 100:.2f}%, 波动 {sg_tan * 100:.2f}%, 夏普 {sharpe:.2f}")

    print("\n    最小方差组合权重：")
    for code, w in zip(rets.columns, w_mvp):
        print(f"      {STOCKS[code]:　<6} {w * 100:6.2f}%")
    print("    最大夏普组合权重：")
    for code, w in zip(rets.columns, w_tan):
        print(f"      {STOCKS[code]:　<6} {w * 100:6.2f}%")

    print("\n[4] Riskfolio-Lib 对照（禁卖空版，l=0）")
    try:
        mvp_rp = riskfolio_long_only(rets, rf)
        print("    最小方差组合：手写(无约束) vs Riskfolio(禁卖空)")
        for code in rets.columns:
            name = STOCKS[code]
            hand = w_mvp[list(rets.columns).index(code)]
            rp_w = float(mvp_rp.get(name, 0.0))
            print(f"      {name:　<6} 手写 {hand * 100:6.2f}%  Riskfolio {rp_w * 100:6.2f}%")
    except Exception as e:
        print(f"    Riskfolio 对照跳过：{e}")

    print("\n[5] 绘图")
    fig1 = fig1_diversification(mu, sigma, corr)
    fig2 = fig2_risk_return_scatter(mu, sigma)
    fig3 = fig3_efficient_frontier(mu, sigma, cov, rf, w_mvp, mu_mvp, sg_mvp,
                                   w_tan, mu_tan, sg_tan)
    plt.close("all")
    print("\n全部完成。")


if __name__ == "__main__":
    main()
