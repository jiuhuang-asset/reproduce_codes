# -*- coding: utf-8 -*-
"""
参数化组合策略（Parametric Portfolio Policies）：权重直接写成特征的函数
========================================================================

对应公众号文章《资产组合 · 参数化组合策略：不估收益，直接估权重》（量海泛舟）
方法论参考：https://www.tidy-finance.org/chapters/parametric-portfolio-policies.html
（Brandt, Santa-Clara & Valkanov 2009）

核心思路：均值-方差要先估每只股票的期望收益与协方差矩阵（几十上百个参数，
误差巨大）；参数化组合策略反其道而行，把组合权重直接写成股票特征的线性函数，
只估计 2 个参数 θ=(θ_mom, θ_size)：

    ω_{i,t} = (1/N_t) · (1 + θ_mom·x̂_mom + θ_size·x̂_size)

其中 x̂ 是横截面标准化后的特征（动量、对数市值）。θ 通过最大化样本内的
幂效用函数（γ=5）估计，施加无卖空约束（截断负权重并重新归一化）。

内容：
  1. 取月度前复权行情 + 月度总市值（约 50 只 A 股大盘股，2015–2026）；
  2. 构建两个特征：12 个月动量（t-13 到 t-2，跳过最近 1 月）、对数市值；
  3. 每个月底横截面标准化，用 L-BFGS-B（θ 限制在 [-10,10]）估计 θ；
  4. 对比参数化组合（最优 θ）与等权基准（θ=0）的累计净值与夏普。

运行方式
--------
1. 安装依赖：  pip install jh-quant matplotlib scipy
2. 设置环境变量（从 https://jiuhuang.xyz 申请 API Key）：
       export JIUHUANG_API_KEY=你的key      # Windows: set JIUHUANG_API_KEY=你的key
   （或在项目根目录放 .env 文件，JHData 会自动读取）
3. 运行：      python parametric_portfolio_policy.py

输出
----
控制台打印最优 θ、参数化组合与等权基准的绩效；并生成 2 张图到 output/：
    - fig1_characteristic_returns.png  动量分组 / 市值分组的月均收益（倾斜依据）
    - fig2_ppp_cumulative.png          参数化组合 vs 等权基准累计净值
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# 统一图表风格（house style，保证公众号文章图表一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from jh_quant.data import JHData, DataTypes  # noqa: E402

# ============================================================
# 全局参数
# ============================================================
START, END = "2015-01-01", "2026-06-30"
GAMMA = 5.0           # 幂效用函数的相对风险厌恶系数（参考原文取 5）
MOM_WINDOW = 12       # 动量窗口：12 个月

# 约 50 只跨行业 A 股大盘/中盘股（月度数据 2015–2026 完整）
STOCKS = {
    "600036.SH": "招商银行", "601398.SH": "工商银行", "601288.SH": "农业银行",
    "601988.SH": "中国银行", "601939.SH": "建设银行", "601166.SH": "兴业银行",
    "002142.SZ": "宁波银行", "601318.SH": "中国平安", "601601.SH": "中国太保",
    "601628.SH": "中国人寿", "600030.SH": "中信证券", "601211.SH": "国泰君安",
    "600519.SH": "贵州茅台", "000858.SZ": "五粮液",   "000568.SZ": "泸州老窖",
    "600809.SH": "山西汾酒", "002304.SZ": "洋河股份", "600887.SH": "伊利股份",
    "300750.SZ": "宁德时代", "002594.SZ": "比亚迪",   "600438.SH": "通威股份",
    "601985.SH": "中国核电", "000002.SZ": "万科A",    "600048.SH": "保利发展",
    "601668.SH": "中国建筑", "000333.SZ": "美的集团", "000651.SZ": "格力电器",
    "600690.SH": "海尔智家", "000100.SZ": "TCL科技",  "600276.SH": "恒瑞医药",
    "603259.SH": "药明康德", "300015.SZ": "爱尔眼科", "000538.SZ": "云南白药",
    "600196.SH": "复星医药", "002415.SZ": "海康威视", "002475.SZ": "立讯精密",
    "000725.SZ": "京东方A",  "000063.SZ": "中兴通讯", "600050.SH": "中国联通",
    "600028.SH": "中国石化", "601857.SH": "中国石油", "601088.SH": "中国神华",
    "600585.SH": "海螺水泥", "600111.SH": "北方稀土", "600019.SH": "宝钢股份",
    "600031.SH": "三一重工", "601766.SH": "中国中车", "601888.SH": "中国中免",
    "300059.SZ": "东方财富", "002714.SZ": "牧原股份", "600900.SH": "长江电力",
}

os.makedirs("output", exist_ok=True)


# ============================================================
# 1. 取数
# ============================================================
def fetch_data():
    jh = JHData()
    codes = ",".join(STOCKS)

    # 月度前复权行情 -> 月收益
    monthly = jh.get_data(
        DataTypes.TS_MONTHLY_QFQ, ts_code=codes, start=START, end=END
    ).to_df()
    monthly = monthly[["ts_code", "trade_date", "close"]].copy()
    monthly["trade_date"] = pd.to_datetime(monthly["trade_date"])
    monthly = monthly.sort_values(["ts_code", "trade_date"])
    monthly["ret"] = monthly.groupby("ts_code")["close"].pct_change()
    rets = monthly.pivot(index="trade_date", columns="ts_code", values="ret")

    # 月度总市值（月末值）
    basic = jh.get_data(
        DataTypes.TS_MONTHLY_BASIC, ts_code=codes, start=START, end=END
    ).to_df()
    basic = basic[["ts_code", "trade_date", "total_mv"]].copy()
    basic["trade_date"] = pd.to_datetime(basic["trade_date"])
    mv = basic.pivot(index="trade_date", columns="ts_code", values="total_mv")

    return rets, mv


# ============================================================
# 2. 特征构建
# ============================================================
def build_characteristics(rets, mv):
    """构建两个滞后特征（月底已知，不偷看未来）：
    - momentum：过去 12 个月收益（t-13 到 t-2，跳过最近 1 个月）
    - size：对数总市值（t-1 月末）
    """
    # 12 个月动量，再滞后 1 个月以跳过最近一个月
    momentum = rets.rolling(MOM_WINDOW).sum().shift(1)
    size = np.log(mv).shift(1)
    return momentum, size


def standardize(df):
    """对每一行（每一个月）做横截面标准化（z-score），返回等形状 DataFrame。"""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


# ============================================================
# 3. 参数化组合策略
# ============================================================
def ppp_portfolio_returns(rets, momentum, size, theta):
    """给定参数 theta，计算每个月的组合收益序列（无卖空截断后）。

    momentum/size 已标准化；组合权重 = (1/N)·(1 + θ_mom·mom + θ_size·size)。
    缺失特征按 0 处理（即该股当月不做倾斜、给等权）；缺失收益按 0 处理。
    返回 (组合月收益 Series, 权重 numpy 数组)。
    """
    mom = momentum.fillna(0.0).values
    siz = size.fillna(0.0).values
    r = rets.fillna(0.0).values
    valid = rets.notna().values                      # 该月有收益的股票

    raw = 1.0 + theta[0] * mom + theta[1] * siz      # (1 + θ'x̂)
    N = valid.sum(axis=1)                            # 每月有效股票数
    raw_w = raw / N[:, None]
    raw_w = np.where(valid, raw_w, 0.0)              # 无收益股票权重 0

    # 无卖空：负权重截断为 0，再归一化
    pos = np.clip(raw_w, 0.0, None)
    w = pos / pos.sum(axis=1, keepdims=True)

    port_ret = (w * r).sum(axis=1)
    return pd.Series(port_ret, index=rets.index), w


def power_utility(ret, gamma=GAMMA):
    """幂效用函数 u(r) = (1+r)^(1-γ)/(1-γ)，对组合收益序列求平均。"""
    return np.mean(((1.0 + ret) ** (1.0 - gamma)) / (1.0 - gamma))


def estimate_theta(rets, momentum, size):
    """最大化平均幂效用，估计 θ。

    用 L-BFGS-B 并把 θ 限制在 [-10, 10]。参考原文用数千只股票的截面，
    θ 落在 O(1) 量级（约 0.27 / -1.66）；本文只有 51 只股票，若不设边界，
    优化器会过度集中仓位（θ 发散到 1e13），这是小截面下的过拟合，故加边界。
    """
    def neg_utility(theta):
        port_ret, _ = ppp_portfolio_returns(rets, momentum, size, theta)
        return -power_utility(port_ret)

    res = minimize(
        neg_utility, x0=np.array([0.0, 0.0]), method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
    )
    return res.x, -res.fun


# ============================================================
# 4. 绘图
# ============================================================
def grouped_mean(rets, rank, lo, hi):
    """按横截面分位分组，返回该组内所有（月 × 股票）收益的均值。"""
    mask = (rank > lo) & (rank <= hi)
    return float(np.nanmean(rets[mask].to_numpy()))


def fig1_characteristic_returns(rets, momentum, size):
    """特征与收益：按动量 / 市值分组，看组间月均收益差异（倾斜的依据）。"""
    momentum = momentum.loc[rets.index]
    size = size.loc[rets.index]
    mom_rank = momentum.rank(axis=1, pct=True)   # 每月横截面分位（0~1）
    size_rank = size.rank(axis=1, pct=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["最低", "2", "3", "4", "最高"]

    mom_means = [grouped_mean(rets, mom_rank, edges[i], edges[i + 1])
                 for i in range(5)]
    ax1.bar(labels, [m * 100 for m in mom_means], color=mpl_style.ACCENT)
    ax1.set_title("动量分组：过去涨得多的，未来月均收益")
    ax1.set_ylabel("月均收益（%）")

    size_means = [grouped_mean(rets, size_rank, edges[i], edges[i + 1])
                  for i in range(5)]
    ax2.bar(labels, [m * 100 for m in size_means], color=mpl_style.ACCENT_2)
    ax2.set_title("市值分组：小市值的未来月均收益")
    ax2.set_ylabel("月均收益（%）")

    # 供文章引用的数字：动量 / 市值各分组的月均收益
    print("    动量分组月均收益（%）：", [round(m * 100, 2) for m in mom_means])
    print("    市值分组月均收益（%）：", [round(m * 100, 2) for m in size_means])

    for ax in (ax1, ax2):
        mpl_style.hide_spines(ax)
    fig.tight_layout()
    fig.savefig("output/fig1_characteristic_returns.png")
    print("    已保存 fig1_characteristic_returns.png")
    return fig


def fig2_ppp_cumulative(rets, momentum, size, theta):
    """参数化组合 vs 等权基准的累计净值。"""
    ppp_ret, _ = ppp_portfolio_returns(rets, momentum, size, theta)
    ew_ret = rets.mean(axis=1)  # 等权基准（θ=0 的近似）

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rets.index, (1 + ppp_ret).cumprod().values, lw=2.0,
            color=mpl_style.RISE, label=f"参数化组合（θ_mom={theta[0]:.2f}, θ_size={theta[1]:.2f}）")
    ax.plot(rets.index, (1 + ew_ret).cumprod().values, lw=2.0,
            color=mpl_style.ACCENT, label="等权基准")
    mpl_style.hide_spines(ax)
    ax.set_title("参数化组合 vs 等权基准（2016–2026）")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值（初始 = 1）")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("output/fig2_ppp_cumulative.png")
    print("    已保存 fig2_ppp_cumulative.png")
    return fig


# ============================================================
# 5. 主流程
# ============================================================
def performance(monthly_ret):
    """月度收益 -> 年化收益、年化波动、夏普（无风险利率近似取 0）。"""
    ret = monthly_ret.mean() * 12
    vol = monthly_ret.std() * np.sqrt(12)
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def main():
    print("=" * 60)
    print("参数化组合策略（Parametric Portfolio Policies）")
    print("=" * 60)

    print("\n[1] 取数：月度行情 + 月度总市值")
    rets, mv = fetch_data()
    # 只保留既有收益又有市值的股票，并把市值对齐到收益的日期索引
    common = rets.columns.intersection(mv.columns)
    rets, mv = rets[common], mv[common]
    rets = rets.dropna(how="all")
    mv = mv.reindex(rets.index).reindex(columns=common)
    print(f"    股票数: {rets.shape[1]}, 月份数: {rets.shape[0]}")

    print("\n[2] 构建特征（动量 + 对数市值，滞后避免前视）")
    momentum, size = build_characteristics(rets, mv)
    momentum_z = standardize(momentum)
    size_z = standardize(size)

    # 只剔除预热期：动量需要 12 个月 + 1 个月滞后，首月无任何股票有有效动量
    valid = momentum_z.notna().any(axis=1)
    rets_v = rets[valid]
    momentum_v = momentum_z[valid]
    size_v = size_z[valid]
    print(f"    有效月份: {len(rets_v)}（{rets_v.index[0].date()} 至 {rets_v.index[-1].date()}）")

    print("\n[3] 估计 θ（最大化平均幂效用，γ=5）")
    theta, util = estimate_theta(rets_v, momentum_v, size_v)
    print(f"    最优 θ：动量 {theta[0]:.3f}，市值 {theta[1]:.3f}")
    print(f"    （θ_mom>0 表示向高动量倾斜，θ_size<0 表示向小市值倾斜）")

    print("\n[4] 绩效对比")
    ppp_ret, _ = ppp_portfolio_returns(rets_v, momentum_v, size_v, theta)
    ew_ret = rets_v.mean(axis=1)
    for name, r in [("参数化组合", ppp_ret), ("等权基准", ew_ret)]:
        ret, vol, sharpe = performance(r)
        print(f"    {name:　<6} 年化收益 {ret * 100:6.2f}%  波动 {vol * 100:6.2f}%  夏普 {sharpe:5.2f}")

    print("\n[5] 绘图")
    fig1 = fig1_characteristic_returns(rets_v, momentum, size)
    fig2 = fig2_ppp_cumulative(rets_v, momentum_v, size_v, theta)
    plt.close("all")
    print("\n全部完成。")


if __name__ == "__main__":
    main()
