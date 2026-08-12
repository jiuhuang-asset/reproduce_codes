# -*- coding: utf-8 -*-
"""
Beta 估计：用 jh_quant 拉取 A 股数据，验证 CAPM 市场模型
==========================================================

对应公众号文章《资产定价入门 · Beta 是什么，用 A 股数据算给你看》（量海泛舟 · 系列第一篇）
方法论参考：https://www.tidy-finance.org/chapters/beta-estimation.html

做什么
------
用市场模型 R_i - R_f = alpha + beta * (R_m - R_f) + e，对三只代表性 A 股做 OLS 回归：
  - 招商银行 600036.SH（银行，预期低 beta / 防御型）
  - 贵州茅台 600519.SH（消费蓝筹，预期 beta 接近 1）
  - 宁德时代 300750.SZ（新能源成长，预期高 beta / 进攻型）
市场基准 = 沪深300（000300.SH）。无风险利率取 SHIBOR 隔夜利率（日频）。

运行方式
--------
1. 安装依赖：  pip install jh-quant matplotlib
2. 设置环境变量（从 https://jiuhuang.xyz 申请 API Key）：
       export JIUHUANG_API_KEY=你的key      # Windows: set JIUHUANG_API_KEY=你的key
   （或在项目根目录放 .env 文件，JHData 会自动读取）
3. 运行：      python beta_estimation_a_share.py

输出
----
控制台打印三只股票的 beta / alpha / R^2，并生成 3 张图：
  - fig1_scatter_regression.png   日收益散点 + 回归线（3 子图）
  - fig2_beta_compare.png         beta 对比柱状图
  - fig3_rolling_beta.png         60 交易日滚动 beta 时序
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 统一图表风格（house style，保证公众号文章图表一致）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpl_style  # noqa: E402

# 数据期间：最近 5 个完整日历年（约 1200 个交易日）
START, END = "2020-01-01", "2024-12-31"

# 市场基准：沪深300（TS 源 ts_code 为 000300.SH）

# 三只代表性股票：代码 -> 简称
STOCKS = {
    "600036.SH": "招商银行",
    "600519.SH": "贵州茅台",
    "300750.SZ": "宁德时代",
}


def fetch_data():
    """用 jh_quant 拉取：个股前复权日线 + 沪深300 + SHIBOR 无风险利率。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()  # 自动从环境变量 JIUHUANG_API_KEY 读取

    # 1) 三只股票前复权日线（close 已复权，可直接算收益）
    stock_prices = jh.get_data(
        DataTypes.TS_DAILY_QFQ,
        ts_code=",".join(STOCKS.keys()),
        start=START,
        end=END,
    ).to_df()
    stock_prices = stock_prices[["ts_code", "trade_date", "close"]]

    # 2) 沪深300 指数日线（TS 源，沪深300 的 ts_code 是 000300.SH）
    market = jh.get_data(
        DataTypes.TS_INDEX_DAILY,
        ts_code="000300.SH",
        start=START,
        end=END,
    ).to_df()
    market = market[["trade_date", "close"]].rename(
        columns={"close": "mkt_close"}
    )

    # 3) SHIBOR 隔夜利率（无风险利率，TS_SHIBOR 用 on 列，日化 /360）
    #    数据量小，bypass_cache 直接拉取：TS_SHIBOR 服务端 DDL 列名尚未同步为 1m/1w
    shibor = jh.get_data(
        DataTypes.TS_SHIBOR, start=START, end=END, bypass_cache=True
    ).to_df()
    shibor = shibor[["date", "on"]].rename(columns={"on": "rf_pct"})
    shibor["rf_pct"] = pd.to_numeric(shibor["rf_pct"], errors="coerce")  # 远程返回字符串，转数值

    return stock_prices, market, shibor


def prepare_returns(stock_prices, market, shibor):
    """算收益 + 超额收益（减无风险利率），按交易日对齐。"""
    # 个股日收益：前复权 close 的百分比变化
    stock_prices = stock_prices.sort_values(["ts_code", "trade_date"])
    stock_prices["ret"] = (
        stock_prices.groupby("ts_code")["close"].pct_change() * 100  # 换算成百分数
    )
    # 沪深300 日收益
    market = market.sort_values("trade_date")
    market["mkt_ret"] = market["mkt_close"].pct_change() * 100

    # SHIBOR 按交易日对齐（隔夜利率年化百分比 -> 日度小数，再对齐后向前填充）
    shibor["date"] = pd.to_datetime(shibor["date"])
    shibor = shibor.dropna(subset=["rf_pct"]).set_index("date").sort_index()
    rf_daily = shibor["rf_pct"] / 100 / 360  # 年化隔夜利率 -> 日收益

    out = []
    for code, name in STOCKS.items():
        s = stock_prices[stock_prices["ts_code"] == code].copy()
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        s = s.set_index("trade_date")
        m = market.set_index(pd.to_datetime(market["trade_date"]))[["mkt_ret"]]
        df = s[["ret"]].join(m, how="inner")  # 只保留共同交易日
        df["rf"] = rf_daily.reindex(df.index).ffill().fillna(0.0)
        df["stock_excess"] = df["ret"] - df["rf"]   # R_i - R_f
        df["mkt_excess"] = df["mkt_ret"] - df["rf"]  # R_m - R_f
        df["symbol"] = code
        df["name"] = name
        out.append(df.reset_index())
    return pd.concat(out, ignore_index=True)


def ols_beta(y, x):
    """手写最小二乘：回归 y = alpha + beta * x。返回 (beta, alpha, r2, t_beta, n)。"""
    X = np.column_stack([np.ones_like(x), x])
    beta_vec, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = beta_vec[0], beta_vec[1]
    resid = y - X @ beta_vec
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = len(y)
    # beta 的标准误与 t 统计量
    dof = max(n - 2, 1)
    sigma2 = ss_res / dof
    sxx = float(((x - x.mean()) ** 2).sum())
    se_beta = np.sqrt(sigma2 / sxx) if sxx > 0 else np.nan
    t_beta = beta / se_beta if se_beta else np.nan
    return beta, alpha, r2, t_beta, n


def rolling_beta(df, window=60):
    """60 个交易日滚动窗口的 beta 时序（收盘对齐，作为 beta 随时间变化的直观演示）。"""
    x = df["mkt_excess"].values
    y = df["stock_excess"].values
    n = len(df)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        out[i - 1] = ols_beta(y[i - window:i], x[i - window:i])[0]
    return out


def main():
    os.makedirs("output", exist_ok=True)
    print(">>> 1/4 拉取数据（jh_quant）...")
    stock_prices, market, shibor = fetch_data()
    print(f"    个股行数: {len(stock_prices)}, 指数行数: {len(market)}, SHIBOR 行数: {len(shibor)}")

    print(">>> 2/4 计算收益并合并...")
    df = prepare_returns(stock_prices, market, shibor)
    print(f"    合并后行数: {len(df)}（三只股票 × {df['trade_date'].nunique()} 个交易日）")

    print(">>> 3/4 全样本 OLS 回归（手写最小二乘）...")
    results = []
    for code, name in STOCKS.items():
        sub = df[df["symbol"] == code].dropna()
        beta, alpha, r2, t_beta, n = ols_beta(sub["stock_excess"], sub["mkt_excess"])
        results.append(
            {"name": name, "code": code, "beta": beta, "alpha": alpha,
             "r2": r2, "t_beta": t_beta, "n": n}
        )
    res = pd.DataFrame(results).set_index("name")
    print(res.round(3).to_string())

    # 对照：jh_quant 内置的 calculate_exposures（同一份数据，应得到几乎一样的 beta）
    print(">>> 3.5 对照：jh_quant.factors.calculate_exposures ...")
    try:
        from jh_quant.factors import calculate_exposures
        stock_ret = df[["symbol", "trade_date", "stock_excess"]].rename(
            columns={"symbol": "symbol", "trade_date": "date", "stock_excess": "return"}
        ).dropna()
        factor_ret = (
            df[["trade_date", "mkt_excess"]].drop_duplicates("trade_date")
            .set_index("trade_date")[["mkt_excess"]].rename(columns={"mkt_excess": "mkt"})
        )
        exposure = calculate_exposures(stock_ret, factor_ret, period="D", lookback=252)
        print("    jh_quant 内置 beta（全样本）:")
        for code, name in STOCKS.items():
            b = exposure[exposure["symbol"] == code]["mkt"].mean()
            print(f"      {name}: {b:.3f}")
    except Exception as e:  # 内置接口或环境缺失时不影响主流程
        print(f"    [跳过] calculate_exposures 不可用: {e}")

    print(">>> 4/4 绘图（mpl_style 统一风格）...")
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (code, name) in zip(axes, STOCKS.items()):
        sub = df[df["symbol"] == code].dropna()
        ax.scatter(sub["mkt_excess"], sub["stock_excess"], s=8, alpha=0.45,
                   color=mpl_style.COLOR_CYCLE[0], label="日收益")
        beta, alpha, r2, _, _ = ols_beta(sub["stock_excess"], sub["mkt_excess"])
        xline = np.linspace(sub["mkt_excess"].min(), sub["mkt_excess"].max(), 50)
        ax.plot(xline, alpha + beta * xline, color=mpl_style.RISE, lw=2.2,
                label=f"回归线 β={beta:.2f}")
        ax.axhline(0, color="#B2BABB", lw=0.8, ls="--")
        ax.axvline(0, color="#B2BABB", lw=0.8, ls="--")
        ax.set_title(f"{name}\nβ={beta:.2f}  R²={r2:.2f}", fontsize=12)
        ax.set_xlabel("沪深300 日收益 (%)")
        ax.set_ylabel("个股日收益 (%)")
        ax.legend(loc="upper left", fontsize=9)
        mpl_style.hide_spines(ax)
    fig1.suptitle("三只股票 vs 沪深300：日收益散点与市场模型回归线", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("output/fig1_scatter_regression.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig1_scatter_regression.png")

    fig2, ax = plt.subplots(figsize=(8, 5))
    names = [r["name"] for r in results]
    betas = [r["beta"] for r in results]
    colors = [mpl_style.FALL if b < 1 else mpl_style.RISE for b in betas]
    bars = ax.bar(names, betas, color=colors, width=0.5)
    ax.axhline(1.0, color="#7F8C8D", lw=1.2, ls="--", label="β = 1（与大盘同步）")
    for bar, b in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{b:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Beta")
    ax.set_title("三只股票的 CAPM 市场 Beta（2020–2024）", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(betas) * 1.25)
    ax.legend(fontsize=10)
    mpl_style.hide_spines(ax)
    fig2.tight_layout()
    fig2.savefig("output/fig2_beta_compare.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_beta_compare.png")

    fig3, ax = plt.subplots(figsize=(11, 5))
    for code, name in STOCKS.items():
        sub = df[df["symbol"] == code].dropna().sort_values("trade_date")
        rb = rolling_beta(sub, window=60)
        ax.plot(sub["trade_date"], rb, lw=1.6, label=f"{name}")
    ax.set_ylabel("60 日滚动 Beta")
    ax.set_title("三只股票的 60 个交易日滚动 Beta（2020–2024）", fontsize=14, fontweight="bold")
    ax.axhline(1.0, color="#7F8C8D", lw=1, ls="--")
    ax.legend(fontsize=10, ncol=3)
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("output/fig3_rolling_beta.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_rolling_beta.png")

    print("\n完成。三张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
