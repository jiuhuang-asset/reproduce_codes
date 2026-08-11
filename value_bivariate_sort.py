# -*- coding: utf-8 -*-
"""
价值因子与二维排序（市值 x 账面市值比，A股月度实证）
=====================================================

对应公众号文章《价值因子与二维排序》（量海泛舟 · 资产定价系列第五篇）
方法论参考：https://www.tidy-finance.org/chapters/value-and-bivariate-sorts.html

做什么
------
把全 A 股按「市值」和「账面市值比（BM = 1/PB，越便宜 BM 越高）」两个维度同时
分组，构造 5x5 共 25 个组合，用市值加权收益研究两件事：
  1. 控制市值后，价值股是否仍有溢价？（value premium）
  2. 市值在组合内部的主导作用有多大？（等权 vs 市值加权的差异）

价值溢价做法（参考 tidy-finance）
--------------------------------
做多最高 BM 的 5 个组合，做空最低 BM 的 5 个组合：
  每个 BM 层内 5 个 size 组合等权平均，构成该 BM 层的收益，
  VMG = 高BM层 - 低BM层。

关键设计（延续系列惯例）
------------------------
- 月度再平衡：t 月末市值/BM 分组，持有 t+1 月收益（滞后一期，避免前视偏差）
- BM 用月末 PB 倒数（TS_MONTHLY_BASIC 的 pb 字段，取月末值；随行情更新，天然月度化）
- 剔除上市不足 12 个月的次新股
- 组合内市值加权；5 个 size 组合之间等权（同 tidy-finance）

运行方式
--------
1. 安装依赖： pip install jh-quant matplotlib
2. 设置 JIUHUANG_API_KEY（从 https://jiuhuang.xyz 申请，或放 .env）
3. 运行：      python value_bivariate_sort.py

输出
----
控制台打印 5x5 平均月收益矩阵与价值溢价统计，
并生成 3 张图：
  - fig1_bm_matrix.png        5x5 平均月收益热力图
  - fig2_vmg_cumulative.png   价值多空(VMG) vs 市值多空(SMB)累计净值
  - fig3_cap_share_matrix.png 5x5 总市值占比热力图（说明市值主导）
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

# 二维排序的层数
N_BUCKETS = 5


def fetch_data():
    """用 jh_quant 拉取：月度行情（含 PB）、月度前复权行情、上市日期。"""
    from jh_quant.data import JHData, DataTypes

    jh = JHData()

    # 上市日期（剔除次新股）
    basic = jh.get_data(DataTypes.TS_STOCK_BASIC).to_df()
    basic = basic[["ts_code", "list_date"]].dropna()
    basic["list_date"] = pd.to_datetime(basic["list_date"], format="%Y%m%d")

    # 月度基本面表：含 total_mv(市值) 与 pb(市净率) —— BM = 1/PB
    mb = jh.get_data(DataTypes.TS_MONTHLY_BASIC, start=CAP_START, end=RET_END).to_df()
    mb = mb[["trade_date", "ts_code", "total_mv", "pb"]].dropna(subset=["total_mv"])
    mb["trade_date"] = pd.to_datetime(mb["trade_date"])
    mb["ym"] = mb["trade_date"].dt.to_period("M")
    # 每月每只股票取当月最后一次记录（市值 + PB）
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

    return basic, mb, mpx


def build_panel(basic, mb, mpx):
    """构建月度面板：收益 + 滞后一期市值/BM + 剔次新。"""
    ret = mpx[(mpx["trade_date"] >= RET_START)].dropna(subset=["ret"]).copy()
    ret["ym"] = ret["trade_date"].dt.to_period("M")

    # 市值 / BM 都滞后一期（t-1 月末已知 -> t 月收益）
    mb["bm"] = 1.0 / mb["pb"]  # 账面市值比 = 1 / 市净率（PB 越高越贵）
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


def assign_quintile(s):
    """横截面五分位：返回 1(最小/最便宜) ... 5(最大/最贵)。"""
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    clean = s.dropna()
    if clean.empty:
        return out
    q = clean.quantile(np.linspace(0.2, 0.8, 4))
    labels = pd.cut(clean, bins=[-np.inf, *q.values, np.inf],
                    labels=False, duplicates="drop") + 1
    out.loc[labels.index] = labels
    return out


def main():
    print(">>> 1/3 拉取数据（jh_quant）...")
    basic, mb, mpx = fetch_data()
    print(f"    月度基本面: {len(mb)} 行, 行情: {len(mpx)} 行")

    print(">>> 2/3 构建面板 + 5x5 二维排序 ...")
    panel = build_panel(basic, mb, mpx)
    print(f"    面板: {len(panel)} 行, 股票 {panel['ts_code'].nunique()} 只")

    # 每月横截面：分别给市值、BM 打五分位（independent sorts）
    panel["size_q"] = panel.groupby("ym")["mktcap"].transform(assign_quintile)
    panel["bm_q"] = panel.groupby("ym")["bm"].transform(assign_quintile)
    panel = panel.dropna(subset=["size_q", "bm_q"])
    # size_q=5 大盘 / size_q=1 小盘；bm_q=5 最便宜(价值) / bm_q=1 最贵(成长)

    # 每月市值加权收益矩阵（行=size 5大..1小, 列=bm 1成长..5价值）
    monthly = {}
    for ym, grp in panel.groupby("ym"):
        if grp.empty:
            continue
        mat = np.full((N_BUCKETS, N_BUCKETS), np.nan)
        for (sq, bq), sub in grp.groupby(["size_q", "bm_q"]):
            w = sub["mktcap"].clip(lower=0)
            mat[int(sq) - 1, int(bq) - 1] = (sub["ret"] * w).sum() / w.sum() if w.sum() > 0 else sub["ret"].mean()
        monthly[ym] = mat
    # 按 size_q 从大到小（5大盘->1小盘）存，方便画图
    arr = np.stack(list(monthly.values()))  # (月, 5, 5)，行=size_q 1..5, 列=bm_q 1..5

    # 平均月收益矩阵（%）
    mean_mat = np.nanmean(arr, axis=0) * 100  # 行=size_q 1(小)..5(大), 列=bm_q 1(成长)..5(价值)
    print("\n5x5 平均月收益矩阵（%，行=size 小->大, 列=bm 成长->价值）：")
    print(pd.DataFrame(mean_mat, index=[f"S{i}" for i in range(1, 6)],
                       columns=[f"B{i}" for i in range(1, 6)]).round(2).to_string())

    # 价值溢价 VMG = 高BM层 - 低BM层（每层 5 个 size 组合等权平均）
    # 每层收益：arr 沿 size 轴平均（等权平均 5 个组合）
    layer_ret = np.nanmean(arr, axis=1)  # (月, bm_q 1..5)
    vmg = layer_ret[:, 4] - layer_ret[:, 0]  # bm_q=5 价值 - bm_q=1 成长
    vmg_mean = vmg.mean() * 100
    vmg_t = vmg.mean() / (vmg.std(ddof=1) / np.sqrt(len(vmg)))
    print(f"\n价值溢价 VMG = 价值层 - 成长层：")
    print(f"  平均月收益 {vmg_mean:.2f}% | t 值 {vmg_t:.2f}（样本 {len(vmg)} 个月）")

    # 市值溢价 SMB = 小盘层 - 大盘层（每层 5 个 BM 组合等权平均）
    size_layer = np.nanmean(arr, axis=2)  # (月, size_q 1..5)
    smb = size_layer[:, 0] - size_layer[:, 4]  # size_q=1 小盘 - size_q=5 大盘
    smb_mean = smb.mean() * 100
    print(f"  同样本期市值溢价 SMB = {smb_mean:.2f}%")

    # 总市值占比矩阵（%）—— 各 (size, bm) 组合的市值占全市场比例
    cap_sum = panel.groupby(["size_q", "bm_q"])["mktcap"].sum()
    cap_total = cap_sum.sum()
    cap_share = cap_sum / cap_total * 100  # Series[(sq, bq)]
    cap_mat = np.zeros((N_BUCKETS, N_BUCKETS))
    for (sq, bq), v in cap_share.items():
        cap_mat[int(sq) - 1, int(bq) - 1] = v

    print("\n5x5 总市值占比矩阵（%，行=size 小->大, 列=bm 成长->价值）：")
    print(pd.DataFrame(cap_mat, index=[f"S{i}" for i in range(1, 6)],
                       columns=[f"B{i}" for i in range(1, 6)]).round(1).to_string())

    print(">>> 3/3 绘图（mpl_style 统一风格）...")
    # 图1：5x5 平均月收益热力图
    fig1, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(mean_mat, cmap="RdBu_r", vmin=-1.5, vmax=3.0, aspect="auto")
    for i in range(5):
        for j in range(5):
            v = mean_mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if abs(v - np.nanmean(mean_mat)) > 0.8 else "black", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"成长\nB{i}" for i in range(1, 6)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"S{i}\n小盘" if i == 1 else f"S{i}" for i in range(1, 6)])
    ax.set_xlabel("账面市值比（B1 成长 → B5 价值）")
    ax.set_ylabel("市值（S1 小盘 → S5 大盘）")
    ax.set_title("5×5 二维排序的平均月收益（% ，2015-2024）", fontsize=14, fontweight="bold")
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    mpl_style.hide_spines(ax)
    fig1.tight_layout()
    fig1.savefig("fig1_bm_matrix.png", dpi=200, bbox_inches="tight")
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
    fig2.savefig("fig2_vmg_cumulative.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig2_vmg_cumulative.png")

    # 图3：5x5 总市值占比热力图（说明市值主导）
    fig3, ax = plt.subplots(figsize=(8, 6.5))
    im3 = ax.imshow(cap_mat, cmap="YlOrBr", aspect="auto")
    for i in range(5):
        for j in range(5):
            v = cap_mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 20 else "black", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"成长\nB{i}" for i in range(1, 6)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f"S{i}\n小盘" if i == 1 else f"S{i}" for i in range(1, 6)])
    ax.set_xlabel("账面市值比（B1 成长 → B5 价值）")
    ax.set_ylabel("市值（S1 小盘 → S5 大盘）")
    ax.set_title("5×5 组合占总市值比例（%）", fontsize=14, fontweight="bold")
    fig3.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    mpl_style.hide_spines(ax)
    fig3.tight_layout()
    fig3.savefig("fig3_cap_share_matrix.png", dpi=200, bbox_inches="tight")
    print("    已保存 fig3_cap_share_matrix.png")

    print("\n完成。三张图与本文对应，可插入公众号文章。")


if __name__ == "__main__":
    main()
