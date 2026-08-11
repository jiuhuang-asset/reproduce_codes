"""
统一 matplotlib 图表风格 —— 量海泛舟（lhfz）

所有公众号文章的图表都必须引用本文件，保证全号视觉统一。
复现代码（../reproduce_codes）引用时，把本文件复制到该仓库并同样导入。

用法：
    import mpl_style  # 或在独立脚本里 exec(open("mpl_style.py").read())
    import matplotlib.pyplot as plt
    ...常规绘图即可，全局风格已生效...

配色约定（A股/中国市场习惯）：
    - 上涨 / 正向用红色 RISE，下跌 / 负向用绿色 FALL
    - 序列线用 COLOR_CYCLE，强调色用 ACCENT
"""

import matplotlib

# 无交互后端也允许存图（复现脚本在无显示环境下可运行）
# 注意：必须先 use() 再 import pyplot，否则后端切换不生效
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# 中文字体：Windows 优先 Microsoft YaHei，macOS/主流发行版回退列表
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# ============ 统一调色板 ============
RISE = "#C0392B"    # 涨 / 正（红，A股习惯）
FALL = "#1E8449"    # 跌 / 负（绿）
ACCENT = "#2C5F8A"  # 主强调色（深蓝，金融数据分析基调）
ACCENT_2 = "#E67E22"  # 次强调色（橙）

COLOR_CYCLE = [
    "#2C5F8A",  # 深蓝
    "#C0392B",  # 红
    "#1E8449",  # 绿
    "#E67E22",  # 橙
    "#8E44AD",  # 紫
    "#16A085",  # 青
    "#7F8C8D",  # 灰
    "#D4AC0D",  # 金
]

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_CYCLE)

# ============ 画布 / 字号 ============
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 200        # 微信端高清
plt.rcParams["savefig.bbox"] = "tight"

plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.edgecolor"] = "#CCCCCC"

# ============ 坐标轴 / 网格 ============
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "#D5D8DC"
plt.rcParams["grid.alpha"] = 0.6
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6

plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.edgecolor"] = "#B2BABB"


def hide_spines(ax, keep=("left", "bottom")):
    """去掉顶部/右侧边框（matplotlib 无法用 rcParams 控制，需逐轴设置）。

    用法：fig, ax = plt.subplots(); mpl_style.hide_spines(ax)
    """
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return ax


# ============ 便捷工具函数 ============
def style_axes(ax, title=None, xlabel=None, ylabel=None, legend_loc="best"):
    """给单个 axes 应用一致的标题/标签样式。"""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc)
    return ax


def savefig(fig, path):
    """统一保存图片，控制台提示。"""
    fig.savefig(path)
    print(f"[mpl_style] 图表已保存: {path}")
