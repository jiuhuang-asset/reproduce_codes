# reproduce_codes

[量海泛舟](https://github.com/jiuhuang-asset/lhfz) 公众号文章的**配套可复现代码仓库**。

每篇公众号文章的分析流程，都能在本仓库找到一个（或多个）独立可跑的 Python 脚本。读者安装依赖、配置 `JIUHUANG_API_KEY` 后，即可一键复现文章中的数据、图表与结论。

## 文章 → 代码索引

| 公众号文章（量海泛舟） | 复现代码 |
|---|---|
| [资产定价入门 · Beta 是什么，用 A 股数据算给你看](https://github.com/jiuhuang-asset/lhfz/blob/main/articles/factor_analysis/001_asset-pricing-beta-a-share.md) | [beta_estimation_a_share.py](beta_estimation_a_share.py) |
| [投资组合排序 · 小盘股真的跑赢大盘股吗？](https://github.com/jiuhuang-asset/lhfz/blob/main/articles/factor_analysis/002_portfolio-sort-size-a-share.md) | [size_portfolio_sorts.py](size_portfolio_sorts.py) |
| [价值因子与三因子模型：从二维排序到 FF3 归因](https://github.com/jiuhuang-asset/lhfz/blob/main/articles/factor_analysis/003_value-factor-and-ff3.md) | [ff3_factor_model.py](ff3_factor_model.py) |
| [因子真的被定价了吗？Fama-MacBeth 回归](https://github.com/jiuhuang-asset/lhfz/blob/main/articles/factor_analysis/004_fama-macbeth-regression.md) | [fama_macbeth_regression.py](fama_macbeth_regression.py) |
| [资产组合入门 · 马科维茨的均值方差理论](https://github.com/jiuhuang-asset/lhfz/blob/main/articles/portfolio/001_mean-variance-theory.md) | [mean_variance_portfolio.py](mean_variance_portfolio.py) |
| [资产组合 · 均值方差理论的边界：从约束回测到参数化策略](https://github.com/jiuhuang-asset/lhfz/blob/main/articles/portfolio/002_from-backtest-to-parametric-policy.md) | [constrained_portfolio_backtest.py](constrained_portfolio_backtest.py) 、 [parametric_portfolio_policy.py](parametric_portfolio_policy.py) |

## 公共文件

- `mpl_style.py` — 全仓库统一的 matplotlib 图表风格（中文字体、涨跌配色、网格、dpi 等），所有脚本从该文件导入，保持文章配图风格一致。
- `output/` — 脚本运行产出的图片目录（脚本会自动创建）。

## 运行方式

1. 安装依赖：`pip install jh-quant matplotlib`（部分脚本另需 `scipy` / `riskfolio-lib`，以脚本头部说明为准）。
2. 配置数据源：设置环境变量 `JIUHUANG_API_KEY`（从 https://jiuhuang.xyz 申请），或在仓库根目录放 `.env` 文件，`JHData` 会自动读取。
3. 运行：`python <脚本名>.py`，控制台打印关键结果，图片统一输出到 `output/`。
