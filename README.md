# 2025 CUMCM A：烟幕干扰弹的投放策略

2025 年高教社杯全国大学生数学建模竞赛 A 题的完整求解仓库：以确定性运动学方程
描述导弹、无人机、干扰弹与烟幕云团的运动，把遮蔽问题转化为「导弹—目标视线」
与烟幕球域的几何相交判定，再对投放参数做全局优化，依次求解问题 1–5，
并给出验证、敏感性分析与论文。

## 交付物

| 交付物 | 位置 |
| --- | --- |
| 论文 PDF（25 页） | [`main.pdf`](main.pdf) |
| 论文 LaTeX 源码与插图 | [`paper/main.tex`](paper/main.tex)、`paper/figs/` |
| 竞赛提交结果（题干列格式） | [`提交结果/`](提交结果)、[`说明.md`](提交结果/说明.md) |
| 完整数值结果 | `results/q1_best.json` … `results/q5_best.json`、`results/result*.xlsx` |
| 结果图表 | `figures/` |
| 题目原文与建模提示词 | `Prompt/` |

`result1.xlsx` → 问题 3（FY1 三弹对 M1），`result2.xlsx` → 问题 4（FY1–FY3 各一弹对 M1），
`result3.xlsx` → 问题 5（5 架无人机、每架至多 3 枚弹对 M1–M3）。

## 结果摘要

| 问题 | 场景 | 决策维度 | 遮蔽时长 (s) |
| --- | --- | ---: | ---: |
| 问题 1 | FY1 固定策略（$v=120\,\mathrm{m/s}$、$\theta=\pi$、$t_d=1.5\,\mathrm{s}$、$\tau=3.6\,\mathrm{s}$）单弹对 M1 | — | 1.410 |
| 问题 2 | FY1 单弹优化对 M1 | 4 | 4.667 |
| 问题 3 | FY1 同机三弹协同对 M1 | 8 | 7.740 |
| 问题 4 | FY1、FY2、FY3 各一弹对 M1 | 12 | 14.335 |
| 问题 5 | 5 机至多 15 弹对 M1–M3 | 40 | 20.947（M1 12.145 / M2 4.855 / M3 3.947） |

表中的遮蔽时长为「各弹遮蔽区间并集」的总长度，与 `results/*.json` 逐位一致。
问题 5 的数值为当前提交版本；该解已知不是全局最优，见文末「已知局限」。

## 模型要点

- **导弹**：以 $300\,\mathrm{m/s}$ 沿初始位置指向假目标中心的直线匀速飞行，
  命中时刻由位置矢量模长与速度之比确定。
- **无人机**：等高度匀速直线飞行，速度 $70\text{–}140\,\mathrm{m/s}$、航向角自由；
  同一架无人机的所有弹共享同一航向与速度。
- **干扰弹**：脱离后仅受重力（忽略空气阻力），起爆时刻 $t_e=t_d+\tau$，
  起爆点由斜抛公式给出。
- **烟幕云团**：起爆瞬间形成半径 $10\,\mathrm{m}$ 的球，中心以 $3\,\mathrm{m/s}$ 匀速下沉，
  起爆后 $20\,\mathrm{s}$ 内有效。
- **遮蔽判据**：某时刻云团球心到「导弹位置—假目标中心」线段的距离不超过
  $10\,\mathrm{m}$ 即判定该时刻遮蔽（另有更严格的圆柱体判据用于敏感性分析）。
- **目标**：总遮蔽时长等于三枚导弹各自遮蔽区间并集长度之和；约束为起爆高度
  $\ge 60\,\mathrm{m}$ 与同机两弹投放间隔 $\ge 1\,\mathrm{s}$。

## 算法要点

- **区间精化**：先在 $0.1\,\mathrm{s}$ 网格上定位遮蔽连通段，再用 `brentq`
  精化区间端点，得到数值精度内的遮蔽时长（`src/shielding.py`）。
- **结构化时序编码**：同机三弹用「排序变量 $+$ $1\,\mathrm{s}$ 递增」编码投放时刻，
  引信延时上界由起爆高度约束反推，硬约束自动满足（`src/objective.py`、论文第 5 节）。
- **全局优化**：几何启发式构造可行初始种群（`src/heuristics.py`），
  差分进化做全局搜索（CPU `scipy` 版本与自实现 GPU 版本），
  再用 L-BFGS-B 与有界 Nelder-Mead 局部精修（`src/optimize.py`）。
- **问题 5 加速**：Triton 批量目标核函数把整个 DE 种群映射到 GPU，
  单代评估由 CPU 顺序的约 $0.65\,\mathrm{s}$ 降到 $0.5\,\mathrm{ms}$ 量级；
  最终解一律用 CPU `brentq` 精化重新计算（`src/triton_objective.py`、`src/optimize_gpu.py`）。
- **问题 1**：按题面固定策略直接计算，并用 $0.005\,\mathrm{s}$ 稠密采样交叉验证，
  两者遮蔽时长与区间端点一致。

## 目录结构

```text
2025_cumcm_A/
├── README.md
├── main.pdf                    # 论文（25 页）
├── paper/                      # 论文 LaTeX 源码
│   ├── main.tex
│   └── figs/                   # 论文插图（与 figures/ 内容一致）
├── Prompt/                     # 题目原文与建模提示词
├── config.yaml                 # 全局参数、优化预算、随机种子
├── requirements.txt
├── src/
│   ├── config.py               # 坐标系、导弹/无人机参数、优化预算
│   ├── geometry.py             # 点—线段距离与区间代数
│   ├── kinematics.py           # 导弹/无人机/弹丸/云团运动学
│   ├── shielding.py            # 遮蔽区间（粗采样 + brentq 精化）
│   ├── objective.py            # Q2–Q5 变量编解码与目标函数
│   ├── heuristics.py           # 几何启发式可行种子
│   ├── optimize.py             # 差分进化 + L-BFGS-B/Nelder-Mead 精修
│   ├── triton_objective.py     # Q5 批量目标 Triton 核函数
│   ├── optimize_gpu.py         # GPU 批量差分进化
│   ├── problems.py             # 问题 1–5 求解入口
│   ├── export.py               # result*.xlsx 导出
│   ├── sensitivity.py          # 敏感性分析与三维响应面
│   └── plotting.py             # 甘特图、三维轨迹、收敛曲线
├── scripts/
│   ├── run_all.py              # 一键求解问题 1–5
│   ├── run_q1.py … run_q5.py   # 逐题运行并出图
│   ├── run_sensitivity.py      # 敏感性分析
│   ├── make_fig5.py            # 论文第 5 节示意图
│   ├── collect_results.py      # 归集提交格式结果
│   └── check_q5_lower_bound.py # 问题 5 解的下界交叉检查（见「已知局限」）
├── tests/                      # 13 项单元测试
├── results/                    # JSON 结果、xlsx、敏感性表
├── 提交结果/                    # 题干格式的 result1-3.xlsx 与说明
└── figures/                    # 结果图表
```

## 环境与依赖

已实测通过的解释器与版本：

```text
Python 3.14.4 / 3.12.13
numpy 2.3.5   scipy 1.18.0   pandas 3.0.3
matplotlib 3.11.1   openpyxl 3.1.5   PyYAML   pytest 9.1.1
（可选 GPU 路径）PyTorch 2.13.0+cu132 + Triton 3.7.1，NVIDIA RTX 5070 Laptop (sm_120)
```

```bash
pip install -r requirements.txt
```

GPU 相关依赖仅在问题 5 的加速路径中使用；`torch`/`triton` 不可用时
`src.problems.solve_q5` 会自动回退到 CPU 差分进化。

## 复现命令

```bash
python -m pytest tests -q          # 13 项单元测试
python scripts/run_all.py          # 一键求解问题 1–5，写出 results/*
python scripts/run_q1.py           # 逐题运行（含出图）
python scripts/run_q2.py
python scripts/run_q3.py           # 同时写出 results/result1.xlsx
python scripts/run_q4.py           # 同时写出 results/result2.xlsx
python scripts/run_q5.py           # 同时写出 results/result3.xlsx
python scripts/run_sensitivity.py  # 敏感性分析 + 三维响应面
python scripts/collect_results.py  # 归集到 提交结果/
python scripts/check_q5_lower_bound.py   # 问题 5 解的下界交叉检查
```

所有随机种子固定为 `2025`（`config.yaml`），问题 2–4 的差分进化、局部精修与
问题 5 的 GPU 批量差分进化均使用该种子，重复运行结果一致。

## 验证情况

- `python -m pytest tests -q` → **13 passed**，覆盖几何距离、运动学、遮蔽区间、
  时序编码与 xlsx 导出。
- 问题 1 直接计算与 $0.005\,\mathrm{s}$ 稠密采样交叉验证：区间端点误差小于
  $0.01\,\mathrm{s}$；重算得到 $1.410197\,\mathrm{s}$，与 `results/q1_best.json` 一致。
- 问题 5 的解分别用 GPU 粗网格、CPU 粗网格与 CPU 精化三套评估交叉检查，
  三种口径的偏差在 $0.3\,\mathrm{s}$ 以内。
- 敏感性分析覆盖 $g$、云团半径、最低起爆高度与圆柱体判据四种扰动
  （见 `results/sensitivity_table.json` 与论文第 6 节）。

## 已知局限

1. **官方 `result*.xlsx` 模板缺失**。题目附件未提供模板，三个结果文件按建模提示词
   第 9.2 节的默认列规范输出；列含义见 [`提交结果/说明.md`](提交结果/说明.md)。
   若拿到官方模板，应由 `src/export.py` 按模板列名与顺序重新导出。
2. **问题 5 的解不是全局最优**。问题 4 得到的 FY1–FY3 各一弹策略在问题 5 的可行域内
   同样可行，因此「FY1–FY3 沿用问题 4 的解、FY4/FY5 沿用问题 5 的解」这一 9 弹方案
   给出了更高的目标值：

   ```text
   $ python scripts/check_q5_lower_bound.py
   组合方案（9 弹）: 可行=True 总遮蔽=25.617 s (M1 21.669 / M2 0.000 / M3 3.947)
   results/q5_best.json 记录的解: 20.947 s
   ```

   即当前 `result3.xlsx` 记录的解可被非最优可行解严格支配，说明 GPU 批量差分进化
   在 40 维空间中收敛不足（其代理目标未含起爆高度惩罚项，且初始种群未包含问题 4 的解）。
   论文中的 $20.947\,\mathrm{s}$ 应理解为「本仓库当前提交版本的结果」，而非全局最优值。
3. **三枚导弹未出现同时遮蔽**（同时遮蔽时长为 $0\,\mathrm{s}$）。当前目标函数为三枚导弹
   遮蔽时长之和，未包含「最差导弹遮蔽时长」或「同时遮蔽」项；若需要同时致盲，
   应把 $\min_m T_m$ 一类指标纳入目标函数重新求解。
4. **问题 5 存在资源冗余**。15 枚弹中仅 3 枚对所属导弹产生非零遮蔽时长，
   其余弹位的贡献为 $0\,\mathrm{s}$；这与局限 2 同源。
5. `data/`、`build/` 为空目录，不纳入版本控制；`TASK.md` 与 `.agent/` 为本地工作流
   记录，已加入 `.gitignore`。

## 许可

仓库沿用初始化时的 [GPL-3.0](LICENSE) 许可。
