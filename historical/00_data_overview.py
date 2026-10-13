# -*- coding: utf-8 -*-
"""
00_data_overview.py
铝合金数据集加载、样本统计、验证样本选择与成分空间可视化（t-SNE）。
输出：data/01_processed_data.csv、data/overview_stats.csv、data/verification_holdout.csv、
      figure/fig1_composition_space.png（图1：(a) t-SNE 成分空间 (b) 三目标分布，4 牌号配色）
环境：LubLLM

数据说明：源文件 strength 两列（yield_strength_mpa / ultimate_tensile_strength_mpa）已满足
YS<=UTS 物理约束（此前存在的两列整体互换已在源文件中修正）。本脚本另剔除 1 个物理一致性
异常样本（al_227：牌号标注 6082，但强度标注接近 7075 水平，详见论文数据审计部分），
保留 288 个有效样本进入后续流程。
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
FIG_DIR = os.path.join(BASE, 'figure')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

XLSX = os.environ.get('ALLOY_SOURCE_WORKBOOK', 'source_workbook_not_deposited.xlsx')

# nature-skill 风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.9,
    'legend.frameon': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# 4 牌号配色（nature-skill PALETTE）；牌号键为整数（与 alloy_or_coating_name 列的 dtype 一致）
GRADE_COLOR = {2024: '#B64342', 5083: '#8BCF8B', 6082: '#0F4D92', 7075: '#9A4D8E'}
GRADE_MARKER = {2024: 'o', 5083: 's', 6082: '^', 7075: 'D'}

COMP_COLS = ['Al_wt_pct', 'Ti_wt_pct', 'Fe_wt_pct', 'Ni_wt_pct', 'Cr_wt_pct',
             'Mn_wt_pct', 'Si_wt_pct', 'Cu_wt_pct', 'Zn_wt_pct', 'Mg_wt_pct',
             'Mo_wt_pct', 'V_wt_pct', 'Nb_wt_pct', 'Zr_wt_pct', 'Sn_wt_pct',
             'Co_wt_pct', 'C_wt_pct', 'N_wt_pct', 'O_wt_pct', 'B_wt_pct',
             'P_wt_pct', 'S_wt_pct']
TARGETS = ['yield_strength_mpa', 'ultimate_tensile_strength_mpa', 'elongation_pct']
TARGET_CN = {'yield_strength_mpa': '屈服强度 YS (MPa)',
             'ultimate_tensile_strength_mpa': '抗拉强度 UTS (MPa)',
             'elongation_pct': '延伸率 EL (%)'}


def load_al_data():
    """加载铝合金板材样本（剔除物理一致性异常样本 al_227 后共 288 个）。"""
    df = pd.read_excel(XLSX, sheet_name='metal_samples', header=1)
    df = df.iloc[7:].copy()
    df = df[df['material_system'].isin(['铝合金'])].copy()
    df = df[df['sample_id'].notna()].copy()
    for c in COMP_COLS + TARGETS:
        df[c] = df[c].apply(lambda v: 0 if v in ('无', '不关注', '/', '', None) else v)
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 数据审计：剔除成分-性能物理一致性异常样本 al_227（6082 成分 + 7075 级强度标注）
    df = df[df['sample_id'] != 'al_227'].copy()

    assert len(df) == 288, f'期望 288 个样本（剔除 al_227 后），实际 {len(df)}'
    assert (df['ultimate_tensile_strength_mpa'] >= df['yield_strength_mpa']).all(), '仍存在 UTS<YS'
    return df.reset_index(drop=True)


def overview_stats(df):
    rows = []
    for grade, sub in df.groupby('alloy_or_coating_name'):
        row = {'牌号': grade, '体系': '铝合金', '样本数': len(sub)}
        for t in TARGETS:
            s = sub[t]
            row[TARGET_CN[t]] = f'{s.min():.0f}~{s.max():.0f} ({s.mean():.0f}±{s.std():.0f})'
        rows.append(row)
    row = {'牌号': '铝合金（合计）', '体系': '铝合金', '样本数': len(df)}
    for t in TARGETS:
        s = df[t]
        row[TARGET_CN[t]] = f'{s.min():.0f}~{s.max():.0f} ({s.mean():.1f}±{s.std():.1f})'
    rows.append(row)
    return pd.DataFrame(rows)


def composition_space_figure(df):
    """图1：(a) t-SNE 成分空间（4 牌号 4 色）；(b) 三目标分布（4 牌号分层直方图）。"""
    comp = df[COMP_COLS].copy()
    comp = comp.loc[:, (comp != 0).any(axis=0)]
    X = StandardScaler().fit_transform(comp)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca',
                learning_rate='auto')
    Xt = tsne.fit_transform(X)

    fig = plt.figure(figsize=(14.4, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.16,
                          left=0.06, right=0.97, top=0.92, bottom=0.13)
    ax = fig.add_subplot(gs[0, 0])
    for grade, color in GRADE_COLOR.items():
        grp = df[df['alloy_or_coating_name'] == grade]
        ax.scatter(Xt[grp.index, 0], Xt[grp.index, 1], c=color, s=60, alpha=0.9,
                   edgecolors='white', linewidths=0.6, marker=GRADE_MARKER[grade], label=grade)
    ax.set_xlabel('t-SNE 1', fontsize=14)
    ax.set_ylabel('t-SNE 2', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc='lower right', ncol=2, columnspacing=0.9,
              handletextpad=0.3, frameon=False)
    ax.text(0.02, 1.05, '(a) Composition space (t-SNE)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom')

    gs_b = gs[0, 1].subgridspec(1, 3, wspace=0.24)
    hist_specs = [('yield_strength_mpa', 'YS (MPa)'),
                  ('ultimate_tensile_strength_mpa', 'UTS (MPa)'),
                  ('elongation_pct', 'EL (%)')]
    for i, (t, xlab) in enumerate(hist_specs):
        axh = fig.add_subplot(gs_b[0, i])
        for grade, color in GRADE_COLOR.items():
            grp = df[df['alloy_or_coating_name'] == grade]
            axh.hist(grp[t], bins=9, density=True, histtype='stepfilled',
                     alpha=0.28, color=color, edgecolor='none')
            axh.hist(grp[t], bins=9, density=True, histtype='step',
                     linewidth=0.9, color=color)
        axh.set_xlabel(xlab, fontsize=14)
        axh.tick_params(labelsize=11)
        if i == 0:
            axh.set_ylabel('Density', fontsize=14)
            axh.legend(list(GRADE_COLOR.keys()), fontsize=11, loc='upper left', frameon=False)
    ax_b0 = fig.axes[1]
    ax_b0.text(0.02, 1.05, '(b) Target distributions', transform=ax_b0.transAxes,
               fontsize=14, fontweight='bold', va='bottom')
    fig.savefig(os.path.join(FIG_DIR, 'fig1_composition_space.png'), dpi=600,
                bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(FIG_DIR, 'fig1_composition_space.png'))


def select_verification_samples(df, n_holdout=4):
    """兼顾度 = 三目标 min-max 归一化后的最小值；每牌号取兼顾度最优 1 名，共 4 名。"""
    norms = {}
    for t in TARGETS:
        lo, hi = df[t].min(), df[t].max()
        norms[t] = (df[t] - lo) / (hi - lo) if hi > lo else 0.0
    score = pd.concat(norms.values(), axis=1).min(axis=1)
    work = df.copy()
    work['balance_score'] = score

    picked_idx = []
    # 每牌号兼顾度最优 1 名（剔除 al_227 后 6082 自动取次优干净样本）
    for grade, sub in work.groupby('alloy_or_coating_name'):
        picked_idx.append(sub['balance_score'].idxmax())

    holdout = work.loc[picked_idx]
    train = work.drop(index=picked_idx)
    assert len(holdout) == n_holdout and len(train) == len(df) - n_holdout
    return holdout, train


def main():
    df = load_al_data()
    holdout, train = select_verification_samples(df, n_holdout=4)
    holdout.to_csv(os.path.join(DATA_DIR, 'verification_holdout.csv'),
                   index=False, encoding='utf-8-sig')
    print('验证保留样本:')
    for _, r in holdout.iterrows():
        print(f"  {r['sample_id']} {r['alloy_or_coating_name']} "
              f"YS={r['yield_strength_mpa']:.0f} UTS={r['ultimate_tensile_strength_mpa']:.0f} "
              f"EL={r['elongation_pct']:.1f} 兼顾度={r['balance_score']:.3f}")

    train = train.drop(columns=['balance_score']).reset_index(drop=True)
    assert len(train) == 284, f'期望 284 个训练样本，实际 {len(train)}'
    print('训练样本数:', len(train))
    print(train['alloy_or_coating_name'].value_counts().to_dict())

    stats = overview_stats(train)
    stats.to_csv(os.path.join(DATA_DIR, 'overview_stats.csv'), index=False, encoding='utf-8-sig')
    train.to_csv(os.path.join(DATA_DIR, '01_processed_data.csv'), index=False, encoding='utf-8-sig')
    composition_space_figure(train)
    print('00 done')


if __name__ == '__main__':
    main()
