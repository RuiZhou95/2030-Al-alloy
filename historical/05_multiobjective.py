# -*- coding: utf-8 -*-
"""
05_multiobjective.py
基于代理模型的多目标权衡分析（铝合金）：
- YS/UTS/EL 三目标均用 RF 作代理（YS 的 RF 测试集 R² 六模型最优；UTS 的 RF 仅次于 GPR 但
  GPR 难以支撑 5×10⁵ 规模采样；EL 的 RF 叶节点均值预测保守稳健）
- 在训练集特征 min-max 包络内以训练样本为锚点扰动采样（质量守恒 Al+各元素=100），
  代理模型预测三目标，非支配排序求 YS-EL 与 UTS-EL 帕累托前沿
- 输出：data/multiobj_*.csv、figure/fig7_multiobjective.png（论文图9）
环境：LubLLM
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
FIG_DIR = os.path.join(BASE, 'figure')

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

TARGETS = ['yield_strength_mpa', 'ultimate_tensile_strength_mpa', 'elongation_pct']

COMP_COLS = ['Ti_wt_pct', 'Fe_wt_pct', 'Ni_wt_pct', 'Cr_wt_pct', 'Mn_wt_pct',
             'Si_wt_pct', 'Cu_wt_pct', 'Zn_wt_pct', 'Mg_wt_pct', 'V_wt_pct', 'Zr_wt_pct']


def pareto_front_2d(x, y, maximize_x=True, maximize_y=True):
    idx = np.argsort(-x if maximize_x else x)
    xs = x[idx]
    ys = y[idx]
    front = []
    best = -np.inf if maximize_y else np.inf
    for i in range(len(xs)):
        if (ys[i] > best) if maximize_y else (ys[i] < best):
            front.append(idx[i])
            best = ys[i]
    return np.array(front)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, '02_engineered_features.csv'), encoding='utf-8-sig')
    feature_cols = [c for c in df.columns
                    if c not in TARGETS + ['sample_id', 'material_system', 'alloy_or_coating_name']]
    X = df[feature_cols].values.astype(float)
    y_all = df[TARGETS].values.astype(float)

    with open(os.path.join(DATA_DIR, 'results.json'), encoding='utf-8') as f:
        results = json.load(f)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_all, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)

    models = {}
    for t, spec in [('yield_strength_mpa', 'RandomForest'),
                    ('ultimate_tensile_strength_mpa', 'RandomForest'),
                    ('elongation_pct', 'RandomForest')]:
        params = results[t][spec]['best_params']
        m = RandomForestRegressor(random_state=42, n_jobs=8, **params)
        m.fit(X_tr_s, y_tr[:, TARGETS.index(t)])
        models[t] = m
        print(f'{t}: {spec} trained')

    # ---------- 包络内采样（训练样本锚点扰动，成分质量守恒 Al+元素=100） ----------
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)
    n_sample = 500000
    rng = np.random.RandomState(42)
    width = hi - lo
    width[width == 0] = 1.0
    j_al = feature_cols.index('Al_wt_pct')
    j_tot = feature_cols.index('total_alloy_wt')
    j_elems = [feature_cols.index(c) for c in COMP_COLS if c in feature_cols]
    n_tr = len(X_tr)
    X_sam = np.empty((n_sample, len(feature_cols)))
    k = 0
    while k < n_sample:
        x = X_tr[rng.randint(0, n_tr)] + \
            rng.uniform(-0.25, 0.25, size=len(feature_cols)) * width
        x = np.clip(x, lo, hi)
        s = x[j_elems].sum()
        al = 100.0 - s
        if lo[j_al] <= al <= hi[j_al]:
            x[j_al] = al
            x[j_tot] = s
            X_sam[k] = x
            k += 1
    print(f'锚点扰动采样完成: {len(X_sam)}')

    X_sam_s = scaler.transform(X_sam)
    ys_pred = models['yield_strength_mpa'].predict(X_sam_s)
    uts_pred = models['ultimate_tensile_strength_mpa'].predict(X_sam_s)
    el_pred = models['elongation_pct'].predict(X_sam_s)

    valid = ys_pred < uts_pred
    ys_pred, uts_pred, el_pred = ys_pred[valid], uts_pred[valid], el_pred[valid]
    X_sam_s = X_sam_s[valid]
    print(f'YS<UTS 物理约束后: {len(ys_pred)}')

    f_ys = pareto_front_2d(ys_pred, el_pred)
    f_uts = pareto_front_2d(uts_pred, el_pred)
    print(f'YS-EL 前沿点数: {len(f_ys)}, UTS-EL 前沿点数: {len(f_uts)}')

    np.savetxt(os.path.join(DATA_DIR, 'multiobj_front_ys.csv'),
               np.column_stack([ys_pred[f_ys], el_pred[f_ys], uts_pred[f_ys]]),
               delimiter=',', header='YS,EL,UTS', comments='')
    np.savetxt(os.path.join(DATA_DIR, 'multiobj_front_uts.csv'),
               np.column_stack([uts_pred[f_uts], el_pred[f_uts], ys_pred[f_uts]]),
               delimiter=',', header='UTS,EL,YS', comments='')

    order = np.argsort(el_pred[f_ys])
    picks = [f_ys[order[int(len(order) * 0.05)]], f_ys[order[int(len(order) * 0.5)]],
             f_ys[order[int(len(order) * 0.95)]]]
    rows = []
    for i, p in enumerate(picks):
        x_orig = scaler.inverse_transform(X_sam_s[p].reshape(1, -1))[0]
        row = {'解': f'解{i + 1}', 'YS/MPa': f'{ys_pred[p]:.0f}', 'UTS/MPa': f'{uts_pred[p]:.0f}',
               'EL/%': f'{el_pred[p]:.1f}'}
        for c in ['Al_wt_pct', 'Cu_wt_pct', 'Mg_wt_pct', 'Zn_wt_pct', 'Si_wt_pct',
                  'Mn_wt_pct', 'rolling_temp_c', 'aging_temp_c']:
            j = feature_cols.index(c)
            row[c] = f'{x_orig[j]:.2f}'
        rows.append(row)
    sol = pd.DataFrame(rows)
    sol.to_csv(os.path.join(DATA_DIR, 'multiobj_solutions.csv'), index=False, encoding='utf-8-sig')
    print(sol[['解', 'YS/MPa', 'UTS/MPa', 'EL/%', 'Al_wt_pct', 'Cu_wt_pct', 'Mg_wt_pct', 'Zn_wt_pct']].to_string(index=False))

    # ---------- 图9 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    sub = rng.choice(len(ys_pred), size=8000, replace=False)
    ax = axes[0]
    ax.scatter(el_pred[sub], ys_pred[sub], s=8, c='#CFCECE', alpha=0.7, edgecolors='none',
               label='Feasible space')
    ax.scatter(el_pred[f_ys], ys_pred[f_ys], s=60, c='#E53935', alpha=0.9,
               edgecolors='white', linewidths=0.6, zorder=4, label='Pareto front')
    ax.scatter(y_tr[:, 2], y_tr[:, 0], s=70, c='#0F4D92', alpha=0.9, edgecolors='white',
               linewidths=0.8, zorder=5, label='Training data')
    ax.set_xlabel('EL (%)', fontsize=14)
    ax.set_ylabel('YS (MPa)', fontsize=14)
    ax.text(0.03, 1.0, '(a) YS–EL trade-off', transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='bottom')
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax = axes[1]
    ax.scatter(el_pred[sub], uts_pred[sub], s=8, c='#CFCECE', alpha=0.7, edgecolors='none',
               label='Feasible space')
    ax.scatter(el_pred[f_uts], uts_pred[f_uts], s=60, c='#E53935', alpha=0.9,
               edgecolors='white', linewidths=0.6, zorder=4, label='Pareto front')
    ax.scatter(y_tr[:, 2], y_tr[:, 1], s=70, c='#0F4D92', alpha=0.9, edgecolors='white',
               linewidths=0.8, zorder=5, label='Training data')
    ax.set_xlabel('EL (%)', fontsize=14)
    ax.set_ylabel('UTS (MPa)', fontsize=14)
    ax.text(0.03, 1.0, '(b) UTS–EL trade-off', transform=ax.transAxes, fontsize=15,
            fontweight='bold', va='bottom')
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc='upper right')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig7_multiobjective.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    print('05 done')


if __name__ == '__main__':
    main()
