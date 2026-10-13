# -*- coding: utf-8 -*-
"""
02_train_models.py
6 模型 × 3 目标训练：Ridge / XGBoost / RandomForest / SVR / GPR / MLP。
超参搜索（5 折 CV）+ 8:2 划分（random_state=42）+ StandardScaler（仅在训练集拟合）。
输出：data/model_summary.csv、data/results.json、figure/fig3_diag_{target}.png
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

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
TARGET_EN = {'yield_strength_mpa': 'YS (MPa)',
             'ultimate_tensile_strength_mpa': 'UTS (MPa)',
             'elongation_pct': 'EL (%)'}
MODEL_NAMES = {'ridge': 'Ridge', 'xgb': 'XGBoost', 'rf': 'RandomForest',
               'svr': 'SVR', 'gpr': 'GPR', 'mlp': 'MLP'}


def get_model(mname):
    if mname == 'ridge':
        return Ridge(random_state=42), {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    if mname == 'xgb':
        return XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4), {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.03, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]}
    if mname == 'rf':
        return RandomForestRegressor(random_state=42, n_jobs=4), {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 4, 8, 12],
            'min_samples_split': [2, 5, 10]}
    if mname == 'svr':
        return SVR(), {
            'C': [0.1, 1.0, 10.0, 100.0, 1000.0],
            'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
            'epsilon': [0.01, 0.05, 0.1, 0.2]}
    if mname == 'gpr':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        n_restarts_optimizer=3, random_state=42), {
            'alpha': [1e-8, 1e-6, 1e-4]}
    if mname == 'mlp':
        return MLPRegressor(random_state=42, max_iter=2000, early_stopping=True), {
            'hidden_layer_sizes': [(16,), (32,), (16, 8), (32, 16)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.0005, 0.001, 0.005]}
    raise ValueError(mname)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, '02_engineered_features.csv'), encoding='utf-8-sig')
    feature_cols = [c for c in df.columns
                    if c not in TARGETS + ['sample_id', 'material_system', 'alloy_or_coating_name']]
    print('特征数:', len(feature_cols))

    summary_rows = []
    results = {}
    for t in TARGETS:
        X = df[feature_cols].values.astype(float)
        y = df[t].values.astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
        results[t] = {}

        for mname in ['ridge', 'xgb', 'rf', 'svr', 'gpr', 'mlp']:
            base, grid = get_model(mname)
            if mname == 'gpr':
                search = GridSearchCV(base, grid, cv=5, scoring='r2', n_jobs=1)
            else:
                search = RandomizedSearchCV(base, grid, n_iter=30, cv=5, scoring='r2',
                                            random_state=42, n_jobs=4)
            search.fit(X_tr_s, y_tr)
            best = search.best_estimator_
            y_tr_pred = best.predict(X_tr_s)
            y_te_pred = best.predict(X_te_s)
            tr_r2 = r2_score(y_tr, y_tr_pred)
            te_r2 = r2_score(y_te, y_te_pred)
            te_rmse = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))
            # 10 折 CV
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            cv_r2 = []
            for i_tr, i_val in kf.split(X_tr_s):
                m, _ = get_model(mname)
                m.set_params(**search.best_params_)
                m.fit(X_tr_s[i_tr], y_tr[i_tr])
                cv_r2.append(r2_score(y_tr[i_val], m.predict(X_tr_s[i_val])))
            summary_rows.append({
                'target': t, 'model': MODEL_NAMES[mname],
                'train_r2': round(tr_r2, 4), 'test_r2': round(te_r2, 4),
                'cv_r2_mean': round(float(np.mean(cv_r2)), 4), 'cv_r2_std': round(float(np.std(cv_r2)), 4),
                'test_rmse': round(te_rmse, 2), 'best_params': str(search.best_params_)})
            results[t][MODEL_NAMES[mname]] = {
                'best_params': search.best_params_,
                'train_r2': tr_r2, 'test_r2': te_r2, 'test_rmse': te_rmse,
                'cv_r2_mean': float(np.mean(cv_r2)),
                'y_test': y_te.tolist(), 'y_test_pred': y_te_pred.tolist(),
                'y_train': y_tr.tolist(), 'y_train_pred': y_tr_pred.tolist()}
            print(f'{t:35s} {MODEL_NAMES[mname]:13s} train={tr_r2:.4f} test={te_r2:.4f} '
                  f'cv={np.mean(cv_r2):.4f}±{np.std(cv_r2):.4f} rmse={te_rmse:.1f}')

        diag_figure(t, y_tr, y_te, results[t])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(DATA_DIR, 'model_summary.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(DATA_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('saved data/model_summary.csv, data/results.json')

    group_variance_figure(df, results)
    print('02 done')


def diag_figure(target, y_tr, y_te, results):
    """图3/4/5：每目标 2×3 六模型对角散点图（英文、无总标题、nature 配色）。"""
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6))
    axes = axes.ravel()
    for i, name in enumerate(['Ridge', 'XGBoost', 'RandomForest', 'SVR', 'GPR', 'MLP']):
        ax = axes[i]
        r = results[name]
        ytr_pred = np.array(r['y_train_pred'])
        yte_pred = np.array(r['y_test_pred'])
        lim = [min(min(y_tr), min(y_te), min(ytr_pred), min(yte_pred)) * 0.95,
               max(max(y_tr), max(y_te), max(ytr_pred), max(yte_pred)) * 1.05]
        ax.scatter(y_tr, ytr_pred, s=45, c='#000000', alpha=0.8, edgecolors='white',
                   linewidths=0.6, label='Training')
        ax.scatter(y_te, yte_pred, s=75, c='#D4AF37', alpha=0.9, edgecolors='white',
                   linewidths=0.8, label='Test')
        ax.plot(lim, lim, 'k--', lw=1.2)
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.text(0.04, 0.93, f'$R^2$ train = {r["train_r2"]:.3f}\n$R^2$ test = {r["test_r2"]:.3f}',
                transform=ax.transAxes, fontsize=12.5, va='top')
        ax.text(0.03, 1.0, '(' + 'abcdef'[i] + ') ' + name,
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.tick_params(labelsize=12)
        if i in (0, 3):
            ax.set_ylabel('Predicted ' + TARGET_EN[target], fontsize=14)
        if i >= 3:
            ax.set_xlabel('Measured ' + TARGET_EN[target], fontsize=14)
        if i == 0:
            ax.legend(fontsize=12, loc='lower right')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'fig3_diag_{target}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def group_variance_figure(df, results):
    """组间/组内方差分解 + 2024 牌号内工艺解析 + 牌号内 R²（论文图3）。
    输出：figure/fig3_group_variance.png、data/variance_decomposition.csv、data/within_grade_r2.csv"""
    feature_cols = [c for c in df.columns
                    if c not in TARGETS + ['sample_id', 'material_system', 'alloy_or_coating_name']]
    grades = [2024, 5083, 6082, 7075]
    grade_cn = {2024: '2024', 5083: '5083', 6082: '6082', 7075: '7075'}
    t_short = {'yield_strength_mpa': 'YS', 'ultimate_tensile_strength_mpa': 'UTS',
               'elongation_pct': 'EL'}

    # (1) 组间/组内方差分解：总方差 = 组间方差 + 组内方差
    var_rows = []
    for t in TARGETS:
        y = df[t].values.astype(float)
        n = len(y)
        total = float(np.var(y))
        between = 0.0
        within_sum = 0.0
        for g in grades:
            yg = df.loc[df['alloy_or_coating_name'] == g, t].values.astype(float)
            ng = len(yg)
            between += ng * (yg.mean() - y.mean()) ** 2
            within_sum += (ng - 1) * float(np.var(yg))
        between /= n
        within = within_sum / n
        var_rows.append({'target': t, 'total_var': total, 'between_var': between,
                         'within_var': within,
                         'between_frac': between / total, 'within_frac': within / total})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(os.path.join(DATA_DIR, 'variance_decomposition.csv'), index=False,
                  encoding='utf-8-sig')
    print('variance decomposition:')
    print(var_df[['target', 'between_frac', 'within_frac']].to_string(index=False))

    # (2) 2024 牌号内 YS 与最相关工艺特征
    g2024 = df[df['alloy_or_coating_name'] == 2024]
    proc_cols = ['homogenization_temp_c', 'homogenization_time_h', 'solution_temp_c',
                 'solution_time_h', 'aging_temp_c', 'aging_time_h', 'annealing_temp_c',
                 'annealing_time_h', 'rolling_temp_c', 'rolling_reduction_pct',
                 'specimen_thickness_mm']
    proc_cn = {'homogenization_temp_c': 'Homogenization temp. (°C)',
               'homogenization_time_h': 'Homogenization time (h)',
               'solution_temp_c': 'Solution temp. (°C)', 'solution_time_h': 'Solution time (h)',
               'aging_temp_c': 'Aging temp. (°C)', 'aging_time_h': 'Aging time (h)',
               'annealing_temp_c': 'Annealing temp. (°C)', 'annealing_time_h': 'Annealing time (h)',
               'rolling_temp_c': 'Rolling temp. (°C)', 'rolling_reduction_pct': 'Rolling reduction (%)',
               'specimen_thickness_mm': 'Thickness (mm)'}
    best_proc, best_corr = None, 0.0
    for p in proc_cols:
        if g2024[p].nunique() <= 1:
            continue
        c = abs(np.corrcoef(g2024[p].values.astype(float),
                            g2024['yield_strength_mpa'].values.astype(float))[0, 1])
        if c > best_corr:
            best_corr, best_proc = c, p
    print('2024 best process feature for YS:', best_proc, '|corr|=%.3f' % best_corr)

    # (3) 牌号内 R²（5 折 CV，RF + 各目标最优超参，仅用该牌号样本）
    r2_rows = []
    for t in TARGETS:
        params = results[t]['RandomForest']['best_params']
        for g in grades:
            sub = df[df['alloy_or_coating_name'] == g]
            val = np.nan
            if len(sub) >= 10:
                Xg = sub[feature_cols].values.astype(float)
                yg = sub[t].values.astype(float)
                preds = np.zeros(len(yg))
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                for i_tr, i_val in kf.split(Xg):
                    m = RandomForestRegressor(random_state=42, n_jobs=4)
                    m.set_params(**params)
                    m.fit(Xg[i_tr], yg[i_tr])
                    preds[i_val] = m.predict(Xg[i_val])
                val = float(r2_score(yg, preds))
            r2_rows.append({'target': t, 'grade': g, 'within_r2': val})
    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(os.path.join(DATA_DIR, 'within_grade_r2.csv'), index=False, encoding='utf-8-sig')
    print('within-grade R²:')
    print(r2_df.to_string(index=False))

    # ---- 图3 ----
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    ax = axes[0]
    x = np.arange(len(TARGETS))
    w = 0.36
    between = [var_df.loc[var_df['target'] == t, 'between_frac'].iloc[0] for t in TARGETS]
    within = [var_df.loc[var_df['target'] == t, 'within_frac'].iloc[0] for t in TARGETS]
    ax.bar(x - w / 2, between, w, color='#0F4D92', label='Between-grade')
    ax.bar(x + w / 2, within, w, color='#8BCF8B', label='Within-grade')
    ax.set_xticks(x)
    ax.set_xticklabels(['YS', 'UTS', 'EL'], fontsize=13)
    ax.set_ylabel('Variance fraction', fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11, loc='upper right')
    ax.tick_params(labelsize=12)
    ax.text(0.02, 1.02, '(a) Between- vs within-grade variance',
            transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')

    ax = axes[1]
    if best_proc is not None:
        xv = g2024[best_proc].values.astype(float)
        yv = g2024['yield_strength_mpa'].values.astype(float)
        ax.scatter(xv, yv, s=55, c='#B64342', alpha=0.85, edgecolors='white', linewidths=0.6)
        ax.set_xlabel(proc_cn.get(best_proc, best_proc), fontsize=13)
    ax.set_ylabel('YS (MPa)', fontsize=13)
    ax.tick_params(labelsize=12)
    ax.text(0.02, 1.02, '(b) 2024 within-grade YS vs process',
            transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')

    ax = axes[2]
    x = np.arange(len(grades))
    w = 0.4
    for i, t in enumerate(['yield_strength_mpa', 'ultimate_tensile_strength_mpa']):
        vals = [r2_df[(r2_df['target'] == t) & (r2_df['grade'] == g)]['within_r2'].iloc[0]
                for g in grades]
        ax.bar(x + (i - 0.5) * w, vals, w, color=['#0F4D92', '#8BCF8B'][i],
               label=t_short[t] + ' within-grade $R^2$')
    ax.set_xticks(x)
    ax.set_xticklabels([grade_cn[g] for g in grades], fontsize=13)
    ax.set_ylabel('Within-grade $R^2$ (5-fold CV)', fontsize=13)
    ax.axhline(0, color='#767676', lw=0.8)
    ax.legend(fontsize=11, loc='lower right')
    ax.tick_params(labelsize=12)
    ax.text(0.02, 1.02, '(c) Within-grade resolution ($R^2$)',
            transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig3_group_variance.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
