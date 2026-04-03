#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用从API获取的真实数据训练模型
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# 生肖配置（顺序：1-马, 2-蛇, 3-龙, 4-兔, 5-虎, 6-牛, 7-鼠, 8-猪, 9-狗, 10-鸡, 11-猴, 12-羊）
ZODIAC_ALL = ["马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"]

# 五行和波色配置
ELEMENT_MAP = {
    1: '火', 2: '火', 3: '土', 4: '木', 5: '木', 6: '土',
    7: '水', 8: '水', 9: '土', 10: '金', 11: '金', 12: '土'
}
COLOR_MAP = {
    1: '红', 2: '蓝', 3: '红', 4: '绿', 5: '蓝', 6: '绿',
    7: '红', 8: '蓝', 9: '绿', 10: '红', 11: '蓝', 12: '绿'
}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取特征（与之前使用相同的特征工程）
    """
    print("\n正在提取特征...")
    
    df = df.copy()
    df['zodiac'] = df['zodiac'].astype(int)
    
    # 计算每个生肖的遗漏值
    for zodiac_id in range(1, 13):
        miss_values = []
        last_seen = {}
        
        for idx, row in df.iterrows():
            current_zodiac = int(row['zodiac'])
            
            for zid in range(1, 13):
                if zid not in last_seen:
                    last_seen[zid] = idx
            
            if zodiac_id in last_seen:
                miss = idx - last_seen[zodiac_id]
            else:
                miss = idx
            
            miss_values.append(miss)
            
            if current_zodiac == zodiac_id:
                last_seen[zodiac_id] = idx
        
        df[f'zodiac_{zodiac_id}_miss'] = miss_values
    
    # 计算遗漏值比例
    max_miss = df[[f'zodiac_{i}_miss' for i in range(1, 13)]].max(axis=1)
    for zodiac_id in range(1, 13):
        df[f'zodiac_{zodiac_id}_miss_ratio'] = df[f'zodiac_{zodiac_id}_miss'] / (max_miss + 1)
    
    # 计算最近N期出现频率
    for window in [5, 10, 20]:
        for zodiac_id in range(1, 13):
            freq_values = []
            for idx in range(len(df)):
                start_idx = max(0, idx - window)
                window_data = df.iloc[start_idx:idx]
                freq = (window_data['zodiac'].astype(int) == zodiac_id).sum() / window if window > 0 else 0
                freq_values.append(freq)
            df[f'zodiac_{zodiac_id}_freq_{window}'] = freq_values
    
    # 计算连开次数
    for zodiac_id in range(1, 13):
        streak_values = []
        current_streak = 0
        
        for idx, row in df.iterrows():
            if int(row['zodiac']) == zodiac_id:
                current_streak += 1
            else:
                current_streak = 0
            streak_values.append(current_streak)
        
        df[f'zodiac_{zodiac_id}_streak'] = streak_values
    
    # 五行特征
    element_order = {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4}
    df['element_code'] = df['zodiac'].apply(lambda x: element_order.get(ELEMENT_MAP.get(int(x), '土'), 4))
    
    # 波色特征
    color_order = {'红': 0, '蓝': 1, '绿': 2}
    df['color_code'] = df['zodiac'].apply(lambda x: color_order.get(COLOR_MAP.get(int(x), '红'), 0))
    
    # 时间特征（使用period的模运算）
    df['month'] = df['period'] % 12 + 1
    df['day_of_week'] = df['period'] % 7
    
    print(f"✓ 特征提取完成，共 {len(df.columns)} 个特征列")
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    准备训练数据
    """
    print("\n正在准备训练数据...")
    
    feature_cols = [col for col in df.columns if 'zodiac_' in col and col != 'zodiac']
    feature_cols.extend(['element_code', 'color_code', 'month', 'day_of_week'])
    
    df_train = df.iloc[30:].copy()
    
    for col in feature_cols:
        if df_train[col].dtype == 'object':
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    
    df_train[feature_cols] = df_train[feature_cols].fillna(0)
    df_train['zodiac'] = pd.to_numeric(df_train['zodiac'], errors='coerce').fillna(0).astype(int)
    
    X = df_train[feature_cols].astype(float).values
    y = df_train['zodiac'].values - 1
    
    print(f"✓ 训练数据准备完成")
    print(f"  样本数量: {len(X)}")
    print(f"  特征数量: {len(feature_cols)}")
    print(f"  类别分布:")
    for i in range(12):
        count = (y == i).sum()
        print(f"    {ZODIAC_ALL[i]}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols


def train_model(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
    训练模型并返回完整报告
    """
    print("\n" + "="*70)
    print("开始模型训练（使用真实API数据）")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"训练模型: {name}")
        print(f"{'='*40}")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        top3_correct = 0
        for i in range(len(y_test)):
            top3_pred = np.argsort(y_pred_proba[i])[-3:]
            if y_test[i] in top3_pred:
                top3_correct += 1
        top3_accuracy = top3_correct / len(y_test)
        
        loss = log_loss(y_test, y_pred_proba)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"\n模型性能:")
        print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Top-3准确率: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
        print(f"  对数损失: {loss:.4f}")
        print(f"  交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'log_loss': loss,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\n{'='*60}")
    print(f"最佳模型: {best_name}")
    print(f"{'='*60}")
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print(f"\nTop 10 重要特征:")
        for i in indices:
            print(f"  {feature_names[i]}: {importances[i]:.4f}")
    
    with open('zodiac_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n✓ 模型已保存: zodiac_model.pkl")
    
    report = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': 'API真实数据 (https://history.macaumarksix.com)',
        'data_stats': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(feature_names)
        },
        'model_comparison': {
            name: {
                'accuracy': float(res['accuracy']),
                'top3_accuracy': float(res['top3_accuracy']),
                'log_loss': float(res['log_loss']),
                'cv_mean': float(res['cv_mean']),
                'cv_std': float(res['cv_std'])
            }
            for name, res in results.items()
        },
        'best_model': best_name,
        'best_model_performance': {
            'accuracy': float(results[best_name]['accuracy']),
            'top3_accuracy': float(results[best_name]['top3_accuracy']),
            'log_loss': float(results[best_name]['log_loss'])
        },
        'feature_importance': [
            {'feature': feature_names[i], 'importance': float(importances[i])}
            for i in np.argsort(importances)[::-1][:10]
        ] if hasattr(best_model, 'feature_importances_') else []
    }
    
    return report


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("使用API真实数据训练模型")
    print("="*70)
    
    df = pd.read_csv('lottery_history_api_unique.csv')
    print(f"\n✓ 加载API数据: {len(df)} 条记录")
    print(f"  期号范围: {df['period'].min()} - {df['period'].max()}")
    
    df = df[['period', 'zodiac']].copy()
    df = df.sort_values('period').reset_index(drop=True)
    
    df_features = extract_features(df)
    X, y, feature_names = prepare_training_data(df_features)
    report = train_model(X, y, feature_names)
    
    import json
    report_file = f"training_report_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 训练报告已保存: {report_file}")
    
    df_features.to_csv('lottery_history.csv', index=False)
    print(f"✓ 训练数据已更新: lottery_history.csv ({len(df_features)} 条记录)")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"\n报告文件: {report_file}")
    print(f"模型文件: zodiac_model.pkl")
    print(f"数据文件: lottery_history.csv")


if __name__ == '__main__':
    main()
