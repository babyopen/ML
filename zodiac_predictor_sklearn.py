#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生肖预测模型 - 使用scikit-learn（无需XGBoost）
包含完整的特征工程和模型训练流程
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 映射表配置 ====================

# 生肖顺序（标准顺序：1鼠、2牛、3虎、4兔、5龙、6蛇、7马、8羊、9猴、10鸡、11狗、12猪）
ZODIAC_ORDER = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
ZODIAC_TO_ID = {z: i+1 for i, z in enumerate(ZODIAC_ORDER)}
ID_TO_ZODIAC = {v: k for k, v in ZODIAC_TO_ID.items()}

# 生肖 → 五行
ZODIAC_WU_XING = {
    1: "水",   # 鼠
    2: "土",   # 牛
    3: "木",   # 虎
    4: "木",   # 兔
    5: "土",   # 龙
    6: "火",   # 蛇
    7: "火",   # 马
    8: "土",   # 羊
    9: "金",   # 猴
    10: "金",  # 鸡
    11: "土",  # 狗
    12: "水"   # 猪
}

# 生肖 → 波色
ZODIAC_COLOR = {
    1: "蓝",   # 鼠
    2: "蓝",   # 牛
    3: "绿",   # 虎
    4: "红",   # 兔
    5: "绿",   # 龙
    6: "蓝",   # 蛇
    7: "红",   # 马
    8: "绿",   # 羊
    9: "红",   # 猴
    10: "红",  # 鸡
    11: "绿",  # 狗
    12: "蓝"   # 猪
}

# 五行相生关系
WU_XING_RELATION = {
    ('木', '火'): '生',
    ('火', '土'): '生',
    ('土', '金'): '生',
    ('金', '水'): '生',
    ('水', '木'): '生',
    ('火', '木'): '克',
    ('土', '火'): '克',
    ('金', '土'): '克',
    ('水', '金'): '克',
    ('木', '水'): '克'
}

def get_wuxing_relation(w1, w2):
    """获取五行关系"""
    if w1 == w2:
        return '同'
    if (w1, w2) in WU_XING_RELATION:
        return WU_XING_RELATION[(w1, w2)]
    if (w2, w1) in WU_XING_RELATION:
        return '克' if WU_XING_RELATION[(w2, w1)] == '生' else '生'
    return '同'

# 辅助属性函数
def get_zodiac_parity(zodiac_id):
    """单双"""
    return "单" if zodiac_id % 2 == 1 else "双"

def get_zodiac_size(zodiac_id):
    """大小（1-6小，7-12大）"""
    return "小" if zodiac_id <= 6 else "大"

def get_zodiac_zone(zodiac_id):
    """区间（1-4/5-8/9-12）"""
    if zodiac_id <= 4:
        return 1
    elif zodiac_id <= 8:
        return 2
    else:
        return 3

def get_zodiac_head(zodiac_id):
    """头数（1-9为0，10-12为1）"""
    return 0 if zodiac_id <= 9 else 1

def get_zodiac_tail(zodiac_id):
    """尾数（mod 10，10→0, 11→1, 12→2）"""
    tail = zodiac_id % 10
    return tail if tail != 0 else 0

# ==================== 2. 数据加载 ====================

def load_data(file_path):
    """读取CSV数据"""
    df = pd.read_csv(file_path)
    df = df.sort_values('period').reset_index(drop=True)
    return df

# ==================== 3. 特征工程 ====================

def build_features(df):
    """
    构建特征向量
    所有特征基于当前期之前的历史数据计算
    """
    features_list = []
    labels = []
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        hist = df.iloc[:i]
        
        label = current['zodiac']
        feat = {}
        
        total_periods = len(hist)
        if total_periods == 0:
            continue
        
        # ========== 3.1 基础统计特征 ==========
        
        for zod_id in range(1, 13):
            zod_occurrences = hist[hist['zodiac'] == zod_id]
            
            # 当前遗漏
            last_occurrence = zod_occurrences.index.max() if len(zod_occurrences) > 0 else None
            if pd.isna(last_occurrence):
                miss = total_periods
            else:
                miss = total_periods - last_occurrence - 1
            feat[f'miss_{zod_id}'] = miss
            
            # 历史最大遗漏
            if len(zod_occurrences) == 0:
                max_miss = total_periods
            else:
                occur_indices = zod_occurrences.index.tolist()
                gaps = [occur_indices[0]]
                for j in range(1, len(occur_indices)):
                    gaps.append(occur_indices[j] - occur_indices[j-1] - 1)
                gaps.append(total_periods - occur_indices[-1] - 1)
                max_miss = max(gaps) if gaps else 0
            feat[f'max_miss_{zod_id}'] = max_miss
            
            # 遗漏比例
            feat[f'miss_ratio_{zod_id}'] = miss / max_miss if max_miss > 0 else 0
            
            # 近期出现次数
            recent_10 = hist.tail(10)
            recent_20 = hist.tail(20)
            recent_50 = hist.tail(50)
            feat[f'recent10_{zod_id}'] = (recent_10['zodiac'] == zod_id).sum()
            feat[f'recent20_{zod_id}'] = (recent_20['zodiac'] == zod_id).sum()
            feat[f'recent50_{zod_id}'] = (recent_50['zodiac'] == zod_id).sum()
            
            # 近期频率
            feat[f'freq10_{zod_id}'] = feat[f'recent10_{zod_id}'] / min(10, total_periods)
            feat[f'freq20_{zod_id}'] = feat[f'recent20_{zod_id}'] / min(20, total_periods)
            feat[f'freq50_{zod_id}'] = feat[f'recent50_{zod_id}'] / min(50, total_periods)
        
        # 热门排名（基于最近20期）
        recent_20 = hist.tail(20)
        counts = recent_20['zodiac'].value_counts()
        for zod_id in range(1, 13):
            rank = counts.get(zod_id, 0)
            rank_value = counts.rank(ascending=False, method='min').get(zod_id, 12)
            feat[f'hot_rank_{zod_id}'] = rank_value
        
        # 连开次数
        for zod_id in range(1, 13):
            consecutive = 0
            for j in range(len(hist)-1, -1, -1):
                if hist.iloc[j]['zodiac'] == zod_id:
                    consecutive += 1
                else:
                    break
            feat[f'consecutive_{zod_id}'] = consecutive if hist.iloc[-1]['zodiac'] == zod_id else 0
        
        # 连断状态
        last_zodiac = hist.iloc[-1]['zodiac']
        for zod_id in range(1, 13):
            is_consecutive = feat[f'consecutive_{zod_id}'] > 0
            feat[f'break_{zod_id}'] = 1 if (is_consecutive and len(hist) > 1 and hist.iloc[-2]['zodiac'] != zod_id) else 0
        
        # ========== 3.2 动态特征（与上期关联） ==========
        
        last_zodiac = hist.iloc[-1]['zodiac']
        last_wuxing = ZODIAC_WU_XING[last_zodiac]
        last_color = ZODIAC_COLOR[last_zodiac]
        last_parity = get_zodiac_parity(last_zodiac)
        last_size = get_zodiac_size(last_zodiac)
        last_zone = get_zodiac_zone(last_zodiac)
        last_head = get_zodiac_head(last_zodiac)
        last_tail = get_zodiac_tail(last_zodiac)
        
        for zod_id in range(1, 13):
            # 位置间隔
            interval = abs(zod_id - last_zodiac)
            interval = min(interval, 12 - interval)
            feat[f'interval_{zod_id}'] = interval
            
            # 五行相生关系
            curr_wuxing = ZODIAC_WU_XING[zod_id]
            relation = get_wuxing_relation(last_wuxing, curr_wuxing)
            relation_map = {'克': 0, '同': 1, '生': 2}
            feat[f'wuxing_relation_{zod_id}'] = relation_map.get(relation, 1)
            
            # 波色相同
            feat[f'same_color_{zod_id}'] = 1 if ZODIAC_COLOR[zod_id] == last_color else 0
            
            # 单双相同
            feat[f'same_parity_{zod_id}'] = 1 if get_zodiac_parity(zod_id) == last_parity else 0
            
            # 大小相同
            feat[f'same_size_{zod_id}'] = 1 if get_zodiac_size(zod_id) == last_size else 0
            
            # 区间相同
            feat[f'same_zone_{zod_id}'] = 1 if get_zodiac_zone(zod_id) == last_zone else 0
            
            # 头数相同
            feat[f'same_head_{zod_id}'] = 1 if get_zodiac_head(zod_id) == last_head else 0
            
            # 尾数相同
            feat[f'same_tail_{zod_id}'] = 1 if get_zodiac_tail(zod_id) == last_tail else 0
        
        # ========== 3.3 时序特征 ==========
        
        for zod_id in range(1, 13):
            zod_occurrences = hist[hist['zodiac'] == zod_id]
            
            # 近期间隔均值/标准差（最近5次）
            if len(zod_occurrences) >= 2:
                occur_indices = zod_occurrences.index.tolist()
                intervals = []
                for j in range(1, len(occur_indices)):
                    intervals.append(occur_indices[j] - occur_indices[j-1])
                
                # 最近5次间隔
                recent_intervals = intervals[-5:] if len(intervals) >= 5 else intervals
                feat[f'interval_mean_{zod_id}'] = np.mean(recent_intervals) if recent_intervals else 0
                feat[f'interval_std_{zod_id}'] = np.std(recent_intervals) if len(recent_intervals) > 1 else 0
            else:
                feat[f'interval_mean_{zod_id}'] = 0
                feat[f'interval_std_{zod_id}'] = 0
        
        # 热度变化（最近2期热门排名差值）
        if len(hist) >= 2:
            recent_20_now = hist.tail(20)
            recent_20_prev = hist.iloc[:-1].tail(20)
            counts_now = recent_20_now['zodiac'].value_counts()
            counts_prev = recent_20_prev['zodiac'].value_counts()
            
            for zod_id in range(1, 13):
                rank_now = counts_now.rank(ascending=False, method='min').get(zod_id, 12)
                rank_prev = counts_prev.rank(ascending=False, method='min').get(zod_id, 12)
                feat[f'hot_change_{zod_id}'] = rank_prev - rank_now
        else:
            for zod_id in range(1, 13):
                feat[f'hot_change_{zod_id}'] = 0
        
        features_list.append(feat)
        labels.append(label)
    
    X = pd.DataFrame(features_list)
    y = np.array(labels) - 1  # 转换为0-11
    
    return X, y

# ==================== 4. 模型训练 ====================

def train_model(X_train, y_train):
    """训练集成模型"""
    # 计算类别权重
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (12 * count) if count > 0 else 1.0 for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    # 使用RandomForest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

# ==================== 5. 模型评估 ====================

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # Top-3准确率
    top3_accuracy = top_k_accuracy_score(y_test, y_pred_proba, k=3)
    
    # 对数损失
    logloss = log_loss(y_test, y_pred_proba)
    
    print("\n" + "="*70)
    print("模型评估结果")
    print("="*70)
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Top-3准确率: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    print(f"对数损失: {logloss:.4f}")
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'logloss': logloss
    }

# ==================== 6. 预测下一期 ====================

def predict_next(model, df):
    """预测下一期概率"""
    # 构建最后一期的特征
    X, _ = build_features(df)
    X_last = X.iloc[[-1]]
    
    # 预测概率
    proba = model.predict_proba(X_last)[0]
    
    # 整理结果
    results = []
    for i in range(12):
        zod_id = i + 1
        zod_name = ID_TO_ZODIAC[zod_id]
        results.append({
            'zodiac_id': zod_id,
            'zodiac_name': zod_name,
            'probability': float(proba[i])
        })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

# ==================== 7. 主程序 ====================

def main():
    print("="*70)
    print("生肖预测模型 (scikit-learn版)")
    print("="*70)
    
    # 加载数据
    print("\n加载数据...")
    try:
        df = load_data('data/lottery_history_recalculated.csv')
        print(f"✓ 加载了 {len(df)} 条记录")
    except FileNotFoundError:
        print("✗ 错误：找不到数据文件 data/lottery_history_recalculated.csv")
        return
    
    # 构建特征
    print("\n构建特征...")
    X, y = build_features(df)
    print(f"✓ 特征维度: {X.shape}")
    print(f"✓ 样本数量: {len(y)}")
    
    # 划分训练集和测试集（时间序列分割）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 训练模型
    print("\n训练模型...")
    model = train_model(X_train, y_train)
    print("✓ 模型训练完成")
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 特征重要性
    print("\n" + "="*70)
    print("Top 10 重要特征")
    print("="*70)
    feature_importance = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 预测下一期
    print("\n" + "="*70)
    print("下一期预测概率")
    print("="*70)
    predictions = predict_next(model, df)
    
    for i, pred in enumerate(predictions[:5]):
        medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
        print(f"{medal} 第{i+1}名: {pred['zodiac_name']} (ID:{pred['zodiac_id']}) - {pred['probability']*100:.2f}%")
    
    # 保存模型
    model_file = f"zodiac_model_sklearn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ 模型已保存: {model_file}")
    
    # 保存特征列表
    with open('feature_names.json', 'w') as f:
        json.dump(list(X.columns), f)
    print("✓ 特征列表已保存: feature_names.json")
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)

if __name__ == "__main__":
    main()
