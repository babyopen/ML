#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征工程生成的特征数量
"""

import pandas as pd
import numpy as np
import pickle
import os

# 加载数据
df = pd.read_csv('data/lottery_history_recalculated.csv')
df = df.sort_values('period').reset_index(drop=True)

# 特征工程
ZODIAC_ORDER = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
ZODIAC_TO_ID = {z: i+1 for i, z in enumerate(ZODIAC_ORDER)}
ID_TO_ZODIAC = {v: k for k, v in ZODIAC_TO_ID.items()}

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

def get_zodiac_parity(zodiac_id):
    return "单" if zodiac_id % 2 == 1 else "双"

def get_zodiac_size(zodiac_id):
    return "小" if zodiac_id <= 6 else "大"

def get_zodiac_zone(zodiac_id):
    if zodiac_id <= 4:
        return 1
    elif zodiac_id <= 8:
        return 2
    else:
        return 3

def get_zodiac_head(zodiac_id):
    return 0 if zodiac_id <= 9 else 1

def get_zodiac_tail(zodiac_id):
    tail = zodiac_id % 10
    return tail if tail != 0 else 0

def build_features(df):
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
        
        # 基础统计特征
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
        
        # 热门排名
        recent_20 = hist.tail(20)
        counts = recent_20['zodiac'].value_counts()
        for zod_id in range(1, 13):
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
        
        # 动态特征
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
            relation_map = {'克': 0, '同': 1, '生': 2}
            feat[f'wuxing_relation_{zod_id}'] = 1  # 简化处理
            
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
        
        features_list.append(feat)
        labels.append(label)
    
    X = pd.DataFrame(features_list)
    y = np.array(labels) - 1
    
    return X, y

# 测试特征工程
X, y = build_features(df)
print(f"特征数量: {X.shape[1]}")
print(f"特征列名: {list(X.columns[:10])}...")
print(f"样本数量: {X.shape[0]}")

# 加载模型并检查期望的特征数量
try:
    with open('models/zodiac_model_optimized.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 检查模型的特征数量
    if hasattr(model, 'n_features_'):
        print(f"模型期望的特征数量: {model.n_features_}")
    else:
        print("无法获取模型的特征数量")
except Exception as e:
    print(f"加载模型失败: {str(e)}")
