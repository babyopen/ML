#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生肖预测模块
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# 生肖配置
ZODIAC_CONFIG = {
    "zodiacs": ["马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"],
    "id_to_name": {
        1: "马", 2: "蛇", 3: "龙", 4: "兔", 5: "虎", 6: "牛",
        7: "鼠", 8: "猪", 9: "狗", 10: "鸡", 11: "猴", 12: "羊"
    },
    "name_to_id": {
        "马": 1, "蛇": 2, "龙": 3, "兔": 4, "虎": 5, "牛": 6,
        "鼠": 7, "猪": 8, "狗": 9, "鸡": 10, "猴": 11, "羊": 12
    },
    "zodiac_to_element": {
        1: "火", 2: "火", 3: "土", 4: "木", 5: "木", 6: "土",
        7: "水", 8: "水", 9: "土", 10: "金", 11: "金", 12: "土"
    },
    "zodiac_to_color": {
        1: "红", 2: "蓝", 3: "红", 4: "绿", 5: "蓝", 6: "绿",
        7: "红", 8: "蓝", 9: "绿", 10: "红", 11: "蓝", 12: "绿"
    }
}

def load_model(model_path='models/zodiac_model.pkl'):
    """加载模型"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def predict_next(model, last_data, df):
    """预测下一期"""
    # 简单的预测逻辑
    last_zodiac = last_data['zodiac']
    next_zodiac = (last_zodiac % 12) + 1
    
    # 生成概率
    probs = np.zeros(12)
    probs[next_zodiac-1] = 0.9
    probs = probs / sum(probs)
    
    return probs

def get_latest_data():
    """获取最新数据"""
    try:
        df = pd.read_csv('data/lottery_history_recalculated.csv')
        df = df.sort_values('period').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"读取数据失败: {e}")
        return None
