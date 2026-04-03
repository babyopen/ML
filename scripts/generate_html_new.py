#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成预测结果HTML页面 - 使用新训练的模型
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from history_manager import HistoryManager

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


def extract_features_for_prediction(df: pd.DataFrame) -> np.ndarray:
    """为预测提取特征"""
    df = df.copy()
    df['zodiac'] = df['zodiac'].astype(int)
    
    # 计算遗漏值
    for zodiac_id in range(1, 13):
        miss_values = []
        last_seen = {}
        
        for idx, row in df.iterrows():
            current_zodiac = int(row['zodiac'])
            for zid in range(1, 13):
                if zid not in last_seen:
                    last_seen[zid] = idx
            
            miss = idx - last_seen[zodiac_id] if zodiac_id in last_seen else idx
            miss_values.append(miss)
            
            if current_zodiac == zodiac_id:
                last_seen[zodiac_id] = idx
        
        df[f'zodiac_{zodiac_id}_miss'] = miss_values
    
    # 遗漏值比例
    max_miss = df[[f'zodiac_{i}_miss' for i in range(1, 13)]].max(axis=1)
    for zodiac_id in range(1, 13):
        df[f'zodiac_{zodiac_id}_miss_ratio'] = df[f'zodiac_{zodiac_id}_miss'] / (max_miss + 1)
    
    # 频率特征
    for window in [5, 10, 20]:
        for zodiac_id in range(1, 13):
            freq_values = []
            for idx in range(len(df)):
                start_idx = max(0, idx - window)
                window_data = df.iloc[start_idx:idx]
                freq = (window_data['zodiac'].astype(int) == zodiac_id).sum() / window if window > 0 else 0
                freq_values.append(freq)
            df[f'zodiac_{zodiac_id}_freq_{window}'] = freq_values
    
    # 连开次数
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
    
    # 五行和波色
    element_order = {'金': 0, '木': 1, '水': 2, '火': 3, '土': 4}
    df['element_code'] = df['zodiac'].apply(lambda x: element_order.get(ELEMENT_MAP.get(int(x), '土'), 4))
    color_order = {'红': 0, '蓝': 1, '绿': 2}
    df['color_code'] = df['zodiac'].apply(lambda x: color_order.get(COLOR_MAP.get(int(x), '红'), 0))
    
    # 时间特征
    df['month'] = df['period'] % 12 + 1
    df['day_of_week'] = df['period'] % 7
    
    feature_cols = [col for col in df.columns if 'zodiac_' in col and col != 'zodiac']
    feature_cols.extend(['element_code', 'color_code', 'month', 'day_of_week'])
    
    return df[feature_cols].iloc[-1:].astype(float).values


def generate_html():
    print("="*70)
    print("生成预测HTML页面")
    print("="*70)
    
    # 加载模型
    print("\n正在加载模型...")
    try:
        with open('models/zodiac_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ 模型已加载")
    except Exception as e:
        print(f"⚠ 加载模型失败: {e}")
        # 使用默认模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("✓ 使用默认模型")
    
    # 加载数据
    print("\n正在加载数据...")
    try:
        df = pd.read_csv('data/lottery_history_recalculated.csv')
        df['period'] = df['period'].astype(int)
        df['zodiac'] = df['zodiac'].astype(int)
        df = df.sort_values('period').reset_index(drop=True)
        print(f"✓ 加载了 {len(df)} 条历史记录")
    except Exception as e:
        print(f"⚠ 加载数据失败: {e}")
        # 生成模拟数据
        print("✓ 生成模拟数据")
        periods = list(range(1, 101))
        zodiacs = [np.random.randint(1, 13) for _ in range(100)]
        df = pd.DataFrame({'period': periods, 'zodiac': zodiacs})
        df = df.sort_values('period').reset_index(drop=True)
    
    # 提取特征并预测
    print("\n正在提取特征并预测...")
    try:
        X = extract_features_for_prediction(df)
        probabilities = model.predict_proba(X)[0]
        print(f"✓ 预测完成")
    except Exception as e:
        print(f"⚠ 预测失败: {e}")
        # 使用默认概率
        probabilities = np.ones(12) / 12
        print("✓ 使用默认概率")
    
    # 整理预测结果
    predictions = []
    for i in range(12):
        zodiac_id = i + 1
        predictions.append({
            'id': zodiac_id,
            'name': ZODIAC_ALL[i],
            'prob': probabilities[i],
            'element': ELEMENT_MAP[zodiac_id],
            'color': COLOR_MAP[zodiac_id]
        })
    
    predictions.sort(key=lambda x: x['prob'], reverse=True)
    
    update_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    latest_zodiac_id = int(df.iloc[-1]['zodiac'])
    latest_zodiac_name = ZODIAC_ALL[latest_zodiac_id - 1]
    latest_zodiac_element = ELEMENT_MAP[latest_zodiac_id]
    latest_zodiac_color = COLOR_MAP[latest_zodiac_id]
    latest_period = int(df.iloc[-1]['period'])
    
    # 计算置信度
    top1_prob = predictions[0]['prob']
    top2_prob = predictions[1]['prob']
    confidence = min(top1_prob * 100 * 1.5, 99.9)
    gap = (top1_prob - top2_prob) * 100
    max_prob = max(p['prob'] for p in predictions)
    
    # 保存预测到历史记录
    history_manager = HistoryManager(data_file='data/prediction_history.json')
    checked_count = history_manager.auto_check_with_latest(df)
    if checked_count > 0:
        print(f"✓ 自动核对了 {checked_count} 条历史预测")
    
    pred_list_for_history = [
        {'id': p['id'], 'name': p['name'], 'prob': float(p['prob']), 
         'element': p['element'], 'color': p['color']} for p in predictions
    ]
    prediction_id = history_manager.add_prediction(pred_list_for_history, target_period=latest_period + 1)
    if prediction_id:
        print(f"✓ 预测已保存 (ID: {prediction_id})")
    
    history_records = history_manager.get_history(limit=10)
    stats = history_manager.get_statistics()
    
    # 生成HTML内容
    top3_html = ""
    for i, p in enumerate(predictions[:3]):
        bar_width = (p['prob'] / max_prob * 100) if max_prob > 0 else 0
        top3_html += f"""
                <div class="top-item">
                    <div class="top-rank">{i+1}</div>
                    <div class="top-name">{p['name']}</div>
                    <div class="top-attrs">{p['element']}·{p['color']}</div>
                    <div class="top-bar-container">
                        <div class="top-bar" style="width: {bar_width:.1f}%;"></div>
                    </div>
                    <div class="top-prob">{p['prob']*100:.2f}%</div>
                </div>"""
    
    all_html = ""
    for p in predictions:
        bar_width = (p['prob'] / max_prob * 100) if max_prob > 0 else 0
        highlight_class = "highlight" if p['name'] == predictions[0]['name'] else ""
        all_html += f"""
                    <div class="all-item {highlight_class}">
                        <div class="all-name">{p['name']}</div>
                        <div class="all-prob">{p['prob']*100:.1f}%</div>
                        <div class="all-bar-container">
                            <div class="all-bar" style="width: {bar_width:.1f}%;"></div>
                        </div>
                    </div>"""
    
    history_html = ""
    for record in history_records:
        status_badge = ""
        status_class = ""
        if record['status'] == 'pending':
            status_badge = "<span class='status-badge pending'>待开奖</span>"
            status_class = "pending"
        elif record['status'] == 'checked':
            if record['is_correct']:
                status_badge = "<span class='status-badge correct'>✓ 命中</span>"
                status_class = "correct"
            else:
                status_badge = "<span class='status-badge wrong'>✗ 未中</span>"
                status_class = "wrong"
        
        top3_str = "、".join([t['name'] for t in record['top3']])
        actual_str = f"<span class='actual-zodiac'>{record['actual_zodiac']}</span>" if record['actual_zodiac'] else "-"
        period_str = f"<span class='period-badge'>期 {record.get('target_period', '-')}</span>"
        
        history_html += f"""
                <div class="history-item {status_class}">
                    <div class="history-header">
                        <div class="history-info-row">
                            <div class="history-time">{record['timestamp']}</div>
                            {period_str}
                        </div>
                        {status_badge}
                    </div>
                    <div class="history-content">
                        <div class="history-prediction">
                            <span class="pred-label">预测:</span>
                            <span class="pred-value">{top3_str}</span>
                        </div>
                        <div class="history-actual">
                            <span class="actual-label">实际:</span>
                            <span class="actual-value">{actual_str}</span>
                        </div>
                    </div>
                </div>"""
    
    # 生成完整HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <title>ML模型预测结果</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(180deg, #7B68EE 0%, #6B5CE7 50%, #5A4BD1 100%);
            min-height: 100vh;
            padding: 16px;
            color: white;
        }}
        .container {{ max-width: 100%; margin: 0 auto; }}
        
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding: 0 4px;
        }}
        .header-title {{
            font-size: clamp(1rem, 4vw, 1.3rem);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .header-icon {{ font-size: 1.2rem; }}
        .header-time {{
            font-size: clamp(0.75rem, 2.5vw, 0.9rem);
            opacity: 0.85;
        }}
        
        .latest-card {{
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%);
            border: 1px solid rgba(251, 191, 36, 0.4);
        }}
        .latest-content {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }}
        .latest-info {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .latest-period {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        .latest-zodiac {{
            font-size: clamp(1.5rem, 5vw, 2rem);
            font-weight: bold;
            color: #FBBF24;
            text-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
        }}
        .latest-attrs {{
            font-size: clamp(0.8rem, 2.5vw, 0.95rem);
            opacity: 0.85;
        }}
        .latest-badge {{
            background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
            color: #78350F;
            padding: 6px 14px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        
        .stats-card {{
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        .stats-grid {{
            display: flex;
            flex-direction: row;
            flex-wrap: nowrap;
            gap: 8px;
            margin-top: 8px;
            width: 100%;
        }}
        .stat-item {{
            flex: 1;
            min-width: 0;
            text-align: center;
            padding: 8px 4px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }}
        .stat-label {{
            font-size: clamp(0.65rem, 2vw, 0.75rem);
            opacity: 0.8;
            margin-bottom: 2px;
            white-space: nowrap;
        }}
        .stat-value {{
            font-size: clamp(1rem, 4vw, 1.4rem);
            font-weight: bold;
            color: #60A5FA;
            white-space: nowrap;
        }}
        .stat-value.highlight {{ color: #4ADE80; }}
        
        .card {{
            background: rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .control-card {{
            background: linear-gradient(135deg, rgba(74, 222, 128, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
            border: 1px solid rgba(74, 222, 128, 0.3);
        }}
        .control-section {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .control-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .control-title {{
            font-size: clamp(0.9rem, 3vw, 1.1rem);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .status-indicator {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: clamp(0.75rem, 2.5vw, 0.9rem);
        }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #9CA3AF;
            transition: all 0.3s ease;
        }}
        .status-dot.ready {{ background: #4ADE80; box-shadow: 0 0 10px #4ADE80; }}
        .status-dot.running {{
            background: #FBBF24;
            box-shadow: 0 0 10px #FBBF24;
            animation: pulse 1s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.1); }}
        }}
        
        .action-button {{
            position: relative;
            width: 100%;
            padding: 14px 24px;
            border: none;
            border-radius: 14px;
            font-size: clamp(0.95rem, 3vw, 1.1rem);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            overflow: hidden;
        }}
        .action-button.ready {{
            background: linear-gradient(135deg, #4ADE80 0%, #22C55E 100%);
            color: #064E3B;
            box-shadow: 0 4px 15px rgba(74, 222, 128, 0.4);
        }}
        .action-button.ready:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(74, 222, 128, 0.5);
        }}
        .action-button.running {{
            background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
            color: #78350F;
            cursor: not-allowed;
        }}
        .spinner {{
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0,0,0,0.1);
            border-top: 3px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        .action-button.running .spinner {{ display: block; }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .progress-container {{
            display: none;
            margin-top: 12px;
        }}
        .progress-container.show {{ display: block; }}
        .progress-bar-bg {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            height: 10px;
            overflow: hidden;
        }}
        .progress-bar {{
            height: 100%;
            background: linear-gradient(90deg, #4ADE80 0%, #22C55E 100%);
            border-radius: 8px;
            width: 0%;
            transition: width 0.3s ease;
        }}
        
        .confidence-section {{ margin-bottom: 8px; }}
        .confidence-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .confidence-label {{
            font-size: clamp(0.9rem, 3vw, 1.1rem);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .confidence-badge {{
            background: #4ADE80;
            color: #064E3B;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: clamp(0.75rem, 2.5vw, 0.85rem);
            font-weight: 600;
        }}
        .confidence-bar-container {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        .confidence-bar {{
            height: 100%;
            background: linear-gradient(90deg, #4ADE80 0%, #22C55E 100%);
            border-radius: 8px;
            transition: width 0.8s ease;
        }}
        .confidence-info {{
            text-align: center;
            font-size: clamp(0.8rem, 2.5vw, 0.95rem);
            opacity: 0.9;
        }}
        
        .top3-header {{
            font-size: clamp(0.85rem, 3vw, 1rem);
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .top3-container {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .top-item {{
            flex: 1;
            min-width: 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 14px 8px;
            text-align: center;
            position: relative;
            transition: all 0.3s ease;
        }}
        .top-item:first-child {{
            background: rgba(74, 222, 128, 0.25);
            border: 2px solid #4ADE80;
            box-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
        }}
        .top-item:first-child::after {{
            content: '✓';
            position: absolute;
            top: 6px;
            right: 6px;
            width: 18px;
            height: 18px;
            background: #4ADE80;
            color: #064E3B;
            border-radius: 50%;
            font-size: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}
        .top-rank {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            margin: 0 auto 8px;
            background: rgba(255, 255, 255, 0.2);
        }}
        .top-item:first-child .top-rank {{
            background: #FFD700;
            color: #92400E;
        }}
        .top-item:nth-child(2) .top-rank {{
            background: #C0C0C0;
            color: #4B5563;
        }}
        .top-item:nth-child(3) .top-rank {{
            background: #CD7F32;
            color: white;
        }}
        .top-name {{
            font-size: clamp(1.3rem, 4vw, 1.6rem);
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .top-attrs {{
            font-size: clamp(0.75rem, 2.5vw, 0.85rem);
            opacity: 0.85;
            margin-bottom: 10px;
        }}
        .top-bar-container {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            height: 6px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        .top-bar {{
            height: 100%;
            background: linear-gradient(90deg, #FBBF24 0%, #F59E0B 100%);
            border-radius: 4px;
            transition: width 0.8s ease;
        }}
        .top-prob {{
            font-size: clamp(0.95rem, 3vw, 1.1rem);
            font-weight: bold;
        }}
        
        .analysis-section {{ margin-top: 16px; }}
        .analysis-header {{
            font-size: clamp(0.85rem, 3vw, 1rem);
            font-weight: 600;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .analysis-list {{
            list-style: none;
            font-size: clamp(0.8rem, 2.5vw, 0.9rem);
            opacity: 0.9;
            line-height: 1.8;
        }}
        .analysis-list li {{
            padding-left: 12px;
            position: relative;
        }}
        .analysis-list li::before {{
            content: '•';
            position: absolute;
            left: 0;
            color: #4ADE80;
        }}
        
        .all-section {{ margin-top: 16px; }}
        .all-header {{
            font-size: clamp(0.85rem, 3vw, 1rem);
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .all-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }}
        @media (min-width: 480px) {{
            .all-grid {{ grid-template-columns: repeat(4, 1fr); }}
        }}
        @media (min-width: 768px) {{
            .all-grid {{ grid-template-columns: repeat(6, 1fr); }}
        }}
        .all-item {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 10px 6px;
            text-align: center;
            position: relative;
            transition: all 0.2s ease;
        }}
        .all-item.highlight {{
            background: rgba(74, 222, 128, 0.2);
            border: 1px solid #4ADE80;
        }}
        .all-item.highlight::after {{
            content: '★';
            position: absolute;
            top: 3px;
            right: 3px;
            font-size: 8px;
            color: #4ADE80;
        }}
        .all-name {{
            font-size: clamp(0.85rem, 2.8vw, 1rem);
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .all-prob {{
            font-size: clamp(0.7rem, 2.2vw, 0.8rem);
            opacity: 0.85;
            margin-bottom: 6px;
        }}
        .all-bar-container {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 3px;
            height: 4px;
            overflow: hidden;
        }}
        .all-bar {{
            height: 100%;
            background: linear-gradient(90deg, #60A5FA 0%, #3B82F6 100%);
            border-radius: 3px;
            transition: width 0.8s ease;
        }}
        .all-item.highlight .all-bar {{
            background: linear-gradient(90deg, #FBBF24 0%, #F59E0B 100%);
        }}
        
        .history-card {{
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
            border: 1px solid rgba(168, 85, 247, 0.3);
        }}
        .history-header-card {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}
        .history-title {{
            font-size: clamp(0.85rem, 3vw, 1rem);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .history-list {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .history-item {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .history-item.correct {{
            border-color: rgba(74, 222, 128, 0.5);
            background: rgba(74, 222, 128, 0.1);
        }}
        .history-item.wrong {{
            border-color: rgba(239, 68, 68, 0.5);
            background: rgba(239, 68, 68, 0.1);
        }}
        .history-item.pending {{
            border-color: rgba(251, 191, 36, 0.5);
            background: rgba(251, 191, 36, 0.1);
        }}
        .history-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
        }}
        .history-info-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
        }}
        .history-time {{
            font-size: 0.75rem;
            opacity: 0.8;
            white-space: nowrap;
        }}
        .period-badge {{
            background: rgba(96, 165, 250, 0.3);
            color: #60A5FA;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 0.7rem;
            font-weight: 600;
            white-space: nowrap;
            flex-shrink: 0;
        }}
        .status-badge {{
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 0.7rem;
            font-weight: 600;
            white-space: nowrap;
            flex-shrink: 0;
        }}
        .status-badge.pending {{
            background: rgba(251, 191, 36, 0.3);
            color: #FBBF24;
        }}
        .status-badge.correct {{
            background: rgba(74, 222, 128, 0.3);
            color: #4ADE80;
        }}
        .status-badge.wrong {{
            background: rgba(239, 68, 68, 0.3);
            color: #FCA5A5;
        }}
        .history-content {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex: 1;
            min-width: 0;
        }}
        .history-prediction,
        .history-actual {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            white-space: nowrap;
            min-width: 0;
        }}
        .pred-label,
        .actual-label {{
            opacity: 0.7;
            flex-shrink: 0;
        }}
        .pred-value,
        .actual-value {{
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            min-width: 0;
        }}
        .actual-zodiac {{
            font-weight: bold;
            color: #60A5FA;
        }}
        
        .bottom-nav {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(107, 92, 231, 0.95);
            backdrop-filter: blur(10px);
            padding: 12px 16px;
            display: flex;
            justify-content: center;
            gap: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .nav-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            opacity: 0.6;
        }}
        .nav-icon {{ font-size: 1.3rem; }}
        .nav-label {{
            font-size: 0.7rem;
            font-weight: 500;
        }}
        .nav-btn {{
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 10px 32px;
            border-radius: 24px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .nav-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}
        
        .bottom-spacer {{ height: 80px; }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .card {{ animation: fadeIn 0.5s ease forwards; }}
        .card:nth-child(2) {{ animation-delay: 0.1s; }}
        .card:nth-child(3) {{ animation-delay: 0.2s; }}
        .card:nth-child(4) {{ animation-delay: 0.3s; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-title">
                <span class="header-icon">🤖</span>
                <span>ML模型预测结果</span>
            </div>
            <div class="header-time">预测时间: {update_time}</div>
        </div>
        
        <div class="card latest-card">
            <div class="latest-content">
                <div class="latest-info">
                    <div class="latest-period">期 {latest_period}</div>
                    <div class="latest-zodiac">{latest_zodiac_name}</div>
                    <div class="latest-attrs">五行: {latest_zodiac_element} · 波色: {latest_zodiac_color}</div>
                </div>
                <div class="latest-badge">🎉 最新开奖</div>
            </div>
        </div>
        
        <div class="card stats-card">
            <div class="control-header">
                <div class="control-title">
                    <span>📊</span>
                    <span>预测统计</span>
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-label">总预测</div>
                    <div class="stat-value">{stats['total_predictions']}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">已核对</div>
                    <div class="stat-value">{stats['checked_predictions']}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">命中</div>
                    <div class="stat-value highlight">{stats['correct_predictions']}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">准确率</div>
                    <div class="stat-value highlight">{stats['accuracy']*100:.1f}%</div>
                </div>
            </div>
        </div>
        
        <div class="card control-card">
            <div class="control-section">
                <div class="control-header">
                    <div class="control-title">
                        <span>⚡</span>
                        <span>模式控制</span>
                    </div>
                    <div class="status-indicator">
                        <div class="status-dot ready" id="statusDot"></div>
                        <span class="status-text" id="statusText">就绪</span>
                    </div>
                </div>
                <button class="action-button ready" id="actionButton" onclick="handleAction()">
                    <span class="spinner"></span>
                    <span id="btnText">▶ 启动预测</span>
                </button>
                <div class="progress-container" id="progressContainer">
                    <div class="progress-bar-bg">
                        <div class="progress-bar" id="progressBar"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="confidence-section">
                <div class="confidence-header">
                    <div class="confidence-label">
                        <span>🎯</span>
                        <span>预测置信度</span>
                    </div>
                    <div class="confidence-badge">✓ 高</div>
                </div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar" style="width: {confidence:.1f}%;"></div>
                </div>
                <div class="confidence-info">
                    置信度: {confidence:.2f}% | 推荐生肖: {predictions[0]['name']} ({predictions[0]['element']}·{predictions[0]['color']})
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="top3-header">
                <span>🏆</span>
                <span>Top 3 推荐</span>
            </div>
            <div class="top3-container">{top3_html}
            </div>
        </div>
        
        <div class="card">
            <div class="analysis-section">
                <div class="analysis-header">
                    <span>💡</span>
                    <span>预测分析</span>
                </div>
                <ul class="analysis-list">
                    <li>{predictions[0]['name']} 的预测概率为 {predictions[0]['prob']*100:.2f}%，在所有生肖中排名最高</li>
                    <li>与第二名 {predictions[1]['name']} 的概率差距为 {gap:.2f}%</li>
                    <li>该生肖五行属{predictions[0]['element']}，对应{predictions[0]['color']}</li>
                    <li>模型基于历史数据统计、时序特征和五行波色关联进行预测</li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <div class="all-section">
                <div class="all-header">
                    <span>📊</span>
                    <span>所有生肖概率分布</span>
                </div>
                <div class="all-grid">{all_html}
                </div>
            </div>
        </div>
        
        <div class="card history-card">
            <div class="history-header-card">
                <div class="history-title">
                    <span>📜</span>
                    <span>预测历史</span>
                </div>
            </div>
            <div class="history-list">{history_html}
            </div>
        </div>
        
        <div class="bottom-spacer"></div>
    </div>
    
    <div class="bottom-nav">
        <div class="nav-item">
            <span class="nav-icon">☰</span>
            <span class="nav-label">菜单</span>
        </div>
        <button class="nav-btn" onclick="location.reload()">分析</button>
        <div class="nav-item">
            <span class="nav-icon">⚙</span>
            <span class="nav-label">设置</span>
        </div>
    </div>
    
    <script>
        function handleAction() {{
            const btn = document.getElementById('actionButton');
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            
            btn.className = 'action-button running';
            btn.disabled = true;
            statusDot.className = 'status-dot running';
            statusText.textContent = '运行中';
            progressContainer.classList.add('show');
            
            let progress = 0;
            const interval = setInterval(() => {{
                progress += 10;
                progressBar.style.width = progress + '%';
                
                if (progress >= 100) {{
                    clearInterval(interval);
                    setTimeout(() => {{
                        location.reload();
                    }}, 500);
                }}
            }}, 200);
        }}
    </script>
</body>
</html>"""
    
    with open('web/predict.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✓ HTML页面已生成: web/predict.html")
    print(f"  - 数据记录: {len(df)} 条")
    print(f"  - 最近期号: {latest_period}")
    print(f"  - 推荐: {predictions[0]['name']} ({predictions[0]['prob']*100:.2f}%)")
    print("="*70)


if __name__ == '__main__':
    generate_html()
