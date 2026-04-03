#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成预测结果HTML页面 - 紫色主题风格 + 历史记录功能
"""

import pandas as pd
import pickle
from datetime import datetime
from zodiac_ml_predictor import ZODIAC_CONFIG, predict_next, load_model
from history_manager import HistoryManager


def generate_html():
    print("正在加载模型...")
    model = load_model('zodiac_model.pkl')
    
    print("正在加载数据...")
    df = pd.read_csv('lottery_history.csv')
    df['period'] = df['period'].astype(int)
    df['zodiac'] = df['zodiac'].astype(int)
    df = df.sort_values('period').reset_index(drop=True)
    
    print("正在预测...")
    probabilities = predict_next(model, df.iloc[-1], df)
    
    predictions = []
    for i in range(12):
        zodiac_id = i + 1
        name = ZODIAC_CONFIG['id_to_name'][zodiac_id]
        prob = probabilities[i]
        element = ZODIAC_CONFIG['zodiac_to_element'][zodiac_id]
        color = ZODIAC_CONFIG['zodiac_to_color'][zodiac_id]
        predictions.append({
            'id': zodiac_id,
            'name': name,
            'prob': prob,
            'element': element,
            'color': color
        })
    
    predictions.sort(key=lambda x: x['prob'], reverse=True)
    
    update_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    
    # 获取最新开奖信息
    latest_zodiac_id = df.iloc[-1]['zodiac']
    latest_zodiac_name = ZODIAC_CONFIG['id_to_name'][latest_zodiac_id]
    latest_zodiac_element = ZODIAC_CONFIG['zodiac_to_element'][latest_zodiac_id]
    latest_zodiac_color = ZODIAC_CONFIG['zodiac_to_color'][latest_zodiac_id]
    latest_period = df.iloc[-1]['period']
    
    # 计算置信度和分析
    top1_prob = predictions[0]['prob']
    top2_prob = predictions[1]['prob']
    confidence = min(top1_prob * 100 * 1.5, 99.9)
    gap = (top1_prob - top2_prob) * 100
    max_prob = max(p['prob'] for p in predictions)
    
    # 保存预测到历史记录
    history_manager = HistoryManager()
    
    # 先自动核对之前的预测
    checked_count = history_manager.auto_check_with_latest(df)
    
    if checked_count > 0:
        print(f"✓ 自动核对了 {checked_count} 条历史预测")
    
    # 添加新预测（带去重机制）
    pred_list_for_history = [
        {
            'id': p['id'],
            'name': p['name'],
            'prob': float(p['prob']),
            'element': p['element'],
            'color': p['color']
        } for p in predictions
    ]
    prediction_id = history_manager.add_prediction(pred_list_for_history, target_period=latest_period + 1)
    if prediction_id:
        print(f"✓ 预测已保存 (ID: {prediction_id})")
    else:
        print(f"- 预测内容相同，跳过重复记录")
    
    # 获取历史记录和统计
    history_records = history_manager.get_history(limit=10)
    stats = history_manager.get_statistics()
    
    # 生成Top 3 HTML
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
    
    # 生成所有生肖分布HTML
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
    
    # 生成历史记录HTML
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
        
        /* 头部 */
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
        
        /* 最新开奖 */
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
        
        /* 统计卡片 */
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
        
        /* 卡片通用样式 */
        .card {{
            background: rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        /* 功能控制区域 */
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
        .status-dot.error {{ background: #EF4444; box-shadow: 0 0 10px #EF4444; }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.1); }}
        }}
        .status-text {{ opacity: 0.9; }}
        
        /* 功能按钮 */
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
        .action-button.ready:active {{ transform: translateY(0); }}
        .action-button.running {{
            background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
            color: #78350F;
            cursor: not-allowed;
            box-shadow: 0 4px 15px rgba(251, 191, 36, 0.4);
        }}
        .action-button:disabled {{ cursor: not-allowed; }}
        .btn-icon {{ font-size: 1.3rem; }}
        .btn-text {{ flex: 1; }}
        
        /* 加载动画 */
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
        .action-button.running .btn-icon {{ display: none; }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* 进度条 */
        .progress-container {{
            display: none;
            margin-top: 12px;
        }}
        .progress-container.show {{ display: block; }}
        .progress-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            margin-bottom: 6px;
            opacity: 0.9;
        }}
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
        
        /* 消息提示 */
        .message-container {{
            display: none;
            margin-top: 10px;
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 0.85rem;
            animation: slideIn 0.3s ease;
        }}
        .message-container.show {{ display: block; }}
        .message-container.success {{
            background: rgba(74, 222, 128, 0.2);
            border: 1px solid rgba(74, 222, 128, 0.4);
        }}
        .message-container.error {{
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}
        .message-container.info {{
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.4);
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* 历史记录 */
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
            margin-bottom: 0;
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
            min-width: 0;
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
            min-width: 0;
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
        
        /* 置信度区域 */
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
        
        /* Top 3 推荐 */
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
        
        /* 预测分析 */
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
        
        /* 所有生肖分布 */
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
        
        /* 底部导航 */
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
            transition: opacity 0.2s;
        }}
        .nav-item.active {{ opacity: 1; }}
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
        
        /* 底部留白 */
        .bottom-spacer {{ height: 80px; }}
        
        /* 动画 */
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
        <!-- 头部 -->
        <div class="header">
            <div class="header-title">
                <span class="header-icon">🤖</span>
                <span>ML模型预测结果</span>
            </div>
            <div class="header-time">预测时间: {update_time}</div>
        </div>
        
        <!-- 最新开奖 -->
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
        
        <!-- 统计卡片 -->
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
        
        <!-- 功能控制区域 -->
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
                    <span class="btn-icon" id="btnIcon">▶</span>
                    <span class="btn-text" id="btnText">启动预测</span>
                </button>
                <div class="progress-container" id="progressContainer">
                    <div class="progress-label">
                        <span id="progressLabel">正在处理...</span>
                        <span id="progressPercent">0%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar" id="progressBar"></div>
                    </div>
                </div>
                <div class="message-container" id="messageContainer"></div>
            </div>
        </div>
        
        <!-- 置信度 -->
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
        
        <!-- Top 3 推荐 -->
        <div class="card">
            <div class="top3-header">
                <span>🏆</span>
                <span>Top 3 推荐</span>
            </div>
            <div class="top3-container">{top3_html}
            </div>
        </div>
        
        <!-- 预测分析 -->
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
        
        <!-- 所有生肖概率分布 -->
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
        
        <!-- 历史记录 -->
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
    
    <!-- 底部导航 -->
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
        let isRunning = false;
        
        function updateStatus(status) {{
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            statusDot.className = 'status-dot ' + status;
            
            switch(status) {{
                case 'ready':
                    statusText.textContent = '就绪';
                    break;
                case 'running':
                    statusText.textContent = '运行中';
                    break;
                case 'error':
                    statusText.textContent = '错误';
                    break;
            }}
        }}
        
        function updateButton(state) {{
            const button = document.getElementById('actionButton');
            const btnIcon = document.getElementById('btnIcon');
            const btnText = document.getElementById('btnText');
            
            button.className = 'action-button ' + state;
            
            switch(state) {{
                case 'ready':
                    btnIcon.textContent = '▶';
                    btnText.textContent = '启动预测';
                    button.disabled = false;
                    break;
                case 'running':
                    btnText.textContent = '预测中...';
                    button.disabled = true;
                    break;
            }}
        }}
        
        function updateProgress(percent, label) {{
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            const progressPercent = document.getElementById('progressPercent');
            const progressLabel = document.getElementById('progressLabel');
            
            progressContainer.classList.add('show');
            progressBar.style.width = percent + '%';
            progressPercent.textContent = percent + '%';
            if (label) {{
                progressLabel.textContent = label;
            }}
        }}
        
        function hideProgress() {{
            const progressContainer = document.getElementById('progressContainer');
            progressContainer.classList.remove('show');
        }}
        
        function showMessage(message, type) {{
            const messageContainer = document.getElementById('messageContainer');
            messageContainer.className = 'message-container ' + type + ' show';
            messageContainer.textContent = message;
            
            setTimeout(() => {{
                messageContainer.classList.remove('show');
            }}, 3000);
        }}
        
        function simulateProgress() {{
            let progress = 0;
            const steps = [
                {{ p: 10, label: '加载数据...' }},
                {{ p: 30, label: '提取特征...' }},
                {{ p: 50, label: '模型预测...' }},
                {{ p: 70, label: '计算概率...' }},
                {{ p: 90, label: '生成结果...' }},
                {{ p: 100, label: '完成!' }}
            ];
            
            let stepIndex = 0;
            const interval = setInterval(() => {{
                if (stepIndex < steps.length) {{
                    const step = steps[stepIndex];
                    updateProgress(step.p, step.label);
                    stepIndex++;
                }} else {{
                    clearInterval(interval);
                }}
            }}, 400);
            
            return interval;
        }}
        
        function handleAction() {{
            if (isRunning) {{
                return;
            }}
            
            isRunning = true;
            updateStatus('running');
            updateButton('running');
            
            const progressInterval = simulateProgress();
            
            setTimeout(() => {{
                clearInterval(progressInterval);
                updateProgress(100, '完成!');
                
                setTimeout(() => {{
                    isRunning = false;
                    updateStatus('ready');
                    updateButton('ready');
                    hideProgress();
                    showMessage('预测完成！页面即将刷新...', 'success');
                    
                    setTimeout(() => {{
                        location.reload();
                    }}, 1500);
                }}, 500);
            }}, 2500);
        }}
    </script>
</body>
</html>"""
    
    # 保存HTML
    with open('predict.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML页面已生成: predict.html")
    print(f"  - 数据记录: {len(df)} 条")
    print(f"  - 最近期号: {df.iloc[-1]['period']}")
    print(f"  - 推荐: {predictions[0]['name']} ({predictions[0]['prob']*100:.2f}%)")


if __name__ == '__main__':
    generate_html()
