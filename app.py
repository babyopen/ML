#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生肖预测模型应用入口点
"""

import http.server
import socketserver
import os
import json
import subprocess
import sys

# 端口设置
PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 处理API请求
        if self.path.startswith('/api/predict'):
            self.handle_api_request()
        else:
            # 处理静态文件请求
            super().do_GET()
    
    def handle_api_request(self):
        """处理预测API请求"""
        try:
            # 执行预测脚本
            result = subprocess.run(
                [sys.executable, '-c', '''
import pandas as pd
import numpy as np
import pickle
import json
import os

# 加载模型
model_path = 'zodiac_model_sklearn_20260403_085812.pkl'
data_path = 'data/lottery_history_recalculated.csv'

if not os.path.exists(model_path):
    print(json.dumps({'success': False, 'error': '模型文件不存在'}))
    exit(1)

if not os.path.exists(data_path):
    print(json.dumps({'success': False, 'error': '数据文件不存在'}))
    exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 加载数据
df = pd.read_csv(data_path)
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

# 预测下一期
X, _ = build_features(df)
if len(X) == 0:
    print(json.dumps({'success': False, 'error': '没有足够的数据进行预测'}))
    exit(1)

X_last = X.iloc[[-1]]
proba = model.predict_proba(X_last)[0]

# 整理结果
results = []
for i in range(12):
    zod_id = i + 1
    zod_name = ID_TO_ZODIAC[zod_id]
    results.append({
        'zodiac_id': zod_id,
        'zodiac_name': zod_name,
        'wuxing': ZODIAC_WU_XING[zod_id],
        'color': ZODIAC_COLOR[zod_id],
        'probability': float(proba[i])
    })

results.sort(key=lambda x: x['probability'], reverse=True)

# 计算统计信息
top3 = results[:3]
top1 = top3[0]

# 生成分析
analysis = [
    f"{top1['zodiac_name']} 的预测概率为 {top1['probability']*100:.2f}%，在所有生肖中排名最高",
    f"与第二名 {top3[1]['zodiac_name']} 的概率差距为 {(top1['probability']-top3[1]['probability'])*100:.2f}%",
    f"该生肖五行属{top1['wuxing']}，对应{top1['color']}",
    "模型基于历史数据统计、时序特征和五行波色关联进行预测"
]

# 输出结果
result = {
    'success': True,
    'prediction': {
        'top3': top3,
        'all': results,
        'confidence': top1['probability'] * 100,
        'recommended': top1,
        'analysis': analysis
    },
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

print(json.dumps(result, ensure_ascii=False))
'''],
                capture_output=True,
                text=True
            )
            
            # 解析结果
            try:
                response_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                response_data = {'success': False, 'error': '预测失败: 无法解析预测结果'}
            
            # 发送响应
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
            
        except Exception as e:
            # 发送错误响应
            error_response = {'success': False, 'error': f'预测失败: {str(e)}'}
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))

# 更改工作目录到项目根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 启动服务器
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"服务器启动在 http://localhost:{PORT}")
    print(f"前端页面: http://localhost:{PORT}/web/predict.html")
    print(f"API端点: http://localhost:{PORT}/api/predict")
    print("按 Ctrl+C 停止服务器")
    httpd.serve_forever()
