#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的生肖预测模型应用入口点
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

try:
    # 加载数据
    data_path = 'data/lottery_history_recalculated.csv'

    if not os.path.exists(data_path):
        print(json.dumps({'success': False, 'error': '数据文件不存在'}))
        exit(1)

    # 加载数据
    df = pd.read_csv(data_path)
    df = df.sort_values('period').reset_index(drop=True)

    # 特征工程（简化版）
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

    # 计算基本统计特征
    recent_data = df.tail(50)
    zodiac_counts = recent_data['zodiac'].value_counts()
    
    # 生成预测结果（基于统计频率）
    results = []
    total_count = len(recent_data)
    
    for i in range(1, 13):
        zod_id = i
        zod_name = ID_TO_ZODIAC[zod_id]
        count = zodiac_counts.get(zod_id, 0)
        probability = count / total_count if total_count > 0 else 0
        
        results.append({
            'zodiac_id': zod_id,
            'zodiac_name': zod_name,
            'wuxing': ZODIAC_WU_XING[zod_id],
            'color': ZODIAC_COLOR[zod_id],
            'probability': float(probability)
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
        "模型基于历史数据统计频率进行预测"
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
except Exception as e:
    print(json.dumps({'success': False, 'error': f'预测失败: {str(e)}'}))
'''],
                capture_output=True,
                text=True
            )
            
            # 打印调试信息
            print(f"预测脚本输出: {result.stdout}")
            print(f"预测脚本错误: {result.stderr}")
            
            # 解析结果
            try:
                response_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                response_data = {'success': False, 'error': f'预测失败: 无法解析预测结果 - {str(e)}'}
            
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
