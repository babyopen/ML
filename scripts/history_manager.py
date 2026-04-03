#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测历史记录管理模块
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

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


class HistoryManager:
    def __init__(self, data_file: str = 'prediction_history.json'):
        self.data_file = data_file
        self.history: List[Dict] = []
        self._load_history()
    
    def _load_history(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"加载历史记录失败: {e}")
                self.history = []
        else:
            self.history = []
    
    def _save_history(self):
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史记录失败: {e}")
    
    def _predictions_equal(self, preds1: List[Dict], preds2: List[Dict]) -> bool:
        """
        比较两个预测列表是否相等
        
        Args:
            preds1: 预测列表1
            preds2: 预测列表2
            
        Returns:
            是否相等
        """
        if len(preds1) != len(preds2):
            return False
        
        for p1, p2 in zip(preds1, preds2):
            if p1['name'] != p2['name']:
                return False
            # 比较概率，允许小误差
            if abs(float(p1['prob']) - float(p2['prob'])) > 0.001:
                return False
        
        return True
    
    def add_prediction(self, predictions: List[Dict], target_period: Optional[int] = None, force: bool = False) -> Optional[str]:
        """
        添加预测记录（带去重机制）
        
        去重规则：
        - 如果已存在相同目标期号的记录，且预测内容相同，则不重复添加
        - 仅当预测内容变更或开奖后才添加新记录
        - force=True 时强制添加
        
        Args:
            predictions: 预测结果列表
            target_period: 目标期号
            force: 是否强制添加
            
        Returns:
            prediction_id: 预测记录ID（如果添加了新记录），否则返回None
        """
        prediction_id = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # 获取前3名推荐
        top3 = predictions[:3]
        
        # 去重检查：检查是否有相同目标期号的记录
        if target_period and not force:
            # 查找相同目标期号的记录
            existing_records = [r for r in self.history if r.get('target_period') == target_period]
            if existing_records:
                # 检查预测内容是否相同（比较前3名）
                existing_top3 = existing_records[-1].get('top3', [])
                new_top3 = [{'name': p['name'], 'prob': float(p['prob'])} for p in top3]
                existing_top3_simple = [{'name': p['name'], 'prob': float(p['probability'])} for p in existing_top3]
                
                # 比较预测内容
                if len(new_top3) == len(existing_top3_simple):
                    all_same = True
                    for np, ep in zip(new_top3, existing_top3_simple):
                        if np['name'] != ep['name'] or abs(np['prob'] - ep['prob']) > 0.001:
                            all_same = False
                            break
                    
                    if all_same:
                        # 预测内容相同，不重复添加
                        print(f"预测内容相同，跳过重复记录 (期号: {target_period})")
                        return None
        
        # 创建新记录
        record = {
            'id': prediction_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'target_period': int(target_period) if target_period else None,
            'predictions': [
                {
                    'id': int(p['id']),
                    'name': p['name'],
                    'prob': float(p['prob']),
                    'element': p['element'],
                    'color': p['color']
                } for p in predictions
            ],
            'top3': [
                {
                    'rank': i + 1,
                    'name': p['name'],
                    'probability': float(p['prob']),
                    'element': p['element'],
                    'color': p['color']
                } for i, p in enumerate(top3)
            ],
            'status': 'pending',
            'actual_zodiac': None,
            'is_correct': None,
            'checked_at': None
        }
        
        self.history.append(record)
        self._save_history()
        
        return prediction_id
    
    def get_history(self, limit: Optional[int] = None, status: Optional[str] = None) -> List[Dict]:
        """
        获取历史记录
        
        Args:
            limit: 限制返回数量
            status: 按状态筛选
            
        Returns:
            历史记录列表
        """
        records = self.history.copy()
        
        if status:
            records = [r for r in records if r['status'] == status]
        
        records.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if limit:
            records = records[:limit]
        
        return records
    
    def check_prediction(self, prediction_id: str, actual_zodiac: str) -> Optional[Dict]:
        """
        核对预测结果
        
        Args:
            prediction_id: 预测记录ID
            actual_zodiac: 实际开奖生肖
            
        Returns:
            更新后的记录
        """
        record = self._find_record(prediction_id)
        if not record:
            return None
        
        record['actual_zodiac'] = actual_zodiac
        record['checked_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 检查是否命中
        top1_name = record['top3'][0]['name'] if record['top3'] else None
        record['is_correct'] = (top1_name == actual_zodiac)
        record['status'] = 'checked'
        
        self._save_history()
        return record
    
    def auto_check_with_latest(self, df: pd.DataFrame) -> int:
        """
        使用最新开奖结果自动核对所有待核对的预测
        
        Args:
            df: 历史数据DataFrame
            
        Returns:
            核对的记录数
        """
        if len(df) == 0:
            return 0
        
        latest_zodiac_id = df.iloc[-1]['zodiac']
        latest_zodiac_name = ZODIAC_CONFIG['id_to_name'][latest_zodiac_id]
        latest_period = df.iloc[-1]['period']
        
        checked_count = 0
        
        for record in self.history:
            if record['status'] == 'pending':
                # 如果有目标期号且匹配，或者没有目标期号（最新预测）
                if (record['target_period'] and record['target_period'] == latest_period) or \
                   (not record['target_period'] and not record['checked_at']):
                    self.check_prediction(record['id'], latest_zodiac_name)
                    checked_count += 1
        
        return checked_count
    
    def _find_record(self, prediction_id: str) -> Optional[Dict]:
        for record in self.history:
            if record['id'] == prediction_id:
                return record
        return None
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计字典
        """
        total = len(self.history)
        checked = [r for r in self.history if r['status'] == 'checked']
        correct = [r for r in checked if r['is_correct']]
        
        return {
            'total_predictions': total,
            'checked_predictions': len(checked),
            'correct_predictions': len(correct),
            'accuracy': len(correct) / len(checked) if checked else 0,
            'pending_predictions': total - len(checked)
        }
    
    def clear_history(self):
        self.history = []
        self._save_history()




if __name__ == '__main__':
    print("HistoryManager 模块")
    print("用法: from history_manager import HistoryManager")
