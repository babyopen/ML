#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件整理脚本 - 自动整理项目文件结构
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

# 定义目录结构
DIRECTORY_STRUCTURE = {
    'data': {
        'description': '数据文件',
        'extensions': ['.csv', '.json'],
        'patterns': ['lottery_history', 'training_report', 'analytics', 'prediction_history']
    },
    'models': {
        'description': '模型文件',
        'extensions': ['.pkl', '.json'],
        'patterns': ['zodiac_model', 'feature_names']
    },
    'scripts': {
        'description': 'Python脚本',
        'extensions': ['.py'],
        'patterns': ['train_', 'generate_', 'fetch_', 'update_', 'check_', 'debug_', 'predict', 'advanced_', 'history_', 'model_', 'data_', 'verify_', 'zodiac_']
    },
    'web': {
        'description': 'Web文件',
        'extensions': ['.html', '.css', '.js'],
        'patterns': ['predict', 'test']
    },
    'docker': {
        'description': 'Docker配置',
        'extensions': [''],
        'patterns': ['Docker', 'docker', '.dockerignore']
    },
    'logs': {
        'description': '日志文件',
        'extensions': ['.log'],
        'patterns': ['.log']
    },
    'backup': {
        'description': '备份文件',
        'extensions': [''],
        'patterns': ['backup', '.backup']
    }
}

# 保留在根目录的文件
KEEP_IN_ROOT = [
    'README.md',
    'requirements.txt',
    'start.sh',
    'organize_files.py'
]

def get_file_category(filename):
    """判断文件应该属于哪个分类"""
    # 检查是否在保留列表
    if filename in KEEP_IN_ROOT:
        return 'root'
    
    # 检查备份文件
    if 'backup' in filename.lower():
        return 'backup'
    
    # 检查Docker相关
    if any(pattern in filename for pattern in ['Docker', 'docker', '.dockerignore']):
        return 'docker'
    
    # 检查日志文件
    if filename.endswith('.log'):
        return 'logs'
    
    # 检查模型文件
    if 'zodiac_model' in filename or filename == 'feature_names.json':
        return 'models'
    
    # 检查数据文件
    if any(pattern in filename for pattern in ['lottery_history', 'training_report', 'analytics', 'prediction_history']):
        return 'data'
    
    # 检查Web文件
    if filename.endswith('.html'):
        return 'web'
    
    # 检查Python脚本
    if filename.endswith('.py'):
        return 'scripts'
    
    return 'others'

def organize_files():
    """整理文件"""
    print("="*70)
    print("文件整理工具")
    print("="*70)
    
    base_dir = Path('/Users/macbook/Documents/open/ML模型')
    
    # 统计信息
    stats = {
        'moved': {},
        'kept': [],
        'others': []
    }
    
    # 获取所有文件
    all_files = [f for f in base_dir.iterdir() if f.is_file()]
    
    print(f"\n发现 {len(all_files)} 个文件")
    print("\n开始整理...")
    
    for file_path in all_files:
        filename = file_path.name
        category = get_file_category(filename)
        
        if category == 'root':
            stats['kept'].append(filename)
            print(f"  [保留] {filename}")
        elif category == 'others':
            stats['others'].append(filename)
            print(f"  [其他] {filename}")
        else:
            # 创建目标目录
            target_dir = base_dir / category
            target_dir.mkdir(exist_ok=True)
            
            # 移动文件
            target_path = target_dir / filename
            
            # 如果目标文件已存在，添加时间戳
            if target_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                target_path = target_dir / new_name
            
            shutil.move(str(file_path), str(target_path))
            
            if category not in stats['moved']:
                stats['moved'][category] = []
            stats['moved'][category].append(filename)
            print(f"  [移动] {filename} -> {category}/")
    
    # 打印统计
    print("\n" + "="*70)
    print("整理完成!")
    print("="*70)
    
    print(f"\n📁 保留在根目录 ({len(stats['kept'])} 个):")
    for f in stats['kept']:
        print(f"  - {f}")
    
    print(f"\n📂 已分类文件:")
    for category, files in stats['moved'].items():
        print(f"\n  {category}/ ({len(files)} 个):")
        for f in files[:5]:  # 只显示前5个
            print(f"    - {f}")
        if len(files) > 5:
            print(f"    ... 还有 {len(files)-5} 个文件")
    
    if stats['others']:
        print(f"\n❓ 未分类文件 ({len(stats['others'])} 个):")
        for f in stats['others']:
            print(f"  - {f}")
    
    print("\n" + "="*70)
    print("建议:")
    print("  1. 检查 'others/' 目录中的文件是否需要手动分类")
    print("  2. 定期清理 'backup/' 目录中的旧备份")
    print("  3. 重要的训练报告可以保留在 'data/' 目录")
    print("="*70)

if __name__ == '__main__':
    organize_files()
