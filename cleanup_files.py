#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理无用文件脚本
"""

import os
import shutil
from pathlib import Path

# 要删除的文件列表
FILES_TO_DELETE = {
    'data': [
        'lottery_history_api.csv',          # 原始API数据，已去重
        'lottery_history_api_corrected.csv', # 修正版本，已合并
        'lottery_history_api_unique.csv',    # 去重版本，已合并
        'lottery_history_optimized.csv',     # 优化特征数据，可重新生成
        'lottery_history.csv',               # 旧数据，已重新计算
    ],
    'scripts': [
        'debug_api_data.py',
        'debug_data.py',
        'debug_features.py',
        'debug_train.py',
        'predict.py',
        'predict_with_new_model.py',
        'train_model_with_api.py',
        'train_with_local_data.py',
        'train_xgboost_complete.py',
        'fetch_api_data.py',
        'data_fetcher.py',
        'check_data.py',
        'model_optimizer.py',
        'update_data.py',
        'verify_config.py',
        'zodiac_ml_predictor.py',
    ]
}

# 要删除的旧训练报告（保留最新的3个）
OLD_REPORTS_TO_DELETE = [
    'training_report_20260403_070832.json',
    'training_report_20260403_070910.json',
    'training_report_20260403_070910.txt',
    'training_report_api_20260403_073853.json',
    'training_report_api_20260403_075359.json',
    'training_report_api_20260403_075849.json',
    'training_report_api_20260403_080511.json',
    'training_report_api_20260403_080738.json',
]

# 要删除的web文件
WEB_FILES_TO_DELETE = [
    'test.html',
]

# 要删除的备份
BACKUP_TO_DELETE = [
    'lottery_history.csv.backup.20260403_010309',
]

def cleanup_files():
    """清理文件"""
    print("="*70)
    print("清理无用文件")
    print("="*70)
    
    base_dir = Path('/Users/macbook/Documents/open/ML模型')
    deleted_count = 0
    
    # 清理数据文件
    print("\n📁 清理数据文件...")
    for filename in FILES_TO_DELETE['data']:
        filepath = base_dir / 'data' / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  ✓ 删除: data/{filename}")
            deleted_count += 1
        else:
            print(f"  ⚠ 不存在: data/{filename}")
    
    # 清理脚本文件
    print("\n📁 清理脚本文件...")
    for filename in FILES_TO_DELETE['scripts']:
        filepath = base_dir / 'scripts' / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  ✓ 删除: scripts/{filename}")
            deleted_count += 1
        else:
            print(f"  ⚠ 不存在: scripts/{filename}")
    
    # 清理旧训练报告
    print("\n📁 清理旧训练报告...")
    for filename in OLD_REPORTS_TO_DELETE:
        filepath = base_dir / 'data' / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  ✓ 删除: data/{filename}")
            deleted_count += 1
        else:
            print(f"  ⚠ 不存在: data/{filename}")
    
    # 清理web文件
    print("\n📁 清理web文件...")
    for filename in WEB_FILES_TO_DELETE:
        filepath = base_dir / 'web' / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  ✓ 删除: web/{filename}")
            deleted_count += 1
        else:
            print(f"  ⚠ 不存在: web/{filename}")
    
    # 清理备份文件
    print("\n📁 清理备份文件...")
    for filename in BACKUP_TO_DELETE:
        filepath = base_dir / 'backup' / filename
        if filepath.exists():
            filepath.unlink()
            print(f"  ✓ 删除: backup/{filename}")
            deleted_count += 1
        else:
            print(f"  ⚠ 不存在: backup/{filename}")
    
    # 删除空目录
    print("\n📁 检查空目录...")
    for subdir in ['backup']:
        dirpath = base_dir / subdir
        if dirpath.exists() and not any(dirpath.iterdir()):
            dirpath.rmdir()
            print(f"  ✓ 删除空目录: {subdir}/")
    
    print("\n" + "="*70)
    print(f"清理完成! 共删除 {deleted_count} 个文件")
    print("="*70)
    
    # 显示剩余文件
    print("\n📋 剩余文件结构:")
    for subdir in ['data', 'scripts', 'models', 'web', 'docker', 'logs', 'docs']:
        dirpath = base_dir / subdir
        if dirpath.exists():
            files = list(dirpath.iterdir())
            print(f"\n  {subdir}/ ({len(files)} 个文件)")
            for f in files[:3]:
                print(f"    - {f.name}")
            if len(files) > 3:
                print(f"    ... 还有 {len(files)-3} 个")

if __name__ == '__main__':
    cleanup_files()
