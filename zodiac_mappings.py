#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生肖完整映射表
包含：五行、波色、单双、大小、区间、头数、尾数
"""

# ==================== 1. 基础配置 ====================

# 十二生肖顺序（逆序）
ZODIAC_ORDER = ["马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"]
ZODIAC_TO_ID = {z: i+1 for i, z in enumerate(ZODIAC_ORDER)}
ID_TO_ZODIAC = {v: k for k, v in ZODIAC_TO_ID.items()}

# 号码 1~49 的五行
NUMBER_WU_XING = {
    1: "水", 2: "火", 3: "火", 4: "金", 5: "金", 6: "土", 7: "土", 8: "木", 9: "木", 10: "火",
    11: "火", 12: "金", 13: "水", 14: "火", 15: "水", 16: "木", 17: "木", 18: "火", 19: "火", 20: "土",
    21: "土", 22: "水", 23: "水", 24: "木", 25: "木", 26: "金", 27: "金", 28: "土", 29: "土", 30: "水",
    31: "水", 32: "火", 33: "火", 34: "金", 35: "金", 36: "土", 37: "土", 38: "水", 39: "水", 40: "火",
    41: "火", 42: "金", 43: "金", 44: "水", 45: "水", 46: "木", 47: "木", 48: "火", 49: "火"
}

# 马年基础分配（本命生肖马有5个号码，其余4个）
BASE_ALLOCATION = {
    "马": [1, 13, 25, 37, 49],
    "蛇": [2, 14, 26, 38],
    "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40],
    "虎": [5, 17, 29, 41],
    "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43],
    "猪": [8, 20, 32, 44],
    "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46],
    "猴": [11, 23, 35, 47],
    "羊": [12, 24, 36, 48]
}

# ==================== 2. 生肖属性映射 ====================

# 生肖 → 五行（基于马年基础分配中主要号码的五行，或传统五行属性）
# 注意：这里使用传统五行属性，与号码五行不同
ZODIAC_WU_XING = {
    "马": "火",   # 午马属火
    "蛇": "火",   # 巳蛇属火
    "龙": "土",   # 辰龙属土
    "兔": "木",   # 卯兔属木
    "虎": "木",   # 寅虎属木
    "牛": "土",   # 丑牛属土
    "鼠": "水",   # 子鼠属水
    "猪": "水",   # 亥猪属水
    "狗": "土",   # 戌狗属土
    "鸡": "金",   # 酉鸡属金
    "猴": "金",   # 申猴属金
    "羊": "土"    # 未羊属土
}

# 生肖 → 波色（基于号码波色的多数原则，或传统配色）
# 波色规则：红波、蓝波、绿波
ZODIAC_COLOR = {
    "马": "红",   # 火属性配红色
    "蛇": "蓝",   # 
    "龙": "绿",   # 
    "兔": "绿",   # 木属性配绿色
    "虎": "蓝",   # 
    "牛": "红",   # 
    "鼠": "蓝",   # 水属性配蓝色
    "猪": "蓝",   # 水属性配蓝色
    "狗": "绿",   # 
    "鸡": "红",   # 金属性配红色（或白色，但波色只有红蓝绿）
    "猴": "红",   # 金属性配红色
    "羊": "绿"    # 
}

# 生肖ID → 单双（奇偶）
# 规则：ID为奇数→单，偶数→双
ZODIAC_PARITY = {
    1: "单",   # 马
    2: "双",   # 蛇
    3: "单",   # 龙
    4: "双",   # 兔
    5: "单",   # 虎
    6: "双",   # 牛
    7: "单",   # 鼠
    8: "双",   # 猪
    9: "单",   # 狗
    10: "双",  # 鸡
    11: "单",  # 猴
    12: "双"   # 羊
}

# 生肖ID → 大小
# 规则：1-6小，7-12大
ZODIAC_SIZE = {
    1: "大",   # 马
    2: "小",   # 蛇
    3: "大",   # 龙
    4: "小",   # 兔
    5: "小",   # 虎
    6: "小",   # 牛
    7: "大",   # 鼠
    8: "大",   # 猪
    9: "大",   # 狗
    10: "大",  # 鸡
    11: "大",  # 猴
    12: "大"   # 羊
}

# 生肖ID → 区间
# 规则：1-4/5-8/9-12 三个区间
ZODIAC_ZONE = {
    1: 1,   # 马 - 第一区间
    2: 1,   # 蛇 - 第一区间
    3: 1,   # 龙 - 第一区间
    4: 1,   # 兔 - 第一区间
    5: 2,   # 虎 - 第二区间
    6: 2,   # 牛 - 第二区间
    7: 2,   # 鼠 - 第二区间
    8: 2,   # 猪 - 第二区间
    9: 3,   # 狗 - 第三区间
    10: 3,  # 鸡 - 第三区间
    11: 3,  # 猴 - 第三区间
    12: 3   # 羊 - 第三区间
}

# 生肖ID → 头数
# 规则：1-9为0，10-12为1
ZODIAC_HEAD = {
    1: 0,   # 马
    2: 0,   # 蛇
    3: 0,   # 龙
    4: 0,   # 兔
    5: 0,   # 虎
    6: 0,   # 牛
    7: 0,   # 鼠
    8: 0,   # 猪
    9: 0,   # 狗
    10: 1,  # 鸡
    11: 1,  # 猴
    12: 1   # 羊
}

# 生肖ID → 尾数
# 规则：mod 10，10→0, 11→1, 12→2
ZODIAC_TAIL = {
    1: 1,   # 马
    2: 2,   # 蛇
    3: 3,   # 龙
    4: 4,   # 兔
    5: 5,   # 虎
    6: 6,   # 牛
    7: 7,   # 鼠
    8: 8,   # 猪
    9: 9,   # 狗
    10: 0,  # 鸡
    11: 1,  # 猴
    12: 2   # 羊
}

# ==================== 3. 号码属性映射（辅助） ====================

def get_number_parity(n):
    """号码 → 单双"""
    return "单" if n % 2 == 1 else "双"

def get_number_size(n):
    """号码 → 大小（1-24小，25-49大）"""
    return "小" if n <= 24 else "大"

def get_number_zone(n):
    """号码 → 区间（1-16/17-32/33-49）"""
    if n <= 16:
        return 1
    elif n <= 32:
        return 2
    else:
        return 3

def get_number_head(n):
    """号码 → 头数（十位数）"""
    return n // 10

def get_number_tail(n):
    """号码 → 尾数（个位数）"""
    return n % 10

def get_number_color(n):
    """号码 → 波色（六合彩标准规则）"""
    red = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
    blue = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
    if n in red:
        return "红"
    elif n in blue:
        return "蓝"
    else:
        return "绿"

# ==================== 4. 年份生肖映射 ====================

def get_year_zodiac(year):
    """获取指定年份的生肖（2026年是马年）"""
    base_year = 2026
    base_idx = ZODIAC_ORDER.index("马")
    year_diff = year - base_year
    idx = (base_idx - year_diff) % 12
    return ZODIAC_ORDER[idx]

def get_allocation_by_year(year):
    """根据年份获取生肖-号码分配表"""
    base_year = 2026
    base_zodiac = "马"
    idx_base = ZODIAC_ORDER.index(base_zodiac)
    year_diff = year - base_year
    idx_current = (idx_base - year_diff) % 12
    current_zodiac = ZODIAC_ORDER[idx_current]
    
    offset = (idx_current - ZODIAC_ORDER.index("马")) % 12
    rotated = {}
    for i, zod in enumerate(ZODIAC_ORDER):
        target = ZODIAC_ORDER[(i + offset) % 12]
        rotated[target] = BASE_ALLOCATION[zod]
    return rotated, current_zodiac

def number_to_zodiac(number, year):
    """根据年份和号码返回生肖"""
    allocation, _ = get_allocation_by_year(year)
    for zod, nums in allocation.items():
        if number in nums:
            return zod
    return None

# ==================== 5. 获取生肖完整属性 ====================

def get_zodiac_attributes(zodiac_id):
    """获取生肖的完整属性"""
    zodiac_name = ID_TO_ZODIAC[zodiac_id]
    return {
        "id": zodiac_id,
        "name": zodiac_name,
        "wuxing": ZODIAC_WU_XING[zodiac_name],
        "color": ZODIAC_COLOR[zodiac_name],
        "parity": ZODIAC_PARITY[zodiac_id],
        "size": ZODIAC_SIZE[zodiac_id],
        "zone": ZODIAC_ZONE[zodiac_id],
        "head": ZODIAC_HEAD[zodiac_id],
        "tail": ZODIAC_TAIL[zodiac_id]
    }

def print_zodiac_table():
    """打印完整的生肖映射表"""
    print("="*80)
    print("十二生肖完整映射表")
    print("="*80)
    print(f"{'ID':<4} {'生肖':<4} {'五行':<4} {'波色':<4} {'单双':<4} {'大小':<4} {'区间':<4} {'头数':<4} {'尾数':<4}")
    print("-"*80)
    
    for i in range(1, 13):
        attr = get_zodiac_attributes(i)
        print(f"{attr['id']:<4} {attr['name']:<4} {attr['wuxing']:<4} {attr['color']:<4} "
              f"{attr['parity']:<4} {attr['size']:<4} {attr['zone']:<4} {attr['head']:<4} {attr['tail']:<4}")
    
    print("="*80)

# ==================== 6. 测试 ====================

if __name__ == "__main__":
    print_zodiac_table()
    
    print("\n2026年（马年）号码分配：")
    allocation, current = get_allocation_by_year(2026)
    print(f"本命生肖：{current}")
    for zod, nums in allocation.items():
        wuxing_list = [f"{n}({NUMBER_WU_XING[n]})" for n in nums]
        print(f"  {zod}: {', '.join(wuxing_list)}")
