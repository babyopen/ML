# 生肖预测模型 Vercel 部署指南

## 项目概述

这是一个基于 scikit-learn 的生肖预测模型，部署在 Vercel 平台上。模型使用历史数据进行训练，通过 API 端点提供预测服务。

## 技术栈

- **前端**: HTML5 + CSS3 + JavaScript
- **后端**: Node.js (Vercel Serverless Functions)
- **机器学习**: scikit-learn (Python)
- **数据处理**: pandas, numpy

## 项目结构

```
├── api/                # API 端点
│   └── predict.js      # 预测接口
├── data/               # 数据文件
│   └── lottery_history_recalculated.csv  # 历史开奖数据
├── models/             # 模型文件
├── web/                # 前端文件
│   └── predict.html    # 预测结果页面
├── vercel.json         # Vercel 配置文件
├── package.json        # 项目配置文件
└── README.md           # 部署指南
```

## 部署步骤

### 1. 准备工作

1. **安装 Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **登录 Vercel**
   ```bash
   vercel login
   ```

### 2. 部署项目

1. **初始化并部署**
   ```bash
   vercel
   ```

2. **生产环境部署**
   ```bash
   vercel --prod
   ```

### 3. 环境配置

Vercel 会自动处理以下配置：
- Python 环境 (3.9)
- Node.js 环境
- API 路由配置
- 静态文件服务

## API 端点

### 预测接口

**URL**: `/api/predict`
**方法**: GET
**响应格式**:

```json
{
  "success": true,
  "prediction": {
    "top3": [
      {
        "zodiac_id": 7,
        "zodiac_name": "马",
        "wuxing": "火",
        "color": "红",
        "probability": 0.0833
      },
      // 更多预测结果...
    ],
    "all": [/* 所有生肖预测 */],
    "confidence": 8.33,
    "recommended": { /* 推荐生肖 */ },
    "analysis": [/* 分析结果 */]
  },
  "timestamp": "2026-04-03 09:08:31"
}
```

## 前端页面

访问部署后的根路径，即可看到预测结果页面。页面包含：
- 预测控制按钮
- 预测结果展示
- Top 3 推荐
- 所有生肖概率分布
- 预测历史记录

## 注意事项

1. **模型文件**：确保 `zodiac_model_sklearn_20260403_085812.pkl` 文件存在于项目根目录
2. **数据文件**：确保 `data/lottery_history_recalculated.csv` 文件存在
3. **Python 依赖**：Vercel 会自动安装 Python 依赖，但如果需要额外依赖，请在 `requirements.txt` 中添加
4. **冷启动**：Serverless 函数可能会有冷启动延迟
5. **资源限制**：Vercel 有函数执行时间和内存限制，请确保模型预测在限制范围内完成

## 本地开发

1. **启动本地开发服务器**
   ```bash
   vercel dev
   ```

2. **访问本地地址**
   - API: http://localhost:3000/api/predict
   - 前端: http://localhost:3000

## 故障排查

1. **API 调用失败**：检查模型文件和数据文件是否存在
2. **预测结果异常**：检查数据格式是否正确
3. **部署失败**：检查 Vercel 配置文件是否正确

## 性能优化

1. **模型优化**：可以考虑使用更轻量级的模型或模型压缩
2. **缓存策略**：对于相同输入的预测，可以考虑缓存结果
3. **异步处理**：对于复杂的预测任务，可以考虑使用异步处理

## 许可证

MIT
