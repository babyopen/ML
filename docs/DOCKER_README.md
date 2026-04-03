# 生肖预测系统 - Docker部署指南

## 📋 目录

- [项目概述](#项目概述)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [配置说明](#配置说明)

## 🎯 项目概述

基于机器学习的生肖预测系统，使用RandomForest算法构建预测模型。

## 📁 文件结构

```
ML模型/
├── Dockerfile              # Docker镜像构建文件
├── docker-compose.yml      # Docker Compose配置文件
├── requirements.txt        # Python依赖包
├── .dockerignore          # Docker忽略文件
├── start.sh               # 容器启动脚本
├── zodiac_ml_predictor.py # 模型预测核心代码
├── zodiac_model.pkl       # 训练好的模型文件
├── lottery_history.csv    # 历史数据文件
├── data_fetcher.py        # API数据获取模块
├── update_data.py         # 数据更新脚本
├── model_optimizer.py     # 模型优化脚本
├── generate_html.py       # HTML生成脚本
├── predict.py             # 预测脚本
├── predict.html           # 预测结果页面
└── DOCKER_README.md       # 本文件
```

## 🚀 快速开始

### 方式一：使用Docker Compose（推荐）

```bash
# 1. 构建并启动容器
docker-compose up -d

# 2. 查看日志
docker-compose logs -f

# 3. 访问页面
# 浏览器打开: http://localhost:8000/predict.html

# 4. 停止容器
docker-compose down
```

### 方式二：使用Docker命令

```bash
# 1. 构建镜像
docker build -t zodiac-prediction .

# 2. 运行容器
docker run -d \
  --name zodiac-prediction \
  -p 8000:8000 \
  -v $(pwd)/lottery_history.csv:/app/lottery_history.csv \
  -v $(pwd)/zodiac_model.pkl:/app/zodiac_model.pkl \
  zodiac-prediction

# 3. 查看日志
docker logs -f zodiac-prediction

# 4. 停止容器
docker stop zodiac-prediction
docker rm zodiac-prediction
```

## 📖 使用方法

### 1. 访问预测页面

容器启动后，在浏览器中打开：

```
http://localhost:8000/predict.html
```

### 2. 更新数据

```bash
# 进入容器
docker exec -it zodiac-prediction bash

# 增量更新数据
python3 update_data.py --mode incremental

# 全量更新数据
python3 update_data.py --mode full

# 重新生成HTML
python3 generate_html.py

# 退出容器
exit
```

### 3. 重新训练模型

```bash
# 进入容器
docker exec -it zodiac-prediction bash

# 优化模型
python3 model_optimizer.py

# 退出容器
exit
```

## ⚙️ 配置说明

### Dockerfile配置

| 配置项 | 说明 |
|--------|------|
| 基础镜像 | python:3.9-slim |
| 工作目录 | /app |
| 暴露端口 | 8000 |
| 时区 | Asia/Shanghai |

### docker-compose.yml配置

| 配置项 | 说明 |
|--------|------|
| 服务名 | zodiac-prediction |
| 端口映射 | 8000:8000 |
| 数据卷 | lottery_history.csv, zodiac_model.pkl, predict.html |
| 重启策略 | unless-stopped |
| 健康检查 | 每30秒检查一次 |

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| TZ | 时区 | Asia/Shanghai |
| PYTHONUNBUFFERED | Python输出缓冲 | 1 |

## 📊 依赖包

| 包名 | 版本要求 | 说明 |
|------|----------|------|
| pandas | >=1.5.0 | 数据处理 |
| numpy | >=1.21.0 | 数值计算 |
| scikit-learn | >=1.0.0 | 机器学习 |
| requests | >=2.28.0 | HTTP请求 |

## 🔒 安全说明

1. 容器使用非root用户运行（可选）
2. 只暴露必要的端口（8000）
3. .dockerignore排除敏感文件
4. 使用alpine/slim基础镜像减小体积

## 📝 注意事项

1. 确保 `lottery_history.csv` 和 `zodiac_model.pkl` 文件存在
2. 首次运行会自动生成 `predict.html`
3. 数据更新建议使用增量模式
4. 模型重新训练需要足够的历史数据

## 🆘 故障排除

### 容器无法启动

```bash
# 查看日志
docker-compose logs

# 检查端口是否被占用
lsof -i :8000
```

### 预测页面无法访问

```bash
# 确认容器正在运行
docker-compose ps

# 检查端口映射
docker-compose port zodiac-prediction 8000
```

### 数据更新失败

```bash
# 进入容器检查
docker exec -it zodiac-prediction bash

# 检查网络连接
ping -c 3 macaumarksix.com

# 检查文件权限
ls -la /app/
```

## 📄 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2026-04-03
