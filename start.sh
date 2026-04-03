#!/bin/bash

set -e

echo "=========================================="
echo "  生肖预测系统"
echo "=========================================="
echo ""

echo "[1/3] 生成预测HTML..."
python3 scripts/generate_html_new.py
echo ""

echo "[2/3] 启动HTTP服务器..."
echo "服务将在 http://localhost:6000/web/predict.html 启动"
echo "按 Ctrl+C 停止服务"
echo ""

python3 -m http.server 8000
