FROM python:3.9-slim

LABEL maintainer="Zodiac Prediction Team"
LABEL version="1.0.0"
LABEL description="生肖预测系统 - 基于机器学习的智能预测模型"

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import pandas, numpy, sklearn; print('OK')" || exit 1

CMD ["/app/start.sh"]
