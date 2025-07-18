# 使用官方 Python 基础镜像
FROM python:3.10-slim

# 安装 RDKit 所需依赖（RDKit-pypi 需要额外依赖）
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libxrender1 libsm6 libxext6 && \
    apt-get clean

# 设置工作目录
WORKDIR /app

# 拷贝项目文件
COPY . .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置 Flask 启动环境变量
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["flask", "run"]
