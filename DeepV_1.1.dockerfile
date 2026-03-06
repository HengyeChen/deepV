# 使用 Python 3.8 基础镜像
FROM python:3.8-slim

# 创建非root用户
RUN groupadd -r deepvuser && useradd -r -g deepvuser deepvuser

# 设置工作目录
WORKDIR /data

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/data


# 安装系统依赖包
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    gcc \
    g++ \
    make \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libhdf5-dev \
    pkg-config \
    tabix \
    parallel \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖包
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas \
    numpy==1.24.3 \
    tensorflow \
    matplotlib \
    scipy \
    scikit-learn \
    h5py \
    keras \
    opencv-python-headless

# 创建必要的目录结构
RUN mkdir -p /data/step1 /data/step2 /data/logs /data/final_results /data/temp_bed_files

# 复制必要的文件
COPY hg38_chromsize.txt /data/
COPY run_deepv_pipeline.sh /data/
COPY step1/ /data/step1/
COPY step2/ /data/step2/
COPY test_file/ /data/test_file/
COPY input_file/ /data/input_file/

# 设置文件权限和所有权
RUN chmod +x /data/run_deepv_pipeline.sh && \
    chmod +x /data/step1/*.py && \
    chmod +x /data/step2/*.py && \
    chown -R deepvuser:deepvuser /data && \
    chmod -R 777 /data

# 切换到非root用户
USER deepvuser

# 暴露端口
EXPOSE 9000

# 设置工作目录
WORKDIR /data

# 设置入口点
ENTRYPOINT ["bash", "/data/run_deepv_pipeline.sh"]

################## METADATA ######################
LABEL tool.baseimage="python:3.8-slim" 
LABEL tool.name="DeepV"  
LABEL tool.version="1.1" 
LABEL tool.summary="A CNN-based deep learning tool capable of graphically identifying transcription factor footprints on chromatin from chromatin accessibility data." 
LABEL tool.license="SPDX:GPL-3.0" 
LABEL tool.tags="Genomics|Bioinformatics|Deep Learning|Chromatin Analysis|Transcription Factor"
LABEL tool.home="https://github.com/example/DeepV" 
LABEL tool.documentation="https://github.com/example/DeepV/wiki" 
LABEL tool.arch="x86_64" 
LABEL tool.structure="basic" 
LABEL tool.port="9000" 
LABEL tool.workdir="/data" 
LABEL tool.uploader="Qifan Zhang" 
LABEL tool.packtime="20260305" 
LABEL tool.module="Bioinformatics" 
LABEL tool.element="Genomic Analysis"
