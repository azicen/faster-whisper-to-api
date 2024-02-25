FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list
RUN sed -i "s@http://.*ports.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list

RUN apt update && apt upgrade -y --no-install-recommends
RUN apt install -y --no-install-recommends \
        ca-certificates \
        netbase \
        net-tools \
        iputils-ping

RUN DEBIAN_FRONTEND=noninteractive apt install -y ffmpeg

ENV PYTHONUNBUFFERED=1

RUN apt install -y python3 python3-pip

RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple pip -U
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

RUN python3 -m pip install --upgrade pip

ADD ./requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

ADD ./app.py /app/app.py

WORKDIR /app/

EXPOSE 8000

ENTRYPOINT ["hypercorn", "main:app"]
CMD ["--workers", "2", "--bind", "0.0.0.0:8000", "--access-log", "-"]
