FROM nvcr.io/nvidia/cuda:11.7.1-runtime-ubuntu22.04
# FROM alpine:3.19.1
LABEL maintainer="Athena ML"
LABEL version="1.0"

# COPY --from=nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 /usr/local/cuda/ /usr/local/cuda/

COPY --from=continuumio/miniconda3:latest /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app/

RUN conda install -c conda-forge -y python=3.10

RUN apt-get update -y && apt-get install -y unzip curl

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" &&\
    unzip awscliv2.zip &&\
    ./aws/install &&\
    rm awscliv2.zip

# COPY models/ /app/models/
COPY /build/llm-server/bin /app/bin/
COPY config.toml /app/config.toml
COPY build/llm-server/requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY build/llm-server/start.py /app/start.py

ENV LLM_SERVER_THREADS=8
ENV LLM_SERVER_THREADS_BATCH=8

CMD ["python3", "/app/start.py"]
