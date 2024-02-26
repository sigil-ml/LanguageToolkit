FROM python:3.10
LABEL maintainer="Athena ML"
LABEL version="1.0"

WORKDIR /app/
COPY build/api/requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY build/api/langchain_mixtral.py /app/langchain_mixtral.py
COPY build/api/app.py /app/app.py

COPY acronyms.csv /app/acronyms.csv
COPY config.toml /app/config.toml

ENV API_HOST="api"
ENV API_PORT=5000

ENV LLM_SERVER_HOST="llm-server"
ENV LLM_SERVER_PORT=7000

# ENV MIXTRAL8X7B_TEMPERATURE=0.2
# ENV MIXTRAL8X7B_TOP_K=0
# ENV MIXTRAL8X7B_TOP_P=1.0
# ENV MIXTRAL8X7B_MIN_P=0.02
# ENV MIXTRAL8X7B_MIROSTAT=2
# ENV MIXTRAL8X7B_MIROSTAT_LR=0.05
# ENV MIXTRAL8X7B_MIROSTAT_END=3.0
# ENV MIXTRAL8X7B_MAX_LENGTH=512

CMD ["python3", "/app/app.py"]
