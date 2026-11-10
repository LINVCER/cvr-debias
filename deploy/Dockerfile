FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUN_DIR=artifacts/runs/main/uc_escm/seed0

WORKDIR /app

# 先装依赖再拷代码，改代码不触发重装
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install -r deploy/requirements.txt

COPY src ./src
COPY configs ./configs
COPY deploy ./deploy

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "deploy.serve:app", "--host", "0.0.0.0", "--port", "8000"]
