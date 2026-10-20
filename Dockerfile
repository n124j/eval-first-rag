FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY corpus/ corpus/
COPY evals/recordings/ evals/recordings/
RUN pip install --no-cache-dir --no-deps .

RUN useradd --system --create-home --uid 10001 evalrag \
    && mkdir /data \
    && chown evalrag:evalrag /data

ENV DATA_DIR=/data
USER evalrag

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "--factory", "evalrag.api:app_from_env", "--host", "0.0.0.0", "--port", "8000"]
