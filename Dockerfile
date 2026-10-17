FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/srv/.hf PYTHONPATH=/srv/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src src
COPY app app
COPY scripts scripts
COPY data/docs.jsonl data/
# Build the index and cache all models at build time so the container starts offline.
RUN python scripts/build_index.py \
 && python -c "from sentence_transformers import CrossEncoder; from privrag import config; CrossEncoder(config.RERANK_MODEL)" \
 && python -c "from privrag.generate import _local_model; _local_model()"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
