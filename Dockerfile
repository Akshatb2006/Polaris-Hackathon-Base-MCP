FROM python:3.12-slim
# MCP_ALLOWED_HOSTS='*': hosting platforms assign the public hostname at deploy time; override to restrict.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 FASTEMBED_CACHE=/app/.models OMP_NUM_THREADS=2 \
    MCP_ALLOWED_HOSTS=*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the embedding model into the image so startup never downloads anything.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.models')"
RUN useradd --uid 10001 --create-home student
COPY --chown=student:student . .
# Build services that clone without Git LFS leave pointer files; fetch and checksum-verify the real ones.
RUN python deploy/fetch_lfs.py data/products.json index/embeddings.npy \
    && echo "05b6aa5e8e8c1c8c997d584f8288e432862454f8820531e13473443e83c8f073  data/products.json" | sha256sum -c -
# Build the dense index at image build time if it was not shipped (fingerprint-checked at startup).
RUN python -c "import json,pathlib; from engine.dense import DenseIndex; \
p=sorted(json.loads(pathlib.Path('data/products.json').read_text()), key=lambda r: r['id']); \
DenseIndex.load(p, allow_build=True)" && chown -R student:student /app/index /app/.models
USER student
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/healthz', timeout=3)"
CMD ["python", "server.py"]
