FROM python:3.12-slim
WORKDIR /app

# gcc/g++ needed for building any C-extension deps (xgboost/prophet's cmdstanpy
# wheels are prebuilt for this base image, but keep a minimal toolchain for safety)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# sentence-transformers pulls torch transitively, and pip's default index
# resolves the CUDA build: ~2.5GB of nvidia-cu* wheels that are dead weight
# here (this service gets no GPU passthrough) and heavy enough to OOM the
# builder. Install the CPU wheel first so the resolver below sees torch as
# already satisfied.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
