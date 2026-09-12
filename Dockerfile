# ============================================================
# Agentic AI Workbench
# LangGraph + RAG + HITL + Streamlit
# ============================================================

FROM python:3.11-slim

# ============================================================
# PYTHON / PIP CONFIGURATION
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501


# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# COPY REQUIREMENTS FIRST
# This allows Docker to cache the dependency layer.
# ============================================================

COPY requirements.txt .


# ============================================================
# INSTALL PYTHON DEPENDENCIES
# ============================================================

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt


# ============================================================
# COPY APPLICATION
# ============================================================

COPY . .


# ============================================================
# STREAMLIT PORT
# ============================================================

EXPOSE 8501


# ============================================================
# HEALTHCHECK
# ============================================================

HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1


# ============================================================
# START STREAMLIT
# ============================================================

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]