FROM python:3.11-slim@sha256:a7bce671c71744cd4dcef2bc9cd9c5eff3d7cfa6b55eca30759765047184bcdb
RUN apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
COPY wheels /wheels
RUN pip install --no-index --find-links=/wheels pytest==8.3.3 text-unidecode==1.3 setuptools==75.8.0 wheel==0.45.1 \
 && pip check && rm -rf /wheels
ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTEST_ADDOPTS="-p no:cacheprovider" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
