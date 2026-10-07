# SimulSI with the web dashboard, as a container.
#
#   docker build -t simulsi .
#   docker run --rm -p 127.0.0.1:8642:8642 simulsi                 # dashboard on http://localhost:8642
#   docker run --rm -v "$PWD:/work" simulsi run /work/model.py     # any CLI command
#
# The dashboard has no authentication: publish the port on 127.0.0.1 only.

FROM node:22-slim AS web
WORKDIR /src/web/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/frontend/ ./
# vite writes the compiled dashboard into the Python package (../../src/simulsi/web/static)
RUN npm run build

FROM python:3.12-slim AS wheel
WORKDIR /src
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md LICENSE NOTICE CHANGELOG.md ./
COPY src ./src
COPY --from=web /src/src/simulsi/web/static ./src/simulsi/web/static
RUN python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
COPY --from=wheel /dist /tmp/dist
RUN pip install --no-cache-dir "$(ls /tmp/dist/*.whl)[viz]" && rm -rf /tmp/dist \
    && useradd --create-home --uid 1000 simulsi \
    && mkdir /work && chown simulsi /work
USER simulsi
WORKDIR /work
EXPOSE 8642
ENTRYPOINT ["simulsi"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8642", "--no-browser"]
