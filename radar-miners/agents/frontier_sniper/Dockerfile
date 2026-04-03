FROM python:3.11-slim
WORKDIR /app
COPY core/ core/
COPY agents/frontier_sniper/run.py .

# affinetes Actor wrapper
COPY env.py /app/env.py

# affinetes HTTP server layer
RUN pip install --no-cache-dir affinetes fastapi uvicorn[standard] httpx pydantic
RUN AFFINETES_TEMPLATES="$(python -c 'import affinetes, pathlib; print(pathlib.Path(affinetes.__file__).parent / "templates")')" \
    && cp "$AFFINETES_TEMPLATES/http_server.py" /tmp/http_server.py
RUN mkdir -p /app/_affinetes && echo "" > /app/_affinetes/__init__.py
RUN cp /tmp/http_server.py /app/_affinetes/server.py

WORKDIR /app
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port ${AFFINETES_PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
