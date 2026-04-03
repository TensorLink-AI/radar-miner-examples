FROM python:3.11-slim
WORKDIR /app
COPY core/ core/
COPY agents/pareto_hunter/run.py .
CMD ["python", "run.py"]
