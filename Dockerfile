FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY crashgap ./crashgap
COPY dashboard ./dashboard
RUN pip install --no-cache-dir .

ENV CRASHGAP_DB=/app/data/crashgap.db
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
