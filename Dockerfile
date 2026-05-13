FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    BACKEND_URL=http://127.0.0.1:8000 \
    STREAMLIT_SERVER_PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["/bin/bash", "./start.sh"]
