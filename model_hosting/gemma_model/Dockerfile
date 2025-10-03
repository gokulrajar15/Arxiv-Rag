FROM python:3.13.5

WORKDIR /app

COPY . /app/

RUN mkdir -p /app/hf_cache /app/models
RUN chmod -R 777 /app/hf_cache /app/models

ENV HF_HOME=/app/hf_cache

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "server.py"]
