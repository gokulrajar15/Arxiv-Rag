FROM python:3.11-slim

RUN useradd -m -u 1000 user

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/code

WORKDIR $HOME/code

COPY --chown=user:user pyproject.toml ./

RUN uv pip install -r pyproject.toml --system

RUN mkdir -p $HOME/code && \
    chown -R user:user $HOME/code

COPY --chown=user:user app/ ./app/
COPY --chown=user:user .env ./.env

RUN chown -R user:user $HOME

USER user
EXPOSE 80

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:80"]
