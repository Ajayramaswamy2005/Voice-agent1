# Public demo image: the clinic and support lines. The browser line is left out by default because it
# runs a real browser on your server for whoever calls (see README, "Host it").
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .

ENV HOST=0.0.0.0 PORT=8000 \
    ENABLED_LINES=clinic,support \
    MAX_CALLS=3 MAX_CALL_SECONDS=300
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "python", "-m", "agent.main"]
