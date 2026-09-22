# A Python app that happens to carry a browser. Most of the image is Chromium
# and the system libraries it needs, which is why there is no point splitting
# this into build and runtime stages — nothing here is build-only.

FROM python:3.12-slim

# PLAYWRIGHT_BROWSERS_PATH is set explicitly because the browser is installed
# as root and run as someone else; the default location is a home directory
# that the runtime user does not have.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    BROWSER_AGENT_CONTAINER=true \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

# Dependencies first, so editing the application does not reinstall Chromium.
COPY pyproject.toml README.md ./
COPY browser_agent/__init__.py browser_agent/__init__.py
RUN pip install --no-cache-dir ".[web]" \
 && playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/*

COPY browser_agent/ browser_agent/
RUN pip install --no-cache-dir --no-deps .

# Chromium will run as this user, not as root. A browser that renders whatever
# a task points it at should not be the most privileged thing on the box.
RUN useradd --create-home --uid 10001 agent \
 && chmod -R a+rX /opt/playwright
USER agent

EXPOSE 8080

# The health check is what turns "the image has no browser" from a mystery a
# visitor reports into a deploy that refuses to go live.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/healthz', timeout=8).status == 200 else 1)"

CMD ["browser-agent-web"]
