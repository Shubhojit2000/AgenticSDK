# The AgenticSDK web service (API + Gradio UI). Build and run:
#   docker build -t agenticsdk .
#   docker run -p 7860:7860 -e GOOGLE_API_KEY=... -e DEMO_MODE=1 agenticsdk        # then open http://localhost:7860/ui
# Generated scripts run as screened subprocesses of this container (allow-list, stripped environment, wall-clock
# limit). For a container-per-script sandbox with no network, run the service on a host that has Docker and set
# AGENTICSDK_SANDBOX=docker (see Dockerfile.sandbox); that mode cannot work from inside this image.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1     && rm -rf /var/lib/apt/lists/*     && useradd --create-home --uid 1000 app
WORKDIR /app
COPY pyproject.toml ./
COPY agenticsdk ./agenticsdk
RUN pip install --no-cache-dir -e ".[serve]" && mkdir -p runs data results && chown -R app:app /app
USER app

ENV PORT=7860 DEMO_MODE=1 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "-c", "uvicorn agenticsdk.service.app:app --host 0.0.0.0 --port ${PORT}"]
