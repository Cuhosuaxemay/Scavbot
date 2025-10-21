# Dockerfile
FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Copy all your application code into the image.
# This includes llmcord.py, requirements.txt, and importantly, config-example.yaml.
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# This command runs when the container starts.
# It checks for config files and creates them if they're missing, then runs the app.
CMD sh -c "[ -f /app/data/config.yaml ] || cp /app/config-example.yaml /app/data/config.yaml; \
           [ -f /app/data/role_states.yaml ] || touch /app/data/role_states.yaml; \
           exec python llmcord.py"