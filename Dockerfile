# Base image
FROM python:3.11-slim

# Arguments
ARG USER=raguser
ARG UID=1000
ARG GID=1000

# Install dependencies
RUN apt-get update && \
    apt-get install -y git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create user
RUN groupadd -g $GID $USER && \
    useradd -m -u $UID -g $GID -s /bin/bash $USER

WORKDIR /app
USER $USER

# Copy project files
COPY --chown=$USER:$USER . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ENV PATH="/home/${USER}/.local/bin:${PATH}"


# Expose API port
EXPOSE 8000

# Entrypoint
CMD ["make", "serve"]
