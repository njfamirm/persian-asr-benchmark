FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*