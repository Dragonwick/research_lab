FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY color_detector_node/ ./color_detector_node/
COPY color-env/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

ENV NODE_URL=http://0.0.0.0:2001/

CMD ["python", "-m", "color_detector_node.node"]
