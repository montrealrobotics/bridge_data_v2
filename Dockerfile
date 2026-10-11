FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git libgl1 libglib2.0-0 \
        libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

RUN userdel -r ubuntu 2>/dev/null || true; useradd -m -u 1000 -s /bin/bash robonet
USER robonet
WORKDIR /home/robonet

ENV PATH=/home/robonet/.local/bin:/home/robonet/venv/bin:$PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv --python 3.12 /home/robonet/venv

COPY --chown=robonet:robonet requirements.txt /tmp/requirements.txt
# includes the widowx_envs robot client and edgeml (see requirements.txt)
RUN uv pip install --no-cache -r /tmp/requirements.txt

COPY --chown=robonet:robonet . /home/robonet/code/bridge_data_v2
ENV PYTHONPATH=/home/robonet/code/bridge_data_v2
WORKDIR /home/robonet/code/bridge_data_v2
