FROM robonet-base:latest

COPY requirements.txt /tmp/requirements.txt
RUN ~/myenv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONPATH=${PYTHONPATH}:/home/robonet/code/bridge_data_v2

# avoid git safe directory errors
RUN git config --global --add safe.directory /home/robonet/code/bridge_data_v2

WORKDIR /home/robonet/code/bridge_data_v2
