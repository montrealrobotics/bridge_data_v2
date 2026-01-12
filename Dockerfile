FROM dustynv/jax:0.5.2-r36.4.0-cu128-24.04

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /root/workspace/src

ENV WORKSPACE=/root/workspace/src

RUN apt-get update && apt-get install --no-install-recommends -y \
	git \
	vim \
	python3-opencv \
	xauth \
	x11-apps \
	cmake \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR $WORKSPACE

RUN git clone https://github.com/youliangtan/edgeml
RUN git clone -b orin https://github.com/montrealrobotics/bridge_data_robot.git
RUN unset PIP_INDEX_URL && cd edgeml && pip install -e .

WORKDIR $WORKSPACE
#RUN git clone -b v2.5.0 https://github.com/pytorch/pytorch
#RUN cd pytorch && git submodule sync && git submodule update --init --recursive

#COPY ./bridge_data_v2/build_torch_orin.sh /root/workspace/src/pytorch
#WORKDIR $WORKSPACE/pytorch
#RUN export CMAKE_POLICY_VERSION_MINIMUM=3.5 && bash ./build_torch_orin.sh
#COPY torch-2.5.0-cp312-cp312-linux_aarch64.whl $WORKSPACE
#WORKDIR $WORKSPACE
#RUN unset PIP_INDEX_URL && pip install torch-2.5.0-cp312-cp312-linux_aarch64.whl
COPY . $WORKSPACE/bridge_data_v2

RUN set -e; \
    unset PIP_INDEX_URL; \
    WHEEL="$WORKSPACE/bridge_data_v2/torch-2.5.0-cp312-cp312-linux_aarch64.whl"; \
    if [ -f "$WHEEL" ]; then \
        echo "Found torch wheel: $WHEEL"; \
        pip install "$WHEEL"; \
    else \
        echo "Torch wheel not found; building torch from source"; \
        cd "$WORKSPACE"; \
        git clone -b v2.5.0 https://github.com/pytorch/pytorch; \
        cd pytorch && git submodule sync && git submodule update --init --recursive; \
        cp "$WORKSPACE/bridge_data_v2/build_torch_orin.sh" ./build_torch_orin.sh; \
        export CMAKE_POLICY_VERSION_MINIMUM=3.5; \
        bash ./build_torch_orin.sh; \
        # adjust this line if your script outputs elsewhere:
        pip install dist/*.whl; \
    fi

COPY ./checkpoints $WORKSPACE/checkpoints
COPY ./requirements.txt $WORKSPACE
COPY ./cusparse_install.sh $WORKSPACE
RUN bash ./cusparse_install.sh
RUN unset PIP_INDEX_URL && pip install -r $WORKSPACE/requirements.txt
