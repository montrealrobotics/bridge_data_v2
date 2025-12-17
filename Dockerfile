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
RUN unset PIP_INDEX_URL && cd edgeml && pip install -e .

WORKDIR $WORKSPACE
#RUN git clone -b v2.5.0 https://github.com/pytorch/pytorch
#RUN cd pytorch && git submodule sync && git submodule update --init --recursive

#COPY ./bridge_data_v2/build_torch_orin.sh /root/workspace/src/pytorch
#WORKDIR $WORKSPACE/pytorch
#RUN export CMAKE_POLICY_VERSION_MINIMUM=3.5 && bash ./build_torch_orin.sh
COPY torch-2.5.0-cp312-cp312-linux_aarch64.whl $WORKSPACE
WORKDIR $WORKSPACE
RUN unset PIP_INDEX_URL && pip install torch-2.5.0-cp312-cp312-linux_aarch64.whl
COPY ./bridge_data_v2 $WORKSPACE/bridge_data_v2
COPY ./bridge_data_robot $WORKSPACE/bridge_data_robot
COPY ./checkpoints $WORKSPACE/checkpoints
COPY ./requirements.txt $WORKSPACE
COPY ./cusparse_install.sh $WORKSPACE
RUN bash ./cusparse_install.sh
RUN unset PIP_INDEX_URL && pip install -r $WORKSPACE/requirements.txt
