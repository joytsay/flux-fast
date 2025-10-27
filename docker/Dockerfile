ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG USERNAME
ARG USERID
ARG GROUPID

RUN echo ${BASE_IMAGE}:${BASE_IMAGE}
RUN echo ${USERNAME}:${USERNAME}
RUN echo ${USERID}:${USERID}
RUN echo ${GROUPID}:${GROUPID}

RUN useradd -m ${USERNAME} -u ${USERID} -g ${GROUPID} -G sudo -s /bin/zsh
RUN echo ${USERNAME}:${USERNAME} | chpasswd
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgl1 \
  libglib2.0-0 \
  libavahi-client-dev

USER ${USERNAME}
ENV PATH=/root/miniconda:${PATH}
USER root
RUN chmod -R 777 /root/miniconda
USER ${USERNAME}

RUN pip install -U huggingface_hub[hf_xet] accelerate transformers
RUN pip install -U diffusers
RUN pip install -U diffusers
RUN pip install sentencepiece
RUN pip install protobuf
# RUN pip install torch --index-url https://download.pytorch.org/whl/cu129
# RUN pip install torchao --index-url https://download.pytorch.org/whl/cu129
# RUN pip install --pre torch==2.9.0.dev20250824+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
# RUN pip install --pre torchao==0.13.0.dev20250819+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
