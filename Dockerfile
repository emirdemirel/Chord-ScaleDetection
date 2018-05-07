#!/usr/bin/env bash

FROM mtgupf/essentia:ubuntu16.04-python3

# Export env settings
ENV TERM=xterm
ENV LANG C.UTF-8

RUN apt-get update -y && apt-get install build-essential -y

ADD apt-packages.txt /tmp/apt-packages.txt
RUN xargs -a /tmp/apt-packages.txt apt-get install -y

ADD requirements.txt /tmp/requirements/requirements.txt
RUN pip3 install -r /tmp/requirements/requirements.txt

RUN mkdir -p /usr/local/lib/vamp \
    && cd /usr/local/lib/vamp \
    && wget https://code.soundsoftware.ac.uk/attachments/download/1693/nnls-chroma-linux64-v1.1.tar.bz2 \
    && tar xvjf nnls-chroma-linux64-v1.1.tar.bz2 \
    && rm nnls-chroma-linux64-v1.1.tar.bz2 \
    && mv nnls-chroma-linux64-v1.1/* /usr/local/lib/vamp/ \
    && rm -r nnls-chroma-linux64-v1.1

EXPOSE 8888
RUN mkdir -p /home/ds/notebooks
ENV HOME=/home/ds
ENV SHELL=/bin/bash
ENV USER=ds
VOLUME /home/ds/notebooks
WORKDIR /home/ds/notebooks

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ADD jupyter_notebook_config.json /home/ds/.jupyter/
CMD ["tini", "--", "jupyter-notebook", "--allow-root", "--no-browser", "--port", "8888", "--ip", "0.0.0.0"]
