FROM nvcr.io/nvidia/pytorch:23.10-py3

COPY . /tmp/crossfit/
RUN cd /tmp/crossfit && \
    python3 -m pip install .[pytorch-dev] && \
    python3 -m pip install -r requirements/gpu.txt --no-cache-dir --extra-index-url https://pypi.nvidia.com && \
    rm -r /tmp/crossfit

ENV CF_HOME /root/.cf
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    cd /tmp && \
    mkdir -p ${CF_HOME}/raw && \
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip && \
    unzip fiqa.zip -d ${CF_HOME}/raw && \
    rm /tmp/fiqa.zip && \
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip && \
    unzip hotpotqa.zip -d ${CF_HOME}/raw && \
    rm /tmp/hotpotqa.zip && \
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip && \
    unzip nq.zip -d ${CF_HOME}/raw && \
    rm /tmp/nq.zip
