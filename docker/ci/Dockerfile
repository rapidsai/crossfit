FROM nvcr.io/nvidia/nemo:23.10

COPY . /tmp/crossfit/
RUN cd /tmp/crossfit && \
    python3 -m pip install .[pytorch-dev] && \
    python3 -m pip install beir && \
    rm -r /tmp/crossfit

ENV CF_HOME /root/.cf
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python3 -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/deberta-v3-base'); AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')" && \
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
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip && \
    unzip scifact.zip -d ${CF_HOME}/raw && \
    rm /tmp/scifact.zip
