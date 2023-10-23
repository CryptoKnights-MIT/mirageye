FROM mambaorg/micromamba
USER root
RUN apt-get update && DEBIAN_FRONTEND=“noninteractive” apt-get install -y --no-install-recommends \
       nginx \
       ca-certificates \
       apache2-utils \
       certbot \
       python3-certbot-nginx \
       sudo \
       cifs-utils \
       && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y install cron
RUN mkdir /opt/deploy
RUN chmod -R 777 /opt/deploy
WORKDIR /opt/deploy
EXPOSE 8501
RUN adduser --disabled-password --gecos '' micromamba -u 1000
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
   micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install transformers googletrans-py google-trans-new pygoogletranslation gTTS
RUN python3 -m pip install transformers --upgrade 
COPY run.sh run.sh
COPY models models
COPY app app
COPY nginx.conf /etc/nginx/nginx.conf
USER root
RUN chmod a+x run.sh
CMD ["./run.sh"]