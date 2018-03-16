FROM cupy/cupy:latest

RUN pip install asv virtualenv

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
