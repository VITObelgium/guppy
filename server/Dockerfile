FROM continuumio/miniconda3 as conda-env

WORKDIR /opt/guppy2/server
COPY environment.yml environment.yml
RUN conda env update --prefix /opt/guppy2/server/conda-env --file environment.yml


FROM debian:bullseye-slim

RUN  echo "**** install packages ****" \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive \
     apt-get install -y \
       curl \
  && rm -rf /var/lib/apt/lists/* \
  && echo

RUN groupadd --system guppy2 && \
    useradd --system -g guppy2 guppy2

WORKDIR /opt/guppy2/server
COPY --from=conda-env /opt/guppy2/server/conda-env conda-env
COPY guppy2 guppy2

USER guppy2

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

EXPOSE 8080
CMD ["conda-env/bin/gunicorn", "--bind", "0.0.0.0:8080", "--worker-tmp-dir", "/dev/shm", "--workers=2", "--timeout=300", "--threads=4", "--worker-class=uvicorn.workers.UvicornWorker", "--log-file=-", "guppy2.server:app"]