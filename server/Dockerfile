FROM continuumio/miniconda3 as conda-env

WORKDIR /opt/guppy/server
COPY environment.yml environment.yml
RUN conda env update --prefix /opt/guppy/server/conda-env --file environment.yml


FROM debian:bullseye-slim

RUN  echo "**** install packages ****" \
	&& apt-get update \
	&& DEBIAN_FRONTEND=noninteractive \
	apt-get install -y \
	curl \
	&& rm -rf /var/lib/apt/lists/* \
	&& echo

RUN groupadd --system guppy && \
	useradd --system -g guppy guppy

WORKDIR /opt/guppy/server
COPY --from=conda-env /opt/guppy/server/conda-env conda-env
COPY guppy guppy

RUN mkdir -p /opt/guppy/server/data
RUN chmod 775 /opt/guppy/server/data
RUN chown -R guppy:guppy /opt/guppy/server/data

USER guppy

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH /opt/guppy/server/conda-env/bin:$PATH

EXPOSE 8080
CMD ["conda-env/bin/gunicorn", "--bind", "0.0.0.0:8080", "--worker-tmp-dir", "/dev/shm", "--workers=2", "--timeout=300", "--threads=4", "--worker-class=uvicorn.workers.UvicornWorker", "--log-file=-", "guppy.server:app"]