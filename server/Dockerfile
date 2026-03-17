FROM continuumio/miniconda3:26.1.1@sha256:b002473017c77d4b8bad6f8c49747e0298c37419a5ec69ca37ebe36d09ee417a as conda-env

WORKDIR /opt/guppy/server
COPY environment.yml environment.yml
RUN conda env update --prefix /opt/guppy/server/conda-env --file environment.yml


FROM debian:bullseye-slim@sha256:4333240150a6924f878e05ec2c998aec95238010e0e4d2fec6161c90128c4652

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
CMD ["fastapi", "run", "guppy/server.py", "--port", "8080" ]