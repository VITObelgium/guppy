FROM ghcr.io/prefix-dev/pixi:0.65.0 AS build

WORKDIR /opt/guppy/server
COPY pixi.lock pixi.toml ./
RUN pixi install --locked


FROM debian:trixie-slim@sha256:26f98ccd92fd0a44d6928ce8ff8f4921b4d2f535bfa07555ee5d18f61429cf0c

RUN echo "**** install packages ****" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && echo

RUN groupadd --system guppy && \
    useradd --system -g guppy guppy

WORKDIR /opt/guppy/server
COPY --from=build /opt/guppy/server/.pixi .pixi
COPY guppy guppy

RUN mkdir -p /opt/guppy/server/data
RUN chmod 775 /opt/guppy/server/data
RUN chown -R guppy:guppy /opt/guppy/server/data

USER guppy

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH /opt/guppy/server/.pixi/envs/default/bin:$PATH

EXPOSE 8080
CMD ["fastapi", "run", "guppy/server.py", "--port", "8080" ]