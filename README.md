# Guppy server

## Installation

### Conda environment

Ensure miniconda is installed on your system. If not, download and install it from [here](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create --file environment.yml
```

## Usage

### Define configuration

Create a configuration file in `server/guppy2/config.yml` with the following content. Adjust for your specific setup.
minimal yaml:

```yaml
database:
  type: sqlite
  db: guppy2.sqlite
deploy:
  path: guppy
```

full yaml:

```yaml
database:
  type: sqlite
  db: guppy2.sqlite
deploy:
  path: guppy
guppy:
  size_limit: 251e+07
geoserver:
  username: geoserver_account
  password: geoserver_password
```

if you prefer a postgresql database change the database section to:

```yaml
database:
  host: example-host.vito.be
  user: johndoe
  passwd: top-secret-password
  db: guppy2
```

### Start server

```bash
# activate conda environment
conda activate guppy2

# start server
python server/guppy2/server.py
```