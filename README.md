# Guppy server

## Installation

### Conda environment

Ensure miniconda is installed on your system. If not, download and install it from [here](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create --file environment.yml
```

## Usage

### Define configuration

Create a configuration file in the directory `server` named `config.yml` with the following content. Adjust for your specific setup.
minimal yaml:

```yaml
# ./server/config.yml
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

# change directory to 'server'
cd server

# start server using defaults
PYTHONPATH=. python guppy2/server.py
```

Your empty Guppy server is availlable at http://127.0.0.1:5000/guppy/docs

Run the server with reload option to automatically restart the server when changes are made to the code.

```bash
python -m uvicorn guppy2.server:app --reload --port 8000 --host 0.0.0.0
```
 
## Access the UI

The Guppy API is the core of the application, some UI is implemented on top of the API to illustrate the functionality. 

The pages already provided as examples are:

| Page | URL |
| --- | --- |
| Upload layer | http://localhost:8000/guppy/admin/upload/ui |
| Layer metadata | http://localhost:8000/guppy/admin/layers |
| Tile statistics | http://localhost:8000/guppy/admin/stats |
| Layers on a map | http://localhost:8000/guppy/admin/stats |




## Access the API
  
Documentation for the API is available at http://127.0.0.1:5000/guppy/docs

Notice deploy_path is defined in the config.yml file. 
The default value is guppy.
<!-- If you change the deploy path in the config.yml file, you should also change the path in the docker-compose.yml file. -->
<!-- The path in the docker-compose.yml file should be the same as the path in the config.yml file. -->