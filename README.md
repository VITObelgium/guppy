<img src=".github/splash.png" alt="Guppy, A Python web API for easy geospatial analysis of raster data" />

## Quickstart

Assuming Docker and docker-compose are installed on your system, you can start the Guppy server with the following commands:

```bash
cd server
docker-compose up 
```

Then go access the demo UI https://github.com/VITObelgium/guppy/tree/develop?tab=readme-ov-file#access-the-ui or interact with the API at http://localhost:8080/guppy/docs

## Installation

Instead of using the docker-compose file, you can also run the Guppy server locally.
This allows you to run the server in a development environment and to make changes to the code.

### Conda environment

Ensure miniconda is installed on your system. If not, download and install it from [here](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create --file environment.yml
```

## Usage

### Configuration


#### Configuration file: `config.yml`

Create a configuration file in the directory `server` named `config.yml` with the following content. Adjust for your specific setup.
minimal yaml:

```yaml
# ./server/config.yml
database:
  type: sqlite
  db: guppy.sqlite
deploy:
  path: guppy
  content: /content
```

full yaml:

```yaml
database:
  type: sqlite
  db: guppy.sqlite
deploy:
  path: guppy
  content: /content
guppy:
  size_limit: 2510000000
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
  db: guppy
```

#### Configuration via environment variables

Alternatively, you can set the configuration via environment variables.
These variables will override the values in the configuration file.

The following environment variables are available:

| Environment variable        | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| GUPPY_CONFIG_FILE           | Path to the configuration file                                   |
| **GUPPY_DEPLOY_PATH**       | Path to deploy the Guppy server. default: guppy                  |
| **GUPPY_DEPLOY_CONTENT**    | Path of the content directory. default: /content                 |
| GUPPY_SIZE_LIMIT            | Maximum input area size in m² for stats/data calls               |
| GUPPY_DATABASE_TYPE         | Database type (sqlite, postgres) default: sqlite                 |
| GUPPY_DATABASE_HOST         | Database host                                                    |
| **GUPPY_DATABASE_DB**       | Database name OR path to Sqlite file.                            |
| GUPPY_DATABASE_USER         | Database user                                                    |
| GUPPY_DATABASE_PASSWD       | Database password                                                |
| GUPPY_DATABASE_PASSWD_FILE  | Path to a file containing the database password                  |
| GUPPY_GEOSERVER_USER        | GeoServer account username                                       |
| GUPPY_GEOSERVER_PASSWD      | GeoServer account password                                       |
| GUPPY_GEOSERVER_PASSWD_FILE | Path to a file containing the password for the GeoServer account |

bold items are mandatory for correct working of the server.
### Start server

```bash
# activate conda environment
conda activate guppy

# change directory to 'server'
cd server

# start server using defaults
PYTHONPATH=. python guppy/server.py
```

Your empty Guppy server is available at http://127.0.0.1:5000/guppy/docs

Run the server with reload option to automatically restart the server when changes are made to the code.

```bash
python -m uvicorn guppy.server:app --reload --port 5000 --host 0.0.0.0
```
 
## Access the UI

The Guppy API is the core of the application, some UI is implemented on top of the API to illustrate the functionality. 

The pages already provided as examples are:

| Page            | URL                                         |
|-----------------|---------------------------------------------|
| Upload layer    | http://localhost:5000/guppy/admin/upload/ui |
| Layer metadata  | http://localhost:5000/guppy/admin/layers    |
| Tile statistics | http://localhost:5000/guppy/admin/stats     |
| Layers viewer   | http://localhost:5000/guppy/admin/map       |




## Access the API
  
Documentation for the API is available at http://127.0.0.1:5000/guppy/docs

Notice deploy_path is defined in the config.yml file. 
The default value is guppy.
<!-- If you change the deploy path in the config.yml file, you should also change the path in the docker-compose.yml file. -->
<!-- The path in the docker-compose.yml file should be the same as the path in the config.yml file. -->