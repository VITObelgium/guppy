import logging
import os
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Database:
    host: str
    user: str
    passwd: str
    db: str


@dataclass(frozen=True)
class _Deploy:
    path: str


@dataclass(frozen=True)
class _Guppy:
    size_limit: int


@dataclass(frozen=True)
class _Geoserver:
    username: str
    password: str


@dataclass(frozen=True)
class _Config:
    deploy: _Deploy
    database: _Database
    guppy: _Guppy
    geoserver: _Geoserver


default_deploy_path: str = '/api'


def sanitize_deploy_path(deploy_path: str) -> str:
    # Remove all leading and tailing slashes
    deploy_path = deploy_path.lstrip('/').rstrip('/')
    return '/' + deploy_path


def parse_config_file(config_file: str) -> _Config:
    logger.info(f"Parsing {config_file}...")
    try:
        yml_file = open(config_file, 'r')
    except IOError:
        logger.info(" not found")
        raise SystemExit("No usable config file found")
    with yml_file:
        yml_data = yaml.load(yml_file, Loader=yaml.SafeLoader)
    logger.info(" OK")
    if 'path' not in yml_data['deploy']:
        yml_data['deploy']['path'] = default_deploy_path
    yml_data['deploy']['path'] = sanitize_deploy_path(yml_data['deploy']['path'])
    return _Config(
        deploy=_Deploy(
            path=yml_data['deploy']['path'],
        ),
        database=_Database(
            host=yml_data['database']['host'],
            user=yml_data['database']['user'],
            passwd=yml_data['database']['passwd'],
            db=yml_data['database']['db'],
        ),
        guppy=_Guppy(size_limit=int(float(yml_data['guppy']['size_limit'])) if 'guppy' in yml_data else 10000),
        geoserver=_Geoserver(username=yml_data['geoserver']['username'], password=yml_data['geoserver']['password'])
    )


config: _Config = None
# The docker way to configure containers is by using environment variables
guppy2_env_vars = {k: v for k, v in os.environ.items() if k.startswith('GUPPY_')}
if guppy2_env_vars:
    if 'GUPPY_CONFIG_FILE' in guppy2_env_vars:
        config = parse_config_file(guppy2_env_vars['GUPPY_CONFIG_FILE'])
    else:
        if 'GUPPY_DATABASE_PASSWD_FILE' in guppy2_env_vars:
            try:
                with open(guppy2_env_vars['GUPPY_DATABASE_PASSWD_FILE']) as f:
                    guppy2_env_vars['GUPPY_DATABASE_PASSWD'] = f.readline().strip()
            except IOError:
                raise SystemExit("Password file '%s' not found!" % guppy2_env_vars['GUPPY_DATABASE_PASSWD_FILE'])
        if 'GUPPY_AUTH_PASSWD_FILE' in guppy2_env_vars:
            try:
                with open(guppy2_env_vars['GUPPY_AUTH_PASSWD_FILE']) as f:
                    guppy2_env_vars['GUPPY_AUTH_PASSWD'] = f.readline().strip()
            except IOError:
                raise SystemExit("Password file '%s' not found!" % guppy2_env_vars['GUPPY_AUTH_PASSWD_FILE'])
        if 'GUPPY_GEOSERVER_PASSWD_FILE' in guppy2_env_vars:
            try:
                with open(guppy2_env_vars['GUPPY_GEOSERVER_PASSWD_FILE']) as f:
                    guppy2_env_vars['GUPPY_GEOSERVER_PASSWD'] = f.readline().strip()
            except IOError:
                raise SystemExit("Password file '%s' not found!" % guppy2_env_vars['GUPPY_GEOSERVER_PASSWD_FILE'])
        if 'GUPPY_AUTH_PUBLIC_KEY_FILE' in guppy2_env_vars:
            try:
                with open(guppy2_env_vars['GUPPY_AUTH_PUBLIC_KEY_FILE']) as f:
                    guppy2_env_vars['GUPPY_AUTH_PUBLIC_KEY'] = f.readline().strip()
            except IOError:
                raise SystemExit("Password file '%s' not found!" % guppy2_env_vars['GUPPY_AUTH_PUBLIC_KEY_FILE'])
        if 'GUPPY_DEPLOY_PATH' not in guppy2_env_vars:
            guppy2_env_vars['GUPPY_DEPLOY_PATH'] = default_deploy_path
        guppy2_env_vars['GUPPY_DEPLOY_PATH'] = sanitize_deploy_path(guppy2_env_vars['GUPPY_DEPLOY_PATH'])
        try:
            config = _Config(
                deploy=_Deploy(
                    path=guppy2_env_vars['GUPPY_DEPLOY_PATH'],
                ),
                database=_Database(
                    host=guppy2_env_vars['GUPPY_DATABASE_HOST'],
                    user=guppy2_env_vars['GUPPY_DATABASE_USER'],
                    passwd=guppy2_env_vars['GUPPY_DATABASE_PASSWD'],
                    db=guppy2_env_vars['GUPPY_DATABASE_DB'],
                ),
                guppy=_Guppy(size_limit=int(float(guppy2_env_vars['GUPPY_SIZE_LIMIT'])) if 'GUPPY_SIZE_LIMIT' in guppy2_env_vars else 10000),
                geoserver=_Geoserver(username=guppy2_env_vars['GUPPY_GEOSERVER_USER'] if 'GUPPY_GEOSERVER_USER' in guppy2_env_vars else '',
                                     password=guppy2_env_vars['GUPPY_GEOSERVER_PASSWD'] if 'GUPPY_GEOSERVER_PASSWD' in guppy2_env_vars else '')
            )
        except KeyError as key_error:
            raise SystemExit("Environment variable '%s' not found!" % key_error)
else:
    # No environment variables declared, read marvin config file
    try:
        config = parse_config_file("/etc/marvin/guppy2.yml")
    except SystemExit:
        # Marvin config file not found, running in dev, read local config file
        config = parse_config_file("config.yml")
