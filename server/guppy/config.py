import logging
import os
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Database:
    type: str
    host: str
    user: str
    passwd: str
    db: str


@dataclass(frozen=True)
class _Deploy:
    path: str
    content: str


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
default_content_path: str = '/content'


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
    if 'content' not in yml_data['deploy']:
        yml_data['deploy']['content'] = default_content_path
    yml_data['deploy']['path'] = sanitize_deploy_path(yml_data['deploy']['path'])
    return _Config(
        deploy=_Deploy(
            path=yml_data['deploy']['path'],
            content=yml_data['deploy']['content']
        ),
        database=_Database(
            type=yml_data['database']['type'] if 'type' in yml_data['database'] else 'postgres',
            host=yml_data['database']['host'] if 'host' in yml_data['database'] else '',
            user=yml_data['database']['user'] if 'user' in yml_data['database'] else '',
            passwd=yml_data['database']['passwd'] if 'passwd' in yml_data['database'] else '',
            db=yml_data['database']['db'],
        ),
        guppy=_Guppy(size_limit=int(float(yml_data['guppy']['size_limit'])) if 'guppy' in yml_data and 'size_limit' in yml_data['guppy'] else 10000),
        geoserver=_Geoserver(username=yml_data['geoserver']['username'] if 'geoserver' in yml_data and 'username' in yml_data['geoserver'] else '',
                             password=yml_data['geoserver']['password'] if 'geoserver' in yml_data and 'password' in yml_data['geoserver'] else '')
    )


config: _Config = None
# The docker way to configure containers is by using environment variables
guppy_env_vars = {k: v for k, v in os.environ.items() if k.startswith('GUPPY_')}
if guppy_env_vars:
    if 'GUPPY_CONFIG_FILE' in guppy_env_vars:
        config = parse_config_file(guppy_env_vars['GUPPY_CONFIG_FILE'])
    else:
        if 'GUPPY_DATABASE_PASSWD_FILE' in guppy_env_vars:
            try:
                with open(guppy_env_vars['GUPPY_DATABASE_PASSWD_FILE']) as f:
                    guppy_env_vars['GUPPY_DATABASE_PASSWD'] = f.readline().strip()
            except IOError:
                raise SystemExit("Password file '%s' not found!" % guppy_env_vars['GUPPY_DATABASE_PASSWD_FILE'])
        if 'GUPPY_GEOSERVER_PASSWD_FILE' in guppy_env_vars:
            try:
                with open(guppy_env_vars['GUPPY_GEOSERVER_PASSWD_FILE']) as f:
                    guppy_env_vars['GUPPY_GEOSERVER_PASSWD'] = f.readline().strip()
            except IOError:
                raise SystemExit("Password file '%s' not found!" % guppy_env_vars['GUPPY_GEOSERVER_PASSWD_FILE'])
        if 'GUPPY_DEPLOY_PATH' not in guppy_env_vars:
            guppy_env_vars['GUPPY_DEPLOY_PATH'] = default_deploy_path
        if 'GUPPY_CONTENT_PATH' not in guppy_env_vars:
            guppy_env_vars['GUPPY_CONTENT_PATH'] = default_content_path
        guppy_env_vars['GUPPY_DEPLOY_PATH'] = sanitize_deploy_path(guppy_env_vars['GUPPY_DEPLOY_PATH'])
        try:
            config = _Config(
                deploy=_Deploy(
                    path=guppy_env_vars['GUPPY_DEPLOY_PATH'],
                    content=guppy_env_vars['GUPPY_CONTENT_PATH'],
                ),
                database=_Database(
                    type=guppy_env_vars['GUPPY_DATABASE_TYPE'] if 'GUPPY_DATABASE_TYPE' in guppy_env_vars else 'sqlite',
                    host=guppy_env_vars['GUPPY_DATABASE_HOST'] if 'GUPPY_DATABASE_HOST' in guppy_env_vars else '',
                    user=guppy_env_vars['GUPPY_DATABASE_USER'] if 'GUPPY_DATABASE_USER' in guppy_env_vars else '',
                    passwd=guppy_env_vars['GUPPY_DATABASE_PASSWD'] if 'GUPPY_DATABASE_PASSWD' in guppy_env_vars else '',
                    db=guppy_env_vars['GUPPY_DATABASE_DB'],
                ),
                guppy=_Guppy(size_limit=int(float(guppy_env_vars['GUPPY_SIZE_LIMIT'])) if 'GUPPY_SIZE_LIMIT' in guppy_env_vars else 10000),
                geoserver=_Geoserver(username=guppy_env_vars['GUPPY_GEOSERVER_USER'] if 'GUPPY_GEOSERVER_USER' in guppy_env_vars else '',
                                     password=guppy_env_vars['GUPPY_GEOSERVER_PASSWD'] if 'GUPPY_GEOSERVER_PASSWD' in guppy_env_vars else '')
            )
        except KeyError as key_error:
            raise SystemExit("Environment variable '%s' not found!" % key_error)
else:
    # No environment variables declared, read marvin config file
    try:
        config = parse_config_file("/etc/marvin/guppy.yml")
    except SystemExit:
        # Marvin config file not found, running in dev, read local config file
        config = parse_config_file("config.yml")
