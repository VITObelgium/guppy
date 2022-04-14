import os
from dataclasses import dataclass

import yaml


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
class _Auth:
    public_key: str
    guppy2_user: str
    guppy2_password: str


@dataclass(frozen=True)
class _Config:
    deploy: _Deploy
    database: _Database
    auth: _Auth


default_deploy_path: str = '/api'


def sanitize_deploy_path(deploy_path: str) -> str:
    # Remove all leading and tailing slashes
    deploy_path = deploy_path.lstrip('/').rstrip('/')
    return '/' + deploy_path


def parse_config_file(config_file: str) -> _Config:
    print("Parsing '%s'..." % config_file, end='')
    try:
        yml_file = open(config_file, 'r')
    except IOError:
        print(" not found")
        raise SystemExit("No usable config file found")
    with yml_file:
        yml_data = yaml.load(yml_file, Loader=yaml.SafeLoader)
    print(" OK")
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
        auth=_Auth(public_key=yml_data['auth']['public_key'], guppy2_user=yml_data['auth']['guppy2_user'], guppy2_password=yml_data['auth']['guppy2_password'])
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
                auth=_Auth(
                    public_key=guppy2_env_vars['GUPPY_AUTH_PUBLIC_KEY'],
                    guppy2_user=guppy2_env_vars['GUPPY_AUTH_USER'],
                    guppy2_password=guppy2_env_vars['GUPPY_AUTH_PASSWD'],
                )
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
