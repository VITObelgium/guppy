from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from guppy2.config import config as cfg

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

public_key = "-----BEGIN PUBLIC KEY-----\n" + cfg.auth.public_key + "\n-----END PUBLIC KEY-----"


async def get_current_user(token: str = Depends(oauth2_scheme)):
    if token:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            data = jwt.decode(token, public_key, algorithms=['RS512'], options={
                'verify_signature': False,
                'verify_exp': False,
            })
            if data['type'] == 'ACCESS' or data['type'] == 'USER_INITIALISATION':
                roles = data['roles']
            else:
                raise credentials_exception
        except JWTError as e:
            print(e)
            raise credentials_exception
        return roles
    return None


def get_user_id_from_token(token: str):
    if token:
        data = jwt.decode(token, public_key, algorithms=['RS512'], options={
            'verify_signature': False,
            'verify_exp': False,
        })
        if data['type'] == 'ACCESS' or data['type'] == 'USER_INITIALISATION':
            return data['id']
        else:
            return None
