from fastapi import HTTPException


def create_error(message, code=400):
    raise HTTPException(status_code=code, detail=message)
