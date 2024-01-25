from fastapi import HTTPException


def create_error(message: str, code: int = 400) -> HTTPException:
    """
    Args:
        message: A string representing the error message.
        code: An integer representing the HTTP status code for the error. Default is 400.

    Raises:
        HTTPException: An exception indicating an HTTP error with the specified error message and status code.

    """
    raise HTTPException(status_code=code, detail=message)
