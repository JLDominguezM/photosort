from fastapi import Request

from services.database import get_db


def get_app_db(request: Request):
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


def get_clip(request: Request):
    return request.app.state.clip


def get_faces(request: Request):
    return request.app.state.faces


def get_hasher(request: Request):
    return request.app.state.hasher
