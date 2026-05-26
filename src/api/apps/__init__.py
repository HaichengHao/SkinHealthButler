# @Time    : 2026/5/26 21:55
# @Author  : hero
# @File    : __init__.py.py
from .up_and_down_load.uad import uad_route
from fastapi import FastAPI

def create_app():
    app=FastAPI(
        title='Upload your file',
        description='Upload your file with the below APIs🥳',
    )
    app.include_router(uad_route)
    return app