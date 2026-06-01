# @Time    : 2026/5/26 21:55
# @Author  : hero
# @File    : __init__.py.py
from pathlib import Path

from .up_and_down_load.uad import uad_route
from .chat_with_llm.chat_with import chat_rt
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

def create_app():
    app=FastAPI(
        title='Upload your file',
        description='Upload your file with the below APIs🥳',
    )
    app.include_router(uad_route)
    app.include_router(chat_rt)

    src_root = Path(__file__).resolve().parents[3]
    project_root = Path(__file__).resolve().parents[4]
    fronted_dir = src_root / "fronted"
    index_file = fronted_dir / "index.html"
    imgs_dir = project_root / "imgs"

    if fronted_dir.exists():
        app.mount("/fronted", StaticFiles(directory=str(fronted_dir)), name="fronted")
    if imgs_dir.exists():
        app.mount("/imgs", StaticFiles(directory=str(imgs_dir)), name="imgs")

    @app.get("/", include_in_schema=False)
    async def home():
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "fronted/index.html not found"}

    return app
