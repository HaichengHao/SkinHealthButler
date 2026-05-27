# @Time    : 2026/5/26 22:12
# @Author  : hero
# @File    : main.py

import uvicorn
from apps import create_app
from fastapi.middleware.cors import CORSMiddleware

if __name__ == '__main__':
    app = create_app()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host='127.0.0.1', port=8099)