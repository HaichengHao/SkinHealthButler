# @Time    : 2026/5/26 22:12
# @Author  : hero
# @File    : main.py

import sys
import argparse
from pathlib import Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from backend.api.apps import create_app

app = create_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main():
    parser = argparse.ArgumentParser(description="SkinHealthButler FastAPI launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8099, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload (dev only)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
