import argparse
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    project_root = Path(__file__).resolve().parent
    src_root = project_root / "src"

    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    load_dotenv(project_root / ".env")

    os.environ.setdefault(
        "SKIN_CLS_EXPORT_DIR",
        str(project_root / "model_/exports/swinv2"),
    )

    from backend.api.main import app

    parser = argparse.ArgumentParser(description="SkinHealthButler launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8099, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
