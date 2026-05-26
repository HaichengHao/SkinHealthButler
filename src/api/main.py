# @Time    : 2026/5/26 22:12
# @Author  : hero
# @File    : main.py

import uvicorn
from apps import create_app

if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, host='127.0.0.1', port=8099)