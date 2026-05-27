# @Time    : 2026/5/26 22:34
# @Author  : hero
# @File    : project_default_configs.py
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
project_path = Path(__file__).parents[1]

EMAIL_PWD = os.getenv('MAIL_VAL')
EMAIL_SENDER = os.getenv('MAIL_SENDER')


