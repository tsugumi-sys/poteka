from dotenv import load_dotenv
from pathlib import Path
import requests
import os
from logging import getLogger, DEBUG, basicConfig
from typing import Optional

dotenv_path = Path(".env")
load_dotenv(dotenv_path=dotenv_path)

logger = getLogger(__name__)
basicConfig(level=DEBUG)
logger.setLevel(DEBUG)


def send_line(msg: str) -> Optional[int]:
    line_notify_token = os.getenv("LINE_TOKEN")

    if line_notify_token is None:
        logger.info("LINE TOKEN is not setted in .env file at your root directory.")
        return

    line_notify_endpoint = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": msg}
    res = requests.post(url=line_notify_endpoint, headers=headers, data=data)

    return res.status_code
