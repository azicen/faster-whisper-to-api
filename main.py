import asyncio
from app import app
from hypercorn.config import Config
from hypercorn.asyncio import serve


if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    config.accesslog = "-"
    config.workers = 4
    asyncio.run(serve(app, config))
