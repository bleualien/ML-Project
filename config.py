# server/config.py
import os
import logging

def setup_logging():
    # Ensure logs directory exists
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s — %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info("Logging initialized successfully.")
