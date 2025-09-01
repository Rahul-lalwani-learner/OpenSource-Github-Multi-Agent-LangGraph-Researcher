from loguru import logger
LOG_FILE = "logs/app_{time}.log"

logger.add(LOG_FILE, 
           rotation="500 MB",
           retention="10 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
