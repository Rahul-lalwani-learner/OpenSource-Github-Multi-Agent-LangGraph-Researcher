import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
from config import settings
from utils import logger

def load_settings():
    logger.info("Loading settings...")
    return settings

if __name__ == "__main__":
    print("Settings loaded:", load_settings())