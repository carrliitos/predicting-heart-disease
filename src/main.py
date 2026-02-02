import os
from pathlib import Path

from scripts import extraction
from utils import logger
from utils import context

directory = context.get_context(os.path.abspath(__file__))
logger_name = Path(__file__).stem
logging = logger.setup_logger(logger_name, f"{directory}/src/logs/main.log")

def main():
  extraction.extract()

if __name__ == "__main__":
  main()
