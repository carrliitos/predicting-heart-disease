import os
import sys
import pandas as pd
import json
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from pathlib import Path
from utils import logger
from utils import connections
from utils import context
from utils import query as qr

from datetime import datetime
today = datetime.today().strftime("%Y-%m-%d")

directory = context.get_context(os.path.abspath(__file__))
logger_name = Path(__file__).stem
extraction_logger = logger.setup_logger(logger_name, f"{directory}\\logs\\main.log")
data_path = f"{directory}\\data"
version_num = "v01"

def extract():
  try:
    extracts_directory = f"{directory}\\sql\\extracts"
    clarity_engine = connections.connect_db("Clarity")
    clarity_connection = clarity_engine.connect()

    demographics_sql = "select * from dst_team_datasets.example_demographics;"
    a1c_sql = qr.open_and_read(f"{extracts_directory}//00-All_Population.sql")

    queries = {
      demographics_sql : f"HUANG-DAB-CIED-Demographics_{version_num}.xlsx",
      a1c_sql : f"HUANG-DAB-CIED-A1C_{version_num}.xlsx",
    }

    for query, file_name in queries.items():
      qr.run_query_and_save(query, file_name, data_path, clarity_connection, extraction_logger)
  except ConnectionError as connection_error:
    extraction_logger.error(f"Unable to connect to db: {connection_error}")
    sys.exit(1)
  except KeyError as key_error:
    extraction_logger.error(f"Incorrect connection keys: {key_error}")
    sys.exit(1)

  extraction_logger.debug("Extraction complete.")
