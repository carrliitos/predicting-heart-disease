import pandas as pd
from pathlib import Path

def run_query_and_save(query, save_file_name, data_path, connection, query_logger):
  df = pd.read_sql_query(query, connection)

  save_path = Path(data_path) / save_file_name
  save_path.parent.mkdir(parents=True, exist_ok=True)

  df.to_csv(save_path, index=False)
  query_logger.info(f"Query saved to: {save_path}")
  query_logger.info(f"Rows returned: {len(df)}")

def open_and_read(file):
  with open(file, "r", encoding="utf-8") as input_file:
    file_content = input_file.read()

  return file_content