import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
from datetime import datetime

from utils import logger
from utils import context

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import xgboost as xgb
from xgboost import XGBClassifier

def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
  df_ = df_.copy()
  df_.columns = (df_.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(" ", "_"))
  return df_

def kaggle_submit(comp: str, file: str | Path, msg: str) -> str:
  file = str(Path(file).expanduser().resolve())
  cmd = ["kaggle", "competitions", "submit", "-c", comp, "-f", file, "-m", msg]
  res = subprocess.run(cmd, check=True, text=True, capture_output=True)
  return (res.stdout or "") + (res.stderr or "")

def go():
  directory = context.get_context(os.path.abspath(__file__))
  logger_name = Path(__file__).stem
  kaggle_logger = logger.setup_logger(logger_name, f"{directory}/logs/main.log")

  train_df = pd.read_csv(f"{directory}/data-raw/train.csv")
  test_df  = pd.read_csv(f"{directory}/data-raw/test.csv")

  train_df = tweak_kag(train_df)
  test_df  = tweak_kag(test_df)

  TARGET = "heart_disease"
  ID_COL = "id"

  if TARGET not in train_df.columns:
    raise ValueError(f"TARGET '{TARGET}' not found. Available columns: {list(train_df.columns)}")

  y = train_df[TARGET]
  X = train_df.drop(columns=[TARGET])

  if ID_COL in X.columns:
    X = X.drop(columns=[ID_COL])

  X_test = test_df.copy()
  if ID_COL in X_test.columns:
    X_test_no_id = X_test.drop(columns=[ID_COL])
  else:
    X_test_no_id = X_test

  X_enc = pd.get_dummies(X, drop_first=False)
  X_test_enc = pd.get_dummies(X_test_no_id, drop_first=False)
  X_enc, X_test_enc = X_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

  if y.dtype == "object":
    uniq = set(y.dropna().unique())

    if uniq == {"No", "Yes"}:
      y_enc = y.map({"No": 0, "Yes": 1})
    elif uniq == {"N", "Y"}:
      y_enc = y.map({"N": 0, "Y": 1})
    elif uniq == {"False", "True"}:
      y_enc = y.map({"False": 0, "True": 1})
    elif uniq == {"Absence", "Presence"}:
      y_enc = y.map({"Absence": 0, "Presence": 1})
    else:
      y_map = {v: i for i, v in enumerate(sorted(uniq))}
      y_enc = y.map(y_map)
      kaggle_logger.info(f"y_map used: {y_map}")
  else:
    y_enc = y.astype(int)

  if y_enc.isnull().any():
    raise ValueError("y_enc has nulls after mapping. Check label values in TARGET column.")
  y_enc = y_enc.astype(int)

  X_tr, X_va, y_tr, y_va = train_test_split(
    X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
  )

  early_stop = xgb.callback.EarlyStopping(
    rounds=200,
    metric_name="logloss",
    save_best=True,
  )

  model = XGBClassifier(
    n_estimators=10_000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    callbacks=[early_stop]
  )

  model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    verbose=100
  )

  best_iter = model.best_iteration
  best_ntree = best_iter + 1
  kaggle_logger.info(f"best_iteration: {best_iter} (best_ntree={best_ntree})")

  va_proba = model.predict_proba(X_va, iteration_range=(0, best_ntree))[:, 1]
  va_pred = (va_proba >= 0.5).astype(int)

  acc = accuracy_score(y_va, va_pred)
  kaggle_logger.info(f"Validation accuracy: {acc:.4f}")

  try:
    auc = roc_auc_score(y_va, va_proba)
    kaggle_logger.info(f"Validation ROC-AUC: {auc:.4f}")
  except Exception as e:
    kaggle_logger.info(f"ROC-AUC not computed: {e}")

  final_model = XGBClassifier(
    n_estimators=best_ntree,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
  )

  final_model.fit(X_enc, y_enc, verbose=False)
  test_proba = final_model.predict_proba(X_test_enc)[:, 1]

  submission = pd.DataFrame({
    ID_COL: test_df[ID_COL] if ID_COL in test_df.columns else range(len(test_df)),
    TARGET: test_proba
  })

  out_path = f"{directory}/data/submission_{datetime.today().strftime('%Y-%m-%d')}.csv"
  submission.to_csv(out_path, index=False)
  kaggle_logger.info(f"Saved submission: {out_path}")

  version = "02"
  kaggle_submit(comp="playground-series-s6e2", file=Path(out_path), msg=f"XGBoost version {version}")

if __name__ == "__main__":
  go()
