# Kaggle Template

Template repo for Kaggle projects with a clean local workflow **and** an easy way to publish notebooks to Kaggle via the 
Kaggle API.

## Quickstart (Local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

Download competition data locally:

```bash
kaggle competitions download -c playground-series-s6e1 -p src/data-raw/
# unzip into src/data-raw/ if needed
```

## Repo Structure

Local dev structure:

* `src/` - project code
  * `main.py` - entry point (optional)
  * `utils/` - local helpers (optional; see Kaggle Utils Dataset section)
* `src/data/` - generated datasets (**gitignored**)
* `src/data-raw/` - raw inputs (**gitignored**)
* `src/notebooks/` - notebooks (local)
* `kaggle/` - Kaggle publishing mirror (one folder per notebook)
* `requirements.txt` - dependencies

Example Kaggle mirror:

```
kaggle/
  01_EDA/
    01_EDA.ipynb
    kernel-metadata.json
  02_Feature_Engineering/
    02_Feature_Engineering.ipynb
    kernel-metadata.json
```

Each folder under `kaggle/` is one Kaggle notebook (kernel).

## Pushing Notebooks to Kaggle (Multiple Notebooks)

Kaggle expects **one notebook per kernel**, each with its own `kernel-metadata.json`.

1. Initialize metadata in the notebook folder:

```bash
kaggle kernels init -p kaggle/01_EDA
```

2. Edit `kaggle/01_EDA/kernel-metadata.json`:

* Set a unique kernel `id` per notebook
* Attach the competition via `competition_sources`

Example:

```json
{
  "id": "benzonsalazar/ps-s6e1-01-eda",
  "title": "PS S6E1 - 01 EDA",
  "code_file": "01_EDA.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["benzonsalazar/kaggle-utils"],
  "competition_sources": ["playground-series-s6e1"],
  "kernel_sources": [],
  "model_sources": []
}
```

3. Push:

```bash
kaggle kernels push -p kaggle/01_EDA
```

Repeat for each notebook with a different kernel `id`, e.g.:

* `benzonsalazar/ps-s6e1-02-feature-engineering`
* `benzonsalazar/ps-s6e1-03-modeling`
* ...

## Reusing Local Helper Python Files on Kaggle (Option 2: Kaggle Dataset)

If notebooks depend on local helper `.py` files (e.g., `stat_funcs.py`), Kaggle will error unless those files are 
available in the Kaggle runtime.

Instead of copying helpers into every kernel folder, create a Kaggle Dataset containing the helper code and mount it 
via `dataset_sources`.

### 1) Create a Kaggle Dataset for helpers

Create a folder like:

```
kaggle_utils/
  stat_funcs.py
  __init__.py
  dataset-metadata.json
```

`kaggle_utils/dataset-metadata.json`:

```json
{
  "title": "Kaggle Utils (Python helpers)",
  "id": "benzonsalazar/kaggle-utils",
  "licenses": [{ "name": "CC0-1.0" }]
}
```

Create the dataset:

```bash
kaggle datasets create -p kaggle_utils -r zip
```

Update later:

```bash
kaggle datasets version -p kaggle_utils -r zip -m "Update helpers"
```

### 2) Attach helpers to each notebook (kernel)

In each `kernel-metadata.json`, include:

```json
"dataset_sources": ["benzonsalazar/kaggle-utils"]
```

### 3) Import helpers inside Kaggle notebooks

Recommended (imports instead of `%run`):

```python
import sys
sys.path.append("/kaggle/input/kaggle-utils")

from stat_funcs import *   # or: import stat_funcs as sf
print("custom functions are now available in the notebook namespace!")
```

If you want `%run`, use an absolute Kaggle path:

```python
%run /kaggle/input/kaggle-utils/stat_funcs.py
```

## Loading Data Locally vs on Kaggle (No Upload Needed)

On Kaggle, competition files are already mounted at:

`/kaggle/input/playground-series-s6e1/`

Use this drop-in loader so the notebook runs in both environments:

```python
from pathlib import Path
import pandas as pd

KAGGLE_COMP_DIR = Path("/kaggle/input/playground-series-s6e1")
LOCAL_DATA_DIR  = Path("../data-raw")  # adjust if needed

if KAGGLE_COMP_DIR.exists():
  train_path = KAGGLE_COMP_DIR / "train.csv"
  test_path  = KAGGLE_COMP_DIR / "test.csv"
else:
  train_path = LOCAL_DATA_DIR / "train.csv"
  test_path  = LOCAL_DATA_DIR / "test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print("Loaded:", train_path, "and", test_path)
```

Optional: confirm what Kaggle mounted:

```python
import os
print(os.listdir("/kaggle/input/playground-series-s6e1"))
```

## Notes / Gotchas

* The Kaggle API cannot accept competition rules for you; you must join/accept on the website before using the API fully.
* `competition_sources` is what mounts competition data in `/kaggle/input/<competition>/`.
* `dataset_sources` is for extra datasets (like your helper code dataset). Keep it empty unless you need additional datasets.

### Quick Cheat Sheet

**Placeholders**
- `<comp>` = competition slug (e.g., `playground-series-s6e1`)
- `<kernel>` = kernel slug (e.g., `youruser/01-eda`)
- `<dir>` = folder containing `kernel-metadata.json` (e.g., `src/notebooks/03_Training/`)
- `<path>` = dataset folder to zip/upload (e.g., `utils/`, `kaggle-artifacts/`)
- `<file>` = submission CSV path (e.g., `src/data/v07-submission.csv`)
- `<msg>` = message string

#### Competitions

```bash
kaggle competitions list
kaggle competitions submit -c <comp> -f <file> -m "<msg>"
````

#### Kernels (Notebooks)

```bash
kaggle kernels init -p <dir>
kaggle kernels push -p <dir>
kaggle kernels push -p <dir> -m "<msg>"
kaggle kernels pull <kernel>
```

### Datasets (zip + version)

```bash
kaggle datasets create -p <path> -r zip
kaggle datasets version -p <path> -m "<msg>"
```
