import os
import re
import pandas as pd
from datasets import Dataset


# PREPARE DATA

# folder = snapshot_download(
#     "cis-lmu/glotcc-v1",
#     repo_type="dataset",
#     local_dir="./glotcc-v1/",
#     allow_patterns="v1.0/ltz-Latn/*"
# )

# Load the dataset from a Parquet file
# Replace the file path with the path to the desired language's Parquet file

data = []
for path, subdirs, files in os.walk("../LOD-Corpus/Texter/ANER_TEXTER"):
    for name in files:
        fpath = os.path.join(path, name)
        with open(fpath) as fin:
            content = fin.read()
        data.append({
            "source": fpath,
            "content": content,
            "content-length": len(content),
            })

data = pd.DataFrame(data)

CLEANR = re.compile('<.*?>')


def cleanhtml(raw_html):
    return re.sub(CLEANR, '', raw_html)


data.content = data.content.apply(cleanhtml)
dataset = pd.read_parquet('./glotcc-v1/v1.0/ltz-Latn/ltz-Latn_0.parquet')
dataset = pd.concat([data, dataset], ignore_index=True)
print(dataset.head())
dataset = Dataset.from_pandas(dataset[["content"]]).shuffle(seed=42)
