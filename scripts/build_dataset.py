import sys
import argparse
from pathlib import Path

# initial argument parsing
parser = argparse.ArgumentParser()
splitgroup = parser.add_mutually_exclusive_group()
splitgroup.add_argument("--full", action="store_true", help="Store dataset as one table. (default)")
splitgroup.add_argument("--split", action="store_true", help="Split dataset into training/validation subsets.")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_PROCESSED = ROOT / "data_processed"

for p in [ROOT, SRC]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from dataset import DatasetBuilder


builder = DatasetBuilder(verbose=True)

if args.split:
    builder.to_parquet("model_table.parquet", split=True)
else:
    builder.to_parquet("model_table.parquet", split=False)

print("Stores covered: ", len(builder.merged.store_id.unique()))
print("Full dataset shape: ", builder.merged.shape)
print("Training dataset: ", builder.train.shape)
print("Validation dataset: ", builder.valid.shape)
