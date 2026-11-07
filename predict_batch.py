"""Small local inference smoke test; input must be curated Parquet with all features."""
import argparse
import time
import joblib
import pandas as pd
from src import config as cfg
from src.scoring import score_pandas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to local curated parquet file or parquet directory")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    art = joblib.load(cfg.OUT_DIR / "models" / "model.joblib")
    df = pd.read_parquet(args.input, columns=art["cols"]).head(args.limit)
    t0 = time.perf_counter()
    result = score_pandas(df, art)
    print(result.head(10).to_string(index=False))
    print(f"Scored {len(df)} rows in {time.perf_counter()-t0:.3f}s")

if __name__ == "__main__":
    main()
