from pathlib import Path

import polars as pl

datapath = Path('./data/preprocessed_skipgram/')

for dataset in datapath.iterdir():
    size = (pl.scan_parquet(dataset / 'context_windows.parquet')
            .select(pl.len())
            .collect())
    print(f'{dataset.name}: {size.item():,}')
