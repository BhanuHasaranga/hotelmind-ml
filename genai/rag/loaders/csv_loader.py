"""CSV document loader: renders each row as a small text blob so tabular
data (e.g. exported warehouse snapshots) can be indexed alongside prose
documents."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, max_rows: int | None = None) -> list[dict]:
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)

    docs = []
    for i, row in df.iterrows():
        text = ", ".join(f"{col}={row[col]}" for col in df.columns)
        docs.append(
            {
                "text": text,
                "metadata": {"source": str(path), "doc_type": "csv", "row": int(i)},
            }
        )
    return docs
