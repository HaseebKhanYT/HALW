"""Dataset loaders.

Each loader returns a pandas DataFrame with two columns:
    - text:  the document text
    - label: 0 = human, 1 = AI
"""

import os

import kagglehub
import pandas as pd


def load_shanegrami(n_per_class=50000, random_state=42):
    """Shanegrami AI-vs-Human dataset, balanced and shuffled."""
    dataset_dir = kagglehub.dataset_download("shanegerami/ai-vs-human-text")
    csv_path = os.path.join(dataset_dir, "AI_Human.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected AI_Human.csv at {csv_path}")

    df = pd.read_csv(csv_path)
    df_ai = df[df["generated"] == 1].sample(n=n_per_class, random_state=random_state)
    df_human = df[df["generated"] == 0].sample(n=n_per_class, random_state=random_state)

    df = (
        pd.concat([df_ai, df_human])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
        .rename(columns={"generated": "label"})
    )
    return df[["text", "label"]]
