"""Feature pipelines.

Each pipeline takes a DataFrame with `text` and `label` and returns a DataFrame
of numeric features with `label` appended. The `text` column is dropped.
"""

import spacy
import textdescriptives as td
from tqdm import tqdm


def extract_textdescriptives(
    df,
    text_col="text",
    label_col="label",
    spacy_model="en_core_web_lg",
    batch_size=1000,
    n_process=4,
):
    """Extract the full textdescriptives feature set (~70 features per text)."""
    nlp = spacy.load(spacy_model)
    nlp.add_pipe("textdescriptives/all")

    docs = list(
        tqdm(
            nlp.pipe(df[text_col].astype(str), batch_size=batch_size, n_process=n_process),
            total=len(df),
            desc="Extracting features",
        )
    )

    features = td.extract_df(docs)
    if text_col in features.columns:
        features = features.drop(columns=[text_col])
    features[label_col] = df[label_col].values
    return features
