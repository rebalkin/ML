# Scientific Impact and Novelty in Theoretical Particle Physics

An analysis of ~140,000 hep-ph papers from [INSPIRE](https://inspirehep.net) (2000-2025) using [SPECTER2](https://huggingface.co/allenai/specter2) embeddings.

**Two questions:**
1. Does novelty predict impact? We define a paper's novelty score as its cosine distance from the centroid of papers published in the preceding 12 months, and ask whether semantic outliers tend to be more or less cited.
2. Can we classify high-impact papers from their text embedding alone? We train a linear classifier on frozen SPECTER2 representations and test it on papers the model never saw during training.

## Repository structure

```
main.ipynb              — Main analysis notebook (start here)
utils.py                — Reusable functions (models, plotting, novelty computation)
parsing_data.py         — Fetches papers from the INSPIRE HEP API
get_embed.py            — Computes SPECTER2 embeddings (requires GPU)
generate_animation.py   — Generates the UMAP time-lapse animation (run once)
data/                   — Data files (not tracked, see data/README.md)
```

## Reproducing the analysis

**1. Get the data**

```bash
pip install requests pandas pyarrow
```

```python
from parsing_data import get_data
import pandas as pd

df = get_data(from_Year=2000, categories=("hep-ph",), verbal=True)
df.to_parquet('data/papers.parquet', index=False)
```

**2. Compute embeddings** (requires a GPU)

```bash
pip install torch transformers adapters
python get_embed.py
```

**3. Run the notebook**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn torch jupyter
jupyter notebook main.ipynb
```

The UMAP projection (`data/embedding_transformed.npy`) is generated separately — see `data/README.md` for the code snippet. Once saved, the notebook loads it directly.

## Interpretation

The negative novelty-impact relationship is real, but what it means is less obvious than it looks.

SPECTER2 is trained on both abstract text and citation relationships, so the embedding reflects a mix of vocabulary and intellectual lineage. The novelty score conflates the two: a paper looks distant from the mainstream if it uses unusual terminology, if it cites a different set of papers, or both. A paper can be semantically distant while being physically identical to mainstream work (different language, same physics), and conversely, a genuinely original idea framed in familiar language and citing the usual references will look perfectly conventional.

So the finding is probably not that physics punishes originality. It is more likely that the field rewards papers that speak the language of their community and engage with the appropriate literature — which is not the same thing. The 2025 out-of-sample predictions support this reading: the classifier's blind spots are concentrated in non-invertible symmetry papers, an emerging subfield with its own specialized vocabulary that sits in an unusual corner of embedding space but is clearly generating real scientific interest.

## Key results

- SPECTER2 embeddings carry modest but real predictive signal for citation impact (ROC-AUC ~0.67 vs. 0.50 random baseline).
- The novelty score does not improve classification performance over the embedding alone — whatever novelty encodes is already captured by the embedding direction.
- The novelty-impact relationship in hep-ph is consistently negative across all subfields: papers that look semantically unusual relative to the recent discourse tend to be cited less, not more.
- Applying the classifier trained on 2000-2024 to 2025 papers reveals a systematic blind spot: non-invertible symmetry papers are underrated by the model but already accumulating citations.
