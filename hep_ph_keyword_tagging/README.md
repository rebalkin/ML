## [Automatic Keyword Tagging for hep-ph Papers](hep_ph_keyword_tagging/main.ipynb)

A multilabel NLP project that predicts INSPIRE-style keywords for high-energy physics phenomenology (hep-ph) abstracts.

Key components include:
- Data collection via the INSPIRE API
- Keyword normalization and filtering
- TF–IDF feature extraction
- Logistic regression trained in a One-vs-Rest multilabel setting
- Threshold tuning using a dedicated cross-validation set
- Evaluation using micro-F1, macro-F1, and top-k metrics
- Qualitative error analysis on randomly selected abstracts

This project serves as a complete example of a **realistic multilabel NLP pipeline**, including label imbalance, sparsity, and threshold calibration.

### Files

- **main.ipynb**  
  End-to-end notebook for the keyword tagging project.  
  Covers data loading, preprocessing, feature extraction, model training, threshold tuning, and evaluation.

- **parsing_data.py**  
  Utilities for querying and parsing raw metadata from the INSPIRE API into a structured format.

- **word_cleanup.py**  
  Keyword normalization and filtering logic used to clean, deduplicate, and prune noisy or uninformative labels before modeling.

