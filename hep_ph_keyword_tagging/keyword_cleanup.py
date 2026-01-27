import re
from collections import Counter, defaultdict
# =========================================================
# PATTERNS AND CURATED SETS
# =========================================================

PACS_RE = re.compile(r"^\d{2}\.\d{2}\.[0-9A-Za-z+\-]+$")

UNITS = {"gev", "tev", "mev", "kev", "ev", "pb", "fb"}

EXPERIMENT_LABELS = {
    "atlas", "cms", "alice", "lhcb", "belle", "babar", "cdf", "d0",
    "cern lhc coll", "brookhaven rhic coll", "desy hera stor",
    "fermilab tev coll", "tevatron", "lhc",
}

GENERIC_SINGLE_WORD = {
    "mass", "density", "interaction", "production", "decay", "scattering",
    "spectrum", "distribution", "model", "parameter", "parameters",
    "structure", "background", "signature", "effect", "process",
    "measurement", "constraints", "limits", "dependence", "sensitivity",
    "benchmark", "anomaly", "unitarity", "ultraviolet", "stability",
    "suppression", "programming", "kinematics", "galaxy", "correlation",
    "tension", "review", "bibliography",
}

KEEP_SINGLETONS = {
    "supersymmetry", "susy", "supergravity",
    "axion", "wimp",
    "baryogenesis", "leptogenesis",
    "inflation",
    "qcd", "qed",
    "higgs", "neutrino",
    "photon", "gluon", "quark", "lepton",
    "hadron", "meson", "baryon",
    "cosmology",
}

MERGE_MAP = {
    "new physics": "beyond standard model",
    "physics beyond the standard model": "beyond standard model",
    "bsm": "beyond standard model",
}

METHOD_LABELS = {
    "numerical calculations",
}

# =========================================================
# NORMALIZATION
# =========================================================

def norm_kw(k: str) -> str:
    k = (k or "").strip().lower()
    k = k.replace("–", "-").replace("—", "-")
    k = re.sub(r"\s+", " ", k)
    return MERGE_MAP.get(k, k)

def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\s\-:]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def looks_code_like(s: str) -> bool:
    letters = sum(ch.isalpha() for ch in s)
    return letters / max(len(s), 1) < 0.5


# =========================================================
# FILTER FUNCTIONS
# =========================================================

def is_junk_keyword(k: str,DROP_IF_CONTAINS_DIGIT: bool) -> bool:
    if not k:
        return True
    if k in METHOD_LABELS:
        return True
    if PACS_RE.match(k):
        return True
    if k in UNITS:
        return True
    if DROP_IF_CONTAINS_DIGIT and any(ch.isdigit() for ch in k):
        return True
    if looks_code_like(k):
        return True
    return False

def is_experiment_keyword(k: str) -> bool:
    return k in EXPERIMENT_LABELS

def is_bad_singleton(k: str) -> bool:
    toks = k.replace(":", " ").split()
    if len(toks) != 1:
        return False
    if k in KEEP_SINGLETONS:
        return False
    return True

def prune_parent_keywords(keywords):
    children = defaultdict(set)
    for k in keywords:
        parts = [p.strip() for p in k.split(":")]
        for i in range(1, len(parts)):
            parent = ": ".join(parts[:i])
            children[parent].add(k)
    return [k for k in keywords if k not in children]


def cleanup(df,DROP_IF_CONTAINS_DIGIT = True):

    cleaned_keywords_per_row = []
    all_cleaned_keywords = []

    for _, row in df.iterrows():

        raw = row.get("keywords", []) or []
        kws = [norm_kw(k) for k in raw]

        # Step 1: experiments
        kws1 = [k for k in kws if not is_experiment_keyword(k)]

        # Step 2: junk
        kws2 = [k for k in kws1 if not is_junk_keyword(k,DROP_IF_CONTAINS_DIGIT)]

        # Step 3: generic singletons
        kws3 = [k for k in kws2 if not is_bad_singleton(k)]


        # Step 4: prune colon parents
        kws4 = prune_parent_keywords(kws3)

        cleaned_keywords_per_row.append(kws4)

        all_cleaned_keywords.extend(kws4)
    
    df["keywords_clean"] = cleaned_keywords_per_row

    return df[df["keywords_clean"].apply(len) > 0].copy(),all_cleaned_keywords
