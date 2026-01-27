from urllib.parse import urlencode, quote
from datetime import date
import requests
import pandas as pd



def build_inspire_url(
    *,
    categories=("hep-ph", "hep-ex"),
    date_from=None,              # e.g. "2025-01-21"
    date_to=None,                # e.g. "2026-01-01" (exclusive bound if you want)
    extra_query=None,            # e.g. 'NOT doc_type:conference'
    sort="mostrecent",           # e.g. "mostrecent", "mostcited"
    size=1000,
    page=1,                      # INSPIRE uses ?page=1,2,3...
    fields=None,                 # list[str]
    base="https://inspirehep.net/api/literature",
):
    """
    Build an INSPIRE literature API URL.

    Notes:
    - arXiv categories are searchable via arxiv_eprints.categories:<cat>
    - date-earliest is searchable via de, e.g. de > 2025-01-20
    """
    if fields is None:
        fields = [
            "titles",
            "abstracts.value",
            "keywords",
            "citation_count",
            "authors.full_name",
            "arxiv_eprints.categories",
            "earliest_date",
        ]

    # Category clause
    cat_clause = " OR ".join([f"arxiv_eprints.categories:{c}" for c in categories])
    if len(categories) > 1:
        cat_clause = f"({cat_clause})"

    # Date clause (using de = date-earliest)
    date_clauses = []
    if date_from:
        date_clauses.append(f"de > {date_from}")
    if date_to:
        date_clauses.append(f"de < {date_to}")

    # Assemble query
    clauses = [cat_clause]
    if date_clauses:
        clauses.append(" AND ".join(date_clauses))
    if extra_query:
        clauses.append(f"({extra_query})" if " " in extra_query else extra_query)

    q = " AND ".join(clauses)

    params = {
        "q": q,
        "sort": sort,
        "size": int(size),
        "page": int(page),
        "fields": ",".join(fields),
    }

    # urlencode with safe characters so the query stays readable but valid
    return f"{base}?{urlencode(params, quote_via=quote, safe='():><=, ')}"

def year_date_windows(start_year: int):
    today = date.today()
    windows = []

    for y in range(start_year, today.year):
        windows.append((f"{y}-01-01", f"{y+1}-01-01"))

    windows.append((f"{today.year}-01-01", today.isoformat()))
    return windows

def get_data(from_Year=2000,
            include_this_year=False,
            verbal=False,
            categories=("hep-ph",),
            extra_query="NOT doc_type:conference",
            sort="mostcited",
            N=1000):
    time_windows = year_date_windows(from_Year)
    if include_this_year==False:
        time_windows = time_windows[:-2]

    records = []
    for date_from, date_to in time_windows:

        url = build_inspire_url(
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            extra_query=extra_query,
            sort=sort,
            size=min(N,1000),  
        )
        page = 0
        cur_hits =0 
        # r = requests.get(url, timeout=30)
        # r.raise_for_status()
        # data = r.json()

        # hits = data["hits"]["hits"]
        # records.extend(hits)

        while url and page < 10:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()

            hits = data.get("hits", {}).get("hits", [])
            records.extend(hits)
            url = data.get("links", {}).get("next")
            page += 1
            cur_hits += len(hits)
            if (cur_hits>= N):
                break



        if verbal:
            print(f"Window {date_from} → {date_to} | fetched {cur_hits}")

    rows = []

    for rec in records:
        md = rec.get("metadata", {})

        title = md.get("titles", [{}])[0].get("title")
        abstract = md.get("abstracts", [{}])[0].get("value")
        citation_count = md.get("citation_count")
        earliest_date = md.get("earliest_date")
        journal = md.get("journal_title")

        keywords = [
            k.get("value") for k in md.get("keywords", [])
            if "value" in k
        ]

        categories = md.get("arxiv_eprints", [{}])[0].get("categories", [])

        rows.append({
            "title": title,
            "abstract": abstract,
            "citation_count": citation_count,
            "journal": journal,
            "earliest_date": earliest_date,
            "keywords": keywords,
            "categories": categories,
        })

    df = pd.DataFrame(rows)

    return(df)

        