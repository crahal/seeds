import os
import csv
import requests
import numpy as np
from tqdm import tqdm
from collections import Counter

BASE = 'https://api.openalex.org/'
FILTER = '&filter=title_and_abstract.search:random%20number'

# ---------- helpers (no plotting) ----------
def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _nan_to_num_list(vals):
    out = []
    for v in vals:
        try:
            vv = float(v)
        except Exception:
            vv = np.nan
        out.append(vv)
    return out

def _top_k(counter, k=10):
    return counter.most_common(k)

def summarize_corpus(filepath, fname, topk=10):
    """
    Read the just-written CSV and print descriptive statistics.
    Does not modify files or alter scraping behavior.
    """
    path = os.path.join(filepath, fname)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"[summary] File not found or empty: {path}")
        return

    n = 0
    dois = []
    years = []
    cites = []
    langs = Counter()
    venues = Counter()
    prim_topics = Counter()
    domains = Counter()

    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            doi = (row.get('doi') or '').strip()
            dois.append(doi if doi else None)

            y = _safe_int(row.get('publication_year'))
            if y is not None:
                years.append(y)

            c = row.get('cited_by_count')
            cites.append(c)

            lang = (row.get('language') or '').strip()
            if lang:
                langs[lang] += 1

            # venue: prefer journal, else conference, else book
            v = (row.get('journal') or '').strip() or (row.get('conference') or '').strip() or (row.get('book') or '').strip()
            if v:
                venues[v] += 1

            pt = (row.get('primary_topic') or '').strip()
            if pt:
                prim_topics[pt] += 1

            dom = (row.get('domain') or '').strip()
            if dom:
                domains[dom] += 1

    # compute numerics
    cites = _nan_to_num_list(cites)
    cites_arr = np.array(cites, dtype=float)
    cites_mean = float(np.nanmean(cites_arr)) if np.isnan(cites_arr).sum() < len(cites_arr) else float('nan')
    cites_median = float(np.nanmedian(cites_arr)) if np.isnan(cites_arr).sum() < len(cites_arr) else float('nan')

    years_min = min(years) if years else None
    years_max = max(years) if years else None
    doi_missing = sum(1 for d in dois if not d)
    doi_missing_prop = doi_missing / n if n else 0.0

    print(f"\n[summary] {fname}")
    print(f"  Records: {n:,}")
    if years_min is not None and years_max is not None:
        print(f"  Year span: {years_min}–{years_max}")
    print(f"  Missing DOIs: {doi_missing:,} ({doi_missing_prop:.1%})")
    if not np.isnan(cites_mean):
        print(f"  Citations: mean = {cites_mean:.2f}, median = {cites_median:.0f}")
    if venues:
        print("  Top venues:")
        for v,c in _top_k(venues, topk):
            print(f"    - {v}: {c}")
    if langs:
        print("  Top languages:")
        for l,c in _top_k(langs, min(topk,5)):
            print(f"    - {l}: {c}")
    if prim_topics:
        print("  Top primary topics:")
        for t,c in _top_k(prim_topics, topk):
            print(f"    - {t}: {c}")
    if domains:
        print("  Top domains:")
        for d,c in _top_k(domains, topk):
            print(f"    - {d}: {c}")

def summarize_year_counts(filepath):
    path = os.path.join(filepath, 'openalex_year_counts.csv')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    years = []
    counts = []
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = _safe_int(row.get('year'))
            c = _safe_int(row.get('count'))
            if y is not None and c is not None:
                years.append(y); counts.append(c)
    if years:
        print("\n[summary] Global year counts")
        print(f"  Range: {min(years)}–{max(years)}; most recent year {max(years)} total = {counts[-1]:,}")

def summarize_domain_counts(filepath):
    path = os.path.join(filepath, 'openalex_domain_counts.csv')
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    dom_counts = []
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = _safe_int(row.get('domain'))
            c = _safe_int(row.get('count'))
            if d is not None and c is not None:
                dom_counts.append((d,c))
    if dom_counts:
        d, c = max(dom_counts, key=lambda t: t[1])
        print("\n[summary] Domain totals")
        print(f"  Largest domain by total works: Domain {d} with {c:,} works")

# ---------- scraping functions (unchanged queries) ----------
def get_domain_counts(filepath):
    csv_file_path = os.path.join(filepath, 'openalex_domain_counts.csv')
    with open(csv_file_path, 'w', newline='', encoding='utf-8', errors='replace') as csvfile:
        fieldnames = ['domain', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for domain in range(1, 5):
            try:
                api_return = requests.get(
                    f'https://api.openalex.org/works?filter=topics.domain.id:{domain}'
                )
                api_return.raise_for_status()
                count = api_return.json()['meta']['count']
                writer.writerow({'domain': domain, 'count': count})
                csvfile.flush()
            except requests.exceptions.RequestException as e:
                print(f"Request failed for domain {domain}: {e}")
            except KeyError:
                print(f"Unexpected response structure for domain {domain}")

def get_domain_year_counts(filepath):
    csv_file_path = os.path.join(filepath, 'openalex_domain_year_counts.csv')
    with open(csv_file_path, 'w', newline='', encoding='utf-8', errors='replace') as csvfile:
        fieldnames = ['year', 'domain', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for year in tqdm(range(1750, 2026)):
            for domain in range(1, 5):
                try:
                    api_return = requests.get(
                        f'https://api.openalex.org/works?filter=publication_year:{year},topics.domain.id:{domain}')
                    api_return.raise_for_status()
                    count = api_return.json()['meta']['count']
                    writer.writerow({'year': year, 'domain': domain, 'count': count})
                    csvfile.flush()
                except requests.exceptions.RequestException as e:
                    print(f"Request failed for year {year}, domain {domain}: {e}")
                except KeyError:
                    print(f"Unexpected response structure for year {year}, domain {domain}")

def get_year_counts(filepath):
    with open(os.path.join(filepath, 'openalex_year_counts.csv'),
              'w',
              newline='',
              encoding='utf-8',
              errors='replace') as csvfile:
        fieldnames = ['year', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for year in tqdm(range(1750, 2026)):
            api_return = requests.get(BASE + f'works?filter=publication_year:{year}')
            if api_return.status_code == 200:
                writer.writerow({'year': year,
                                 'count': api_return.json()['meta']['count']})
            else:
                print(api_return.status_code)

def get_papers(filepath, FILTER, fname):
    out_path = os.path.join(filepath, fname)
    with open(out_path, 'w', newline='', encoding='utf-8', errors='replace') as csvfile:
        fieldnames = ['doi', 'display_name', 'publication_year', 'pub_date', 'language',
                      'journal', 'conference', 'book', 'cited_by_count',
                      'primary_topic', 'subfield', 'field', 'domain']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        next_cursor = 'START'
        page = 0
        while next_cursor is not None:
            print(f'Working on page {page + 1}')
            if next_cursor == 'START':
                url = BASE + 'works?per-page=100' + FILTER + '&cursor=*'
                api_return = requests.get(url)
                try:
                    print(f"Getting {api_return.json()['meta']['count']} for {fname}")
                except Exception:
                    pass
            else:
                url = BASE + 'works?per-page=100' + FILTER + '&cursor=' + next_cursor
                api_return = requests.get(url)

            meta = api_return.json().get('meta', {})
            next_cursor = meta.get('next_cursor', None)

            if api_return.status_code == 200:
                for paper in api_return.json().get('results', []):
                    # Reset venue placeholders per paper (prevents carry-over)
                    journal = np.nan
                    conference = np.nan
                    book = np.nan

                    paper_name = paper['display_name'] if paper.get('display_name') is not None else np.nan
                    pub_year = paper['publication_year'] if paper.get('publication_year') is not None else np.nan
                    pub_date = paper['publication_date'] if paper.get('publication_date') is not None else np.nan
                    language = paper['language'] if paper.get('language') is not None else np.nan
                    cited = paper['cited_by_count'] if paper.get('cited_by_count') is not None else np.nan

                    try:
                        if paper.get('primary_topic') is not None:
                            pt = paper['primary_topic']
                            primary_topic = pt.get('display_name') if pt.get('display_name') is not None else np.nan
                            subfield = pt['subfield']['display_name'] if pt.get('subfield') is not None else np.nan
                            field = pt['field']['display_name'] if pt.get('field') is not None else np.nan
                            domain = pt['domain']['display_name'] if pt.get('domain') is not None else np.nan
                        else:
                            primary_topic = subfield = field = domain = np.nan
                    except Exception:
                        print('No primary topic found!')
                        primary_topic = subfield = field = domain = np.nan

                    for loc in paper.get('locations', []):
                        if loc.get('source') and loc['source'].get('type'):
                            if loc['source']['type'] == 'journal':
                                journal = loc['source'].get('display_name', journal)
                            elif loc['source']['type'] == 'conference':
                                conference = loc['source'].get('display_name', conference)
                            elif loc['source']['type'] == 'book series':
                                book = loc['source'].get('display_name', book)

                    writer.writerow({
                        'doi': paper.get('doi'),
                        'display_name': paper_name,
                        'publication_year': pub_year,
                        'pub_date': pub_date,
                        'language': language,
                        'journal': journal,
                        'conference': conference,
                        'book': book,
                        'cited_by_count': cited,
                        'primary_topic': primary_topic,
                        'subfield': subfield,
                        'field': field,
                        'domain': domain,
                    })
            else:
                print(api_return.status_code)
            page += 1

# ---------- main orchestration (all your filters preserved) ----------
def main():
    filepath = os.path.join(os.getcwd(), '..', 'data', 'openalex_returns')
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Baselines
    get_domain_counts(filepath)
    get_domain_year_counts(filepath)
    get_year_counts(filepath)
    summarize_year_counts(filepath)
    summarize_domain_counts(filepath)

    # Corpora
    FILTER = '&filter=title_and_abstract.search:"random%20number"'
    fname = 'openalex_rn_papers.csv'
    get_papers(filepath, FILTER, fname)
    summarize_corpus(filepath, fname)

    FILTER = '&filter=title_and_abstract.search:"random%20number",title_and_abstract.search:quantum'
    fname = 'openalex_rn_and_quantum_papers.csv'
    get_papers(filepath, FILTER, fname)
    summarize_corpus(filepath, fname)

    FILTER = '&filter=title_and_abstract.search:"random%20number",title_and_abstract.search:hardware'
    fname = 'openalex_rn_and_hardware_papers.csv'
    get_papers(filepath, FILTER, fname)
    summarize_corpus(filepath, fname)

    FILTER = '&filter=title_and_abstract.search:"random%20number",title_and_abstract.search:pseudo'
    fname = 'openalex_rn_and_pseudo_papers.csv'
    get_papers(filepath, FILTER, fname)
    summarize_corpus(filepath, fname)

    FILTER = '&filter=title_and_abstract.search:"random%20number",title_and_abstract.search:quasi'
    fname = 'openalex_rn_and_quasi_papers.csv'
    get_papers(filepath, FILTER, fname)
    summarize_corpus(filepath, fname)

if __name__ == '__main__':
    main()
