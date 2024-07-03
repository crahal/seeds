import os
import csv
import requests
import numpy as np
from tqdm import tqdm
BASE = 'https://api.openalex.org/'
FILTER = '&filter=title_and_abstract.search:random%20number'

def get_domain_year_counts(filepath):

    csv_file_path = os.path.join(filepath, 'openalex_domain_year_counts.csv')
    with open(csv_file_path, 'w', newline='', encoding='utf-8', errors='replace') as csvfile:
        fieldnames = ['year', 'domain', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for year in tqdm(range(1750, 2025)):
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
        fieldnames = ['year',
                      'count'
                      ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for year in tqdm(range(1750, 2025)):
            api_return = requests.get(BASE+f'works?filter=publication_year:{year}')
            if api_return.status_code == 200:
                writer.writerow({'year': year,
                                 'count': api_return.json()['meta']['count']
                                 }
                                )
            else:
                print(api_return.status_code)
def main():


    filepath = os.path.join(os.getcwd(), '..', 'data', 'openalex_returns')
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    get_domain_year_counts(filepath)
    get_year_counts(filepath)

    FILTER = '&filter=title_and_abstract.search:random%20number'
    fname = 'openalex_rn_papers.csv'
    get_papers(filepath, FILTER, fname)

    FILTER = '&filter=title_and_abstract.search:random%20number,title_and_abstract.search:quantum'
    fname = 'openalex_rn_and_quantum_papers.csv'
    get_papers(filepath, FILTER, fname)

    FILTER = '&filter=title_and_abstract.search:random%20number,title_and_abstract.search:hardware'
    fname = 'openalex_rn_and_hardware_papers.csv'
    get_papers(filepath, FILTER, fname)

    FILTER = '&filter=title_and_abstract.search:random%20number,title_and_abstract.search:pseudo'
    fname = 'openalex_rn_and_pseudo_papers.csv'
    get_papers(filepath, FILTER, fname)


def get_papers(filepath, FILTER, fname):

    with (open(os.path.join(filepath, fname),
              'w',
              newline='',
              encoding='utf-8',
              errors='replace') as csvfile):
        fieldnames = ['doi',
                      'display_name',
                      'publication_year',
                      'pub_date',
                      'language',
                      'journal',
                      'conference',
                      'book',
                      'cited_by_count',
                      'primary_topic',
                      'subfield',
                      'field',
                      'domain',
                      ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        next_cursor='START'
        page = 0
        while next_cursor is not None:
            print(f'Working on page {page+1}')
            if next_cursor == 'START':
                url = BASE + 'works?per-page=100' + FILTER + '&cursor=*'
                api_return = requests.get(url)
                print(f"Getting {api_return.json()['meta']['count']} for {fname}")
            else:
                url = BASE + 'works?per-page=100' + FILTER + '&cursor=' + next_cursor
                api_return = requests.get(url)
            if api_return.json()['meta']["next_cursor"]:
                next_cursor = api_return.json()['meta']["next_cursor"]
            else:
                next_cursor = None

            journal = np.nan
            conference = np.nan
            book = np.nan
            if api_return.status_code == 200:
                for paper in api_return.json()['results']:
                    if paper['display_name'] is not None:
                        paper_name = paper['display_name']
                    else:
                        paper_name = np.nan
                    if paper['publication_year'] is not None:
                        pub_year = paper['publication_year']
                    else:
                        pub_year = np.nan
                    if paper['publication_date'] is not None:
                        pub_date = paper['publication_date']
                    else:
                        pub_date = np.nan
                    if paper['language'] is not None:
                        language = paper['language']
                    else:
                        language = np.nan
                    if paper['cited_by_count'] is not None:
                        cited = paper['cited_by_count']
                    else:
                        cited = np.nan
                    try:
                        if paper['primary_topic'] is not None:
                            if paper['primary_topic']['display_name'] is not None:
                                primary_topic = paper['primary_topic']['display_name']
                            else:
                                primary_topic = np.nan
                            if paper['primary_topic']['subfield'] is not None:
                                subfield = paper['primary_topic']['subfield']['display_name']
                            else:
                                subfield = np.nan
                            if paper['primary_topic']['field'] is not None:
                                field = paper['primary_topic']['field']['display_name']
                            else:
                                field = np.nan
                            if paper['primary_topic']['domain'] is not None:
                                domain = paper['primary_topic']['domain']['display_name']
                            else:
                                domain = np.nan
                    except:
                        print('No primary topic found!')
                        primary_topic = np.nan
                        subfield = np.nan
                        field = np.nan
                        domain = np.nan

                    for loc in paper['locations']:
                        if (loc['source'] is not None):
                            if loc['source']['type'] is not None:
                                if loc['source']['type'] == 'journal':
                                    journal = loc['source']['display_name']
                                elif loc['source']['type'] == 'conference':
                                    conference = loc['source']['display_name']
                                elif loc['source']['type'] == 'book series':
                                    book = loc['source']['display_name']
                    writer.writerow(
                        {
                            'doi': paper['doi'],
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
                        }
                    )
            else:
                print(api_return.status_code)
            page+=1


if __name__ == '__main__':
    main()