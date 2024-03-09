
import os
import sys
from timeout_decorator import timeout
import re
from pathlib import Path
from typing import List, Union
from google.cloud import bigquery
import pandas as pd
import traceback
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from loguru import logger
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

try:
    from cuml.cluster import HDBSCAN
    logger.info("Using cuml's implementation of HDBSCAN")
except ImportError:
    from hdbscan import HDBSCAN
    logger.info("Using regular implementation of HDBSCAN")
try:
    from cuml.manifold import UMAP
    logger.info("Using cuml's implementation of UMAP")
except ImportError:
    from umap import UMAP
    logger.info("Using regular implementation of UMAP")

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def load_file(data_raw, article_headers, filename):
    return pd.read_csv(os.path.join(data_raw, filename),
                       usecols=[0, 1, 2, 3, 7],
                       names=article_headers)


def clean_free_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub("[^a-zA-Z]+", " ", s)
    return s


def preprocess_data(df, journal):
    logger.info(f'Length of raw {journal} dataframe: {len(df)}')
    df = df[df['abstract'].notnull() &
            (df['abstract'].str.len() >= 500)].sort_values(by='date_normal',
                                                           ascending=False)
    df["cleaned_full_text"] = df["abstract"].apply(clean_free_text)
    logger.info(f'Length of processed {journal} dataframe: {len(df)}')
    df.to_csv(os.path.join(os.getcwd(),
                           '..',
                           'data',
                           'bibliometric',
                           'processed',
                           journal+'processed.csv')
              )
    return df


def get_seed_list():
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


def calculate_silhouette_score(topic_model, embeddings, topics):
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for topic in topics if topic != -1]
    return silhouette_score(X, labels)


def run_bert(
        docs: List[str],
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
        random_state: int,
        journal: str,
        path_metadata_csv: Union[str, Path],
        min_topic_size: int = 25,
        nr_topics: Union[None, str, int] = "auto",
):

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=10,
        init="random",
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )

    topic_model = BERTopic(
        language="english",
        verbose=False,
        calculate_probabilities=False,
        n_gram_range=(1, 2),
        min_topic_size = min_topic_size,
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hsdb_model,
        ctfidf_model=ctfidf_model,
        nr_topics=nr_topics,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    topics_counter = Counter(topics)
    outliers_count = topics_counter.get(-1, 0)
    topics_count = (
        len(topics_counter) - 1 if outliers_count > 0 else len(topics_counter)
    )
    silhouette = calculate_silhouette_score(
        topic_model, embeddings, topic_model.topics_
    )
    metadata = {
        "journal": journal,
        "random_state": random_state,
        "n_neighbors": n_neighbors,
        "nr_topics": nr_topics,
        "topics_count": topics_count,
        "outliers_count": outliers_count,
        "silhouette_score": silhouette
    }
    logger.info(metadata)
    pd.DataFrame([metadata]).to_csv(
        path_metadata_csv,
        mode="a",
        header=not path_metadata_csv.exists(),
        index=False
    )


def dim_query(issn, client, journal):
    print('Getting articles for :', journal)
    QUERY = """
            SELECT id, doi, date_normal, category_for, journal,
            title.preferred, abstract.preferred, concepts
            FROM `dimensions-ai.data_analytics.publications`
            WHERE journal.issn IN UNNEST(%s)""" %(issn)
    query_job = client.query(QUERY)
    rows = query_job.result()
    return rows


def get_abstracts():
    MY_PROJECT_ID = "dimensionsv3"
    client = bigquery.Client(project=MY_PROJECT_ID)
    data_out = os.path.join('..', 'data', 'bibliometric')

    file_name = 'science_articles.csv'
    science_issn = ['0036-8075']
    science = dim_query(science_issn, client, 'science').to_dataframe()
    science.to_csv(os.path.join(data_out, file_name), mode='w', header=False)

    file_name = 'nature_articles.csv'
    nature_issn = ['0028-0836']
    nature = dim_query(nature_issn, client, 'nature').to_dataframe()
    nature.to_csv(os.path.join(data_out, file_name), mode='w', header=False)

    file_name = 'pnas_articles.csv'
    pnas_issn = ['0027-8424']
    pnas = dim_query(pnas_issn, client, 'pnas').to_dataframe()
    pnas.to_csv(os.path.join(data_out, file_name), mode='w', header=False)

    file_name = 'nejm_articles.csv'
    nejm_issn = ['0028-4793']
    nejm = dim_query(nejm_issn, client, 'nejm').to_dataframe()
    nejm.to_csv(os.path.join(data_out, file_name), mode='w', header=False)


@timeout(500)
def run_bert_with_timeout(docs, embedding_model, embeddings, state, name, output_path):
    run_bert(docs, embedding_model, embeddings, state, name, output_path)


def bad_seed(journal, random_state, target_dir):
    logger.info(f'Bad seed found: {random_state}!')
    logger.info(traceback.print_exc())
    path_metadata_csv = Path(target_dir) / f"metadata_{journal}.csv"
    metadata = {
        "journal": journal,
        "random_state": random_state,
        "n_neighbors": n_neighbors,
        "nr_topics": np.nan,
        "topics_count": np.nan,
        "outliers_count": np.nan,
        "silhouette_score": np.nan
    }
    pd.DataFrame([metadata]).to_csv(
        path_metadata_csv,
        mode="a",
        header=not path_metadata_csv.exists(),
        index=False
    )

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == 'Get_Abstracts':
        logger.info('Getting abstract data and initializing GBQ')
        get_abstracts()
    else:
        logger.info('Not getting abstracts')
    if len(sys.argv) < 3:
        print("Need information on which journal to model!")
        sys.exit()
    else:
        print(f'Begining to work on {sys.argv[2]}!')

    article_headers = ['id', 'doi', 'journal.issn',
                       'date_normal', 'abstract']
    data_raw = os.path.join(os.getcwd(), '..', 'data', 'bibliometric', 'raw')
    df = load_file(data_raw, article_headers, sys.argv[2] + '_articles.csv')
    df = preprocess_data(df, sys.argv[2])
    seed_limit = 1000
    n_neighbors = 10
    seed_list = get_seed_list()[:seed_limit]
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    target_dir = os.path.join(os.getcwd(),
                              '..',
                              'data',
                              'bibliometric',
                              'meta_data'
                              )
    meta_path = Path(target_dir) / f"metadata_{sys.argv[2]}.csv"
    if os.path.exists(meta_path):
        os.remove(meta_path)
    docs = df["cleaned_full_text"].tolist()
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    logger.info(f'Working on {sys.argv[2]} dataset of cleaned abstracts')
    representation_model = KeyBERTInspired()
    hsdb_model = HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        prediction_data=False,
        cluster_selection_method='eom',
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    for state in tqdm(seed_list):
        try:
            run_bert_with_timeout(docs,
                                  embedding_model,
                                  embeddings,
                                  state,
                                  sys.argv[2],
                                  meta_path
                                  )
        except:
            bad_seed(sys.argv[2], state, target_dir)
            pass