import os
import sys
from timeout_decorator import timeout
import re
from pathlib import Path
from typing import List, Union
import pandas as pd
from tqdm_joblib import tqdm_joblib
from google.cloud import bigquery
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
from joblib import Parallel, delayed  # Importing joblib

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
                       names=article_headers,
                       compression='zip'
                       )


def clean_free_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub("[^a-zA-Z]+", " ", s)
    return s


def preprocess_data(df, journal):
    logger.info(f'Length of raw {journal} dataframe: {len(df)}')
    df = df[df['abstract'].notnull() &
            (df['abstract'].str.len() >= 470)].sort_values(by='date_normal',
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
    logger.info(f'Beggining to build model with: {len(df)} rows of data.')
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
        nr_topics: Union[None, str, int] = "auto"):

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
        min_topic_size=min_topic_size,
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
            WHERE journal.issn IN UNNEST(%s)""" % (issn)
    query_job = client.query(QUERY)
    rows = query_job.result()
    return rows


def get_abstracts():
    MY_PROJECT_ID = "dimensionsv3"
    client = bigquery.Client(project=MY_PROJECT_ID)
    data_out = os.path.join('..', 'data', 'bibliometric')


    file_name = 'popstudies_articles.zip'
    popstudies_issn = ['0032-4728']
    popstudies = dim_query(popstudies_issn, client, 'popstudies').to_dataframe()
    popstudies.to_csv(os.path.join(data_out, file_name), mode='w', header=False, compression='zip')

    file_name = 'science_articles.zip'
    science_issn = ['0036-8075']
    science = dim_query(science_issn, client, 'science').to_dataframe()
    science.to_csv(os.path.join(data_out, file_name), mode='w', header=False, compression='zip')

    file_name = 'nature_articles.zip'
    nature_issn = ['0028-0836']
    nature = dim_query(nature_issn, client, 'nature').to_dataframe()
    nature.to_csv(os.path.join(data_out, file_name), mode='w', header=False, compression='zip')

    file_name = 'pnas_articles.zip'
    pnas_issn = ['0027-8424']
    pnas = dim_query(pnas_issn, client, 'pnas').to_dataframe()
    pnas.to_csv(os.path.join(data_out, file_name), mode='w', header=False, compression='zip')

    file_name = 'nejm_articles.zip'
    nejm_issn = ['0028-4793']
    nejm = dim_query(nejm_issn, client, 'nejm').to_dataframe()
    nejm.to_csv(os.path.join(data_out, file_name), mode='w', header=False, compression='zip')

@timeout(5000)
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


def process_state(state, docs, embedding_model, embeddings, journal, meta_path, target_dir):
    try:
        run_bert_with_timeout(docs, embedding_model, embeddings, state, journal, meta_path)
    except:
        bad_seed(journal, state, target_dir)
        pass


def prepare_full_texts(df, columns_for_analysis):
    df["full_text"] = df.apply(
        lambda row: "\n".join(str(row[col]) for col in columns_for_analysis), axis=1
    )
    df["cleaned_full_text"] = df["full_text"].apply(clean_free_text)
    return df


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == 'Get_Abstracts':
        logger.info('Getting abstract data and initializing GBQ')
        get_abstracts()
    else:
        logger.info('Not getting abstracts')

    seed_limit = 1000
    seed_list = get_seed_list()[0:seed_limit]
    seed_list.append(77)
    columns_for_analysis = [
        "1. Summary of the impact",
        "2. Underpinning research",
        "3. References to the research",
        "4. Details of the impact",
        "5. Sources to corroborate the impact",
    ]
    df = pd.read_csv(os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'bibliometric',
                                  'raw',
                                  'enhanced_ref_data.csv')
                    )
    df = df[(df['Main panel'] == 'C') |
            (df['Unit of assessment number'] == 4) |
            (df['Main panel'] == 'D')]

    df = prepare_full_texts(df, columns_for_analysis)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    target_dir = os.path.join(os.getcwd(),
                              '..',
                              'data',
                              'bibliometric',
                              'meta_data'
                              )
    logger.info(f'Working on SHAPE')
    output_path = Path(target_dir) / f"metadata_shape.csv"
    docs = df["cleaned_full_text"].tolist()
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    logger.info(f'Working on SHAPE dataset')
    representation_model = KeyBERTInspired()
    hsdb_model = HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        prediction_data=False,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    with tqdm_joblib(tqdm(desc="Processing States", total=len(seed_list))):
        with Parallel(n_jobs=sys.argv[2]) as parallel:
            parallel(
                delayed(process_state)(
                    state, docs, embedding_model, embeddings, output_path
                ) for state in seed_list
            )
    article_headers = ['id', 'doi', 'journal.issn',
                       'date_normal', 'abstract']
    data_raw = os.path.join(os.getcwd(), '..', 'data', 'bibliometric', 'raw')
    n_neighbors = 10
    for journal in ['popstudies', 'pnas', 'nature', 'nejm', 'science']:
        if journal == 'popstudies':
            min_topic_size = 25
        else:
            min_topic_size = 125
        df = load_file(data_raw, article_headers, journal + '_articles.zip')
        df = preprocess_data(df, journal)
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        target_dir = os.path.join(os.getcwd(),
                                  '..',
                                  'data',
                                  'bibliometric',
                                  'meta_data'
                                  )
        meta_path = Path(target_dir) / f"metadata_{journal}.csv"
        docs = df["cleaned_full_text"].tolist()
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        logger.info(f'Working on {journal} dataset of cleaned abstracts')
        representation_model = KeyBERTInspired()
        hsdb_model = HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            prediction_data=False,
            cluster_selection_method='eom',
        )
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        with tqdm_joblib(tqdm(desc="Processing States", total=len(seed_list))):
            Parallel(n_jobs=sys.argv[2])(
                delayed(process_state)(
                    state, docs, embedding_model, embeddings, journal, meta_path, target_dir
                ) for state in seed_list
            )