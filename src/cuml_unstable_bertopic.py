from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import os
import umap
from loguru import logger
from cuml.manifold import UMAP as CUMLUMAP
from collections import Counter


def main():
    logger.info('Loading dataset...')
    alltweets = fetch_20newsgroups(subset='all', remove=('headers',
                                                         'footers',
                                                         'quotes'))['data']
    my_stopwords = list(["rt",
                         "RT",
                         "&",
                         "amp",
                         "&amp",
                         "http",
                         "https",
                         "http://",
                         "https://",
                         "fav",
                         "FAV"])
    logger.info('Initialising vectorizer...')
    vectorizer_model = CountVectorizer(ngram_range=(1, 2),
                                       stop_words=my_stopwords)
    min_clusters = round(len(alltweets) * 0.0017)
    logger.info('Initialising hdbscan...')
    hdbscan_model = HDBSCAN(min_cluster_size=min_clusters,
                            metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True,
                            min_samples=5)
    logger.info('Initialising sentence model...')
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    logger.info('Calculating embeddings...')
    embeddings = sentence_model.encode(alltweets)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    spectral_flag = 'Dont use me!'
    logger.info('Beginning iteration...')
    for iteration in range(0, 100):
        if spectral_flag == 'Dont use me!':
            logger.info('Using init="spectral"')
            umap_model = CUMLUMAP(n_neighbors=15,
                                  n_components=5,
                                  min_dist=0.0,
                                  metric='cosine',
                                  init="spectral",
                                  random_state=42
                                  )
        elif spectral_flag == 'Use me!':
            logger.info('Using init="random"')
            umap_model = CUMLUMAP(n_neighbors=15,
                                  n_components=5,
                                  min_dist=0.0,
                                  init="random",
                                  metric='cosine',
                                  random_state=42
                                  )

        topic_model = BERTopic(nr_topics='auto', umap_model=umap_model,
                               hdbscan_model=hdbscan_model,
                               embedding_model=sentence_model,
                               vectorizer_model=vectorizer_model,
                               low_memory=True,
                               calculate_probabilities=True)
        topics, probs = topic_model.fit_transform(alltweets, embeddings)
        topics_counter = Counter(topics)
        outliers_count = topics_counter.get(-1, 0)
        topics_count = (
            len(topics_counter) - 1 if outliers_count > 0 else len(topics_counter)
        )
        logger.info(f"Run {iteration}: Found {topics_count} topics and {outliers_count} outliers")
        logger.info(f"Run {iteration}: Hash of str of probs is {hash(str(probs))}")
        if spectral_flag == 'Dont use me!':
            spectral_flag = 'Use me!'
        elif spectral_flag == 'Use me!':
            spectral_flag = 'Dont use me!'


if __name__ == "__main__":
    main()