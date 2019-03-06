#!/usr/bin/env python
"""Given a set of Etsy shops, determine each shop's top 5 terms.

To set up environment: `pip install -r requirements.txt`

To build documentation: `cd docs; make html`

`./terms.py -h` provides usage information with options.

The full pipeline consists of the following steps:

1. Download data via Etsy API for shops and their listings.

2. Extract a list of candidate phrases from each shop's listings' tags.

3. Exploit structural information -- make use of title and announcement.

4. Perform topic modeling on product descriptions with NMF or LDA.

5. Do a mapping of candidate phrases to topic keywords

Optionally, the --quick option can be passed to omit steps 4 and 5 and simply return the top stemmed and lemmatized stags.
"""

from collections import Counter, OrderedDict, defaultdict
import argparse
import sys
from tabulate import tabulate
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import threading
from time import time, sleep
import random
from tqdm import tqdm, trange
from nltk.tokenize import word_tokenize, sent_tokenize
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import etsy
import nlp_utils

def parse_CLI_args():
    """Parses CLI arguments for various options.
    Returns:
        argparse.Namespace: Variables from CLI argument parsing.
    """
    parser = argparse.ArgumentParser(description='Determine top terms for a set of Etsy shops.')
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--quiet", help="suppress verbose output", action="store_true")
    output_group.add_argument("--debug", help="show extended debugging output", action="store_true")
    stemmer_group = parser.add_mutually_exclusive_group()
    stemmer_group.add_argument("--stemmer", help="stemming algorithm", choices=['porter', 'lancaster'])
    topic_group = parser.add_mutually_exclusive_group()
    topic_group.add_argument("--quick", help="skip topic modeling", action="store_true")
    topic_group.add_argument("--method", help="topic modeling method", choices=['NMF', 'LDA'])
    return parser.parse_args()

def progress_bar_offset():
    """ Prints blank lines to scroll down past progress bars."""
    for i in range(len(shop_names) + 1):
        print()

def download_data(shop_names):
    """Downloads data via Etsy API for shops and their listings.

    Uses multiprocessing to retrieve multiple shop's data at a time.

    Args:
        shop_names (list): Shop names.
    Returns:
        dict: Shop data indexed by shop name

    Todo:
        Cache downloaded data for reprocessing without redownloading.
    """
    t0 = time()
    pool = ThreadPool(N_CONCURRENT_DOWNLOADS)
    lock = threading.Lock() # for use with non-thread-safe tqdm updates
    if verbose:
        print()
        print('--- Downloding Data for Shop Product Listings ---')

    def download_shop_data(shop_name, position, lock):
        """Retrieves data for a specified shop.
        
        Args:
            shop_name (str): Shop name.
            position (int): Position in list of names, for positioning of progress bar.
            lock (_thread.lock): Thead lock for non-thread-safe tqdm progress bar updates.
        Returns:
            dict: Shop data including title, announcement, and product listings.
        """
        sleep(2 * random.random()) # Random 0-2s delay to avoid surpassing Etsy API rate limits
        shop = etsy_client.get_shop(shop_name)['results'][0]
        announcement = shop['announcement']

        # Retrieve shop listings data.
        shop_listings = etsy_client.get_shop_listings(lock, shop_name, position)
        shop_data = {
            'title': shop['title'],
            'announcement': shop['announcement'],
            'listings': shop_listings,
        }
        return shop_data

    data_by_shop = {}
    for i, shop_name in enumerate(shop_names, 1):
        shop_data = pool.apply_async(download_shop_data, args=(shop_name, i, lock))
        data_by_shop[shop_name] = shop_data
    pool.close()
    pool.join()

    for shop_name, async_download in data_by_shop.items():
        data_by_shop[shop_name] = async_download.get()

    if verbose:
        progress_bar_offset()
        print(f'Retrieved all shop data in {int(time() - t0)} seconds.')
        print()
    return data_by_shop


def select_top_tags(shop_data):
    """Extracts a list of candidate phrases from the specified shop's listings' tags.

    * Rank by frequency of occurance across a shop's products

    * Select top 100 tags

    * Perform stemming and lemmatizing

    Args:
        shop_data (list): Shop and listings data.
    Returns:
        list: The list of tuples containing:
            * stemmed and lemmatized top tags,
            * tags sorted by frequency (useful for later secondary ranking).
    """
    title = shop_data['title']
    announcement = shop_data['announcement']
    shop_listings = shop_data['listings']
    # Use listing tags as candidate phrases.
    product_tags = [item['tags'] for item in shop_listings]
    tags = [item for tags in product_tags for item in tags]
    unique_tags = set(tags)
    tags_by_count = Counter(tags)
    # The shop's 100 most common tags will be the starting point for candidate phrases
    top_100_tags = sorted(tags_by_count.items(), key=lambda kv: kv[1], reverse=True)[:100]

    # Perform stemming to reduce redundancy in tags.
    top_tags_by_stem = {}
    # Reverse ordering of top tags so that more common tags will override less common ones with the same stem.
    for tag, count in reversed(top_100_tags):
        # Stem each word in the tag
        stem = nlp_utils.apply_fn_to_tokenized_phrase(tag, stemmer.stem)
        top_tags_by_stem[stem] = tag

    # Perform lemmatizing to further reduce redundancy in tags.
    top_tags_by_lemma = {}
    # Sort stem:tag dict by increasing tag count
    # so that more common tags will override less common ones with the same lemma.
    sorted_top_tags_by_stem = OrderedDict(sorted(top_tags_by_stem.items(), key=lambda t: tags_by_count[t]))
    for stem, tag in sorted_top_tags_by_stem.items():
        # Lemmatize each word in the tag.
        lemma = nlp_utils.apply_fn_to_tokenized_phrase(tag, lemmatizer.lemmatize)
        top_tags_by_lemma[lemma] = tag
    #shop_top_tags[shop_name] = (top_tags_by_lemma, tags_by_count)
    return (top_tags_by_lemma, tags_by_count)

def exploit_structural_info(shop_name_and_tags):
    """Re-ranks a shop's top 100 tags based on:
        * Tag contains a word present in shop announcement AND shop title.
        * Tag contains a word present in shop title.
        * Tag contains a word present in shop announcement.

    Args:
        shop_name_and_tags (tuple):
            * shop name (str) for selecting shop data from data_by_shop.
            * shop tags (tuple) two lists
                - stemmed and lemmatized top tags,
                - tags sorted by frequency (used for secondary ranking).
    Returns:
        list: List of top tags, reordered by ranking criteria.
    """
    global data_by_shop
    shop_name, shop_tags = shop_name_and_tags
    shop_data = data_by_shop[shop_name]
    title = shop_data['title']
    announcement = shop_data['announcement']
    top_tags_by_lemma, tags_by_count = shop_tags
    title_tags = set()
    for lemma, tag in top_tags_by_lemma.items():
        for word in word_tokenize(tag):
            if word.lower() in title.lower():
                title_tags.add(tag)
    announcement_tags = set()
    for lemma, tag in top_tags_by_lemma.items():
        if tag.lower() in announcement.lower():
            announcement_tags.add(tag)
    title_announcement_tags = title_tags.intersection(announcement_tags)

    # Create a sorted (by tag count) list of each set, and concatenate them according to ranking criteria.
    title_tags = sorted(
        list(title_tags.difference(title_announcement_tags)),
        key=lambda t: tags_by_count[t], reverse=True
    )
    announcement_tags = sorted(
        list(announcement_tags.difference(title_announcement_tags)),
        key=lambda t: tags_by_count[t], reverse=True
    )
    title_announcement_tags = sorted(
        list(title_announcement_tags),
        key=lambda t: tags_by_count[t], reverse=True
    )
    ranked_tags = title_announcement_tags + title_tags + announcement_tags

    # As a fail-safe for shops that have blank titles and announcements, append the rest of the tags
    ranked_tags += [tag for tag in top_tags_by_lemma.values() if tag not in ranked_tags]

    return ranked_tags


def perform_topic_modeling(data_by_shop, method='NMF'):
    """ Performs topic modeling on product descriptions using one of two methods:
        Non-negative Matrix Factorization or Latent Dirichlet Allocation.

    With default configuration:
        * Extracts N_TOPICS = N_TOP_TERMS = 5
        * Selects top N_TOP_WORDS = 20 keywords for each topic

    Args:
        data_by_shop (dict): Shop data indexed by shop name
        method (str): Topic modeling method, 'NMF' or 'LDA'
    Returns:
        dict: Lists of shop topics (which are each lists of words), indexed by shop name

    Todo:
        Make use of mutual information across shops.

        Parallelize.
    """
    shop_topics = {}
    if verbose and not debug:
        print()
        print(f'Performing automatic topic modeling from product listings, using {topic_modeling_method} method:')
        pbar = tqdm(total=len(data_by_shop))

    for shop_name, shop_data in data_by_shop.items():
        product_descriptions = [item['description'] for item in shop_data['listings']]

        if method == 'LDA':
            # Use TF vectorizer for LDA
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=N_FEATURES, stop_words='english')
            model = LatentDirichletAllocation(
                n_components=N_TOPICS, max_iter=5, learning_method='online', learning_offset=50., random_state=0
            )
        else:
            # Use TF-IDF vectorizer for NMF
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=N_FEATURES, stop_words='english')
            model = NMF(n_components=N_TOPICS, random_state=1, alpha=.1, l1_ratio=.5)
        feature_vector = vectorizer.fit_transform(product_descriptions)
        model.fit(feature_vector)
        feature_names = vectorizer.get_feature_names()
        topic_top_words = []
        if debug:
            print('-' * 40)
            print(f'Topics for {shop_name}:')
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-N_TOP_WORDS - 1:-1]]
            topic_top_words.append(top_words)
            if debug:
                print(f'Topic #{topic_idx}: ' + ' '.join(top_words))
        shop_topics[shop_name] = topic_top_words
        if verbose and not debug:
            pbar.update()
            pbar.refresh()
    if verbose and not debug:
        pbar.close()
    return shop_topics

def compare_tags_to_topic_terms(shop_name, position, tags, topics, lock):
    """ Does a mapping of candidate phrases (tags) to topic keywords.

    * Scans for matches of tags in order of their ranking.

    * Scans topics for matches of each given tag.

      - Does a serpentine scanning of topics, in order to maintain balance across topics.

      - Identifies tags that match a topic word.

    * Stops when N_TOP_TERMS tags have been identified.

    Args:
        shop_name (str): Shop name.
        position (int): Position in list of names, for positioning of progress bar.
        tags (list): Ranked list of tags for the shop.
        topics (list): List of the shop's topics (which are each lists of words)
        lock (_thread.lock): Thead lock for non-thread-safe tqdm progress bar updates.
    Returns:
        list: Tags that match the shop's topic, padded with tags if less than N_TOP_TERMS.
    """
    topic_indices = list(range(N_TOPICS))
    serpentine_indices = (topic_indices + list(reversed(topic_indices))) * 10
    """ serpentine indices (e.g.: 0,1,2,3,4,4,3,2,1,0...) are used to evenly distribute ranked tags to topics"""

    matching_tags = []
    if verbose and not debug:
        pbar = tqdm(
            total=N_TOP_TERMS,
            desc='{0: <21}'.format(shop_name),
            position=position
        )
    for tag in tags:
        if debug:
            print(f'    Searching {shop_name} topic for {tag}')
        match_found = False
        for topic_idx in serpentine_indices:
            topic = topics[topic_idx]
            for topic_word in topic:
                tag_stem = nlp_utils.apply_fn_to_tokenized_phrase(tag, stemmer.stem)
                for tag_word in word_tokenize(tag_stem):
                    if tag_word == stemmer.stem(topic_word):
                        if tag not in matching_tags:
                            matching_tags.append(tag)
                            match_found = True
                            if verbose and not debug:
                                with lock:
                                    pbar.update()
                                    pbar.refresh()
                        break
                if match_found:
                    break
            if match_found:
                break
        if len(matching_tags) == N_TOP_TERMS:
            break

    # As a fail-safe, if not enough matching tags have been found, append some extra tags
    if len(matching_tags) < N_TOP_TERMS:
        extra_tags = [tag for tag in tags if tag not in matching_tags][:N_TOP_TERMS - len(matching_tags)]
        matching_tags += extra_tags
        if verbose and not debug:
            with lock:
                pbar.update(len(extra_tags))
                pbar.refresh()

    if verbose and not debug:
        with lock:
            pbar.close()
    return matching_tags

if __name__ == '__main__':
    shop_names = [
        'ClassicMetalSigns',
        'QuoteMyWall',
        'BabyLoveUK',
        'LovelyBabyPhotoProps',
        'KennedyClaireCouture',
        'MalizBIJOUX',
        'JewelsByMoonli',
        'FranJohnsonHouse',
        'MadMarchMoon',
        'FerkosFineJewelry',
        'mimetik',
    ]

    N_FEATURES = 1000
    N_TOPICS = 5
    N_TOP_WORDS = 20
    N_TOP_TERMS = 5
    N_CONCURRENT_DOWNLOADS = 5

    # Process CLI arguments
    args = parse_CLI_args()
    debug = args.debug
    verbose = (not args.quiet) or debug
    quick_run = args.quick
    stemming_algorithm = args.stemmer.lower() if args.stemmer else 'porter'
    topic_modeling_method = args.method.upper() if args.method else 'NMF'

    # Set up stemming and lemmatizing
    nlp_utils.update_nltk_resources(debug)
    stemmer = nlp_utils.provision_stemmer(stemming_algorithm, debug)
    lemmatizer = nlp_utils.provision_lemmatizer('wordnet')


    # Download shop/listings data
    etsy_client = etsy.API_client(verbose)
    data_by_shop = download_data(shop_names)
    for shop_name, shop_data in list(data_by_shop.items()):
        if not shop_data['listings']:
            print(f'WARNING: Shop {shop_name} has no active listings -- excluding it from processing.', file=sys.stderr)
            print()
            del data_by_shop[shop_name]

    # Process data
    t0 = time()
    with Pool() as p:
        shop_names = data_by_shop.keys()
        top_tags = p.map(select_top_tags, [data_by_shop[shop_name] for shop_name in shop_names])
        shop_top_tags = dict(zip(shop_names, top_tags))
        ranked_tags = p.map(exploit_structural_info, shop_top_tags.items())
        shop_ranked_tags = dict(zip(shop_names, ranked_tags))

    if quick_run:
        top_terms = { shop_name: shop_ranked_tags[shop_name][:5] for shop_name in shop_ranked_tags }
    else:
        shop_topics = perform_topic_modeling(data_by_shop, method=topic_modeling_method)
        if verbose:
            print()
            print(f'Matching product listing tags against words selected from topic modeling:')

        pool = ThreadPool(N_CONCURRENT_DOWNLOADS)
        lock = threading.Lock() # for use with non-thread-safe tqdm updates
        shop_matching_tags = {}
        for i, shop_name in enumerate(shop_topics, 1):
            tags = shop_ranked_tags[shop_name]
            topics = shop_topics[shop_name]
            input_args = (shop_name, i, tags, topics, lock)
            shop_matching_tags[shop_name] = pool.apply_async(compare_tags_to_topic_terms, args=input_args)
        pool.close()
        pool.join()

        for shop_name, async_matches in shop_matching_tags.items():
            shop_matching_tags[shop_name] = async_matches.get()

        top_terms = { shop_name: shop_matching_tags[shop_name] for shop_name in shop_ranked_tags }

        if verbose and not debug:
            progress_bar_offset()

    # Print output to terminal
    if verbose:
        print()
        print(f'Processing complete in {int(time() - t0)} seconds.')
        print()
    print(tabulate(top_terms.items(), headers=['Shop', 'Top Terms']))

