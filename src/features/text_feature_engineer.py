import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_bow(corpus, nfeatures=10000, max_df=0.8, min_df=0.2, vect=None):
    """ Convert a collection of text documents to a matrix of token counts

    Args:
        corpus ([type]): [description]
        nfeatures (int, optional): [description]. Defaults to 10000.
        max_df (float, optional): [description]. Defaults to 0.8.
        min_df (float, optional): [description]. Defaults to 0.2.
        as_dataframe (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if vect is None:
        vect = CountVectorizer(max_features=nfeatures, max_df=max_df, min_df=min_df)
    
    count_matrix = vect.fit_transform(corpus)

    return pd.DataFrame(count_matrix.toarray(), columns = vect.get_feature_names()), vect
    



def get_tfidf(corpus, nfeatures=10000, max_df=0.8, min_df=0.2, vect=None):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Args:
        corpus ([type]): [description]
        nfeatures (int, optional): [description]. Defaults to 10000.
        max_df (float, optional): [description]. Defaults to 0.8.
        min_df (float, optional): [description]. Defaults to 0.2.
        as_dataframe (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    if vect is None:
        vect = TfidfVectorizer(max_features=nfeatures, max_df=max_df, min_df=min_df)
    tfidf_matrix = vect.fit_transform(corpus)

    return pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names()), vect



def word_count(text):
    """Return number of words in the text

    Args:
        text (string): Text from where you want to count the number of words 

    Returns:
        Integer: Number of words found
    """
    return len(re.findall(r'\w+',text))