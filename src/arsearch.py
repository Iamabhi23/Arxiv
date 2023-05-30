# Written By Nathan Rigoni for the Fall Capstone of the DAAN 888

from gensim.models.doc2vec import Doc2Vec
import sys
sys.path.append('../src')
from ParsePDF import PDF_loader
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class ArchiveSearchPapers:
    """
    This is a class for proforming a semantic search on the arxiv papers dataset.
    This repo has 2 models that operate on this class. One is a model that is made
    from the metadata and is modeled on the abastracts of all of the papers as of 10/28/2022.
    The other is a model made aon a subset (~1500) of the actual PDF documents for the CS 
    topic in arxiv.
    """

    def __init__(self, path:str=None): #every class has to have an init
        if path[-1]!='/':
            path = path+'/'
        self.model = None
        self.path = path
        self.load_model()

    def load_model(self):
        """
        This method loads the models from the path that is specified in the init.
        models must have the appropriate name from below in order to be loaded.
        """
        if os.path.isdir(self.path): # check if the path supplied is really a path
            self.model = Doc2Vec.load(self.path + 'archive_model')
            self.files = pd.read_csv(self.path + 'names.csv', dtype=str)
            self.meta = pd.read_csv(self.path + 'cs_meta_data.csv')
            with open(self.path + 'UMAP', 'rb') as f:
                self.umap_model = joblib.load(f)
            with open(self.path + 'clusters', 'rb') as f:
                self.cluster = joblib.load(f)
            self._generate_topic_vectors()

    def _vectorize_string(self, text: str):
        # This internal function operates to take in a string 
        # and convert it to an average aggregated vector for
        # comparing to document vectors in the search
        words = text.lower().split(' ')
        word_vecs = [self.model.wv[word.lower()] for word in words if word in self.model.wv.key_to_index.keys()]
        search_vector = np.mean(word_vecs, axis=0)
        return search_vector

    def _get_similarity(self, vector):
        # This internal method is used to calculate the similarity 
        # of the search vector vs all of the vectors for the documents 
        # in th emodel
        norm_vec = normalize(vector.reshape(1,-1))
        norm_dv = normalize(self.model.dv.vectors)
        similar = np.inner(norm_vec, norm_dv)
        return similar

    def search(self, text:str, top_n:int=5):
        """
        Method provided to take in text string and 
        search using cosine similarity agains the 
        vectors for each document.

        Args:
            text: str - This is the input search string space delimited.

            top_n: int - This is the nuimber of the top results, by cosine 
            similarity, that will be returned to the user
        """
        search_vector = self._vectorize_string(text) #convert the search text to a vector
        sims = self._get_similarity(search_vector)[0] # find the similar documents by vector somparison
        ids = np.flip(np.argsort(sims)) # sort the similar document ids
        papers = self.files.loc[ids[:top_n], 'id'].astype(str).copy()
        results = self.meta.loc[self.meta.loc[:, "id"].astype(str).isin(papers)].copy() # use the top ids to grab the documents to show from the metadata
        results['similarity'] = sims[ids[: results.shape[0]]] # tack on the similarity score to show the user
        return results 

    def _generate_topic_vectors(self):
        # internal method to find the center of mass of each topic cluster
        unique_labels = set(self.cluster.labels_)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.topic_vectors = self._l2_normalize(
            np.vstack([self.model.dv.vectors[np.where(self.cluster.labels_ == label)[0]]
                      .mean(axis=0) for label in unique_labels]))

    def get_topic_wordcloud(self, topic_num):
        """
        This method generates a wordcloud from a selected topic. 
        Words for the topic are selected by their cosine similarity 
        to the topic center of mass vector.

        Args:
            topic_num: int - This is the number of the topic you want to see a word cloud for.
        """
        words, scores = self._find_topic_words_and_scores(self.topic_vectors[topic_num])
        word_score_dict = dict(zip(words, scores))

        fig = plt.figure(figsize=(16, 4),
                   dpi=200)
        plt.axis("off")
        plt.imshow(
            WordCloud(width=1600,
                      height=400,
                      background_color='black').generate_from_frequencies(word_score_dict))
        plt.title("Topic " + str(topic_num), loc='left', fontsize=25, pad=20)
        return fig

    def _find_topic_words_and_scores(self, topic_vectors):
        # internal method for finding the words that are close to the 
        # center of mass for a topic and then returning them to the 
        # wordcloud plot in order to visualize them
        topic_words = []
        topic_word_scores = []

        res = np.inner(topic_vectors, self.model.wv.vectors)
        top_words = np.flip(np.argsort(res))
        top_scores = np.flip(np.sort(res))

        for words, scores in zip(top_words, top_scores):
            topic_words.append(self.model.wv.index_to_key[words])
            topic_word_scores.append(scores)

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores

    @staticmethod
    def _l2_normalize(vectors):
        # internal method used to normalize the vectors for cosine similarity analysis
        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]


class ArchiveSearchAbstracts:
    """
    This is a class for proforming a semantic search on the arxiv papers dataset.
    This repo has 2 models that operate on this class. One is a model that is made
    from the metadata and is modeled on the abastracts of all of the papers as of 10/28/2022.
    The other is a model made aon a subset (~1500) of the actual PDF documents for the CS 
    topic in arxiv.
    """

    def __init__(self, path:str=None): #every class has to have an init
        if path[-1]!='/':
            path = path+'/'
        self.model = None
        self.path = path
        self.load_model()

    def load_model(self):
        """
        This method loads the models from the path that is specified in the init.
        models must have the appropriate name from below in order to be loaded.
        """
        if os.path.isdir(self.path): # check if the path supplied is really a path
            self.model = Doc2Vec.load(self.path + 'archive_model')
            self.meta = pd.read_feather('../data/abstracts_meta.feather')
            with open(self.path + 'UMAP', 'rb') as f:
                self.umap_model = joblib.load(f)
            with open(self.path + 'clusters', 'rb') as f:
                self.cluster = joblib.load(f)
            self._generate_topic_vectors()

    def _vectorize_string(self, text: str):
        # This internal function operates to take in a string 
        # and convert it to an average aggregated vector for
        # comparing to document vectors in the search
        words = text.split(' ')
        word_vecs = [self.model.wv[word] for word in words if word in self.model.wv.key_to_index.keys()]
        search_vector = np.mean(word_vecs, axis=0)
        return search_vector

    def _get_similarity(self, vector):
        # This internal method is used to calculate the similarity 
        # of the search vector vs all of the vectors for the documents 
        # in th emodel
        norm_vec = normalize(vector.reshape(1,-1))
        norm_dv = normalize(self.model.dv.vectors)
        similar = np.inner(norm_vec, norm_dv)
        return similar

    def search(self, text:str, top_n:int=5):
        """
        Method provided to take in text string and 
        search using cosine similarity agains the 
        vectors for each document.

        Args:
            text: str - This is the input search string space delimited.

            top_n: int - This is the nuimber of the top results, by cosine 
            similarity, that will be returned to the user
        """
        search_vector = self._vectorize_string(text) #convert the search text to a vector
        sims = self._get_similarity(search_vector)[0] # find the similar documents by vector somparison
        ids = np.flip(np.argsort(sims)) # sort the similar document ids
        # papers = self.files.loc[ids[:top_n], 'id'].astype(str).copy()
        # results = self.meta.loc[self.meta.loc[:, "id"].astype(str)].copy()
        results = self.meta.loc[ids[: top_n]] # use the top ids to grab the documents to show from the metadata
        results['similarity'] = sims[ids[: top_n]] # tack on the similarity score to show the user
        return results 

    def _generate_topic_vectors(self):
        # internal method to find the center of mass of each topic cluster
        unique_labels = set(self.cluster.labels_)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.topic_vectors = self._l2_normalize(
            np.vstack([self.model.dv.vectors[np.where(self.cluster.labels_ == label)[0]]
                      .mean(axis=0) for label in unique_labels]))

    def get_topic_wordcloud(self, topic_num):
        """
        This method generates a wordcloud from a selected topic. 
        Words for the topic are selected by their cosine similarity 
        to the topic center of mass vector.

        Args:
            topic_num: int - This is the number of the topic you want to see a word cloud for.
        """
        words, scores = self._find_topic_words_and_scores(self.topic_vectors[topic_num])
        word_score_dict = dict(zip(words, scores))

        fig = plt.figure(figsize=(16, 4),
                   dpi=200)
        plt.axis("off")
        plt.imshow(
            WordCloud(width=1600,
                      height=400,
                      background_color='black').generate_from_frequencies(word_score_dict))
        plt.title("Topic " + str(topic_num), loc='left', fontsize=25, pad=20)
        return fig

    def _find_topic_words_and_scores(self, topic_vectors):
        # internal method for finding the words that are close to the 
        # center of mass for a topic and then returning them to the 
        # wordcloud plot in order to visualize them
        topic_words = []
        topic_word_scores = []

        res = np.inner(topic_vectors, self.model.wv.vectors)
        top_words = np.flip(np.argsort(res))
        top_scores = np.flip(np.sort(res))

        for words, scores in zip(top_words, top_scores):
            topic_words.append(self.model.wv.index_to_key[words])
            topic_word_scores.append(scores)

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores

    @staticmethod
    def _l2_normalize(vectors):
        # internal method used to normalize the vectors for cosine similarity analysis
        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]
