""" Functions and classes in this file perform stylometric analysis """
import re, os, sys, platform, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from cleaning import clean
from tokenize import Tokenize

class Stylometry:
    # The initialization method loads the corpus into memory from a folder
    def __init__(self, corpus_folder, tokenization_method="char", label_div="_",
                label_types=('title', 'genre', 'era', 'author'), 
                file_extension_length=4, encoding="utf8", 
                error_handling="ignore", cleaning=True):

        self.label_types = label_types

        # Check if the corpus folder exists
        if not os.path.isdir(corpus_folder):
            print(f"The specificed directory, {corpus_folder} was not found")
            sys.exit()

        # Containers for the data
        self.texts = []
        self.labels = []

        # Iterate through each item in the folder and save 
        for root, dirs, files in os.walk(corpus_folder):
            for i, f in enumerate(files):
                
                # append the label information
                self.labels.append(f[:file_extension_length].split(label_div))

                # Open the file and save the text
                with open(os.path.join(root,f), "r", encoding=encoding, 
                    errors=error_handling) as rf:
                    text = rf.read()
                    if cleaning:
                        text = clean(text)
                    tokens = Tokenize(text)
                    text = tokens.get_tokenized()
                    self.texts.append(text)

    # Use sklearn's vectorizor to create a bag of words vectorizer. other 
    # options will be available in the future
    def bow_vectorize(self, common_terms=1000, ngrams=1, use_idf=False,
                        vocab=None):
        vec = TfidfVectorizer(max_features=common_terms, use_idf=use_idf, 
                                analyzer='word', token_pattern='\S+',
                                ngram_range=(ngrams,ngrams), vocabulary=vocab)

        self.vectors = vec.fit_transform(self.texts).toarray()

        self.vocab = vec.get_feature_names()

    def pca(self, n_components=2):
        pca = PCA(n_components=n_components)
        self.pca = pca.fit_transform(self.vectors)
        self.loadings = pca.components_

    def plot_pca(self, output_dim=(10,7), output_name=None, color_value="genre",
                    label_value="title"):

        color_label_index = self.label_types.index(color_value)
        
        # Set the font
        if platform.system() == "Darwin":
            font = matplotlib.font_manager.FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
            matplotlib.rcParams['pdf.fonttype'] = 42
        elif platform.system() == "Windows":
            font = matplotlib.font_manager.FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc")
        elif platform.system() == "Linux":
            # This assumes you have wqy zenhei installed
            font = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

        # create figure
        plt.figure(figsize=output_dim)

        # get unique values to generate a color pallette
        unique_label_values = set()
        for label_list in self.labels:
            unique_label_values.add(label_list[color_label_index])

        # make this a list to fix the order
        unique_label_values = list(unique_label_values)

        # create a color dicitonary for the labels with seaborn
        color_dictionary = dict(zip(unique_label_values,
                                sns.color_pallette("husl", 
                                len(unique_label_values)).as_hex()))

        # get integers for the labels to allow for numpy filtering
        label_integer = [i for i in range(len(unique_label_values))]

        label_to_integer = dict(zip(unique_label_values, label_integer))

        texts_with_integer_label = np.array(label_to_integer[label[color_label_index]] for label in self.labels)

        colors = [color_dictionary[label] for label in unique_label_values]