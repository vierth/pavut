""" Functions and classes in this file perform stylometric analysis """
import re, os, sys, platform, json
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from pavdhutils.cleaning import clean
from pavdhutils.tokenize import Tokenize
from pavdhutils.rename import c_rename
class Stylometry:
    # The initialization method loads the corpus into memory from a folder
    def __init__(self, corpus_folder="corpus", tokenization_method="char", label_div="_",
                label_types=('title', 'genre', 'era', 'author'),
                color_value=1, label_value = 0,
                file_extension=".txt", encoding="utf8", 
                error_handling="ignore", cleaning=True, verbose=True,
                renamefiles=True):

        # save the metadata label types
        self.label_types = label_types
        # set the index of the metadata caegory to use as the default color pallette
        self.color_value = color_value
        # set the index of the metadata caegory to use as the default labels
        self.label_value = label_value
        # Check if the corpus folder exists
        if not os.path.isdir(corpus_folder):
            print(f"The specificed directory, {corpus_folder} was not found")
            sys.exit()

        # Containers for the data
        self.texts = []
        self.labels = []

        # Iterate through each item in the folder and save 
        for root, dirs, files in os.walk(corpus_folder):
            print(files)
            if renamefiles:
                templabels = c_rename(files)
            else:
                templabels = files
            for i, f in enumerate(files):
                # append the label information
                self.labels.append(templabels[i][:-len(file_extension)].split(label_div))

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
    def vectorize(self, common_terms=500, ngrams=1, use_idf=False,
                        vocab=None):
        vec = TfidfVectorizer(max_features=common_terms, use_idf=use_idf, 
                                analyzer='word', token_pattern='\S+',
                                ngram_range=(ngrams,ngrams), vocabulary=vocab)

        # Create dense matrix
        self.vectors = vec.fit_transform(self.texts).toarray()

        # Get corpus vocabulary
        self.vocab = vec.get_feature_names()

    # function that handles generating colors for the script
    def generate_plot_info(self):
        # generate a color palate for plotting

        # find all the unique values for each of the label types
        self.unique_label_values = [set() for i in range(len(self.label_types))]
        for label_list in self.labels:
            for i, label in enumerate(label_list):
                self.unique_label_values[i].add(label)

        # create color dictionaries for all labels
        self.color_dictionaries = []
        for unique_labels in self.unique_label_values:
            color_palette = sns.color_palette("husl",len(unique_labels)).as_hex()
            self.color_dictionaries.append(dict(zip(unique_labels,color_palette)))

        

    # Do PCA
    def pca(self, n_components=2):
        # instantiate pca object
        pca = PCA(n_components=n_components)
        # fit and transform the data
        self.pca = pca.fit_transform(self.vectors)
        # get the loadings information
        self.loadings = pca.components_
        # get the explained variance
        self.explained_variance = pca.explained_variance_
        # generate plot info
        self.generate_plot_info()

    def pca_to_js(self,jsoutfile="data.js"):
        
        data = []
        for datapoint in self.pca:
            pc_dict = {}
            for i, dp in enumerate(datapoint):
                pc_dict[f"PC{str(i + 1)}"] = dp
            data.append(pc_dict)

            js_loadings = []
            for i, word in enumerate(self.vocab):
                temp_loading = {}
                for j,dp in enumerate(self.loadings):
                    temp_loading[f"PC{str(j+1)}"] = dp[i]
                js_loadings.append([word, temp_loading])

            color_dictionary_list = []
            for cd in self.color_dictionaries:
                cdlist = [v for v in cd.values()]
                color_dictionary_list.append(cdlist)

            color_strings = json.dumps(color_dictionary_list)
            label_strings = json.dumps(self.labels, ensure_ascii=False)
            value_types = json.dumps([k for k in data[0].keys()], ensure_ascii=False)
            data_strings = json.dumps(data, ensure_ascii=False)

            limited_label_types = []
            label_counts = []
            for i, t in enumerate(self.label_types):
                label_counts.append(len(self.unique_label_values[i]))
                if len(self.unique_label_values[i]) <= 20:
                    limited_label_types.append(t)
            if label_counts[self.color_value] > 20:
                print("More than 20 label values, defaulting to minimum item")
                self.color_value = min(range(len(label_counts)), key=label_counts.__getitem__)

            cat_type_strings = json.dumps(limited_label_types, ensure_ascii=False)
            loading_strings = json.dumps(js_loadings, ensure_ascii=False)
            stringlist = [f"var colorDictionaries = {color_strings};", f"var labels = {label_strings};",
                        f"var data = {data_strings};", f"var categoryTypes = {list(self.label_types)};", 
                        f"var loadings = {js_loadings};", f"var valueTypes = {value_types};",
                        f"var limitedCategories = {limited_label_types};",
                        f"var activecatnum = {self.color_value};", f"var activelabelnum = {self.label_value};",
                        f"var explainedvariance = [{round(self.explained_variance[0],3)},{round(self.explained_variance[1],3)}]"]


            with open(jsoutfile, "w", encoding="utf8") as wf:
                wf.write("\n".join(stringlist))


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