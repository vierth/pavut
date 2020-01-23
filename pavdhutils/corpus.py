'''
This file contains functions and classes related to importing corpora into
a script.
'''

import os, re, sys
from pavdhutils.cleaning import clean, toremove
from pavdhutils.tokenize import Tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class Corpus:
    ''' This takes the path to a corpus folder and returns a list of strings
    containing the contents of the internal files. It also contains a metadata
    information about the internal texts.
    '''
    
    def __init__(self, corpus_path, meta_types=None, meta_div="_",
                do_clean=(toremove, True, "~BREAK~"), text_encoding='utf8', 
                error_handling='ignore', extension=".txt", filters=None,
                ignore={'.DS_Store', 'README.md', 'LICENSE'}, load_texts=True):

        self.corpus_path, self.meta_types = corpus_path, meta_types
        self.meta_div, self.do_clean = meta_div, do_clean
        self.text_encoding,self.error_handling = text_encoding, error_handling
        self.extension, self.filters, self.ignore = extension, filters, ignore
        self.load_texts = load_texts

        # collect metadata from the files in the corpus path
        self.parse_metadata()
        # expand the ignore list depending on input filters if desired
        if self.filters:
            self.filter_texts()

            # refilter the file list
            self.parse_metadata()

        # you can optionally not load the texts from file if you just want to
        # inspect the metadata associated with the corpus.
        if self.load_texts:
            self.load_texts_from_file()    

    def load_texts_from_file(self):
        # create a container for the text data:
        self.texts = []

        # iterate through the files and load them into memory
        for root, dirs, files in os.walk(self.corpus_path):

            # filter the files
            files = [f for f in files if f not in self.ignore]
            for i,file_name in enumerate(files):

                with open(os.path.join(root, file_name), 
                            encoding=self.encoding,
                            errors=self.error_handling) as rf:
                    text = rf.read()

                    # clean the texts if specified
                    if self.do_clean:
                        text = clean(text, *self.do_clean)
                    self.texts.append(text)
                

                sys.stdout.write(f"{i + 1} documents of {len(files)} completed\r")
                sys.stdout.flush()
        
    def filter_texts(self):
        ''' Takes an input of files to meta, a dictionary of filters, the types 
        of labels in the order they appear, and an ignore set and outputs a new 
        ignore set, which allows the script to not load files that should not
        be analyzed.
        
        filters should be formated where the keys are the label categories and 
        the values are the acceptable values'''

        # iterate through the metadata dictionary
        for file_name, metadata_list in self.file_meta.items():
            if file_name not in self.ignore:
                # go through the filters and values
                for filt, values in self.filters.items():
                    # get the index location of the label under question
                    location = self.meta_types.index(filt)
                    
                    # If the item is not in the filter list then add to ignore
                    if metadata_list[location] not in values:
                        self.ignore.add(file_name)
                        # break the loop because we don't need to look at the
                        # other items if one of the categories is amiss
                        break

    def parse_metadata(self):
        ''' extract metadata from the corpus filenames '''

        # create containers for the meta data, clear them if they already exist
        self.meta_data, self.file_meta = {}, {}

        for root, dirs, files in os.walk(self.corpus_path):
            # remove files if they are in the ignore set
            files = [f for f in files if f not in self.ignore]
            
            # add the filenames as a type of metadata
            self.meta_data["filenames"] = files

        

            # if there are meta_types provided then add them to the dictionary
            if self.meta_types:
                # add the metatypes as keys in the dictionary
                for meta_type in self.meta_types:
                    self.meta_data[meta_type] = []

                # iterate through all the files in the file_list
                for file_name in files:
                    # extract the meta data from the file name
                    # remove the file extension and split into units
                    meta = file_name[:file_name.rfind(self.extension)].split(self.meta_div)
                    
                    # iterate throught the meta values and append to pertinent list
                    for meta_type, meta_value in zip(self.meta_types, meta):
                        self.meta_data[meta_type].append(meta_value)
                    self.file_meta[file_name] = meta
        
        
    def tokenize(self, method='char', removewhitespace=True, n=1):
        ''' Tokenize the texts into lists of ngrams '''
        self.ngrams = []
        for text in self.texts:
            tokens = Tokenize(text, method=method, 
                                removewhitespace=removewhitespace)
            tokens.ngrams(n=n)
            self.ngrams.append(tokens.get_ngrams_string())

    def vectorize(self, common_terms=500, ngrams=1, use_idf=False, vocab=None):
        ''' Vectorize with the Tfidf algorithm. If use_idf is false, this
        will just use the frequency. '''
        # set up vectorizer
        vec = TfidfVectorizer(max_features=common_terms, use_idf=use_idf, 
                                analyzer='word', token_pattern='\S+',
                                ngram_range=(ngrams,ngrams), vocabulary=vocab)

        # Create dense matrix
        self.vectors = vec.fit_transform(self.texts).toarray()

        # Get corpus vocabulary
        self.vocab = vec.get_feature_names()

    def get_cooccuring_labels(self, limit_labels=None):
        ''' get cooccuring labels, optionally limiting the types of labels '''
        cooccuring_labels = {}

        # capture the indices of the labels we need to use
        use_indices = []
        for meta_type in self.meta_types:

            # if limited labels are specified, capture where they will be in 
            # the meta data lists
            if limit_labels:
                if meta_type in limit_labels:
                    use_indices.append(self.meta_types.index(meta_type))
            # otherwise, just use all of the indices
            else:
                use_indices = [i for i in range(len(self.meta_types))]

        # iterate through the file metadata and capture cooccuring values
        for key, values in self.file_meta.items():
            # limit to the values specified in limit_labels
            use_values = [v for i,v in enumerate(values) if i in use_indices]

            # pair the values (but do not pair a value with itself)
            pairs = [(val_1, val_2) 
                    for val_1 in use_values 
                    for val_2 in use_values
                    if val_1 != val_2
                    and val_1 != "" and val_2 != ""]

            # go through each pair and add it to the labels dictionary
            for pair in pairs:
                if pair[0] in cooccuring_labels:
                    cooccuring_labels[pair[0]].add(pair[1])
                else:
                    cooccuring_labels[pair[0]] = {pair[1]}

        return cooccuring_labels

    def get_label_info(self, exclude=set()):
        ''' get information about the types of labels that exist within a corpus
        folder. this depends on the file names containing relevant metadata. this
        can exclude certain categories that might have too many values (like 
        titles, authors, and filenames, for example).
        '''
       
        # save the unique values
        unique_meta_data = {}

        # go through each type of metadata and return the unique values, excluding
        # the categories that contain too many values
        for key, values in self.meta_data.items():
            if key not in exclude:
                unique_meta_data[key] = set(values)
        return unique_meta_data

    def get_labels(self, label):
        return self.meta_data[label]

    def get_labels_as_integers(self, label):
        labels = self.meta_data[label]
        unique_labels = set(labels)
        label_dictionary = {}
        for i,label in enumerate(list(unique_labels)):
            label_dictionary[label] = i
        integer_label_info = {"keys":label_dictionary}
        integer_list = [label_dictionary[label] for label in labels]
        integer_label_info['labels'] = integer_list
        return integer_label_info
