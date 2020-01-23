"""Classes in this file tokenize input strings"""

from pavdhutils.errors import CustomException
import sys

class Tokenize:
    def __init__(self, text, method="char", lang="zh", regex=r"\w+", 
                removewhitespace=True):
        self.methods = ["char", "word", "regex"]
        self.mstring = ", ".join(self.methods[:-1]) + f" or {self.methods[-1]}"
        self.languages = ["zh", "en"]

        # Check to see if specified methods is available
        if method not in self.methods:
            raise CustomException(f"{method} is not a valid option. Try {self.mstring}")
        
        # if character tokenization, turn into list
        if method == "char":
            # remove whitespace if desired
            if removewhitespace:
                self.tokens_ = list(text.replace(" ", ""))
            else:
                self.tokens_ = list(text)

        # elif method == "word":
        #     if lang == "zh":
        #         print("No word tokenization yet")
        #         sys.exit()
        #     elif lang == "en":

    def ngrams(self, n=1, gram_div=""):
        self.ngrams_ = [gram_div.join(self.tokens_[i:i+n]) for i in 
                        range(len(self.tokens_)-(n-1))]

    def get_tokens(self):
        return self.tokens_

    def get_ngrams(self):
        return self.ngrams_

    def get_ngrams_string(self, div=" "):
        return div.join(self.ngrams_)

    def get_tokenized(self, version="tokens"):
        if version == "tokens":
            return " ".join(self.tokens_)
        elif version == "ngrams":
            try:
                return " ".join(self.ngrams_)
            except AttributeError:
                print("No ngrams found yet, returning one grams by default")
                self.ngrams(2)
                return " ".join(self.ngrams_)

    def chineseTokenize(text):
        pass

    def englishTokenize(text):
        pass