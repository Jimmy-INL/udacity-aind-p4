import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('inf')
        best_model = None

        n_features = len(self.sequences[0])
        logN = np.log(len(self.X))

        for n_states in range(self.min_n_components, self.max_n_components+1):   
  
            score = float('inf') 
            p = n_states*n_states + (2*n_states*n_features - 1) # number of parameters
 
            hmm = GaussianHMM(n_components=n_states, \
                              covariance_type="diag", \
                              n_iter=1000, \
                              random_state=self.random_state, \
                              verbose=False)
            try:
                model = hmm.fit(self.X, self.lengths)
                score = -2 * model.score(self.X, self.lengths) + p * logN
            except:             
                #print("Fail", self.this_word, n_states) 
                continue 

            #print("Pass", self.this_word, n_states, score) 
            if score < best_score:
                best_score = score
                best_model = model

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None

        for n_states in range(self.min_n_components, self.max_n_components+1):   
  
            score = float('inf') 
 
            hmm = GaussianHMM(n_components=n_states, \
                              covariance_type="diag", \
                              n_iter=1000, \
                              random_state=self.random_state, \
                              verbose=False)
            try:
                model = hmm.fit(self.X, self.lengths)
                likelyhood_this = model.score(self.X, self.lengths)
                likelyhood_others = []
                for w,v in self.hwords.items():
                    if w != self.this_word:
                        X, lengths = v
                        likelyhood_others.append(model.score(X,lengths))
                score = likelyhood_this - sum(likelyhood_others)/(len(self.hwords)-1)
            except:             
                #print("Fail", self.this_word, n_states) 
                continue 

            #print("Pass", self.this_word, n_states, score) 
            if score > best_score:
                best_score = score
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_n_states = None

        split_method = KFold(min(len(self.lengths), 3))

        for n_states in range(self.min_n_components, self.max_n_components+1):   
  
            n_splits = 0
            s_score = 0

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                X_train, len_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, len_test   = combine_sequences(cv_test_idx, self.sequences)

                hmm = GaussianHMM(n_components=n_states, \
                                  covariance_type="diag", \
                                  n_iter=1000, \
                                  random_state=self.random_state, \
                                  verbose=False)
                try:
                    model = hmm.fit(X_train, len_train)
                    s_score += model.score(X_test, len_test)
                    n_splits += 1             
                except:             
                    continue 

            # average score from all splits
            if n_splits > 0:
                score = s_score / n_splits
                if score > best_score:
                    best_score = score
                    best_n_states = n_states

        model = None
        if best_n_states != None: 
            model = self.base_model(best_n_states)

        return model
