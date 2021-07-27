#!/usr/bin/env python
# coding: utf-8


### THIS SCRIPT IS BASED ON THE WORK BY SCHWABL (see reference in TOIS paper)

#get_ipython().system('pip install fasttext')
get_ipython().system('pip install keras')
get_ipython().system('pip install -U imbalanced-learn')


#import fasttext
#import fasttext.util
import pandas as pd
import numpy as np
import re
import json
import math
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import h5py
import warnings
#import fasttext.util
from statistics import mean
import os
import torch
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras
from keras import backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.layers import *
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D

from IPython.display import display, Markdown
import nltk
from nltk.corpus import stopwords
from imblearn.combine import SMOTEENN
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings, BytePairEmbeddings, StackedEmbeddings






def clean_str(in_str):
    in_str = str(in_str)
    # replace urls with 'url'
    in_str = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "url", in_str)
    in_str = re.sub(r'([^\s\w]|_)+', '', in_str)
    return in_str.strip().lower()


def convert_to_embeddings(data, doc_embeddings):
    we_corpus = np.empty(shape=[0, 400])

    for turn in data:
        current_turn = Sentence(turn)
        doc_embeddings.embed(current_turn)
        current_turn_embedding =current_turn.embedding.cpu().numpy()
        we_corpus = np.append(we_corpus, [current_turn_embedding], axis=0)

    print(f"Word Embedding corpus has length {len(we_corpus)}")
    print(f"Sample one has length {len(we_corpus[1])}")
    print(f"Sample two has length {len(we_corpus[10])}")
    return we_corpus
    


def remove_stop_words(utterance):
    stop_words = list(stopwords.words('german'))
    splitted_utterance = utterance.split()
    cleaned_utterance = [word for word in splitted_utterance if word not in stop_words]
    return " ".join(cleaned_utterance)


def perform_text_classification(data,
                                info_need,
                                condition,
                                resample = False,
                                remove_stopwords = False,
                                classifier = "SVM", 
                                verbose = 1, 
                                data_source = "cooking", 
                                random_state = 42,
                                warning_on_off = "off"):
    
    
    # =====================================================================
    # ================= data preperation and other things =================
    # =====================================================================
    nltk.download('stopwords')


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # use the third GPU
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")



    fast_text_embeddings = StackedEmbeddings(
        [
            WordEmbeddings('de'),
            BytePairEmbeddings('de')
        ])
    document_embeddings = DocumentPoolEmbeddings([fast_text_embeddings])

    if remove_stopwords:
        print("Remove stopwords...")
        data['utterance'] = data['utterance'].apply(remove_stop_words)
    
    label_encoder = LabelEncoder()
    y = data['level_1']
    

    # Control warnings
    if warning_on_off == "off":
        warnings.filterwarnings("ignore")

    elif warning_on_off == "on":
        warnings.simplefilter('always')
    
    
    
    # =================================================
    # ================= featurization =================
    # =================================================
    
    if verbose == 1:
        print("=== Starting embedding process...")
    word_embedding_corpus = convert_to_embeddings(data['utterance'], document_embeddings)
    
    
    

    # =======================================================            
    # ================= building classfiers =================
    # ======================================================= 
    scoring = {'f1_infoneed': make_scorer(f1_score, pos_label=info_need),
               'f1_other': make_scorer(f1_score, pos_label="Other"),
               'recall_infoneed': make_scorer(recall_score, pos_label=info_need),
               'precision_infoneed': make_scorer(precision_score, pos_label=info_need),
               'recall_other': make_scorer(recall_score, pos_label="Other"),
               'precision_other': make_scorer(precision_score, pos_label="Other"),
               'f1_micro':'f1_micro',
               'f1_macro':'f1_macro',
               'f1_weighted':'f1_weighted',
               'precision_micro': 'precision_micro', 
               'precision_macro': 'precision_macro', 
               'precision_weighted': 'precision_weighted',
               'recall_micro': 'recall_micro', 
               'recall_macro': 'recall_macro', 
               'recall_weighted' :'recall_weighted'
              }
    
    if resample:
        print("Resample data...")
        smote_enn = SMOTEENN(random_state=42)
        word_embedding_corpus, y = smote_enn.fit_resample(word_embedding_corpus, y)
        print(sorted(Counter(y).items()))
        
    # ================================ NAIVE BAYES =====================================
    if classifier == "NB":

        if verbose == 1:
            print("=== Building Classifier:", classifier)
        clf = GaussianNB()
        clf_scores = cross_validate(clf, word_embedding_corpus, y, cv=10, scoring=scoring)
        print(f"Classifier: NB -- Condition: {condition} -- Info Need: {info_need} -- F1 Average: {mean(clf_scores['test_f1_infoneed'])}")


  # =========================== SUPPORT VECTOR MACHINE ================================
    elif classifier == "SVM":

        if verbose == 1:
            print("=== Building Classifier:", classifier)

        clf = SVC(gamma='scale')
        clf_scores = cross_validate(clf, word_embedding_corpus, y, cv=10, scoring=scoring)
        print(f"Classifier: SVM -- Condition: {condition} -- Info Need: {info_need} -- F1 Average: {mean(clf_scores['test_f1_infoneed'])}")




    # ================================ RANDOM FOREST ======================================
    elif classifier == "RF":

        if verbose == 1:
            print("=== Building Classifier:", classifier)

        clf = RandomForestClassifier(class_weight="balanced")
        clf_scores = cross_validate(clf, word_embedding_corpus, y, cv=10, scoring=scoring)
        print(f"Classifier: RF -- Condition: {condition} -- Info Need: {info_need} -- F1 Average: {mean(clf_scores['test_f1_infoneed'])}")

    
    
    df_classification_report = pd.DataFrame(columns=['loss', 'task_name', 'info_need','model', 'num_epochs', 'condition',
                            'acc','f1_other','f1_infoneed','precision_infoneed','recall_infoneed',
                            'recall_other','precision_other','recall_macro','precision_macro',
                            'recall_micro','precision_micro','recall_weighted','precision_weighted',
                            'f1_weighted','f1_macro','f1_micro','mcc','report','preds','labels','stopwords_removed','resampled'])


    for fold in range(10):
        df_classification_report.loc[fold]=['None', 'text_classification', info_need, classifier, 'None', condition, 'None', clf_scores['test_f1_other'][fold],
                                        clf_scores['test_f1_infoneed'][fold], clf_scores['test_precision_infoneed'][fold], clf_scores['test_recall_infoneed'][fold],
                                        clf_scores['test_recall_other'][fold], clf_scores['test_precision_other'][fold], clf_scores['test_recall_macro'][fold],
                                        clf_scores['test_precision_macro'][fold], clf_scores['test_recall_micro'][fold], clf_scores['test_precision_micro'][fold],
                                        clf_scores['test_recall_weighted'][fold], clf_scores['test_precision_weighted'][fold], clf_scores['test_f1_weighted'][fold],
                                        clf_scores['test_f1_macro'][fold], clf_scores['test_f1_micro'][fold], "None", "None", "None", "None",remove_stopwords,resample]
    return df_classification_report



def create_dataset_for(info_need, condition, dataset):
    dataset_copy = dataset.copy(deep=True)
    dataset_copy.loc[dataset_copy.level_1 != info_need, 'level_1'] = 'Other'
    return dataset_copy


def run_baseline_models():
    random_states_list = [42]

    classifier_list = ["NB", 
                       "RF", 
                       "SVM"]

    featurization_list = ["fasttext"]

    remove_stopwords = [False, True]
    resampling = [False, True]



    no_context_corpus = pd.read_csv("corpus_no_context.csv", sep=";", decimal=",")
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \?', '?')
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \!', '!')
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \.', '.')
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \,', ',')
    no_context_corpus = no_context_corpus[['utterance','level_1']]

    corpus_one_prev_utterance = pd.read_csv("corpus_one_prev_utterance.csv", sep=";", decimal=",")
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \?', '?')
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \!', '!')
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \.', '.')
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \,', ',')
    corpus_one_prev_utterance = corpus_one_prev_utterance[['utterance','level_1']]

    corpus_all_prev_utterances = pd.read_csv("corpus_all_prev_utterances.csv", sep=";", decimal=",")
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \?', '?')
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \!', '!')
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \.', '.')
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \,', ',')
    corpus_all_prev_utterances = corpus_all_prev_utterances[['utterance','level_1']]

    label_list = ['Temperature', 'Cooking technique', 'Recipe', 'Equipment', 
                  'Meal', 'Time', 'Knowledge', 'Miscellaneous', 'Preparation', 'Amount', 'Ingredient']

    context_conditions = {"no_context":no_context_corpus,
                         "one_prev_utterance":corpus_one_prev_utterance,
                         "all_prev_utterance":corpus_all_prev_utterances
                         }

    # A list to store the results
    baseline_models_results = []
    for classifier_name in classifier_list:
        for do_resample in resampling:
            for do_remove_stopword in remove_stopwords:
                for condition,dataset in context_conditions.items():
                    for info_need in label_list:
                            current_dataset = create_dataset_for(info_need, condition, dataset)
                            test_results = perform_text_classification(current_dataset,
                                                                       info_need,
                                                                       condition,
                                                                       classifier = classifier_name,
                                                                       resample = do_resample,
                                                                       remove_stopwords = do_remove_stopword,
                                                                       verbose = 1, 
                                                                       random_state = 42,
                                                                       warning_on_off = "off")

                            baseline_models_results.append(test_results) 


        # Concatenate dataframes list to one single dataframe
        models_results = pd.concat(baseline_models_results)

        # Write results to a file
        models_results.to_csv("results.csv", index=True)

if __name__ == "__main__":
    run_baseline_models()