#!/usr/bin/env python
# coding: utf-8

### NOTE: THIS SCRIPT IS BASED ON THE ONE PROVIDED BY FARM: https://github.com/deepset-ai/FARM/blob/master/examples/doc_classification_crossvalidation.py ###

# Install FARM
#get_ipython().system('pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html')
#get_ipython().system('pip install farm==0.7.1')
#get_ipython().system('pip install --upgrade scikit-learn')


import torch
import json
import os
from pathlib import Path
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.data_silo import DataSilo, DataSiloForCrossVal
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
#from farm.train import Trainer
from farm.utils import MLFlowLogger
from farm.train import Trainer, EarlyStopping
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from farm.eval import Evaluator
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score
from farm.evaluation.metrics import simple_accuracy, register_metrics
import numpy as np
import csv
import itertools
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler 
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from collections import Counter
import logging


def perform_info_need_prediction(level_needs,level):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" # use the third GPU
    # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(1))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    
    ## Load different context conditions ##
    no_context_corpus = pd.read_csv("corpus_no_context.csv", sep=";", decimal=",")
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \?', '?')
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \!', '!')
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \.', '.')
    no_context_corpus['utterance']=no_context_corpus['utterance'].str.replace(' \,', ',')
    no_context_corpus = no_context_corpus[['utterance',level]]

    corpus_one_prev_utterance = pd.read_csv("corpus_one_prev_utterance.csv", sep=";", decimal=",")
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \?', '?')
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \!', '!')
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \.', '.')
    corpus_one_prev_utterance['utterance']=corpus_one_prev_utterance['utterance'].str.replace(' \,', ',')
    corpus_one_prev_utterance = corpus_one_prev_utterance[['utterance',level]]

    corpus_all_prev_utterances = pd.read_csv("corpus_all_prev_utterances_cropped.csv", sep=";", decimal=",")
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \?', '?')
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \!', '!')
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \.', '.')
    corpus_all_prev_utterances['utterance']=corpus_all_prev_utterances['utterance'].str.replace(' \,', ',')
    corpus_all_prev_utterances = corpus_all_prev_utterances[['utterance',level]]

    ## Create train-test split ## 
    no_context_train_data, no_context_test_data = train_test_split(no_context_corpus, test_size = .15, random_state = 42, stratify=no_context_corpus[level])
    corpus_one_prev_utterance_train_data, corpus_one_prev_utterance_test_data = train_test_split(corpus_one_prev_utterance, test_size = .15, random_state = 42, stratify=corpus_one_prev_utterance[level])
    corpus_all_prev_utterances_train_data, corpus_all_prev_utterances_test_data = train_test_split(corpus_all_prev_utterances, test_size = .15, random_state = 42, stratify=corpus_all_prev_utterances[level])


    def init_logging():
        logger = logging.getLogger(__name__)
        logging.basicConfig(
          format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
          datefmt="%m/%d/%Y %H:%M:%S",
          level=logging.INFO)
        # reduce verbosity from transformers library
        logging.getLogger('transformers').setLevel(logging.WARNING)
        ml_logger = MLFlowLogger(tracking_uri="logs")
        return logger, ml_logger

    def perform_fine_tuning(current_info_need,
                            bert_model,
                            label_list,
                            num_epochs,
                            condition,
                            folds=10,
                            stratified=True,
                            learning_rate=2e-5,
                            batch_size=32,
                            embeds_dropout_prob=.1):
        
        ## Define evaluation metrics ##
        def evaluation_metrics(preds, labels):
            acc = simple_accuracy(preds, labels).get("acc")
            f1other = f1_score(y_true=labels, y_pred=preds, pos_label="Other")
            f1infoneed = f1_score(y_true=labels, y_pred=preds, pos_label=current_info_need)
            recall_infoneed = recall_score(y_true=labels, y_pred=preds, pos_label=current_info_need)
            precision_infoneed = precision_score(y_true=labels, y_pred=preds, pos_label=current_info_need)
            recall_other = recall_score(y_true=labels, y_pred=preds, pos_label="Other")
            precision_other = precision_score(y_true=labels, y_pred=preds, pos_label="Other")
            recall_macro = recall_score(y_true=labels, y_pred=preds, average="macro")
            precision_macro = precision_score(y_true=labels, y_pred=preds, average="macro")
            recall_micro = recall_score(y_true=labels, y_pred=preds, average="micro")
            precision_micro = precision_score(y_true=labels, y_pred=preds, average="micro")
            recall_weighted = recall_score(y_true=labels, y_pred=preds, average="weighted")
            precision_weighted = precision_score(y_true=labels, y_pred=preds, average="weighted")
            f1macro = f1_score(y_true=labels, y_pred=preds, average="macro")
            f1micro = f1_score(y_true=labels, y_pred=preds, average="micro")
            mcc = matthews_corrcoef(labels, preds)
            f1weighted = f1_score(y_true=labels, y_pred=preds, average="weighted")

            return {
              "info_need": current_info_need,
              "model": bert_model,
              "num_epochs": num_epochs,
              "condition": condition,
              "acc": acc,
              "f1_other": f1other,
              "f1_infoneed": f1infoneed,
              "precision_infoneed": precision_infoneed,
              "recall_infoneed": recall_infoneed,
              "recall_other": recall_other,
              "precision_other": precision_other,
              "recall_macro": recall_macro,
              "precision_macro": precision_macro,
              "recall_micro": recall_micro,
              "precision_micro": precision_micro,
              "recall_weighted": recall_weighted,
              "precision_weighted": precision_weighted,
              "f1_weighted": f1weighted,
              "f1_macro": f1macro,
              "f1_micro": f1micro,
              "f1_weighted": f1weighted,
              "mcc": mcc
            }
        register_metrics(f'eval_metrics_{current_info_need}_{bert_model}_{condition}__{num_epochs}_epochs', evaluation_metrics)
        metric = f'eval_metrics_{current_info_need}_{bert_model}_{condition}__{num_epochs}_epochs'
        set_all_seeds(seed=42)
        device, n_gpu = initialize_device_settings(use_cuda=True)
        logger, ml_logger = init_logging()
        tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=bert_model,
        do_lower_case=False)

        processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=256,
                                            train_filename=f"{current_info_need}_{condition}_{num_epochs}_epochs_train.csv",
                                            test_filename=f"{current_info_need}_{condition}_{num_epochs}_epochs_test.csv",
                                            data_dir="data/",
                                            label_list=label_list,
                                            metric=metric,
                                            text_column_name="utterance",
                                            label_column_name=level,
                                            delimiter=";")

        data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

        silos = DataSiloForCrossVal.make(data_silo, n_splits=folds, sets=['train','test'])

        # the following steps should be run for each of the folds of the cross validation, so we put them
        # into a function
        def train_on_split(silo_to_use,
                        n_fold,                     
                        save_dir):
            logger.info(f"############ Crossvalidation: Fold {n_fold} ############")
            # Create an AdaptiveModel
            # a) which consists of a pretrained language model as a basis
            language_model = LanguageModel.load(bert_model)
            # b) and a prediction head on top that is suited for our task => Text classification
            prediction_head = TextClassificationHead(
              class_weights=data_silo.calculate_class_weights(task_name="text_classification"),
              num_labels=len(label_list))

            model = AdaptiveModel(
              language_model=language_model,
              prediction_heads=[prediction_head],
              embeds_dropout_prob=embeds_dropout_prob,
              lm_output_types=["per_sequence"],
              device=device)

            # Create an optimizer
            model, optimizer, lr_schedule = initialize_optimizer(
              model=model,
              learning_rate=learning_rate,
              device=device,
              n_batches=len(silo_to_use.loaders["train"]),
              n_epochs=num_epochs,
              use_amp=None)

            # Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
            # Also create an EarlyStopping instance and pass it on to the trainer

            # An early stopping instance can be used to save the model that performs best on the dev set
            # according to some metric and stop training when no improvement is happening for some iterations.
            # NOTE: Using a different save directory for each fold, allows us afterwards to use the
            # nfolds best models in an ensemble!
            save_dir = Path(str(save_dir) + f"-{n_fold}")
            earlystopping = EarlyStopping(
              metric="f1_infoneed", mode="max",   # use the metric from our own metrics function instead of loss
              save_dir=save_dir,  # where to save the best model
              patience=5    # number of evaluations to wait for improvement before terminating the training
            )

            trainer = Trainer(
              model=model,
              optimizer=optimizer,
              data_silo=silo_to_use,
              epochs=num_epochs,
              n_gpu=n_gpu,
              lr_schedule=lr_schedule,
              evaluate_every=100,
              device=device,
              early_stopping=earlystopping,
              evaluator_test=False)

            # train it
            trainer.train()

            return trainer.model

        # for each fold, run the whole training, earlystopping to get a model, then evaluate the model
        # on the test set of each fold
        # Remember all the results for overall metrics over all predictions of all folds and for averaging
        allresults = []
        all_preds = []
        all_labels = []
        bestfold = None
        bestf1_info_need = -1
        language_model_name = bert_model
        if language_model_name.find("/")!=-1:
            language_model_name = language_model_name.replace("/","_")
        save_dir = Path(f"saved_models/{current_info_need}-{condition}-{num_epochs}_epochs-cook-{language_model_name}")
        for num_fold, silo in enumerate(silos):
            model = train_on_split(silo, num_fold, save_dir)

            # do eval on test set here (and not in Trainer),
            #  so that we can easily store the actual preds and labels for a "global" eval across all folds.
            evaluator_test = Evaluator(
              data_loader=silo.get_data_loader("test"),
              tasks=silo.processor.tasks,
              device=device
            )
            result = evaluator_test.eval(model, return_preds_and_labels=True)
            evaluator_test.log_results(result, "Test", steps=len(silo.get_data_loader("test")), num_fold=num_fold)

            allresults.append(result)
            all_preds.extend(result[0].get("preds"))
            all_labels.extend(result[0].get("labels"))

            # keep track of best fold
            f1_info_need = result[0]["f1_infoneed"]
            if f1_info_need > bestf1_info_need:
                bestf1_info_need = f1_info_need
                bestfold = num_fold

            # emtpy cache to avoid memory leak and cuda OOM across multiple folds
            model.cpu()
            torch.cuda.empty_cache()


        # Save the per-fold results to json for a separate, more detailed analysis
        with open(f"classification_results/test/{current_info_need}-{language_model_name}-{condition}-{num_epochs}_epochs-{folds}-fold-cv.results.json", "wt") as fp:
            json.dump(allresults, fp)

        # calculate overall metrics across all folds
        xval_f1_other = f1_score(all_labels, all_preds, labels=label_list, pos_label="Other")
        xval_f1_info_need = f1_score(all_labels, all_preds, labels=label_list, pos_label=current_info_need)  
        xval_f1_micro = f1_score(all_labels, all_preds, labels=label_list, average="micro")
        xval_f1_macro = f1_score(all_labels, all_preds, labels=label_list, average="macro")
        xval_mcc = matthews_corrcoef(all_labels, all_preds)

        xval_overall_results = {
            "xval_f1_other": xval_f1_other, 
            f"xval_f1_infoneed": xval_f1_info_need,
            "xval_f1_micro": xval_f1_micro, 
            "xval_f1_macro": xval_f1_macro,
            "xval_f1_mcc": xval_mcc
        }

        logger.info(f"XVAL F1 MICRO: {xval_f1_micro}")
        logger.info(f"XVAL F1 MACRO: {xval_f1_macro}")
        logger.info(f"XVAL F1 OTHER: {xval_f1_other}")
        logger.info(f"XVAL F1 {current_info_need} {condition} {num_epochs} epochs:   {xval_f1_info_need}")
        logger.info(f"XVAL MCC: {xval_mcc}")

        # -----------------------------------------------------
        # Just for illustration, use the best model from the best xval val for evaluation on
        # the original (still unseen) test set.
        logger.info("###### Final Eval on hold out test set using best model #####")
        evaluator_origtest = Evaluator(
          data_loader=data_silo.get_data_loader("test"),
          tasks=data_silo.processor.tasks,
          device=device
        )
        # restore model from the best fold
        lm_name = model.language_model.name
        save_dir = Path(f"saved_models/{current_info_need}-{condition}-{num_epochs}_epochs-cook-{language_model_name}-{bestfold}")
        model = AdaptiveModel.load(save_dir, device, lm_name=lm_name)
        model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

        result = evaluator_origtest.eval(model)
        logger.info("TEST F1 MICRO: {}".format(result[0]["f1_micro"]))
        logger.info("TEST F1 MACRO: {}".format(result[0]["f1_macro"]))
        logger.info("TEST F1 OTHER: {}".format(result[0]["f1_other"]))
        logger.info("TEST F1 {0}: {1}".format(current_info_need, result[0]["f1_infoneed"]))
        logger.info("TEST MCC:  {}".format(result[0]["mcc"]))

        test_set_results = {
            "test_f1_other": result[0]["f1_other"], 
            "test_f1_infoneed": result[0][f"f1_infoneed"],
            "test_f1_micro": result[0]["f1_micro"], 
            "test_f1_macro": result[0]["f1_macro"],
            "test_f1_mcc": result[0]["mcc"]
        }


    def create_dataset_for(info_need, condition, dataset, num_epochs, level):
        info_need_train = dataset[0].copy(deep=True)
        info_need_test = dataset[1].copy(deep=True)
        info_need_train.loc[info_need_train[level] != info_need, level] = 'Other'
        info_need_test.loc[info_need_test[level] != info_need, level] = 'Other'
        info_need_train.to_csv(f'data/{info_need}_{condition}_{num_epochs}_epochs_train.csv', index=False,sep=";")
        info_need_test.to_csv(f'data/{info_need}_{condition}_{num_epochs}_epochs_test.csv', index=False,sep=";")

    ## Define BERT models to use ##
    model_list = ["deepset/gbert-base",
                  "xlm-roberta-base",
                  "bert-base-multilingual-cased"              
                 ]
    
    ## Define conditions/condition data to use ##
    context_conditions = {"no_context":(no_context_train_data, no_context_test_data),
                         "one_prev_utterance":(corpus_one_prev_utterance_train_data, corpus_one_prev_utterance_test_data),
                         "all_prev_utterance":(corpus_all_prev_utterances_train_data, corpus_all_prev_utterances_test_data)
                         }
    epochs = [4]
    
    ## Here the experiments with its different conditons are started ##
    for epoch in epochs:
        for model in model_list:
            for condition,dataset in context_conditions.items():    
                for info_need in level_needs:
                    create_dataset_for(info_need, condition, dataset, epoch, level) 
                    perform_fine_tuning(info_need,model,[info_need, 'Other'], num_epochs = epoch, condition=condition)

if __name__ == "__main__":
    level_0_needs = ['fact','competence']
    level_1_needs = ['Temperature', 
                  'Cooking technique', 
                  'Recipe', 
                  'Equipment', 
                  'Meal', 
                  'Time', 
                  'Knowledge', 
                  'Miscellaneous', 
                  'Preparation', 
                  'Amount', 
                  'Ingredient']
    # Add the need list to want as first argument and specify on which taxonomy level these needs are
    perform_info_need_prediction(level_1_needs,'level_1')