# Task 4

import pandas as pd
from collections import Counter
import re
import time
import numpy as np
import pickle
import string

## Reload INVERTED INDEX and VOCAB LIST 
## -------------------------------------------------
## PICKLE -------------------------------------------------
file_name = 'inverted_index.pkl'
with open(file_name, 'rb') as file:
    inverted_index = pickle.load(file)
file_name = 'vocab_list.pkl'
with open(file_name, 'rb') as file:
    vocab_list = pickle.load(file)
file_name = 'cleaned_cp_docs.pkl'
with open(file_name, 'rb') as file:
    cleaned_cp_docs = pickle.load(file)
file_name = 'querytodocs.pkl'
with open(file_name, 'rb') as file:
    query_to_docs = pickle.load(file)

## PICKLE -------------------------------------------------
## -------------------------------------------------
queries = pd.read_table("test-queries.tsv", header=None)
queries = queries.rename(columns={0: 'qindex', 1: 'query'})
import string
trans = str.maketrans("", "", string.punctuation)
queries.iloc[:,1] = [string.translate(trans) for string in queries.iloc[:,1]]

V = len(vocab_list)
cleaned_cp_docs = cleaned_cp_docs[['index', 'passage']]

# ===============================================================
# ===============================================================
# 4.1 Laplace smoothing =========================================
# ===============================================================
# ===============================================================

## Method 3: ================================================
# Compute doc term frequencies on the fly 

## Make a function that directly computes the Laplace smoothed score
## MINI FUNCTION 
def laplace_smoothing(query_terms, doc_terms, doc_term_freq, V):
    score = 0
    for term in query_terms: ## takes in a string query
        if term in doc_terms: ## common term 
            tf_doc = doc_term_freq[term] 
        else:
            tf_doc = 0
        score += np.log((tf_doc + 1) / (len(doc_terms) + V)) ## append to score
    return score 


## Make the overall function that iterates through everything then calls mini fucntion
# - THIS FUNCTION WILL CALL ON LAPLACE SMOOTHING MINI FUNCTION
# USE FOR THE PD ROW LATER - FEED IN QUERIES
def compute_score_laplace(row): 
    qid = row['qindex']
    query_string = row['query']
    scores = []
    
    ## Get only relevant docs 
    document_relevant_subset_indices = query_to_docs[qid]
    document_relevant_subset = cleaned_cp_docs.loc[cleaned_cp_docs['index'].isin(document_relevant_subset_indices)]

    ## ITERATE OVER RELEVANT DOCS
    for docid, doc_string in document_relevant_subset[['index', 'passage']].itertuples(index=False):
    
    ## CHECK IF ALREADY CALCULATED BEFORE --------- (REPEATED DOCS)
    ## compute on the fly method 
        if docid not in doc_term_freq:
            ## INITALISE 
            doc_term_freq[docid] = {}
            for term in doc_string.split():
                if term not in doc_term_freq[docid]:
                    doc_term_freq[docid][term] = 1
                else:
                    doc_term_freq[docid][term] += 1
        ## Feed into mini function now
        doc_terms = set(doc_string.split())
        query_terms = set(query_string.split())
        score = laplace_smoothing(query_terms, doc_terms, doc_term_freq[docid], V)
        scores.append(score)
    
    result = pd.DataFrame({
        'qid': [qid] * len(document_relevant_subset),
        'pid': document_relevant_subset['index'],
        'score': scores,
    })
    return result


## Call the function(s) on the data to get output!
doc_term_freq = {}
s = time.time()
laplace_scores = queries.apply(compute_score_laplace, axis=1)
laplace_results = pd.concat(list(laplace_scores), axis=0, ignore_index=True)
print("Time to build Laplace is ", time.time() - s)

laplace_top_docs = laplace_results.sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(100)

## FINAL OUTPUT laplace smoothing -- ARRANGE THE QUERY IDS AS PER TEST QUERIES
## ---------------------------------------
# Load test queries
proper_qid_order = pd.read_table("test-queries.tsv", header = None)[0]
# Create a new category datatype from pandas for the right order
category_dtype = pd.api.types.CategoricalDtype(categories=proper_qid_order, ordered=True)
# Convert our query index column to this data type (categorical)
laplace_top_docs['qid'] = laplace_top_docs['qid'].astype(category_dtype)
# FINALLY: ARRANGE! 
laplace_top_docs = laplace_top_docs.sort_values('qid').reset_index(drop=True)
laplace_top_docs = laplace_top_docs.groupby('qid', sort = False).apply(lambda x: x.sort_values('score', ascending=False))
laplace_top_docs = laplace_top_docs[['qid','pid','score']]
laplace_top_docs.to_csv("laplace.csv", index= False, header = False)
## ---------------------------------------


# ===============================================================
# ===============================================================
# 4.2 Lidstone smoothing =========================================
# ===============================================================
# ===============================================================

## CHANGE THE LAPLACE SMOOTHING --> LIDSTONE SMOOTHING FUNCTION

## LIDSTONE SMOOTHING 
## Set hyperparameter e (small value )
e = 0.1 ## can be changed 

## Modify the function
# MINI FUNCTION 
def lidstone_smoothing(query_terms, doc_terms, doc_term_freq, V, e = 0.1):
    score = 0
    for term in query_terms: ## takes in a string query
        if term in doc_terms: ## common term 
            tf_doc = doc_term_freq[term] 
        else:
            tf_doc = 0
        score += np.log((tf_doc + e) / (len(doc_terms) + e*V)) ## append to score
    return score 

## OVERALL FUNCTION - USE PD DATAFRAME APPLY 
# - this function calls on LIDSTONE_SMOOTHING MINI FUNCTION
def compute_score_lidstone(row): ## Make function that applies to pandas
    qid = row['qindex']
    query_string = row['query']
    scores = []
    
    ## Get only relevant docs 
    document_relevant_subset_indices = query_to_docs[qid]
    document_relevant_subset = cleaned_cp_docs.loc[cleaned_cp_docs['index'].isin(document_relevant_subset_indices)]

    ## Apply same logic as before
    for docid, doc_string in document_relevant_subset[['index', 'passage']].itertuples(index=False):
        if docid not in doc_term_freq:
            doc_term_freq[docid] = {} ## INITIALISE
            for term in doc_string.split():
                if term not in doc_term_freq[docid]:
                    doc_term_freq[docid][term] = 1
                else:
                    doc_term_freq[docid][term] += 1
        
        ## feed into mini function
        doc_terms = set(doc_string.split())
        query_terms = set(query_string.split())
        score = lidstone_smoothing(query_terms, doc_terms, doc_term_freq[docid], V)
        scores.append(score)

    result = pd.DataFrame({
        'qid': [qid] * len(document_relevant_subset),
        'pid': document_relevant_subset['index'],
        'score': scores,
    })
    return result

## Call the function(s) on the data to get output!
doc_term_freq = {}
s = time.time()
lidstone_scores = queries.apply(compute_score_lidstone, axis=1)
lidstone_results = pd.concat(list(lidstone_scores), axis=0, ignore_index=True)
print("Time to build Lidstone is ", time.time() - s)

lidstone_top_docs = lidstone_results.sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(100)

## FINAL OUTPUT laplace smoothing -- ARRANGE THE QUERY IDS AS PER TEST QUERIES
## ---------------------------------------
lidstone_top_docs['qid'] = lidstone_top_docs['qid'].astype(category_dtype)
# FINALLY: ARRANGE! 
lidstone_top_docs = lidstone_top_docs.sort_values('qid').reset_index(drop=True)
lidstone_top_docs = lidstone_top_docs.groupby('qid', sort = False).apply(lambda x: x.sort_values('score', ascending=False))
lidstone_top_docs = lidstone_top_docs[['qid','pid','score']]
lidstone_top_docs.to_csv("lidstone.csv", index=False, header = False)
## ---------------------------------------

# ===============================================================
# ===============================================================
# 4.3 Dirichlet smoothing =========================================
# ===============================================================
# ===============================================================


## STEP 1. CORPUS MLE : construct dictionary that gives corpus MLE 
## - THIS IS THE CORPUS MLE 
def CORPUS_MLE_DICT(inverted_index = inverted_index):
    term_counter = {}
    total_terms = 0
    ## OPEN UP ALL WORDS + POSTINGS 
    for term, posting in inverted_index.items():
        ## For each word, count occurences in all docs 
        term_count = sum(termfreq for (docid, termfreq) in posting)
        term_counter[term] = term_count
        ## Conveniently get total word count
        total_terms += term_count 
    ## This is what we want 
    corpus_mle_dict = {}
    for term, count in term_counter.items():
        corpus_mle_dict[term] = count/total_terms
    return(corpus_mle_dict)

## Intialise dictionary 
corpus_mle_dic = CORPUS_MLE_DICT()

## STEP 1.1 : INTRODUCE a term frequency function to speed things up 
# THIS HELPS TO REUSE THE SAME RESULTS OF THE FUNCTION UP TO 128 CALLS (since it is reused many times)
#from functools import lru_cache
#@lru_cache(maxsize=128)
#def get_term_frequency(term, docid):
#    for (term_docid, term_freq) in inverted_index[term]:
#        if term_docid == docid:
#           return term_freq
#   return 0

## STEP 2. MINI FUNCTION : dirichlet score (does the actua calculation for each query string and doc string)
# Will call on CORPUS MLE for smoothing
def compute_dirichlet_score(args):
    query_string, doc_string, mu, docid = args
    query_terms = query_string.split()
    doc_terms = doc_string.split()
    score = 0
    for term in query_terms:
        tf_doc_i = 0

        ## for terms in the query: 
        if term in inverted_index and term in doc_terms : # check that its a common term 
            for (term_docid, term_freq) in inverted_index[term]:
                if term_docid == docid:
                    tf_doc_i = term_freq
                    break ## just to retrieve term frequency for the particular doc id 

            ## APPLY SMOOTHING TO TERM THAT EXISTS IN DOC
            score_smoothed = (tf_doc_i + mu * corpus_mle_dic[term]) / (len(doc_terms) + mu)
        elif term in corpus_mle_dic:
            ## OTHERWISE (IF TERM NOT IN DOC), USE CORPUS MLE 
            score_smoothed = mu * corpus_mle_dic[term] / (len(doc_terms) + mu)
        else:
            score_smoothed = 0
        ## update score 
        if score_smoothed > 0:
            score += np.log(score_smoothed)
    return (docid, score)


## STEP 3. OVERALL FUNCTION: iterate through datasets and apply mini function
def compute_dirichlet_scores(qid, query_string, document_relevant_subset):
    mu = 50
    dirichlet_scores = []


    for doc_id, doc_string in document_relevant_subset[['index', 'passage']].itertuples(index=False):
    ## RECALL 2 OUTPUTS FROM COMPUTE SCORE FUNCTION
        docid, score = compute_dirichlet_score((query_string, doc_string, mu, doc_id))
        dirichlet_scores.append((qid, doc_id, score))

    return pd.DataFrame(dirichlet_scores, columns=['qid', 'pid', 'score'])

## Call the functions(s) on the data to get Lirichlet output
## Call the function(s) on the data to get Dirichlet output
s = time.time()
dirichlet_scores = []
for qid, query_string in queries.itertuples(index=False):
    ## Get only relevant docs 
    document_relevant_subset_indices = query_to_docs[qid]
    document_relevant_subset = cleaned_cp_docs.loc[cleaned_cp_docs['index'].isin(document_relevant_subset_indices)]
    scores = compute_dirichlet_scores(qid, query_string, document_relevant_subset)
    dirichlet_scores.append(scores)

dirichlet_results = pd.concat(dirichlet_scores, keys=queries.index)
print("Time to build Dirichlet is ", time.time() - s)

dirichlet_top_docs = dirichlet_results.sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(100)

## FINAL OUTPUT laplace smoothing -- ARRANGE THE QUERY IDS AS PER TEST QUERIES
## ---------------------------------------
dirichlet_top_docs['qid'] = dirichlet_top_docs['qid'].astype(category_dtype)
# FINALLY: ARRANGE! 
dirichlet_top_docs = dirichlet_top_docs.sort_values('qid').reset_index(drop=True)
dirichlet_top_docs = dirichlet_top_docs.groupby('qid', sort = False).apply(lambda x: x.sort_values('score', ascending=False))
dirichlet_top_docs = dirichlet_top_docs[['qid','pid','score']]
dirichlet_top_docs.to_csv("dirichlet.csv", index = False, header = False)
## ---------------------------------------


## EVALUATING PERFORMANCE 
## Test parameters 


def return_top_doc(queryid):
    test_query = queries.iloc[queryid]
    test_query_id = test_query[0]
    document_relevant_subset_indices = query_to_docs[test_query['qindex']]
    document_relevant_subset = cleaned_cp_docs.loc[cleaned_cp_docs['index'].isin(document_relevant_subset_indices)]
    listabc = []
    ## Laplace
    firstplace_pid_lap = laplace_top_docs[laplace_top_docs['qid'] == test_query_id]['pid'].iloc[0]
    a = list(document_relevant_subset [document_relevant_subset['index'] == firstplace_pid_lap]['passage'])
    ## Lidstone
    firstplace_pid_lid = lidstone_top_docs[lidstone_top_docs['qid'] == test_query_id]['pid'].iloc[0]
    b = list(document_relevant_subset [document_relevant_subset['index'] == firstplace_pid_lid]['passage'])
    ## Dirichlet 
    firstplace_pid_diri = dirichlet_top_docs[dirichlet_top_docs['qid'] == test_query_id]['pid'].iloc[0]
    c = list(document_relevant_subset [document_relevant_subset['index'] == firstplace_pid_diri]['passage'])
    listabc.append((test_query[1],test_query[1],test_query[1]))
    listabc.append((a,b,c))
    print("Question: ", test_query[1])
    print("Lap: ", a)
    print("Lid: ", b)
    print("Diri: ", c)
    return pd.DataFrame(listabc, columns = ['Laplace', 'Lidstone', 'Dirichlet'])

np.random.seed(10)
## Evaluation
result1 = return_top_doc(np.random.randint(200))

np.random.seed(11)
result2 = return_top_doc(np.random.randint(200))

np.random.seed(12)
result3 = return_top_doc(np.random.randint(200))
