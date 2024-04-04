import pandas as pd
from collections import Counter, defaultdict
import numpy as np 
import time
import pickle
import random

## Read data
valid_data = pd.read_table("validation_data.tsv")

## Take only a sample of the data (BM25 wasnt able to run on the full data)
def takesample(valid_data, n = 200):
    unique_qid = valid_data['qid'].unique()
    subsetqid = random.sample(sorted(unique_qid), n)
    sampled_data = valid_data[valid_data['qid'].isin(subsetqid)]
    return(sampled_data)

#valid_data = takesample(valid_data,200)

## 1 --- BM25 MODEL ================================================================================================

## 1.1 setence cleaner - clean passages
def clean_list_strings(strings):
    special_words = ['ar15', 'ww1', '2010']
    result = []
    for string in strings:
        words = string.split()
        processed_words = []
        for word in words:
            word = word.lower()
            if word.isalpha() or word in special_words:
                processed_words.append(word)
        result.append(" ".join(processed_words))
    return result

passages = valid_data[['pid','passage']].drop_duplicates(ignore_index=True)
passages['passage'] = clean_list_strings(passages['passage'])
cleaned_unique_passages = passages

## 1.2 create inverted index
cleaned_unique_passages['words'] = cleaned_unique_passages['passage'].str.split()
words_list = cleaned_unique_passages['words'].tolist() ## list of lists

s = time.time()
inverted_index = defaultdict(list)
for index, words in zip(cleaned_unique_passages['pid'], cleaned_unique_passages['words']): ## iterate over (tokenised) sentences
    ## Create Counter Object: 
    document_counter = Counter(words)
    for word,count in document_counter.items():
            inverted_index[word].append((index, count))

print("time for inverted index building in mins ", (time.time() - s)/60)

## 1.3 query to docs for relevance subset
indexes_only = valid_data.iloc[:,[0,1]] ## We need the original data, since the original data has unique pairs btn qid and did
query_to_docs = {}
for query_id, group in indexes_only.groupby('qid'):
    query_to_docs[query_id] = group['pid'].tolist()

## 1.4 hyperparameters
avg_doc_length = np.array([len(d) for d in cleaned_unique_passages['passage']]).mean()
b = 0.75; k1 = 1.2; k2 = 100
N = len(cleaned_unique_passages)  # total number of documents
list_queries = valid_data[['qid', 'queries']].drop_duplicates(ignore_index=True)
documents = cleaned_unique_passages.iloc[:,[0,1]]

## 1.5 BM25 function 
def BM25_generator_MORE_EFFICIENT():

    scores = []
    ## Loop over all queries 
    for query_id, query in list_queries[['qid', 'queries']].itertuples(index=False):
        query_terms = Counter(query.split()) ## This for freq term in query 
        
        ## For each query, get only relevant documents  
        document_relevant_subset_indices = query_to_docs[query_id]
        document_relevant_subset = documents.loc[documents['pid'].isin(document_relevant_subset_indices)]
        
        ## Loop through relevant docs 
        for passage_id, passage in document_relevant_subset[['pid', 'passage']].itertuples(index=False):
            doc_terms = passage.split()
            doc_length = len(doc_terms)

            # Precompute information for each document: counts the query terms in the document
            query_term_counts = defaultdict(int)
            for term in doc_terms:
                if term in query_terms and term in inverted_index:
                    query_term_counts[term] += 1

            # Compute BM25 score
            score = 0

            for query_term in query_terms:
                if query_term in inverted_index and query_term in query_term_counts:
                    ni = len(inverted_index[query_term]) ## number of docs with the term i
                    fi = query_term_counts[query_term] ## freq of term i in docs
                    qfi = query_terms[query_term]
                    K = k1 * ((1 - b) + b * (doc_length / avg_doc_length))
                    
                    
                    A = np.log((N - ni + 0.5) / (ni + 0.5))
                    B = ((k1 + 1) * fi) / (K + fi)
                    C = ((k2 + 1) * qfi) / (k2 + qfi)
                    
                    score += A * B * C
            
            scores.append((query_id, passage_id, score))
            
    result = pd.DataFrame(scores, columns=['qid', 'pid', 'score'])
    top_docs = (result
            .sort_values(['qid', 'score'], ascending=[True, False])
            .groupby('qid')
            .head(500)) ## TAKE THE TOP 500 DOCUMENTS 
    return(top_docs)

## 1.6 generate data
s = time.time()
bm25 = BM25_generator_MORE_EFFICIENT()
print("time to get bm25: ", time.time() - s)


## 1.7 save BM25 to pickle object 
file_name = 'BM25_validationdata.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(bm25, file)
    print(f'Object successfully saved to "{file_name}"')
file_name = 'inverted_index.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(inverted_index, file)
    print(f'Object successfully saved to "{file_name}"')


## 2 --- Merge: Add in relevancy scores ================================================================================================
bm25_relevancy = pd.merge(valid_data, bm25, on= ['qid', 'pid']).sort_values(by=['qid', 'score'], ascending=[True, False])
bm25_relevancy.groupby('qid').sum('relevancy')

## 3 --- CALCULATE AVERAGE PRECISION
## Group by query
by_qid = bm25_relevancy.groupby('qid')
average_precisions = []
## For every query, calculate the average precision (therefore end up with # queries X AP for each query)
for _, group in by_qid:
    ## For each query... get all doc relevancies
    relevancy_scores = group['relevancy'] ## vector of (0,1,0,0,1...)
    num_relevant = np.sum(relevancy_scores) ## sum all 1s (total number of relevant documents)
    if num_relevant == 0:
        average_precisions.append(0)
    else:
        num_documents = len(relevancy_scores) 
        ## GENERATE THE PRECISIONS FOR EVERY ROW (RELEVANT OR NOT)
        precision = np.cumsum(relevancy_scores) / np.arange(1, num_documents + 1)

        ## USE RELEVANCY SCORES VECTORS (1S AND 0S) TO ONLY COMPUTE THE 1S I.E. rows that are actually relevant 
        average_precision = np.sum(precision * relevancy_scores) / num_relevant
        average_precisions.append(average_precision)

## FIND THE MEAN AP ACROSS ALL QUERIES 
BM25_MAP = np.mean(average_precisions)
print('The Mean Average Precision of the BM25 Ranked Results is ', BM25_MAP)

## ============================================================================================
## NOTE THAT: Since I only included top 500 documents, this is equivalent Precision at rank 500
## ============================================================================================

## 3 --- Find NDCG ================================================================================================
ndcgs = []
for _, group in by_qid:
    relevancy_scores = group['relevancy']

    ## GET IDEALISED RANKING -- I.E. ALL 1s ON TOP
    ideal_sorted_relevancy = sorted(relevancy_scores, reverse=True)
    k = len(relevancy_scores)

    ## Recall that top ranked divides by 1, while second onwards divides by log(2), log(3) ...
    dcg = relevancy_scores.iloc[0] + np.sum(relevancy_scores.iloc[1:k] / np.log2(np.arange(2, k+1))) ## this is essentially gain vector (relevance vector) discounted by log vector
    idcg = ideal_sorted_relevancy[0] + np.sum(ideal_sorted_relevancy[1:k] / np.log2(np.arange(2, k+1)))
#    if idcg == 0:
#        ndcg = 0
#    else:
#        ndcg = dcg / idcg
    ndcg = dcg / idcg if idcg != 0 else 0
    ndcgs.append(ndcg)

## FIND THE MEAN AP ACROSS ALL QUERIES 
BM25_NDCG = np.mean(ndcgs)
print('The Mean NDCG of the BM25 Ranked Results is ', BM25_NDCG)



############### TESTING PURPOSES ONLY 
def MAP(data, topk):
    by_qid = data.groupby('qid')
    average_precisions = []
## For every query, calculate the average precision (therefore end up with # queries X AP for each query)
    for _, group in by_qid:
        group = group.head(topk)
    # Compute precision and recall for top 100 documents
        relevancy_scores = group['relevancy'] ## vector of (0,1,0,0,1...)
        num_relevant = np.sum(relevancy_scores) ## sum all 1s (total number of relevant documents)
        if num_relevant == 0:
            average_precisions.append(0)
        else:
            num_documents = len(relevancy_scores) 
            ## GENERATE THE PRECISIONS FOR EVERY ROW (RELEVANT OR NOT)
            precision = np.cumsum(relevancy_scores) / np.arange(1, num_documents + 1)

            ## USE RELEVANCY SCORES VECTORS (1S AND 0S) TO ONLY COMPUTE THE 1S I.E. rows that are actually relevant 
            average_precision = np.sum(precision * relevancy_scores) / num_relevant
            average_precisions.append(average_precision)
    return np.mean(average_precision)

print(MAP(bm25_relevancy, 10), 'MAP@10')
print(MAP(bm25_relevancy, 100), 'MAP@100')

def NDCG(data, topk):
    by_qid = data.groupby('qid')
    ndcgs = []
    for _, group in by_qid:
        group = group.head(topk)
        relevancy_scores = group['relevancy']

        ## GET IDEALISED RANKING -- I.E. ALL 1s ON TOP
        ideal_sorted_relevancy = sorted(relevancy_scores, reverse=True)
        k = len(relevancy_scores)

        ## Recall that top ranked divides by 1, while second onwards divides by log(2), log(3) ...
        dcg = relevancy_scores.iloc[0] + np.sum(relevancy_scores.iloc[1:k] / np.log2(np.arange(2, k+1))) ## this is essentially gain vector (relevance vector) discounted by log vector
        idcg = ideal_sorted_relevancy[0] + np.sum(ideal_sorted_relevancy[1:k] / np.log2(np.arange(2, k+1)))
    #    if idcg == 0:
    #        ndcg = 0
    #    else:
    #        ndcg = dcg / idcg
        ndcg = dcg / idcg if idcg != 0 else 0
        ndcgs.append(ndcg)

    return np.mean(ndcgs)

print(NDCG(bm25_relevancy, 10), 'NDCG@10')
print(NDCG(bm25_relevancy, 100), 'NDCG@100')