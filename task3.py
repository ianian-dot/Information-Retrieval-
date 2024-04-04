## =============================================================================
## TASK 3: CREATE TFIDF ===============================================
## =============================================================================
import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict

## Task 2 data 
cptop1000 = pd.read_table("candidate-passages-top1000.tsv",header=None)
cp_docs = cptop1000.iloc[:,([1,3])]
cp_docs = cp_docs.rename(columns={1: 'index', 3: 'passage'})

## Reload INVERTED INDEX and VOCAB LIST 
## -------------------------------------------------
## PICKLE -------------------------------------------------
import pickle
file_name = 'inverted_index.pkl'
with open(file_name, 'rb') as file:
    inverted_index = pickle.load(file)
file_name = 'vocab_list.pkl'
with open(file_name, 'rb') as file:
    vocab_list = pickle.load(file)
file_name = 'cleaned_cp_docs.pkl'
with open(file_name, 'rb') as file:
    cleaned_cp_docs = pickle.load(file)

## PICKLE -------------------------------------------------
## -------------------------------------------------

## READ QUERIES, SUBSET REAL INDEX AND QUERIES
## ----------------------
queries = pd.read_table("test-queries.tsv", header=None)
queries = queries.rename(columns={0: 'qindex', 1: 'query'})
import string
trans = str.maketrans("", "", string.punctuation)
queries.iloc[:,1] = [string.translate(trans) for string in queries.iloc[:,1]]
## ----------------------


## CREATE IDF MAPPER -- common vector 
#idf_dict = {word: math.log(total_no_documents / len(postings)) for word, postings in inverted_index.items()}
total_no_documents = len(set(cp_docs.iloc[:,1]))
doc_freq = [len(inverted_index[word])+1 for word in vocab_list] ## +1 since somehow there are words with 0 postings in the inverted index
IDF_vector = np.log(np.array(total_no_documents)/np.array(doc_freq))

## ----------------------------------
# MAP: WORD INDEX FOR COLUMNS : gives column number for a word according to vocabulary order
word_index = {word: i for i, word in enumerate(vocab_list)}
## Unqiue documents 
unqiue_cpdocs = cp_docs.drop_duplicates(ignore_index=True)
## MAP : Create a mapping from document index to row in the matrix
doc_index_to_row = {doc_index: i for i, doc_index in enumerate(unqiue_cpdocs['index'])}


from scipy.sparse import csc_matrix
## =====================================================================================
## CREATE TFIDF FOR PASSAGES -- SPARSE MATRIX due to dimensions
## =====================================================================================

## Initialize the sparse matrix with the number of rows (documents) and columns (vocabulary size)
rows = [] ## initialise entry metadata for sparse matrix entry insertion 
cols = []
data = []
## fill up placeholders per entry 
for word, posting_list in inverted_index.items():
    col = word_index[word]
    for doc_index, word_count in posting_list:
        row = doc_index_to_row[doc_index]
        rows.append(row)
        cols.append(col)
        data.append(word_count)

## MAKE THE SPARSE MATRIX --
passages_matrix = csc_matrix((data, (rows, cols)), shape=(len(doc_index_to_row), len(vocab_list)))
import numpy as np
## Calculate the TF-IDF values for the passages
tfidf_matrix = (passages_matrix.multiply(np.log(len(cp_docs)/np.array(doc_freq)))).tocsr()
tfidf_matrix1 = (passages_matrix.multiply(IDF_vector)).tocsr()

## Change for better name 
passages_tfidf = tfidf_matrix

## =====================================================================================
## CREATE TFIDF FOR queries -- normal matrix 
## =====================================================================================

## row mapper now
q_index_to_row = {qindex: i for i, qindex in enumerate(queries['qindex'])}
## matrix np placeholder
query_word_counts = np.zeros((len(queries), len(vocab_list)))

## FILL IN MATRIX 
for i, query in queries['query'].items():
    words = query.split()
    word_set = set(words)
    for word in word_set:
        if word in vocab_list:
            j = word_index[word]
            query_word_counts[i][j] = words.count(word)

## Calculate the TF-IDF values for the queries
query_tfidf = query_word_counts * (IDF_vector)

## Check shapes -- check num columns
query_tfidf.shape
passages_tfidf.shape

## GET COSINE SIMILARITY - full matrix 
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarities = cosine_similarity(query_tfidf, passages_tfidf)
cosine_similarities.shape ## queries by passages

## MAPPER - get back ACTUAL document index 
document_indices = {row: doc_index for doc_index, row in doc_index_to_row.items()}
## Get most relevant - sort score (descending -ve sign) 
## Take only 100 
topdocs_indices = np.argsort(-cosine_similarities, axis=1)[:, :100]
## check: top 100 docs (cols) per query (row)
topdocs_indices.shape

## Need to also get back ACTUAL QUERY INDEX 
## reorganise data -- prepare for pandas df
results = [] 
for i, query_id in enumerate(queries['qindex']): ## prepare actual query id 
    # get all scores for query -- full long vector
    query_row = cosine_similarities[i,:]

    ## retrieve from 200 by 100 array -- get row only 
    top_docs = topdocs_indices[i,:]    ## 100 most relevant
    for j, doc_index in enumerate(top_docs):
        doc_id = document_indices[doc_index]
        score = query_row[doc_index]
        results.append([query_id, doc_id, score])

TFIDF_FINAL = pd.DataFrame(results)
TFIDF_FINAL = TFIDF_FINAL.rename(columns={0: 'qid', 1:'pid', 2: 'score'})

## FINAL OUTPUT TFIDF COSINE SCORES -- ARRANGE THE QUERY IDS AS PER TEST QUERIES
## ---------------------------------------
## Load test queries
proper_qid_order = pd.read_table("test-queries.tsv", header = None)[0]
## Create a new category datatype from pandas for the right order
category_dtype = pd.api.types.CategoricalDtype(categories=proper_qid_order, ordered=True)
## Convert our query index column to this data type (categorical)
TFIDF_FINAL['qid'] = TFIDF_FINAL['qid'].astype(category_dtype)
# FINALLY: ARRANGE! 
TFIDF_FINAL = TFIDF_FINAL.sort_values('qid').reset_index(drop=True)
TFIDF_FINAL = TFIDF_FINAL.groupby('qid', sort = False).apply(lambda x: x.sort_values('score', ascending=False))
TFIDF_FINAL = TFIDF_FINAL[['qid','pid','score']]
TFIDF_FINAL.to_csv("tfidf.csv", index = False, header = False)
## ---------------------------------------



## ======================================================================= ##
## ======================================================================= ##
## ======================================================================= ##
## Task 3.2 
## BM25


## HYPERPARAMETERS 
avg_doc_length = np.array([len(d) for d in cleaned_cp_docs['passage']]).mean()
b = 0.75; k1 = 1.2; k2 = 100
N = len(cleaned_cp_docs)  # total number of documents

## MAP QUERY ID TO RELEVANT DOCS ID ONLY
cptop1000 = pd.read_table("candidate-passages-top1000.tsv", header = None) 
cp_indexes_only = cptop1000.iloc[:,[0,1]] ## We need the original data, since the original data has unique pairs btn qid and did
cp_indexes_only = cp_indexes_only.rename({0: 'Queryindex', 1: 'Docindex'}, axis=1)
query_to_docs = {}
for query_id, group in cp_indexes_only.groupby('Queryindex'):
    query_to_docs[query_id] = group['Docindex'].tolist()

## document indices (unqiue only )
set_cp_docs_index = set(cptop1000.iloc[:,1])

## ------------------------------------------------------------------------
## TRY A MORE EFFICIENT METHOD


## ONTO MAIN FUNCTION 
def BM25_generator_MORE_EFFICIENT(list_queries = queries, documents = cleaned_cp_docs):

    scores = []
    for query_id, query in list_queries[['qindex', 'query']].itertuples(index=False):
        query_terms = Counter(query.split()) ## This for freq term in query 
        
        ## Get only relevant docs 
        document_relevant_subset_indices = query_to_docs[query_id]
        document_relevant_subset = documents.loc[documents['index'].isin(document_relevant_subset_indices)]
        
        ## Loop through relevant docs 
        for passage_id, passage in document_relevant_subset[['index', 'passage']].itertuples(index=False):
            doc_terms = passage.split()
            doc_length = len(doc_terms)

            ## Precompute information for each document
            query_term_counts = defaultdict(int)
            for term in doc_terms:
                if term in query_terms and term in inverted_index:
                    query_term_counts[term] += 1

            ## Compute BM25 score
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
            .head(100))
    return(top_docs)
## GENERATE the improved BM25

s = time.time()
BM25_result = BM25_generator_MORE_EFFICIENT()
print("Time to build BM25 is ", time.time() - s)

## FINAL OUTPUT BM25-- ARRANGE THE QUERY IDS AS PER TEST QUERIES
## ---------------------------------------
# Load test queries
BM25_result['qid'] = BM25_result['qid'].astype(category_dtype)
# FINALLY: ARRANGE! 
BM25_result = BM25_result.sort_values('qid').reset_index(drop=True)
BM25_result = BM25_result.groupby('qid', sort = False).apply(lambda x: x.sort_values('score', ascending=False))
BM25_result = BM25_result[['qid','pid','score']]
BM25_result.to_csv('bm25.csv', index = False, header = False)
## ---------------------------------------



## Task OUTPUTS: inverted index, cleaned list of passages from top candidate
## -------------------------------------------------
## PICKLE -------------------------------------------------

file_name = 'querytodocs.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(query_to_docs, file)
    print(f'Object successfully saved to "{file_name}"')
