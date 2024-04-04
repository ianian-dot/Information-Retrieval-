## TASK 2: CREATE INVERTED INDEX ===============================================
## =============================================================================
import numpy as np
import time 
import pickle
import pandas as pd
from collections import defaultdict, Counter
import re


## Task 2 data 
cptop1000 = pd.read_table("candidate-passages-top1000.tsv",header=None)

## Data contains both queries and associated 1000 documents 
## GET DOCUMENT DATA ONLY 

## Subset for only passages/document data: passage index and passage itself
cp_docs = cptop1000.iloc[:,([1,3])]
cp_docs = cp_docs.rename(columns={1: 'index', 3: 'passage'})


## Reload data from previous task 
## -------------------------------------------------
## PICKLE -------------------------------------------------
file_name = 'vocab_list.pkl'
with open(file_name, 'rb') as file:
    vocab_list = pickle.load(file)
## PICKLE -------------------------------------------------
## -------------------------------------------------

def clean_list_strings(strings):
    special_words = ['ar15', 'ww1', '2010']
    result = []
    for string in strings:
        string = re.sub(r'[^\w\s]', '', string)
        words = string.split()
        processed_words = []
        for word in words:
            word = word.lower()
            if word.isalpha() or word in special_words:
                processed_words.append(word)
        result.append(" ".join(processed_words))
    return result


## --------------------------- ## INVERTED INDEX: METHOD 2 -- 'VECTORISED': start
cp_docs = cp_docs.drop_duplicates(ignore_index=True)
s = time.time()
cp_docs['passage'] = clean_list_strings(cp_docs['passage']) ## clean the passages for better speed?
print("time for to clean passages in mins ", (time.time() - s)/60)
## Create a new column with the tokenised strings of each passage
cp_docs['words'] = cp_docs['passage'].str.split()

## Huge list of lists of all words ! (2 layered list)
words_list = cp_docs['words'].tolist() ## list of lists

## Create inverted index
s = time.time()
inverted_index = defaultdict(list)
for index, words in zip(cp_docs['index'], cp_docs['words']): ## iterate over (tokenised) sentences
    ## Create Counter Object: 
    document_counter = Counter(words)
    for word,count in document_counter.items():
        if word in vocab_list:
            inverted_index[word].append((index, count))

print("time for inverted index building in mins ", (time.time() - s)/60)

## NOT NEEDED ANYMORE: Remove duplicates : IN CASE DUPLICATES IN INVERTED INDEX
# def remove_duplicates(inverted_index):
#     unique_tuples = defaultdict(set)
#     for word, tuples in inverted_index.items():
#         unique_tuples[word].update(set(tuples))
#     return unique_tuples


## --------------------------- ## INVERTED INDEX: METHOD 2 -- 'VECTORISED': end

## TRY A METHOD THAT DOESNT RESULT IN DUPLICATE TUPLES IN INVERTED INDEX 



## Task 2 OUTPUTS: inverted index, cleaned list of passages from top candidate
## -------------------------------------------------
## PICKLE -------------------------------------------------
file_name = 'inverted_index.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(inverted_index, file)
    print(f'Object successfully saved to "{file_name}"')
file_name = 'cleaned_cp_docs.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(cp_docs, file)
    print(f'Object successfully saved to "{file_name}"')

## PICKLE -------------------------------------------------
## ----------
