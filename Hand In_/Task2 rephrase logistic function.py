from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
import pandas as pd
import re
import time
import random
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

## ================ Data
training_data = pd.read_table("train_data.tsv")

## ================ Check data
training_data

def takesample(data, n_questions = 300):
    unique_qid = data['qid'].unique()
    subsetqid = random.sample(sorted(unique_qid), n_questions)
    sampled_data = data[data['qid'].isin(subsetqid)]
    return(sampled_data)
# Take sample
random.seed(42)
training_data = takesample(training_data).reset_index(drop=True)
#training_data
## Check data 
## Do all queries have a 1000 documents? 
list(training_data.groupby('qid')['pid'].count())
all(element == 1000 for element in list(training_data.groupby('qid')['pid'].count()))
## - Result: not all queries have a 1000 -- some have 670, 7, 5....

################################################################################################################################
# Try using SPACY
# disclaimer: Spacy has a lot of dimensions for the word embeddings 
# Hence takes very long to embed 
################################################################################################################################

## ================ 1. Preprocessing step: for all documents and queries ========================================================

## Set of stopwords
stop_words = set(stopwords.words('english'))

## Preprocessing step : Augment to include removing stopwords, for better embedding performance
def clean_list_strings(strings):
    result = []
    for string in strings:
        string = re.sub(r'[^\w\s]', '', string) ## 1. REMOVE PUNCTUATION
        words = string.split() # get tokens 
        processed_words = [] # placeholder for words in a sentence 
        for word in words: # get each token
            word = word.lower() ## 2. LOWERCASE 
            if word.isalpha() and word not in stop_words: ## 3. REMOVE STOPWORDS 
                processed_words.append(word) # add back to placeholder sentence (list of words)
        result.append(" ".join(processed_words)) ## form back into a sentence 
    return result


## CLEAN VALID DATA 
# Preprocess passages : to prepare for word embedding
print('cleaning queries and passages')
s = time.time()
training_data['preprocessed_passage'] = clean_list_strings(training_data['passage'])
print('time to clean documents: ', time.time() - s)
s = time.time()
training_data['preprocessed_queries'] = clean_list_strings(training_data['queries'])
print('time to clean queries: ', time.time() - s)


## TESTING ONLY 
list(training_data['preprocessed_passage'][0:50])


## TESTING ONLY 
#training_data['embedded_queries']
#words = training_data['preprocessed_queries'][1].split()
#embeddings = [model(word).vector for word in words if word in model.vocab and model(word).has_vector]
#avg_embedding = sum(embeddings) / len(embeddings) if embeddings else None
#len(model.vocab)
#'three' in model.vocab
#model('three').has_vector

################################################################################################################################
# Try using GENSIM 
# Spacy returning too many None vectors
################################################################################################################################

import gensim.downloader as api

## Load pre-trained GloVe model
model = api.load('glove-wiki-gigaword-100')

## Define GENSIM function to compute average embedding (after taking each words normalised vectors)
def compute_avg_embedding(text):
    words = text.split()
    ## vectorise all words 
    vectors = [model[key] for key in words if key in model.key_to_index]
    ## normalise the vectors for each word 
    normalised_vects = [vec / np.linalg.norm(vec) for vec in vectors] if vectors else None
    ## finally, take the average of all word vectors to represent the query/passage
    avg_vector = sum(normalised_vects) / len(normalised_vects) if normalised_vects else None
    return avg_vector

## Get average vector representation for all queries and documents 
## Get: embedded_queries and embedded_passages
print("Embedding queries with spacy model...")
s = time.time()
training_data['embedded_queries'] = training_data['preprocessed_queries'].apply(compute_avg_embedding) ## splits into tokens, then turn all tokens into embeddings, then take average
print("Time taken to embed queries: ", time.time() - s)
s = time.time()
training_data['embedded_passages'] = training_data['preprocessed_passage'].apply(compute_avg_embedding) ## splits into tokens, then turn all tokens into embeddings, then take average
print("Time taken to embed passages: ", time.time() - s)

## Check 
training_data
print(training_data.shape)

## ------------------------------------------------------------------------------------

## SOME EMBEDDINGS ARE NA -- REMOVE THEM BEFORE FEEDING INTO THE MODEL 
training_data.dropna(subset=['embedded_queries', 'embedded_passages'], inplace=True)
print(training_data.shape) ## about less than 300 rows dropped


## CHECK FOR NA DATA
training_data['embedded_passages'].isna().sum()
training_data['embedded_queries'].isna().sum()

## ------------------------------------------------------------------------------------

## ================ 3. CODE THE LOGISTIC REGRESSION MODEL  ========================================================

## HELPER FUNCTIONS
def innerpdt(X, weights): ##this is for inner products
    if len(weights) == (X.shape[-1] + 1): ## if there is one more weight than the number of columns in the data matrix 
        x0 = np.expand_dims(np.ones(X.shape[:-1], X.dtype), axis=-1) ## add a ones column to the data matrix 
        X =  np.concatenate((x0, X), axis=-1)    
    return (X @ weights)


def log_step( X, w ):
    return 1/(1 + np.exp(-(innerpdt(X, w))))


def log_loss ( X, y, w, eps=1e-10 ):
    g = log_step(X, w)
    ## Binary loss cross entropy 
    return (np.dot(-y, np.log(g + eps)) - np.dot((1 - y), np.log(1 - g + eps)))/len(y)

def log_grad ( X, y, w ): ## returns the gradient with respect to w, given ground truth
    g = log_step(X, w)
    ## recall that sigmoid gradient 
    return X.T @ (g - y)

def GD ( z, loss_func, grad_func, lr=0.01,
                       loss_stop=1e-4, z_stop=1e-4, max_iter=100 ):
    losses = [ loss_func(z) ]
    zs = [ z ]
    
    d_loss = np.inf
    d_z = np.inf

    ## Stopping mechanism : number of iterations or loss magnitude or change in parameters
    while (len(losses) <= max_iter) and (d_loss > loss_stop) and (d_z > z_stop):
        zs.append(zs[-1] - lr * grad_func(zs[-1]))
        losses.append(loss_func(zs[-1]))
        
        d_loss = np.abs(losses[-2] - losses[-1])
        d_z = np.linalg.norm(zs[-2] - zs[-1])
    
    return zs[1:], losses[1:]

def log_regress ( X, y, w0=None, lr=0.05,
                          loss_stop=1e-4, weight_stop=1e-4, max_iter=100 ):
    assert(len(X.shape)==2)
    assert(X.shape[0]==len(y))
    
    ## initialise weights 
    if w0 is None: w0 = np.zeros(X.shape[-1])
    
    return GD ( w0,
                              loss_func = lambda z: log_loss(X, y, z),
                              grad_func = lambda z: log_grad(X, y, z),
                              lr = lr,
                              loss_stop=loss_stop, z_stop=weight_stop, max_iter=max_iter )

############# PREPARE DATA ################################################################
## Get X data getting COSINE SIMILARITY between query and passage vectors  

training_data['embedded_queries'].shape
training_data['embedded_passages'].shape

## Convert query and passage series into matrices first 
#queries_matrix_vectors = np.vstack(training_data['embedded_queries'].to_numpy())
#passages_matrix_vectors = np.vstack(training_data['embedded_passages'].to_numpy())

## Get cosine similarity 
print("Get cosine similarity between each row's query and passage avg vector...")
s = time.time()
## Function for cosine similarity --- JUST IN CASE NEEDED (not needed for now)
def np_cos_similarity(x,y):
    dotpdt = np.dot(x,y)
    normalised = dotpdt/(np.linalg.norm(x)*np.linalg.norm(y))
    return normalised
## ----------------------------------

## List comprehension to get a vector of scores 
training_data['cosine_similarity'] = [np_cos_similarity(row['embedded_queries'], row['embedded_passages']) for _,row in training_data.iterrows()]
print("Time to compute cos sim scores for query-doc pairs..." ,time.time() - s)

## Check
training_data['cosine_similarity']

X = np.array(training_data['embedded_queries'].to_list()) + np.array(training_data['embedded_passages'].to_list())
X = np.stack(X)
#X.shape

#y = np.array(training_data['relevancy'])
#y.shape

############# Save object: training data avg embed sum ################################################################
training_data['avg_embed_sum'] = X.tolist()
file_name = 'prepared_training_data.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(training_data, file)
    print(f'Object successfully saved to "{file_name}"')



##############################################################################
## TRAIN THE MODEL         ###################################################
##############################################################################

## Prepare data to feed into model
X = training_data['cosine_similarity'].to_numpy()
y = training_data['relevancy'].to_numpy()

## Subset some data firt
# X = X[0:10000]
# y = y[0:10000]

## The functions are robust to multivariate inputs -- hence reshape first
X = X.reshape(-1,1)

## Check data inputs
X.shape
y.shape

## Train the model
print('feeding data into logistic model...')
s = time.time()
weights, _ = log_regress(X, y)
print('Time to train model is ', time.time() - s)

## TAKE LAST SET OF WEIGHTS AS OPTIMISED WEIGHTS 
## This is our fitted model 
optimised_weights = weights[-1]


##############################################################################
############# RUN TRAINED MODEL ON VALIDATION DATA 
##############################################################################

valid_data = pd.read_table("validation_data.tsv")

## Repeat all steps for validation data: preprocess, embed, remove na
## Preprocess
print("Prepare validation data: preprocessing... ")
print('Cleaning the passages and queries ')
valid_data['preprocessed_passage'] = clean_list_strings(valid_data['passage'])
valid_data['preprocessed_queries'] = clean_list_strings(valid_data['queries'])

## Embed and compute average 
print("Embedding and taking average")
s = time.time()
valid_data['embedded_queries'] = valid_data['preprocessed_queries'].apply(compute_avg_embedding) ## splits into tokens, then turn all tokens into embeddings, then take average
valid_data['embedded_passages'] = valid_data['preprocessed_passage'].apply(compute_avg_embedding) ## splits into tokens, then turn all tokens into embeddings, then take average
print("Total time to embed the average for validation data queries and passages...", time.time() - s)

## Drop NAs before running through 
valid_data.dropna(subset=['embedded_queries', 'embedded_passages'], inplace=True)

## Check
valid_data['embedded_passages'].isna().sum()
valid_data['embedded_queries'].isna().sum()

## Get the X_validation (i.e. cosine similarity scores )
print("Get cos sim scores for validation data (X input)...")
s = time.time()

## List comprehension to get a vector of scores 
valid_data['cosine_similarity'] = [np_cos_similarity(row['embedded_queries'], row['embedded_passages']) for _,row in valid_data.iterrows()]

print("Time to compute cos sim scores for query-doc pairs..." ,time.time() - s)

## Check
valid_data['cosine_similarity']

##############################################################################
## RUN THE DATA THROUGH THE MODEL TO MAKE PREDICTIONS, USING OPTIMISED WEIGHTS
##############################################################################
X_valid = valid_data['cosine_similarity'].to_numpy()
X_valid = X_valid.reshape(-1,1)
y_valid = valid_data['relevancy'].to_numpy()
## Run the model
print("Running the model on validation data")
s = time.time()
valid_data['predicted_relevancy'] = log_step(X_valid, w = optimised_weights)
print("Time taken to run the model on the valid data...", time.time() - s)

## Apply thresholding to discretise 
valid_data['predicted_relevancy'].value_counts()
# - There are values which are very small but not 0 
# - therefore set them to 0 
#valid_data.loc[valid_data['predicted_relevancy'] <0.5,'predicted_relevancy']  = 0 
#valid_data.loc[valid_data['predicted_relevancy'] >0.5,'predicted_relevancy']  = 1 

## Check 
valid_data['predicted_relevancy'].value_counts() 


## CHECK VALIDATION DATA: WITH PREDICTION AND REAL RELEVACNY
valid_data.columns
valid_data_results_only = valid_data[['qid', 'pid', 'predicted_relevancy', 'relevancy']]

############# MODEL EVALUATION USING AP AND NDCG ################################################################

## Sort values by qid and relevancy scores
## GROUPS BY QUERY, THEN SORTS OUT BY PREDICTED SCORE WITHIN QUERY
valid_data_results_only.sort_values(['qid', 'predicted_relevancy'], ascending=[True, False])

## ========================================================================
## ========== AVERAGE PRECISION =============================================================
## ========================================================================
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


## FIND THE MEAN AP ACROSS ALL QUERIES 
print(MAP(valid_data_results_only, 10), 'MAP@10')
print(MAP(valid_data_results_only, 100), 'MAP@100')




## ========================================================================
## ========== NDCG =============================================================
## ========================================================================
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

print(NDCG(valid_data_results_only, 10), 'NDCG@10')
print(NDCG(valid_data_results_only, 100), 'NDCG@100')

########################################################################
########################################################################
########################################################################

########################################
############# Save object: valid data avg embed sum ################################################################
#valid_data['avg_embed_sum'] = X_valid.tolist()

## Subset for only necessary columns
#valid_data = valid_data[['qid', 'pid','avg_embed_sum','relevancy']]

## Save pickle
file_name = 'valid_data_with_avgembedsum.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(valid_data, file)
    print(f'Object successfully saved to "{file_name}"')


######################################################################
## More extensive way of doing hyperparam tuning : try more values
######################################################################

def best_LR_finder(lower_LR_log = -6, upper_LR_log = 2):
    num_intervals = np.abs(lower_LR_log)+ np.abs(upper_LR_log)+1
    learning_rates = np.logspace(lower_LR_log,upper_LR_log,num_intervals)
    final_losses = []
    number_iterations = []
    for i in learning_rates:
        print ("learning rate is: ",i)
        _, loss = log_regress(X, y, lr = i)
        final_losses.append(loss[-1])
        number_iterations.append(len(loss))

    lr_table = pd.DataFrame({
        'LR': learning_rates,
        'final_losses' : (final_losses),
        'number_iterations' : (number_iterations)})
    lr_table.sort_values(by = ['LR'])

    ## Plot

    # Plot the final loss against log(LR)
    fig, ax = plt.subplots()
    ax.scatter(x=np.log(lr_table['LR']), y=lr_table['final_losses'])
    ax.plot(np.log(lr_table['LR']), lr_table['final_losses'], color='red')
    ax.set_ylabel("Final (binary) loss")
    ax.set_xlabel("Log(Learning Rate)")
    ## Set location of x ticks
    ax.set_xticks(np.log(learning_rates))
    ax.set_xticklabels(np.linspace(lower_LR_log, upper_LR_log, num_intervals))
    plt.show()

    return(lr_table)

## Run hyperparam tuning
best_LR_finder()



#########################################################
#########################################################
## Now, use logistic model to rerank the passages and queries
## in candidate dataset 
#########################################################
#########################################################

test_set = pd.read_table('candidate_passages_top1000.tsv',
                         header=None,
                         names = ['qid', 'pid', 'query', 'passage'])
test_set = test_set.groupby('qid').filter(lambda x: len(x) == 100)

## For submitting later -- need the proper order of qid 
proper_qid_order = test_set['qid']


test_set.head()

## ----- 1. Clean the passages and queries
test_set['preprocessed_passage'] = clean_list_strings(test_set['passage'])
test_set['preprocessed_queries'] = clean_list_strings(test_set['query'])

## ----- 2. Embed and take average
test_set['embedded_queries'] = test_set['preprocessed_queries'].apply(compute_avg_embedding) ## splits into tokens, then turn all tokens into embeddings, then take average
test_set['embedded_passages'] = test_set['preprocessed_passage'].apply(compute_avg_embedding) ## splits into tokens, then turn all tokens into embeddings, then take average

## - check embeddings 
test_set['embedded_passages'].isna().sum()
test_set['embedded_queries'].isna().sum()

test_set[test_set['embedded_queries'].isna()]['query'].drop_duplicates()
test_set[test_set['embedded_passages'].isna()]['query'].drop_duplicates()

## -- drop na  
test_set.dropna(subset=['embedded_queries', 'embedded_passages'], inplace=True)

## Check
test_set['embedded_passages'].isna().sum()
test_set['embedded_queries'].isna().sum()

## Get cosine similarity scores
test_set['cosine_similarity'] = [np_cos_similarity(row['embedded_queries'], row['embedded_passages']) for _,row in test_set.iterrows()]

## Check
test_set['cosine_similarity']

## Run through the model to get predicted relevancy
X_test = test_set['cosine_similarity'].to_numpy()
X_test = X_test.reshape(-1,1)
test_set['predicted_relevancy'] = log_step(X_test, w = optimised_weights)

## Check
test_set['predicted_relevancy'].value_counts()
test_set['predicted_relevancy'].sum()

## Sort accoriding to the initial order of qids, and sort in descending order of scores
## Create a new category datatype from pandas for the right order
proper_qid_order = proper_qid_order.unique()
category_dtype = pd.api.types.CategoricalDtype(categories=proper_qid_order,
                                                ordered=True)
## Convert our query index column to this data type (categorical)
test_set['qid'] = test_set['qid'].astype(category_dtype)
# FINALLY: ARRANGE! 
test_set = test_set.sort_values('qid').reset_index(drop=True)
test_set = test_set.groupby('qid', sort = False).apply(lambda x: x.sort_values('predicted_relevancy', ascending=False))[:100].reset_index(drop=True)

## Insert 2nd col: A2, insert 
test_set['A2'] = 'A2'

## Insert ranking 
test_set['ranking'] = test_set.groupby('qid', sort=False).cumcount() + 1

## Insert algo type 
test_set['Algo'] = 'LR'

test_set = test_set[['qid','A2','pid','ranking','predicted_relevancy',
                     'Algo']]
test_set.to_csv("LR.txt", index = False, header = False,
                sep=" ")

print("test set printed to txt")