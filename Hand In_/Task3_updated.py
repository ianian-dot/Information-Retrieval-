import pandas as pd
import pickle
import time 
import numpy as np

## ================ Use same word embeddings as before
## ================ Reload TRAINING avgembedsum and VALID avgembedsum 
file_name = 'training_data_with_avgembedsum.pkl'
with open(file_name, 'rb') as file:
    train_data = pickle.load(file)
file_name = 'valid_data_with_avgembedsum.pkl'
with open(file_name, 'rb') as file:
    valid_data = pickle.load(file)


## Check data
train_data.columns
valid_data.columns

## ================ Bring in LambdaMart model
import xgboost as xgb


## ================ Preparing the data: training x data, grouping by query, training y data 
## Sort training data by query
train_data = train_data.sort_values(['qid'], ascending = [True])
## Group the training data by query -- this is for the XGB ranker model
groups = train_data.groupby('qid')['pid'].count().to_numpy()

model = xgb.XGBRanker(  
    objective='rank:ndcg', ## objective function to train the model
    random_state=42, ## set seed 
    eta=0.05, ## similar to learning rate
    max_depth=6, ## max depth of the tree -- complexity of thre tree 
    n_estimators=100, ## number of trees -- final pred is combination of all predictions 
    subsample=0.75 ## not nec but can experiment?
    )

## ================ FIT MODEL
X_train = (np.array(train_data['avg_embed_sum'].to_list()))
y_train = (train_data['relevancy'])

s = time.time()
model.fit(X_train,y_train, group = groups)
print("time taken to fit lambdamart model ", time.time()-s)

## --- Save model using pickle 
with open('xgb_ranker_model', 'wb') as file:
    pickle.dump(model, file)


## ================ TAKE MODEL, TEST ON VALIDATION DATA, SEE AP AND NCDG
## (Do the same as before)
## Sort the valid data by qid before getting group numbers via groupby
valid_data = valid_data.sort_values(['qid'], ascending = [True])
valid_groups = valid_data.groupby('qid')['pid'].count().to_numpy()

## Prepare the input for the model from valid data
X_test = np.array(valid_data['avg_embed_sum'].to_list())
y_test = np.array(valid_data['relevancy'])
qid_test = valid_data['qid']
d_test = xgb.DMatrix(X_test, label=y_test)
d_test.set_group(valid_groups)

## Run the model for predictions 
valid_data['xgb_pred_relevancy'] = model.predict(d_test)


## ====== Sort the dataframe according to predicted relevancy to be able to run the evaluation
valid_data_sorted = valid_data.sort_values(['qid', 'xgb_pred_relevancy'], 
                                           ascending = [True, False])
## === AP evaluation
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
print(MAP(valid_data_sorted, 10), 'MAP@10')
print(MAP(valid_data_sorted, 100), 'MAP@100')

## === NDCG evaluation
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

print(NDCG(valid_data_sorted, 10), 'NDCG@10')
print(NDCG(valid_data_sorted, 100), 'NDCG@100')
## =========================================================
## ================ HYPERPARAMETER TUNING ==================
## =========================================================

## the parameters we had used were: 
# eta (learning rate) = 0.05, 
# maxdepth (tree complexity) = 6 
# n_est (number of trees) = 100
# subsample = 0.75

## use gridsearch to find the best parameter values, and use cross validation to compare the models 
from sklearn.model_selection import GridSearchCV

params_grid = {
    'max_depth': [4, 5, 7 ,8],
    'learning_rate': [ 0.005, 0.5, 1],
    'n_estimators': [50, 75, 200],
    'gamma': [ 0, 0.1, 1],
    'subsample' : [0.5, 1]
}

## Instantiate
model_tester = xgb.XGBRanker()

ndcg_grid = GridSearchCV(estimator = model_tester,
                        param_grid= params_grid,
                        scoring = 'ndcg', 
                        cv = 5)
## Fit the gridsearchCV to the valid data - it contains the model structure which will be trained and tested on the data
ndcg_grid.fit(X_test, y_test, valid_groups)

## Best model
optimal_param_model = ndcg_grid.best_estimator_
print(optimal_param_model)


