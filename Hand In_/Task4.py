import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import pickle
## Load in data with average word embeddings 
file_name = 'prepared_training_data.pkl'
with open(file_name, 'rb') as file:
	train_data = pickle.load(file)


##############################################################
##############################################################
## Pytorch network 
##############################################################
##############################################################


class RelevancyPredictionNN(nn.Module):
	def __init__(self, num_filters, kernel_size, hidden_size,
				 pooling_size):

		'''
		** My Plan: 
		- one entry for query vector
		- one entry for passage vector
		- join the two of them (add or cat?) --> then fully connected layer 

		Note that: vector size - kernel size + 1 before max pooling
		max pooling: /2 (max pooling size)
		'''
		
		super().__init__()

		## ENTRY 1: Query vector
		## Convolution layer -- identify patterns
		self.conv_query = nn.Sequential(
			## in channels == number of colours for images -- just 1 for this case 
			nn.Conv1d(in_channels=1, 
					  out_channels=num_filters, ## out channels == number of kernels/filters 
					  kernel_size=kernel_size), ## size of the 'stamp'
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=pooling_size) ## CHOOSE 2 SO THAT IT IS EASIER TO FIND INPUT DIM FOR NEXT STAGE
		)
		## 1.2 fully connected layer after 
		self.fc_query = nn.Sequential(
			nn.Linear((num_filters * (100 - kernel_size + 1) // pooling_size), ## IMPORTANT: 49 BECAUSE MAX POOLING: 100 - 3 + 1 = 98, 98/2 = 49 
					  hidden_size),
		nn.ReLU()
		)

		## ENTRY 2: Passage vector
		## Convolution layer -- identify patterns
		self.conv_passage = nn.Sequential(
			nn.Conv1d(in_channels=1, 
					  out_channels=num_filters, 
					  kernel_size=kernel_size),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=pooling_size) ##
		)
		## 2.2 fully connected layer after
		self.fc_passage = nn.Sequential(
			nn.Linear((num_filters * (100 - kernel_size + 1) // pooling_size), ## IMPORTANT: 49 BECAUSE MAX POOLING: 100 - 3 + 1 = 98, 98/2 = 49 
					  hidden_size),
		)

		## JOINT FINAL output layer
		## 2 layers?
		self.output = nn.Sequential(
			nn.Linear(hidden_size * 2, hidden_size),
			nn.ReLU(),


			## Final output size is just 1 -- scalar
			nn.Linear(hidden_size, 1),
			nn.Sigmoid() ## Binary output
		)

	def forward(self, x_query, x_passage):

		## FOR BATCHING STEP LATER -- NEED 3 DIMENSIONS (BATCH, inchannel, vector_size)
		## Need to increase by 1 dimension 
		
		# x_query = x_query.unsqueeze()  
		# x_passage = x_passage.unsqueeze() 
		x_query = x_query.unsqueeze(1)  ## shape: (batch_size, 1, 100)
		x_passage = x_passage.unsqueeze(1)  ## shape: (batch_size, 1, 100)

		## Convolution for QUERY
		x_query = self.conv_query(x_query)
		
		## INPUT FOR FC: (BATCH SIZE, NO KERNELS, 49)
		## FLATTEN TO : (BATCH SIZE, --), -- = no. kernels x 49
		## NEED TO FLATTEN 
		x_query = x_query.view(x_query.size(0), -1)
		x_query = self.fc_query(x_query)

		## Convolution for PASSAGE
		x_passage = self.conv_passage(x_passage)

		## NEED TO FLATTEN
		x_passage = x_passage.view(x_passage.size(0), -1)
		x_passage = self.fc_passage(x_passage)

		## Add/concat, then run through final layer 
		x = torch.cat((x_query, x_passage), dim=1)
		x = self.output(x)
		return x



## Split into inputs and targets
x_query = torch.tensor(train_data['embedded_queries'].tolist())
x_passage = torch.tensor(train_data['embedded_passages'].tolist())
y_train = torch.tensor(train_data['relevancy'].tolist(), dtype=torch.float32)

## Instantiate the model
## DONT CHANGE POOLING SIZE = 2 
## 
my_model = RelevancyPredictionNN(num_filters=10, ## have 10 filters/ kernels 
								 kernel_size=5,  ## try 3? can change 
								 hidden_size=64,
								 pooling_size=2) ## hidden layer

## PREPARE FOR TRAINING: loss and optimiser 
loss_function = nn.BCELoss()
optimiser = optim.SGD(my_model.parameters(), lr=0.001)

## Create DataLoader for our data - add batching 
train_dataset = torch.utils.data.TensorDataset(x_query, x_passage, y_train)
train_dataloader = DataLoader(train_dataset, 
							  batch_size=32, ## can change?
							  shuffle=True)

## Train model!!
num_epochs = 5
for epoch in range(num_epochs):
	## iterate through each batch
	for batch_idx, (batch_x_query, batch_x_passage, batch_y) in enumerate(train_dataloader):
		optimiser.zero_grad() ## 0 out saved gradients 

		## forward prop
		output = my_model(batch_x_query, batch_x_passage)
		## find loss
		## BCE loss -- try view(-1,1)
		loss = loss_function(output, batch_y.view(-1, 1))
		## backprop
		loss.backward()
		## update
		optimiser.step()

	## Print loss after each epoch
	print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

my_model

##############################################################
##############################################################
## Test on validation data 
##############################################################
##############################################################

## ===== Prepare validation data
file_name = 'valid_data_with_avgembedsum.pkl'
with open(file_name, 'rb') as file:
    valid_data = pickle.load(file)

## Check 
valid_data.columns

## 1. Turn data into tensors 
valid_query = torch.tensor(valid_data['embedded_queries'].tolist())
valid_passage = torch.tensor(valid_data['embedded_passages'].tolist())

## 2. Run Model on data tensors  
print(my_model)
with torch.no_grad():
	## make predict 
	valid_predicted_relevancy = my_model(valid_query, valid_passage)

## Check outputs 
valid_predicted_relevancy.unique(return_counts = True)

## Dont actually have to use thresholding -- we just want the model to give us 
# some ranking that we can use 

valid_data['nn_predicted_relevancy'] = valid_predicted_relevancy


##############################################################
##############################################################
## EVALUATION OF NN PREDICTED RELEVANCY
##############################################################
##############################################################
valid_data.sort_values(['qid', 'nn_predicted_relevancy'], ascending=[True, False])

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
print(MAP(valid_data, 10), 'MAP@10')
print(MAP(valid_data, 100), 'MAP@100')

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

print(NDCG(valid_data, 10), 'NDCG@10')
print(NDCG(valid_data, 100), 'NDCG@100')

## ========================================================================
## ========== USE MODEL TO PREDICT AND RANK TEST DATA =============================================================
## ========================================================================
test_set = pd.read_table('candidate_passages_top1000.tsv',
                         header=None,
                         names = ['qid', 'pid', 'query', 'passage'])
## For submitting later -- need the proper order of qid 
proper_qid_order = pd.read_table('candidate_passages_top1000.tsv',
                                 header=None)[0]

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

## vector
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

## 1. Turn data into tensors 
test_query = torch.tensor(test_set['embedded_queries'].tolist())
test_passage = torch.tensor(test_set['embedded_passages'].tolist())

with torch.no_grad():
	## make predict 
	test_predicted_relevancy = my_model(test_query, test_passage)

## Append to dataframe 
test_set['predicted_relevancy'] = test_predicted_relevancy

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
test = test_set.groupby('qid', sort = False).apply(lambda x: x.sort_values('predicted_relevancy', ascending=False))
test_set = test_set[['qid','pid','predicted_relevancy']]
test_set.to_csv("NN.csv", index = False, header = False)


