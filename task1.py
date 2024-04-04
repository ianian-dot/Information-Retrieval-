import pandas as pd
from collections import Counter
import re
import time
import numpy as np
import pickle
import string
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from nltk.corpus import stopwords


## Task 1 data 
collection = pd.read_table("passage-collection.txt",
                          header=None) ## no real index
# Convert to list
list_sentences = [collection.values[i][0] for i in range(len(collection))]

## Some functions ===============================
# =============================================== 
## Function 1 : More important for later parts, where preserving the sentences 
# is important 
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

## Function 2: creates a single list, appends all possible words to it
# cleans it (lower case, removes non alpha characters, removes full digits)
def clean1(list_of_strings):
    corpus_list = []
    for i in range(len(list_of_strings)):
        word = list_of_strings[i].lower().split()  # lowercase + token
        corpus_list += word
    corpuslist = [re.sub(r'[^A-Za-z0-9 ]+', '', word)
                         for word in corpus_list if not word.isdigit() or word == '2010']  # removes weird charac
    return(corpuslist)

## Function 3: Extra function because upon seeing results, we still ended up with a lot of 'weird words'
def removeweirdwords(list_of_words):
    return [word for word in list_of_words if word.isalpha() or word == 'ar15' or word == 'ww1']
# ===============================================
# ===============================================


## Cleaning list of sentences 
s = time.time()
maincorpus1 = clean1(list_sentences)
maincorpus11 = removeweirdwords(maincorpus1)
print("time to clean list sentences, ", time.time() - s)
## --------------------------------------------------



## IMPLEMENT FUNCTION THAT 
# 1. COUNTS # OCCURRENCES OF TERMS IN THE DATASET 
# 2. PLOTS PROB OF OCCURRENCE (NORMALISED FREQ) AGAINST RANKING 

### VERSION 2 
def without_stopwords(word_freq): 
    # REMOVE STOPWORDS!
    new_word_freq = word_freq
    stopwordsset = set(stopwords.words('english'))

    # remove them from our word count
    for word in stopwordsset:
        if word in new_word_freq:
            del new_word_freq[word]

    # Re-sort everything, renormalise + add word count
    sorted_word_counter_list = sorted(
        new_word_freq.items(), key=lambda x: x[1], reverse=True)
    word_counter_df = pd.DataFrame(
        sorted_word_counter_list, columns=["word", "freq"])
    word_counter_df["rank"] = range(1+0, 1+len(word_counter_df))
    word_counter_df['freq'] = word_counter_df['freq']/sum(word_counter_df['freq'])

    y = (word_counter_df["freq"])
    x = (word_counter_df["rank"])
    C = sum([(1/x) for x in word_counter_df["rank"]])

    return x,y,C

def counter_and_plotter():
    ## Step 1: Counter 
    word_freq = Counter()
    word_freq.update(maincorpus11)  # saved as a dictionary type

    ## Step1.1: Vocab list 
    vocab_list = list(word_freq.keys())
    print(f"The vocabulary list length is {len(vocab_list)}")

    ## Step 2.1: preprocessing the data for plotting 
    # Sort out based on word count
    sorted_word_counter_list = sorted(
        word_freq.items(), key=lambda x: x[1], reverse=True)
    word_counter_df = pd.DataFrame(
    sorted_word_counter_list, columns=["word", "freq"])
    # Add rank column for Zipfs law
    word_counter_df["rank"] = range(1+0, 1+len(word_freq))

    # Normalise the frequency! I.e. get density
    word_counter_df['freq'] = word_counter_df['freq']/sum(word_counter_df['freq'])
    y_log = np.log(word_counter_df["freq"])
    x_log = np.log(word_counter_df["rank"])
    C = sum([(1/x) for x in word_counter_df["rank"]])
    theoretical_y_log = -x_log-np.log(C)

    ## Step 2.2: Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    # Plot original Zipfs law (no log)
    axs[0,0].scatter(x=word_counter_df["rank"], y=word_counter_df["freq"])
    axs[0,0].set_ylabel("Word Frequency Normalised")
    axs[0,0].set_xlabel("Word rank")
    axs[0,0].set_title("With stopwords: Raw data / Double log")
    ## theoretical line 
    axs[0,0].plot(word_counter_df["rank"], 1/(C*word_counter_df['rank']), color='red')

    # Plot linear scatterplot with theoretical line
    axs[1,0].scatter(x_log, y_log)
    axs[1,0].set_xlabel("Log(Word Rank)")
    axs[1,0].set_ylabel("Log(Word Frequency)")
    axs[1,0].plot(x_log, theoretical_y_log, color="red")

    ## Quantify difference 
    mse_with_stopwords = mean_squared_error(y_log, theoretical_y_log)
    mape_with_stopwords = mean_absolute_percentage_error(y_log, theoretical_y_log)
    rscore_with_stopwords = r2_score(y_log, theoretical_y_log)
    print(f"The R^2 between our data and Zipfs line is with stopwords: {rscore_with_stopwords}]\n")
    print(f"The MSE difference is with stopwords: {mse_with_stopwords}]\n")
    print(f"The MAPE (Mean Absolute Percentage Error) difference is with stopwords: {mape_with_stopwords}]\n")

    ## PART 2: REMOVING STOPWORDS 
    x_without_stopwords, y_without_stopwords, C_without_stopwords = without_stopwords(word_freq)

    # Plot without stopwords
    axs[0,1].scatter(x_without_stopwords, y_without_stopwords)
    axs[0,1].set_xlabel("Word Rank")
    axs[0,1].set_ylabel("Word Frequency Normalised")
    ## theoretical line 
    axs[0,1].plot(x_without_stopwords, 1/(C_without_stopwords*x_without_stopwords), color = 'red')
    axs[0,1].set_title("Without stopwords: Raw data / Double log")

    # Plot linear scatterplot with theoretical line
    xws_log = np.log(x_without_stopwords)
    yws_log = np.log(y_without_stopwords)
    theoretical_yws_log = -xws_log - np.log(C_without_stopwords)
    
    axs[1,1].scatter(xws_log, yws_log)
    axs[1,1].set_xlabel("Log(Word Rank)")
    axs[1,1].set_ylabel("Log(Word Frequency)")
    ## theoretical line 
    axs[1,1].plot(xws_log, theoretical_yws_log, color = 'red')

    ## Quantify difference 
    mse_without_stopwords = mean_squared_error(yws_log,theoretical_yws_log )
    mape_without_stopwords = mean_absolute_percentage_error(yws_log, theoretical_yws_log)
    rscore_with_stopwords = r2_score(yws_log, theoretical_yws_log)
    print(f"The R^2 between our data and Zipfs line is with stopwords: {rscore_with_stopwords}]\n")
    print(f"The MSE difference is without stopwords: {mse_without_stopwords}")
    print(f"The MAPE difference is without stopwords: {mape_without_stopwords}")

    ## Save plots 
    plt.savefig('zipfplots.pdf')

    ## Task 1 outputs: vocabulary list
    ## -------------------------------------------------
    ## PICKLE -------------------------------------------------
    file_name = 'vocab_list.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(vocab_list, file)
        print(f'Object successfully saved to "{file_name}"')
    ## PICKLE -------------------------------------------------
    ## -------------------------------------------------

    ## Plot all 
    plt.show()


## RUN THE FUNCTIONS : GET PLOTS AND MSE DIFFERENCE
counter_and_plotter()
