from os import listdir
import math
import os
from os.path import isfile, join
import jsonlines
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import unicodedata
import numpy as np
import random
import csv
import pickle
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import pearsonr, spearmanr

from nltk.tokenize import RegexpTokenizer

valid_file = 'fever.dev.jsonl'
train_file = 'fever.train.jsonl'

TOP_N = 20
MIN_FREQ = 5
NGRAM = 2

def get_single_stopwords(dataset, ngram=1):

    tokenizer = RegexpTokenizer(r'\w+')
    fp = open(dataset, "r", encoding='utf-8')
    reader = jsonlines.Reader(fp)

    global_word_counter = defaultdict(int)
    phrases = 0

    for dictionary in tqdm(reader.iter()):
        claim = dictionary['claim']

        words = tokenizer.tokenize(claim.lower())
       
        for i  in range(len(words) + 1):
            if i > ngram - 1:
                phrase = ' '.join(words[i - ngram:i])
                global_word_counter[phrase] += 1

    counter = Counter(global_word_counter)
    stop_words = counter.most_common(10)
    print (sum(counter.values()))
    stop_words = [word[0] for word in stop_words]
    print (stop_words)
    
    return stop_words

def get_counters(dataset, ngram=NGRAM):

    stop_words = get_single_stopwords(train_file, ngram=1)

    tokenizer = RegexpTokenizer(r'\w+')
    fp = open(dataset, "r", encoding='utf-8')
    reader = jsonlines.Reader(fp)

    global_word_counter = defaultdict(int)
    global_label_counter = defaultdict(int)
    phrases = 0

    label_word_counter = defaultdict(lambda: defaultdict(int)) 

    for dictionary in tqdm(reader.iter()):
        label = dictionary['label']
        claim = dictionary['claim']

        #words = word_tokenize(claim.lower())
        words = tokenizer.tokenize(claim.lower())
        words = [words[i] for i in range(len(words)) if words[i] not in stop_words]

        bigrams = ngrams(words, NGRAM)
        
        """ 
        for word in words:
            global_word_counter[word] += 1
            global_label_counter[label] += 1
            label_word_counter[label][word] += 1
            phrases += 1
        
        """

        for bigram in bigrams:
            bigram = ' '.join(bigram)
                
            global_word_counter[bigram] += 1
            global_label_counter[label] += 1
            label_word_counter[label][bigram] += 1
            phrases += 1
        
        
        """
        for i  in range(len(words) + 1):
            if i > ngram - 1:
                phrase = ' '.join(words[i - ngram:i])
                global_word_counter[phrase] += 1
                global_label_counter[label] += 1
                label_word_counter[label][phrase] += 1
                phrases += 1
        """

    
    print ('Total count: ' + str(phrases))
    return global_word_counter, label_word_counter, global_label_counter, phrases

valid_global_word_counter, valid_label_word_counter, valid_global_label_counter, valid_words = get_counters(valid_file)
train_global_word_counter, train_label_word_counter, train_global_label_counter, train_words = get_counters(train_file)

corr = {'SUPPORTS': [], 'REFUTES': [], 'NOT ENOUGH INFO': []}

for label in train_label_word_counter.keys():
    words = []
    scores = []
    pmis = []
    valid_pmis = []
    valid_scores = []
    freqs = []
    valid_freqs = []
    p_l_train = train_global_label_counter[label] / train_words
    p_l_valid = valid_global_label_counter[label] / valid_words
    print (train_words)

    word_counter = train_label_word_counter[label]
    for w in word_counter:
        if train_global_word_counter[w] < MIN_FREQ:
            continue

        # p(label | word)
        score = word_counter[w] / train_global_word_counter[w]
        pmi = math.log(score / p_l_train)
        #pmi = max(0, pmi)

        if w in valid_global_word_counter:
            valid_score = valid_label_word_counter[label][w] / valid_global_word_counter[w]
            if valid_score == 0:
                valid_pmi = float('inf')
            else:
                valid_pmi = math.log(valid_score / p_l_valid)
                #valid_pmi = max(0, math.log(valid_score / p_l_valid))
        else:
            valid_score = 0
            valid_pmi = float('inf')

        words.append(w)
        scores.append(score)
        pmis.append(pmi)
        freqs.append(word_counter[w])
        valid_freqs.append(valid_label_word_counter[label][w])
        valid_scores.append(valid_score)
        valid_pmis.append(valid_pmi)

    assert(len(words) == len(scores) == len(freqs) == len(valid_freqs) == len(valid_scores) == len(pmis))

    pmis_x_freq = list(np.array(pmis)*freqs/train_words)
    valid_pmis_x_freq = list(np.array(valid_pmis)*valid_freqs/valid_words)
    pmis_x_freq, pmis, scores, freqs, words, valid_scores, valid_pmis, valid_pmis_x_freq, valid_freqs = (list(t) for t in zip(*sorted(zip(pmis_x_freq, pmis, scores, freqs, words, valid_scores, valid_pmis, valid_pmis_x_freq, valid_freqs), reverse=True)))

    print("")
    print("---- {}".format(label))
    print("{:20} | {:7} | {:5} | {:7} | {:5}".format('word', 'lmi', 'p(l|w)', 'valid_lmi', 'valid_p(l|w)'))

    #print("{:20} | {:6} | {:7} | {:7} | {:4} | {:11} | {:10} | {:10} | {:10}".format('word', 'score', 'pmi', 'lmi', 'freq', 'valid score', 'valid_pmi', 'valid_lmi', 'valid freq'))
    print ("-"*80)

    #filepath = 'top_20_lmi_p_2_' + label + '.csv'
    filepath = 'top_1000_unigram_' + label + '.csv'
    with open(filepath, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(min(TOP_N, len(words))):
            #print("{:20} | {:6} | {:7} | {:7} | {:4} | {:11} | {:10} | {:10} | {:10}".format(words[i], round(scores[i], 3), round(pmis[i],3), round(pmis_x_freq[i],3), freqs[i], round(valid_scores[i],3), round(valid_pmis[i],3), round(valid_pmis_x_freq[i],3), valid_freqs[i]))
            if not math.isnan(valid_pmis_x_freq[i]): 
                print("{:20} | {:7} | {:5} | {:7} | {:5}".format(words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), int(round(valid_pmis_x_freq[i]*10**6)), round(valid_scores[i],2)))
                csv_writer.writerow([words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), freqs[i], int(round(valid_pmis_x_freq[i]*10**6)), round(valid_scores[i],2), valid_freqs[i]])
            else:
                print("{:20} | {:7} | {:5} | {:7} | {:5}".format(words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), valid_pmis_x_freq[i], round(valid_scores[i],2)))
                csv_writer.writerow([words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), freqs[i], valid_pmis_x_freq[i], round(valid_scores[i],2), valid_freqs[i]])

        '''
        extra_words = ['did not', 'yet to', 'does not', 'refused to', 'failed to', 'unable to', 'incapable being', 'united states', 'least one', 'at least', 'person who', 'stars actor', 'least one', 'won award', 'played for']
        #extra_words = ['at least one']
        for w in extra_words:
            i = words.index(w)
            if not math.isnan(valid_pmis_x_freq[i]): 
                print("{:20} | {:7} | {:5} | {:7} | {:5}".format(words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), int(round(valid_pmis_x_freq[i]*10**6)), round(valid_scores[i],2)))
                csv_writer.writerow([words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), freqs[i], int(round(valid_pmis_x_freq[i]*10**6)), round(valid_scores[i],2), valid_freqs[i]])
            else:
                print("{:20} | {:7} | {:5} | {:7} | {:5}".format(words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), valid_pmis_x_freq[i], round(valid_scores[i],2)))
                csv_writer.writerow([words[i], int(round(pmis_x_freq[i]*10**6)), round(scores[i],2), freqs[i], valid_pmis_x_freq[i], round(valid_scores[i],2), valid_freqs[i]])
        '''
    
    
    limits = [10, 20, 50, 100, 200, 500, 1000]
    #corr_filepath = 'corr_unigram_1000.pkl'
    corr_ind = []
    for limit in limits:
        pears = pearsonr(scores[0:limit], valid_scores[0:limit])
        print ("pearson correlation for top {}: {} (p-value: {})".format(limit, round(pears[0],3), round(pears[1],3)))
        corr_ind.append(round(pears[0],3))

        spear = spearmanr(scores[0:limit], valid_scores[0:limit])
        print ("spearman correlation for top {}: {} (p-value: {})".format(limit, round(spear[0],3), round(spear[1],3)))

    corr[label] = [limits, corr_ind]

    #pickle.dump(corr, open(corr_filepath, 'wb'))


