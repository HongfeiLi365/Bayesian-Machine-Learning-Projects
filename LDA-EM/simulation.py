# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 19:06:20 2018

@author: HL
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import reuters_main
import lda
import time
from matplotlib import pyplot as plt

np.random.seed(1)
num_docs = 10
topics = ['a','b','c','d']
K = len(topics)
doc_length = 20

alpha = [1]*4
vocabulary = list(range(80))
V = len(vocabulary)
eta = np.random.rand(V)

beta = np.zeros((K,V))
for i in range(K):
    beta[i,:]= np.random.dirichlet(eta)
    
    
theta = np.zeros((num_docs, K))
z = np.zeros((num_docs, doc_length), dtype=np.int16)
for i in range(num_docs):
    theta[i,:] = np.random.dirichlet(alpha)
    for j in range(doc_length):
        z[i,j] = np.where(np.random.multinomial(1,theta[i,:])==1)[0][0]

s_corpus = np.zeros((num_docs, doc_length), dtype=np.int16)
for i in range(num_docs):
    for j in range(doc_length):
        s_corpus[i,j] = int(np.where(np.random.multinomial(1,beta[int(z[i,j]),:])==1)[0][0])

s_doc_term_mat = np.zeros((num_docs,V), dtype=np.int32)
for i in range(num_docs):
    for word in list(s_corpus[i,:]):
        s_doc_term_mat[i, word] += 1 

    
WS, DS = lda.utils.matrix_to_lists(s_doc_term_mat)
    
my_lda_start = time.time()
model = reuters_main.LDA_Model(num_topics = K, num_terms = V)      

docs = []
for d in range(num_docs):
    docs.append(reuters_main.Document(doc_id = None, K = model.num_topics, words_id = WS[DS==d], 
                         label = theta[d,:]))

reuters_main.EM(docs,model,EM_MAX_ITER = 700)
     
my_lda_end = time.time()
my_lda_time = my_lda_end-my_lda_start
print("My LDA runtime: {}".format(my_lda_time))

def get_gamma_label(docs):
    X = []
    Y = []
    for i in range(num_docs):
        #print('gamma: {} \n label: {}'.format(docs[i].gamma, docs[i].label))
        X.append(docs[i].gamma)
        Y.append(docs[i].label)
    return X, Y

gammas, Y = get_gamma_label(docs)
    
x_axis = [1,2,3,4]


fig = plt.figure()
#plt.title('Topics distribution for first 9 documents')
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    ax.plot(x_axis, softmax(gammas[i]),'r-' , x_axis, Y[i], 'b-')
    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

#plt.axis('off')