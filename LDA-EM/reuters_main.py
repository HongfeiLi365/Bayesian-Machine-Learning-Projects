# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:00:25 2018

@author: HL
"""

from nltk.corpus import reuters, stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from scipy.special import digamma
import numpy as np
import lda
from reuters_utils import softmax, log_gamma, log_sum, trigamma, opt_alpha
import time
from matplotlib import pyplot as plt
#reuters.fileids()[-10:]
#len(reuters.categories())
#reuters.categories('training/9989')
#print(*reuters.words(idlist[-1]), sep = '\n')
    

class LDA_Model:
    def __init__(self, num_topics, num_terms, alpha=1.0):
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.alpha = alpha
        self.log_prob_w = np.random.random([num_topics, num_terms]) + 1/num_terms
  

class Document:
    def __init__(self,doc_id, words_id, K,label = None):
        # words_id: a list of positions of words in vocabulary
        self.words = words_id
        self.length = len(words_id)
        self.terms, self.term_counts = np.unique(words_id, return_counts = True)
        self.num_terms = len(self.terms)
        self.doc_id = doc_id
        self.label = label
        self.gamma = np.zeros(K)
        self.phi = np.zeros([self.num_terms,K])

class LDA_Suffstats:
    def __init__(self,model):
        self.class_total = np.zeros(model.num_topics)
        self.class_term = np.zeros([model.num_topics, model.num_terms])
        self.num_docs = 0
        self.alpha_suffstats = 0


def compute_likelihood(doc, model, phi, gamma):
    likelihood = 0
    dig = digamma(gamma)
    gamma_sum = np.sum(gamma)
    digsum = digamma(gamma_sum)
    likelihood = (likelihood + log_gamma(model.alpha * model.num_topics) 
                             - model.num_topics * log_gamma(model.alpha)
                             - log_gamma(gamma_sum))
    
    likelihood = (likelihood + model.alpha * np.sum(dig - digsum)
                             - np.sum((gamma - 1) * (dig - digsum)))

    for k in range(model.num_topics):
        likelihood += log_gamma(gamma[k])

    for n in range(doc.num_terms):
        for k in range(model.num_topics):
            if phi[n,k] > 0:
                likelihood = (likelihood + doc.term_counts[n] * (phi[n,k] * ((dig[k] - digsum) - np.log(phi[n,k]) + model.log_prob_w[k,doc.terms[n]])))
                            
    return likelihood


def variational_inference(doc, model,  max_iter, converge_thresh = 1e-6 ):
    '''
    doc: the dth document
    gamma: 1 x K (number_of_topics)
    digamma_gamma: 1 x K
    phi: N_d(number of terms in dth document) x K
    '''
    gamma = doc.gamma
    phi = doc.phi
    converged = 1
    phisum = 0
    likelihood = 0
    likelihood_old = 0
    oldphi = np.zeros([model.num_topics])
    digamma_gamma = np.zeros([model.num_topics])
    
    # initialization of gamma and phi
    gamma = np.ones(model.num_topics) * (model.alpha + doc.length / model.num_topics)
    digamma_gamma = digamma(gamma)
    phi = np.ones([doc.num_terms,model.num_topics]) / model.num_topics
    #print('digamma:{}'.format(digamma_gamma))
    # inference loop
    i_iter = 0
    while (converged > converge_thresh) and (i_iter < max_iter or max_iter == -1):
        i_iter = i_iter + 1
        for n in range(doc.num_terms):
            phisum = 0
            for k in range(model.num_topics):     
                oldphi[k] = phi[n,k]
                phi[n,k] = digamma_gamma[k] + model.log_prob_w[k, doc.terms[n]]

                if k > 0:
                    phisum = log_sum(phisum, phi[n,k])
                else:
                    phisum = phi[n, k]
                    
            #End For
            
            phi[n,:] = np.exp(phi[n,:] - phisum)
            gamma = gamma +  doc.term_counts[n] * (phi[n,:] - oldphi)
            digamma_gamma = digamma(gamma)

        #End For
        likelihood = compute_likelihood(doc, model, phi, gamma)
        if i_iter > 1:
            converged = (likelihood_old - likelihood) / likelihood_old
        likelihood_old = likelihood
        
    #End while
    #print("[Variational Inference Summary] Iteration: {0} Likelihood: {1:.0f} Coverged: {2:.8f}".format(i_iter, likelihood, converged), file=text_file)
    
    doc.gamma = gamma
    doc.phi = phi
    return likelihood # Check: scope of gamma and phi


def doc_e_step(doc, model, ss, VAR_MAX_ITER):
    
    likelihood = variational_inference(doc, model, max_iter = VAR_MAX_ITER);

    # update sufficient statistics

    gamma_sum = np.sum(doc.gamma)
    ss.alpha_suffstats = (ss.alpha_suffstats + np.sum(digamma(doc.gamma))
                           - model.num_topics * digamma(gamma_sum))

    for n in range(doc.num_terms):
        for k in range(model.num_topics):
            ss.class_term[k,doc.terms[n]] = ss.class_term[k,doc.terms[n]]+ doc.term_counts[n]*doc.phi[n,k]
            ss.class_total[k] = ss.class_total[k] + doc.term_counts[n]*doc.phi[n,k]
    
    ss.num_docs = ss.num_docs + 1
    
    return likelihood



def lda_mle(model, ss, estimate_alpha):
    for k in range(model.num_topics):
        for w in range( model.num_terms):
            if (ss.class_term[k,w] > 0):
                model.log_prob_w[k,w] =np.log(ss.class_term[k,w]) - np.log(ss.class_total[k])
            else:
                model.log_prob_w[k,w] = -100

    if estimate_alpha:
        model.alpha = opt_alpha(ss.alpha_suffstats,ss.num_docs, model.num_topics)
        print("new alpha = {0:.3}\n".format(model.alpha))



def EM(docs, model, EM_CONVERGED = 1e-4, EM_MAX_ITER = 2000):
    i = 0
    likelihood_old = 0
    converged = 1
    VAR_MAX_ITER = 20
    while (((converged < 0) or (converged > EM_CONVERGED) or (i <= 2)) and (i <= EM_MAX_ITER)):
        i = i + 1
        print("****** EM iteration {} ****** timestamp:{:.0f}".format(i, time.time()))
        likelihood = 0
        ss = LDA_Suffstats(model)

        # e-step

        for doc in docs:
            likelihood = likelihood + doc_e_step(doc, model, ss, VAR_MAX_ITER)

        # m-step
        lda_mle(model,ss, estimate_alpha = True)
        
        # check for convergence
        if i >1:
            converged = (likelihood_old - likelihood) / likelihood_old
            if (converged < 0): VAR_MAX_ITER = VAR_MAX_ITER * 2 
            
        likelihood_old = likelihood

def infer( test_labels, test_corpus):
    test_doc_term_mat = vectorizer.transform(test_corpus)
    WS, DS = lda.utils.matrix_to_lists(test_doc_term_mat)
    t_docs = []
    for d in range(len(test_corpus)):
        t_docs.append(Document(doc_id = test_idlist[d], K = model.num_topics, words_id = WS[DS==d], 
                             label = reuters.categories(test_idlist[d])))
    for doc in t_docs: 
        variational_inference(doc, model, 20)
    return t_docs, test_doc_term_mat
    
# def main():
        

if __name__ == '__main__':
    
    text_file = open("Output.txt", "w")

    number_of_docs = 10788
    #num_train_docs = int(number_of_docs*0.8)
    num_train_docs = int(10788-3019)

    np.random.seed(6)
    
#    idlist = list(np.random.choice(reuters.fileids(), number_of_docs))
#    train_idlist = idlist[:num_train_docs]
#    test_idlist = idlist[num_train_docs:]

    idlist = reuters.fileids()
    train_idlist = idlist[3019:]
    test_idlist = idlist[:3019]

    stop_words = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    
    def read_document(idlist, stop_words=stop_words, stemmer=p_stemmer, V=None):
        labels = []
        corpus = []
        for id in idlist:
            labels.append(reuters.categories(id))
            stopped_tokens = [word.lower() for word in reuters.words(id) if word.isalpha() and (word.lower() not in stop_words)]
            reuters.words(id).close()
            stemmed_tokens = [p_stemmer.stem(token) for token in stopped_tokens] 
            if V:
                stemmed_tokens = [token for token in stemmed_tokens if token in V]
            corpus.append(' '.join(stemmed_tokens))
        return labels, corpus
        
    labels, corpus = read_document(train_idlist)
    labelsets = list(set(sum(labels,[]))) # concatnate the lists in labels and keep the uniques
    number_of_topics = 10 #len(labelsets)
        
    vectorizer = CountVectorizer(lowercase = False)
    doc_term_mat = vectorizer.fit_transform(corpus)
    #np.sum(doc_term_mat)
    #doc_term_mat[99,1787]
    Vocabulary = vectorizer.get_feature_names()
    
    WS, DS = lda.utils.matrix_to_lists(doc_term_mat)
    
    my_lda_start = time.time()
    model = LDA_Model(num_topics = number_of_topics, num_terms = len(Vocabulary))      

    docs = []
    for d in range(num_train_docs):
        docs.append(Document(doc_id = train_idlist[d], K = model.num_topics, words_id = WS[DS==d], 
                             label = reuters.categories(train_idlist[d])))
    
    EM(docs,model,EM_MAX_ITER = 1000)
    
    my_lda_end = time.time()
    my_lda_time = my_lda_end-my_lda_start
    print("My LDA runtime: {}".format(my_lda_time))

    text_file.close()
    
    # calculate the top 10 terms of each topic
    
    topicwords = []
    maxTopicWordsNum = 10
    for z in range(0, model.num_topics):
        ids = model.log_prob_w[z, :].argsort()
        topicword = []
        for j in ids:
            topicword.insert(0, Vocabulary[j])
        topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])




    np.set_printoptions(threshold=10000)
    X = []
    Y = []
    for i in range(num_train_docs):
        #print('gamma: {} \n label: {}'.format(docs[i].gamma, docs[i].label))
        X.append(docs[i].gamma)
        Y.append(docs[i].label)
    
    three_classes = []
    for label in Y:
        if 'earn' in label:
            three_classes.append('0')
        elif 'acq' in label:
            three_classes.append('1')
        else:
            three_classes.append('2')
            
    
    test_labels, test_corpus = read_document(test_idlist, V=Vocabulary)
    test_docs, test_mat = infer(test_labels, test_corpus)
    
    test_X = []
    test_Y = []
    for i in range(number_of_docs-num_train_docs):
        #print('gamma: {} \n label: {}'.format(docs[i].gamma, docs[i].label))
        test_X.append(test_docs[i].gamma)
        test_Y.append(test_docs[i].label)


    mul_binarizer = MultiLabelBinarizer()
    bin_Y = mul_binarizer.fit_transform(three_classes)
    
#    le = LabelEncoder()
#    vec_Y = le.fit_transform(three_classes)
    new_X = np.vstack(X)

    SVM_classifier = OneVsRestClassifier(svm.SVC())
    SVM_classifier.fit(new_X,bin_Y)
    pred = SVM_classifier.predict(test_X)
    pred = np.array(pred)    
    pred_label = np.argmax(pred,axis=1)
        
    test_classes=[]
    for label in test_Y:
        if 'earn' in label:
            test_classes.append(0)
        elif 'acq' in label:
            test_classes.append(1)
        else:
            test_classes.append(2)
    
    test_classes = np.array(test_classes)
    np.sum(test_classes[pred_label==0]==2)

#    table = np.argmax(X, axis=1)
#    Y[:20]
#    plt.plot(list(range(38)),test_X[3])
#    plt.xlabel('Topic index')
#    pred = np.argmax(test_X, axis=1)
#        
#    for p in table[:20]:
#        #print(Y[np.where(table==p)[0][0]])
#        print(topicwords[p])
    
#    np.save('model_alpha', model.alpha)
#    np.save('model_log_prob_w', model.log_prob_w)
#    np.save('model_num_topics', model.num_topics)
#    np.save('model_num_terms', model.num_terms)
#    
#    np.save('train_gammas',X)
#    np.save('train_labels',Y)
#    np.save('test_gammas',test_X)
#    np.save('test_labels',test_Y)

    
#    '''
#    LDA Package
#    '''
#    
#    pkg_lda_start = time.time()
#    model_p = lda.LDA(n_topics=number_of_topics)
#    model_p.fit(doc_term_mat)
#    
#    pkg_lda_end = time.time()
#    pkg_lda_time =  pkg_lda_end - pkg_lda_start
#    
#    
#    print("Package LDA runtime: {}".format(pkg_lda_time))
#    print("My LDA runtime: {}".format(my_lda_time))
#
    
    
    
    
    
    
    
    
    
    
    