# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 19:29:45 2018

@author: HL
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from bayesian_optimization import BayesianOptimization
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.ensemble import RandomForestClassifier

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


mnist.train.images.shape
mnist.test.images.shape
mnist.validation.images.shape

mnist.train.labels.shape
mnist.test.labels.shape
mnist.validation.labels.shape

rfc = RandomForestClassifier(n_estimators=10)

rfc.fit(mnist.train.images, mnist.train.labels)
