#!/usr/bin/env python
"""
@author: Damien Iggiotti
"""

#%%
###############################################################################
#  Import the necessary stuff.
###############################################################################
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier

#%%
###############################################################################
#  Load the raw text dataset.
###############################################################################
print("*** Loading train dataset...")
text = ['comment_text']
raw_text_df = pd.read_csv('data/train.csv', usecols=text)

#%%
###############################################################################
#  Extract the vector of the comments.
###############################################################################
print("*** Vectorizing the IF-IDF words...")
vectorizer = TfidfVectorizer(max_df=0.5, max_features=100000, min_df=2,
                             stop_words='english', use_idf=True)

# Build the tfidf vectorizer from the training data, and apply it 
train_tfidf = vectorizer.fit_transform(raw_text_df['comment_text'])

# Get the words that correspond to each of the features.
feat_names = vectorizer.get_feature_names()

print("  * Actual number of TF-IDF features: %d" % train_tfidf.get_shape()[1])
    
#%%
###############################################################################
#  Run the SVD decomposition on the vector to obtain LSA.
###############################################################################
print("*** Performing dimensionality reduction using LSA...")
t0 = time.time()

# Project the tfidf vectors onto the first N principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
svd       = TruncatedSVD(100)
lsa       = make_pipeline(svd, Normalizer(copy=False))
train_lsa = lsa.fit_transform(train_tfidf)

print("  * SVD Reduction done in %.3f seconds" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("  * Explained variance by SVD: {}%".format(int(explained_variance * 100)))

#%%
###############################################################################
#  Run classification of the test articles
###############################################################################
print("*** Classifying the LSA vector...")

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 
# and brute-force calculation of distances.
knn_tfidf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn_tfidf.fit(train_tfidf, y_train)

# Classify the test vectors.
p = knn_tfidf.predict(X_test_tfidf)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("  done in %.3fsec" % elapsed)


print("\nClassifying LSA vectors...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 
# and brute-force calculation of distances.
knn_lsa = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
knn_lsa.fit(X_train_lsa, y_train)

# Classify the test vectors.
p = knn_lsa.predict(X_test_lsa)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)    
print("    done in %.3fsec" % elapsed)



#%%
###############################################################################
#  Test data
###############################################################################
print("*** Loading test dataset...")
text = ['comment_text']
raw_test_df = pd.read_csv('data/test.csv', usecols=text)

# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(X_test_raw)
X_test_lsa = lsa.transform(X_test_tfidf)
