import random
import nltk
from nltk.corpus import movie_reviews

# build a list of documents
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

# shuffle the documents
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

# We' ll use the 4000 most common words as features
word_features = list(all_words.keys())[:4000]

def find_features(document):
    words = set(document)
    features={}
    for w in word_features:
        features[w] = (w in words)
    
    return features
        
featuresets = [(find_features(rev), category) for (rev, category) in documents]


# we can split the feature set into train and tesr using sklearn
from sklearn import model_selection

# define a seed for reproducablity
seed = 1

# split the data into training and test
train, test = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)


# using sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC


model = SklearnClassifier(SVC(kernel = 'linear'))

# train model on training data
model.train(train)

# test on test dataset
accuracy = nltk.classify.accuracy(model,test)
print('SVC Accuracy: {}'.format(accuracy))

# accuracy obtained - > 0.706
