import pandas as pd
import numpy as np
import re
import csv

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

dataset.shape

dataset.head()

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

len(corpus)

corpus[0:10]

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

X.shape

X[0:2]

y.shape

y[0:10]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train[0:2]

X_test[0:2]

y_train[0:2]

y_test[0:2]

gauss_classifier = GaussianNB()
gauss_classifier

gauss_classifier.fit(X_train, y_train)

y_pred = gauss_classifier.predict(X_test)

y_pred.shape

y_pred[0:10]

cm = confusion_matrix(y_test, y_pred)
print(cm)

dataset.to_csv('Restaurant_Reviews.csv', index=False)

with open('corpus1.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(corpus)

dict = {'Cleaned_Reviews': corpus}
df_corpus = pd.DataFrame(dict)
df_corpus.to_csv('corpus2.csv', index=False)

df_x = pd.DataFrame(X)
df_x.to_csv('X.csv', index=False)

# np.set_printoptions(suppress=True,precision=3)
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
np.savetxt("X2.csv", X, delimiter=",", fmt='%1.3f')

