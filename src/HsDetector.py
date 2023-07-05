#https://www.kaggle.com/code/gabrielbchacon/nlp-model-to-predict-hate-speech

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

#to data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#NLP tools
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#train split and fit models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.metrics import confusion_matrix, accuracy_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

file_path=os.getcwd()
file_path+='/src'

dataset = pd.read_csv(file_path+'/labeled_data.csv')

dt_trasformed = dataset[['class', 'tweet']]
y = dt_trasformed.iloc[:, :-1].values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = np.array(ct.fit_transform(y))
y_df = pd.DataFrame(y)
y_hate = np.array(y_df[0])
y_offensive = np.array(y_df[1])

corpus = []
for i in range(0, 24783):
  review = re.sub('[^a-zA-Z]', ' ', dt_trasformed['tweet'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()

pkl_filename = "/countvectorizer.pkl"
with open(file_path+pkl_filename, 'wb') as fout:
    pickle.dump(cv, fout)


X_train, X_test, y_train, y_test = train_test_split(X, y_hate, test_size = 0.20, random_state = 0)

classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)

#Linear Regression
y_pred_lr = classifier_lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)
#print(cm)

lr_score = accuracy_score(y_test, y_pred_lr)
#print("lrscore:",lr_score)

pkl_filename = "/lrhatespeech_hate.pkl"
with open(file_path+pkl_filename, 'wb') as file:
    pickle.dump(classifier_lr, file)

    #####

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_offensive, test_size = 0.20, random_state = 0)

classifier_lrb = LogisticRegression(random_state = 0)
classifier_lrb.fit(X_train2, y_train2)

#Linear Regression
y_pred_lrb = classifier_lrb.predict(X_test2)
cm2 = confusion_matrix(y_test2, y_pred_lrb)
#print(cm2)

lr_score2 = accuracy_score(y_test2, y_pred_lrb)
#print("lrscore:",lr_score2)
#Linear Regression Accuracy:

pkl_filename = "/lrhatespeech_offensive.pkl"
with open(file_path+pkl_filename, 'wb') as file:
    pickle.dump(classifier_lrb, file)


