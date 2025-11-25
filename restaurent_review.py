import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv(r"C:\Users\Sanket kshirsagar\Downloads\Restaurant_Reviews (1).tsv", 
                      delimiter='\t', quoting=3)

# Import necessary libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (run once)
nltk.download('stopwords')

# Initialize corpus
corpus = []

# Data cleaning and text preprocessing
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

# Dependent variable
y = dataset.iloc[:, 1].values

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Training the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:\n", cm)
print("\nAccuracy:", ac)


