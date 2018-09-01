import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder

with open('data/titanic.csv', 'rb') as csvfile:
    titanic_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

    # Header contains feature names

    row = titanic_reader.next()
    feature_names = np.array(row)

    # Load dataset, and target classes

    titanic_X, titanic_y = [], []
    for row in titanic_reader:
        titanic_X.append(row)
        titanic_y.append(row[1])  # The target value is survived

    titanic_X = np.array(titanic_X)
    titanic_y = np.array(titanic_y)

# print(feature_names)
# print(titanic_X[0], titanic_y[0])

# We keep class, age and sex

titanic_X = titanic_X[:, [0, 4, 3]]
feature_names = feature_names[[0, 4, 3]]

print(feature_names)
print(titanic_X[15], titanic_y[15])

# We have missing values for age
# Assign the mean value

ages = titanic_X[:, 1]
mean_age = np.mean(titanic_X[ages != '', 1].astype(np.float))
titanic_X[titanic_X[:, 1] == '', 1] = mean_age

# Encode sex

enc = LabelEncoder()
label_encoder = enc.fit(titanic_X[:2, 2])

print("Categorical classes:", label_encoder.classes_)

integer_classes = label_encoder.transform(label_encoder.classes_)

print("Integer classes:", integer_classes)

t = label_encoder.transform(titanic_X[:2, 2])
titanic_X[:2, 2] = t

print(feature_names)
print(titanic_X[15], titanic_y[15])







