"""
COMP 472 Artificial Intelligence Summer 2025
Mini Project 1

Team Members:
- Johnny Dang 40245598
- Julia Trinh 40245980
- Oviya Sinnathamby 40249479
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns

#Fetching the data 
# - Dividing by 255, pixel values 
mnist= fetch_openml('mnist_784', version=1)
X=mnist.data.values.astype('float32')/255.0
Y=mnist.target.values


print(X.dtype,Y.dtype)
print(X.shape,Y.shape)

def plot_images(images, labels):
    nb_cols= min(5, len(images))
    nb_rows= len(images)// nb_cols
    fig = plt.figure(figsize=(8,8))

    for i in range (nb_rows * nb_cols):
        sp = fig.add_subplot(nb_rows,nb_cols, i+1)
        plt.axis('off')
        plt.imshow(images[i], cmap=plt.cm.gray)
        sp.set_title(labels[i])
    plt.show()

# plot first 20
p=np.random.permutation(len(X))
p=p[:20]
plot_images(X[p].reshape(-1,28,28), Y[p])

# split 80% for training and 20% for testing ??
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
train_X.shape, test_X.shape

# Train the model
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(max_iter=1000)
cls.fit(train_X, train_Y)

# Evaluate the model
cls.score(test_X, test_Y)

predictions = cls.predict(test_X)
print(classification_report(test_Y, predictions))
confusionMatrix = confusion_matrix(test_Y, predictions)

print(confusionMatrix)

# plot predictions
p = np.random.permutation(len(test_X))
p = p[:20]
plot_images(test_X[p].reshape(-1, 28, 28), predictions[p])

# Calculate confusion matrix
cm = confusion_matrix(test_Y, predictions)

# Visualize confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Digit')
plt.ylabel('Actual Digit')
plt.show()