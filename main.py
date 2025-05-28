import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml

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

#plot first 20
p=np.random.permutation(len(X))
p=p[:20]
plot_images(X[p].reshape(-1,28,28), Y[p])