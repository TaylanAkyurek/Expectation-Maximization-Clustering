import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as st
import scipy.linalg as linalg

data_set = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw05_data_set.csv" ,delimiter = ",")

initial_centroids = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw05_initial_centroids.csv", delimiter = ",")

X = data_set
K = initial_centroids.shape[0]
N = X.shape[0]


class_means = np.array([[+0.0, +5.5], 
                        [-5.5, +0.0], 
                        [+0.0, +0.0],
                        [+5.5, +0.0],
                        [+0.0, -5.5]])

class_covariances = np.array([[[+4.8, +0.0], 
                               [+0.0, +0.4]],
                              [[+0.4, +0.0], 
                               [+0.0, +2.8]],
                              [[+2.4, +0.0], 
                               [+0.0, +2.4]],
                              [[+0.4, +0.0], 
                               [+0.0, +2.8]],
                              [[+4.8, +0.0], 
                               [+0.0, +0.4]]])

class_sizes = np.array([275, 150, 150, 150, 275])

K = initial_centroids.shape[0]
means = initial_centroids

distances = spa.distance_matrix(means, X)
memberships = np.zeros((N, K))
memberships[range(N), np.argmin(distances, axis=0)] = 1

covariances = np.zeros((K, X.shape[1], X.shape[1]))
for k in range(K):
    
    covariances[k] = np.cov(X - means[k], rowvar=False, aweights=memberships[:,k])
  
prior_probabilities = np.sum(memberships, axis=0) / N

print("prior probabilities: ",prior_probabilities)
print("")
print("covariances: ",covariances)
print("")

def E_step(means, covariances, prior_probabilities, X):
    

    probabilities = np.zeros((N, K))
    for k in range(K):
        probabilities[:, k] = prior_probabilities[k] * st.multivariate_normal.pdf(X,  means[k], covariances[k])
    
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)    

    
    return probabilities

def M_step(memberships, X):
   
    D = X.shape[1]

    means = np.zeros((K, D))
    for k in range(K):
        means[k] = np.sum(np.multiply(X, memberships[:,k, None]), axis=0) / np.sum(memberships[:, k])
    
    covariances = np.zeros((K, D, D))
    for k in range(K):
        covariances[k] = np.cov(X - means[k], rowvar=False, aweights=memberships[:,k])
    
    prior_probabilities = memberships.mean(axis=0)
    
    return means, covariances, prior_probabilities


for i in range(100):
    
    means, covariances, prior_probabilities = M_step(memberships, X)
    
    memberships = E_step(means, covariances, prior_probabilities, X)

print("means: ",means)


colors = np.array(["blue", "green", "red", "orange", "purple"])

fig = plt.figure(figsize = (6, 6))

final_assignments = np.argmax(memberships, axis = 1)

x1 = np.linspace(-8,8,1601)  
x2 = np.linspace(-8,8,1601)

XX, YY = np.meshgrid(x1, x2) 
pos = np.empty(XX.shape + (2,))                
pos[:, :, 0] = XX; pos[:, :, 1] = YY

for c in range(K):
    plt.plot(X[final_assignments == c, 0], X[final_assignments == c, 1], ".", markersize = 10, 
                color = colors[c])
    plt.contour(XX, YY, st.multivariate_normal(class_means[c], class_covariances[c]).pdf(pos), 1, colors = "black", linestyles = "dashed")
    plt.contour(XX, YY, st.multivariate_normal(means[c], covariances[c]).pdf(pos), 1, colors = colors[c], linestyles = "solid")
    
plt.show()
