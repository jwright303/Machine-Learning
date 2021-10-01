import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# Toy problem with 3 clusters for us to verify k-means is working well
def toyProblem():
  # Generate a dataset with 3 cluster
  X = np.random.randn(150,2)*1.5
  X[:50,:] += np.array([1,4])
  X[50:100,:] += np.array([15,-2])
  X[100:,:] += np.array([5,-2])

  # Randomize the seed
  np.random.seed()

  # Apply kMeans with visualization on
  k = 3
  max_iters=20
  centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)
  plotClustering(centroids, assignments, X, title="Final Clustering")
  
  # Print a plot of the SSE over training
  plt.figure(figsize=(16,8))
  plt.plot(SSE, marker='o')
  plt.xlabel("Iteration")
  plt.ylabel("SSE")
  plt.text(k/2, (max(SSE)-min(SSE))*0.9+min(SSE), "k = "+str(k))
  plt.show()


  #############################
  # Q5 Randomness in Clustering
  #############################
  k = 5
  max_iters = 20

  SSE_rand = []
  # Run the clustering with k=5 and max_iters=20 fifty times and 
  # store the final sum-of-squared-error for each run in the list SSE_rand.
  for i in range(0, 50):
    centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)
    SSE_rand.append(SSE)
  #raise Exception('Student error: You haven\'t implemented the randomness experiment for Q5.')
  

  # Plot error distribution
  plt.figure(figsize=(8,8))
  plt.hist(SSE_rand, bins=20)
  plt.xlabel("SSE")
  plt.ylabel("# Runs")
  plt.show()

  ########################
  # Q6 Error vs. K
  ########################

  SSE_vs_k = []
  # Run the clustering max_iters=20 for k in the range 1 to 150 and 
  # store the final sum-of-squared-error for each run in the list SSE_vs_k.
  for i in range(0, 150):
    centroids, assignments, SSE = kMeansClustering(X, k=(i+1), max_iters=max_iters, visualize=False)
    SSE_vs_k.append(SSE)
  #raise Exception('Student error: You haven\'t implemented the randomness experiment for Q5.')

  # Plot how SSE changes as k increases
  plt.figure(figsize=(16,8))
  plt.plot(SSE_vs_k, marker="o")
  plt.xlabel("k")
  plt.ylabel("SSE")
  plt.show()
  return


def imageProblem():
  np.random.seed()
  # Load the images and our pre-computed HOG features
  data = np.load("img.npy")
  img_feats = np.load("hog.npy")


  # Perform k-means clustering
  k=10
  centroids, assignments, SSE = kMeansClustering(img_feats, k, 30, min_size=0)

  # Visualize Clusters
  print("SSE: " + str(SSE[-1]))
  for c in range(len(centroids)):
    # Get images in this cluster
    members = np.where(assignments==c)[0].astype(np.int)
    imgs = data[np.random.choice(members,min(50, len(members)), replace=False),:,:]
    
    # Build plot with 50 samples
    print("Cluster "+str(c) + " ["+str(len(members))+"]")
    _, axs = plt.subplots(5, 10, figsize=(16, 8))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img,plt.cm.gray)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # Fill out plot with whitespace if there arent 50 in the cluster
    for i in range(len(imgs), 50):
      axs[i].axes.xaxis.set_visible(False)
      axs[i].axes.yaxis.set_visible(False)
    plt.show()
  return


##########################################################
# initializeCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   k --  integer number of clusters to make
#
# Outputs:
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
##########################################################

def initalizeCentroids(dataset, k):
  centroids = []
  #print(k)
  randInds = np.random.choice(len(dataset), k, replace=False)
  for i in range(0, k):
    centroids.append(dataset[randInds[i]])
  #print(centroids)
    
  #raise Exception('Student error: You haven\'t implemented initializeCentroids yet.')
  return np.array(centroids)

##########################################################
# computeAssignments
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#
# Outputs:
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
##########################################################

def computeAssignments(dataset, centroids):
  assignments = []
  centDistances = []
  for i in range(0, len(centroids)):
    x = dataset - centroids[i]
    y = np.linalg.norm(x, axis=1)
    centDistances.append(y)

  sortCentr = np.argsort(np.transpose(centDistances), axis=1)
  #print(sortCentr)
  assignments = sortCentr[:,0]
  #print(assignments)
  #print(assignments.shape)
  #raise Exception('Student error: You haven\'t implemented initializeCentroids yet.')
  #print(assignments)

  return assignments

##########################################################
# updateCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   centroids -- n x d matrix of centroid points where the
#                 j'th row represents c_j after being updated
#                 as the mean of assigned points
#   counts -- k x 1 matrix where the j'th entry is the number
#             points assigned to cluster j
##########################################################

def updateCentroids(dataset, centroids, assignments):
  newCentr = np.zeros(centroids.shape)
  totalCentrCount = np.zeros((len(centroids), 1))
  for i in range(0, len(centroids)):
    posSum = np.zeros(dataset[0].shape)
    centrCount = 0
    for j in range(0, len(dataset)):
      if assignments[j] == i:
	centrCount += 1
	posSum = posSum + dataset[j]
	#print(posSum)
   
    #print(posSum/centrCount)
    newCentr[i] = (posSum/centrCount)
    totalCentrCount[i] = (int(centrCount))
    #print(newCentr)

    newCentroids = np.array(newCentr)
    centroidCounts = np.array(totalCentrCount)
  #print(newCentroids.shape)
  #print(centroidCounts.shape)
  #print(newCentroids)
  #print(centroidCounts)

  #raise Exception('Student error: You haven\'t implemented updateCentroids yet.')
  return newCentroids, centroidCounts
  

##########################################################
# calculateSSE
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   sse -- the sum of squared error of the clustering
##########################################################

def calculateSSE(dataset, centroids, assignments):
  sse = 0
  for i in range(0, len(dataset)):
    x = dataset[i] - centroids[assignments[i]]
    #print(x)
    y = np.linalg.norm(x)
    sse += y

  #raise Exception('Student error: You haven\'t implemented calculateSSE yet.')
  return sse
  

########################################
# Instructor Code: Don't need to modify 
# beyond this point but should read it
########################################

def kMeansClustering(dataset, k, max_iters=10, min_size=0, visualize=False):
  
  # Initialize centroids
  centroids = initalizeCentroids(dataset, k)
  
  # Keep track of sum of squared error for plotting later
  SSE = []

  # Main loop for clustering
  for i in range(max_iters):

    # Update Assignments Step
    assignments = computeAssignments(dataset, centroids)
    
    # Update Centroids Step
    centroids, counts = updateCentroids(dataset, centroids, assignments)

    # Re-initalize any cluster with fewer then min_size points
    for c in range(k):
      if counts[c] <= min_size:
        centroids[c] = initalizeCentroids(dataset, 1)
    
    if visualize:
      plotClustering(centroids, assignments, dataset, "Iteration "+str(i))
    SSE.append(calculateSSE(dataset,centroids,assignments))

    # Get final assignments
    assignments = computeAssignments(dataset, centroids)

  return centroids, assignments, SSE

def plotClustering(centroids, assignments, dataset, title=None):
  plt.figure(figsize=(8,8))
  plt.scatter(dataset[:,0], dataset[:,1], c=assignments, edgecolors="k", alpha=0.5)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=5, edgecolors="k", s=250)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=2, edgecolors="w", s=200)
  if title is not None:
    plt.title(title)
  plt.show()
  return


if __name__=="__main__":
  #toyProblem()
  imageProblem()
