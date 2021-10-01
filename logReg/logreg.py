import numpy as np
import sys
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
#step_size=0.0001
step_size=0.00014
max_iters=3500

def main():

  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()

  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train @ w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  X_train_bias = dummyAugment(X_train)
 
  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias,y_train)
  y_pred_train = X_train_bias@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))

  #X_test_bias = dummyAugment(X_test)
  #y_pred_test = X_test_bias@w >= 0
  #makePred(y_pred_test)

  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  logging.info("\n---------------------------------------------------------------------------\n")

  logging.info("Running cross-fold validation for bias case:")

  # Perform k-fold cross
  #kvals = [2, 3, 4, 5, 10, 20, 50]
  kvals = [50]
  for k in kvals:
    cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
    logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

  ####################################################
  # Write the code to make your test submission here
  ####################################################

  #raise Exception('Student error: You haven\'t implemented the code in main() to make test predictions.')
  return


def dummyAugment(X):
  Xlen = len(X)
  #newCol = np.ones((Xlen, 1))
  newCol = np.ones(Xlen)
  newM = np.insert(X, 0, newCol, axis=1)
  return newM
  #raise Exception('Student error: You haven\'t implemented dummyAugment yet.')


def sigmoidCalc(val):
  return 1.0 / (1.0 + np.exp(-val))


def makePred(arr):
  sys.stdout = open("logR.csv", 'w')
  print("id,type")
  guess = 0
  for i in range(0, len(arr)):
    if arr[i]:
      guess = 1
    else:
      guess = 0
    print(str(i) + "," + str(guess))

  return


def calculateNegativeLogLikelihood(X,y,w):
  #print("x vec", X[0])
  #print("w vec", w)
  #print(np.shape(np.transpose(w)))
  #print(np.shape(X[0]))

  logSum = 0
  logH = 0.0000000000001
  #logH = 0

  for i in range(0, len(X)):
    curV = np.transpose(w) @ X[i]
    sigV = sigmoidCalc(curV)
    itVal = (y[i] * np.log(sigV+logH)) + ((1 - y[i]) * np.log((1 - sigV) + logH))
    logSum = logSum + itVal

  #print("logSum", -logSum[0])

  return -logSum[0]
  #raise Exception('Student error: You haven\'t implemented the negative log likelihood calculation yet.')
 

def trainLogistic(X,y, max_iters=max_iters, step_size=step_size):

  # Initialize our weights with zeros
  w = np.zeros( (X.shape[1],1) )
    
  # Keep track of losses for plotting
  losses = [calculateNegativeLogLikelihood(X,y,w)]
    
  # Take up to max_iters steps of gradient descent
  for i in range(max_iters):
  
    # Make a variable to store our gradient
    w_grad = np.zeros( (X.shape[1],1) )
        
    # Compute the gradient over the dataset and store in w_grad
    # .
    # . Implement equation 9.
    # .
    #print(np.shape(w_grad))

    for j in range(0, len(X)):
      curV = np.transpose(w) @ X[j]
      sigV = sigmoidCalc(curV)
      finCalc = (sigV - y[j]) * X[j]
      finCalc = finCalc[:,np.newaxis]
      #print(finCalc)
      #w_grad[j] = finCalc
      w_grad += finCalc
      #print(w_grad)
    #print(np.shape(finCalc))
    #print(np.shape(w_grad))
    #w_grad = gradSum

    #DO something

    #raise Exception('Student error: You haven\'t implemented the gradient calculation for trainLogistic yet.')
    # This is here to make sure your gradient is the right shape
    assert(w_grad.shape == (X.shape[1],1))

    # Take the update step in gradient descent
    w = w - step_size*w_grad
        
    # Calculate the negative log-likelihood with the 
    # new weight vector and store it for plotting later
    losses.append(calculateNegativeLogLikelihood(X,y,w))
        
  return w, losses




##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
