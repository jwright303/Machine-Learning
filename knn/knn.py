import csv
import sys
import numpy

def calcDistance(npArr, nVal):
	print(npArr.shape, nVal.shape)
	exit(0)
	x = npArr - nVal
	y = numpy.linalg.norm(x, axis=1)

	sortInd = numpy.argsort(y)

	return [y, sortInd]


#Weighting function for the cross validation 
#Essnetially this is what I used to play around with the different weighing funcitons
def evalFold(indecies, k, data, pRes, dRes, distances):
	outcome = 0
	classifList = [dRes[indecies[i]] for i in range(0, k)]
	outcome = numpy.sum(classifList)
	base = 0.55
	maxD = distances[indecies[k-1]]

	#This is more similar to the weighting function that I used as my final results for the kaggle competition
	#for i in range(0, k/12):
	#	dx = distances[indecies[i]]
	#	#zWeight = 0.56 * (1 - (dx/(maxD*3/4)))
	#	DWeight = 0.5 * (1 - (dx/(maxD*3/4)))
	#	zWeight = 0.55
	#	if data[indecies[i]][-1] == 0:
	#		#if zWeight >= 0:
	#		outcome -= 1 * (1 - (dx)/maxD)
	#	elif data[indecies[i]][-1] == 1:
	#		#if DWeight >= 0:
	#		outcome += (1.8 * (1 - (dx)/maxD))

	for i in range(0, 5):
		dx = distances[indecies[i]]
		#print(dx, maxD)
		DWeight = 2.65 * (maxD - dx)/maxD
		ZWeight = 1.05 * (maxD - dx)/maxD
		if data[indecies[i]][-1] == 1:
			outcome += (DWeight + 0.35)

	if (float(outcome)/len(classifList) >= 0.5 and pRes == 1) or (float(outcome)/len(classifList) < 0.5 and pRes == 0):
		return 1
	else:
		return 0

	return

#Weight function and descicion function for the new test points
def nearestNeighbors(indecies, k, data, distances):
	outcome = 0
	classifList = [data[indecies[i]][-1] for i in range(0, k)]
	outcome = numpy.sum(classifList)
	maxD = 0

	for i in range(0, k):
		if distances[indecies[i]] > maxD:
			maxD = distances[indecies[i]]

	for i in range(0, k/15):
		dx = distances[indecies[i]]
		zWeight = 0.56 * (1 - (dx/(maxD*3/4)))
		DWeight = 0.5 * (1 - (dx/(maxD*3/4)))
		
		if data[indecies[i]][-1] == 0:
			if zWeight >= 0:
				outcome -= zWeight
		#elif data[indecies[i]][-1] == 1:
		#	if zWeight >= 0:
		#		outcome += DWeight
	
	if (float(outcome)/len(classifList) >= 0.5):
		return 1
	else:
		return 0

	return


#Fcuntion was used for the actual test data for kaggle
#Takes in the test data and the train data and finds the estimated results and prints them to the screen
def testData(train, test, k):
	#Do something
	trainD = train[ : , 1:-1 ]
	
	sys.stdout = open("knnOut6.csv", 'w')
	print("id,income")
	for i in range(0, len(test)):
		distInd = calcDistance(trainD, test[i])
		distances = distInd[0]
		indecies = distInd[1]
		res = nearestNeighbors(indecies, k, train, distances)
		print(str(i) + "," + str(res))

	return


#Function that implements cross validation for the given k value on the given data
#Prints out the accuracy on each fold as well as the average, and the variance
def crossVal(data, k):
	correct = 0
	acc = []

	#train = data[ : , 1:-1 ]
	#trainRes = data[ : , -1]
	#for i in range(0, len(data)):
	#	results = data[i, -1]
	#	test = data[i, 1:-1 ]
	#	distInd = calcDistance(train, test)
		#distances = distInd[0]
		#indecies = distInd[1]
	#	correct += evalFold(distInd[1], k, data, results, trainRes, distInd[0])
	#percent = correct/8000.0
	#print(percent, k)
	#acc.append(percent)
	#return


	for i in range(0, 4):
		percent = 0
		correct = 0
		
		if i == 0:
			#SOmeti
			train = data[:6000, 1:-1 ]
			test = data[6000:8000, 1:-1 ]
			results = data[6000:8000, -1]
			trainRes = data[:6000, -1]
		elif i == 1:
			#Something
			train = data[2000:8000, 1:-1 ]
			test = data[:2000, 1:-1 ]
			results = data[:2000, -1]
			trainRes = data[2000:8000, -1]
		elif i == 2:
			#Something
			train = numpy.concatenate((data[4000:8000, 1:-1 ], data[:2000, 1:-1 ]), axis=0)
			test = data[2000:4000, 1:-1 ]
			results = data[2000:4000, -1]
			trainRes = numpy.concatenate((data[4000:8000, -1], data[:2000, -1]), axis=0)
		elif i == 3:
			#Somethings
			train = numpy.concatenate((data[6000:8000, 1:-1 ], data[:4000, 1:-1 ]), axis=0)
			test = data[4000:6000, 1:-1 ]
			results = data[4000:6000, -1]
			trainRes = numpy.concatenate((data[6000:8000, -1], data[:4000, -1]), axis=0)

		for i in range(0, len(test)):
			distInd = calcDistance(train, test[i])#Collects the distances of the closest neighbors as well as their indecies
			distances = distInd[0]
			indecies = distInd[1]
			correct += evalFold(distInd[1], k, data, results[i], trainRes, distInd[0])#Checks to see if its correct

		percent = correct/2000.0
		acc.append(percent)
	
	print("Accuracies, " + str(k))
	print(acc)
	print("Mean: " + str(numpy.mean(acc)))
	print("Var: " + str(numpy.var(acc)))

	return


def main():
	data = []
	testD = []
	k = [40, 45, 50, 55, 60, 65, 70, 75]
	k = [80, 85, 90, 95, 100, 105, 120, 130, 140]
	#k = [1, 3, 5, 7, 9, 99, 999, 6000]
	#k = [60, 65]

	#testArr = numpy.array([[1, 2, 3], [4, 5, 6], [1, 5, 2], [7, 1, 2]])

	with open('train.csv') as csv_file:
		csvReader = csv.reader(csv_file, delimiter=',')
		lineC = 0
		for row in csvReader:
			if lineC != 0:
				vector = [float(row[i]) for i in range(0, len(row))]
				data.append(vector)
			lineC += 1

	with open('test_pub.csv') as csv_file:
		csvReader = csv.reader(csv_file, delimiter=',')
		lineC = 0
		for row in csvReader:
			if lineC != 0:
				vector = [float(row[i]) for i in range(1, len(row))]
				testD.append(vector)
			lineC += 1


	npArr = numpy.array(data)
	testArr = numpy.array(testD)
	#print(testArr)
	
	#testData(npArr, testArr, 65)
	#k = 85
	#for i in range(0, 10):
		#k = k + 5
	#for val in k:
	#	crossVal(npArr, val)
	crossVal(npArr, 65)

	return

main()
