import sys
from scipy import stats

def myMode(v1, v2, v3):
	myDict = {}
	greatestC = 0
	index = v1

	if v1 in myDict.keys():
		myDict[v1] += 1
	else:
		myDict[v1] = 1
	
	for key in myDict.keys():
		if myDict[key] > greatestC:
			greatestC = myDict[key]
			index = key
		
	return index

def parseFile(fName):
	f = open(fName, 'r')
	f.readline()
	first = []
	for line in f:
		val = line.split(",")
		intV = int(val[1])
		first.append(intV)
	
	return first

def main():
	first = parseFile("NBres.csv")
	second = parseFile("NBres1.csv")
	third = parseFile("NBres2.csv")


	final = []
	for i in range(0, len(first)):
		val = myMode(first[i], second[i], third[i])
		#val = stats.mode([first[i], second[i], third[i]])
		#print(int(val[0][0]))
		#return
		final.append(val)

	sys.stdout = open("ensamble.csv", 'w')	
	print("id,digit")
	for i in range(0, len(final)):
		print(str(i) + "," + str(final[i]))

	return

main()
