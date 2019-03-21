# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_house_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

	# determine (1) the unique zip codes and (2) the number of data
	# points with each zip code
	zipcodes = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().tolist()

	# loop over each of the unique zip codes and their corresponding
	# count
	for (zipcode, count) in zip(zipcodes, counts):
		# the zip code counts for our housing dataset is *extremely*
		# unbalanced (some only having 1 or 2 houses per zip code)
		# so let's sanitize our data by removing any houses with less
		# than 25 houses per zip code
		if count < 25:
			idxs = df[df["zipcode"] == zipcode].index
			df.drop(idxs, inplace=True)

	# return the data frame
	return df

def process_house_attributes(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["bedrooms", "bathrooms", "area"]

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])

	# one-hot encode the zip code categorical data (by definition of
	# one-hot encoing, all output features are now in the range [0, 1])
	zipBinarizer = LabelBinarizer().fit(df["zipcode"])
	trainCategorical = zipBinarizer.transform(train["zipcode"])
	testCategorical = zipBinarizer.transform(test["zipcode"])
	print("x",trainCategorical.shape)
	try:
		zipBinarizer = OneHotEncoder().fit(df["zipcode"])
		trainCategorical = zipBinarizer.transform(train["zipcode"])
		testCategorical = zipBinarizer.transform(test["zipcode"])
	except:
		print(df['zipcode'])
		print(len(df['zipcode']))	
	# print(trainCategorical.shape)
	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])

	# return the concatenated training and testing data
	return (trainX, testX)

def load_house_images(df,inputPath):
	#initializing our images array
	images=[]

	for i in df.index.values:
		basePath=os.path.sep.join([inputPath,"{}_*".format(i+1)])
		housePaths=sorted(list(glob.glob(basePath)))

		inputImages=[]
		outputImage=np.zeros((64,64,3),dtype="uint8")

		#loop over the input house paths
		for housePath in housePaths:
			image=cv2.imread(housePath)
			image=cv2.resize(image,(32,32))
			inputImages.append(image)

		outputImage[0:32,0:32]=inputImages[0]
		outputImage[0:32,32:64]=inputImages[1]
		outputImage[32:64,32:64]=inputImages[2]
		outputImage[32:64,0:32]=inputImages[3]

		images.append(outputImage)

	return np.array(images)

