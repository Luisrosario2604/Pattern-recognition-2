#!/usr/bin/python3

# Importing python3 from local, just use "python3 <binary>" if is not the same location

# Imports
import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Global variables

# Class declarations

# Function declarations


def main():
	datasets = pd.read_csv('./Datasets/reto2_X.csv', header=None)

	for i in range(3980, 4000):
		plt.imshow(datasets.iloc[i].values.reshape(28, 28), 'Blues')
		plt.draw()
		a = plt.waitforbuttonpress(0) # this will wait for indefinite time
		print(a)
		plt.close()

# Main body
if __name__ == '__main__':
	main()