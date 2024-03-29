#!/usr/bin/python3

# Importing python3 from local, just use "python3 <binary>" if is not the same location

# Imports
import pandas as pd
from matplotlib import pyplot as plt


def main():
	datasets = pd.read_csv('./predictions/reto1_X.csv', header=None)

	for i in range(200):
		plt.imshow(datasets.iloc[i].values.reshape(28, 28), 'Blues')
		plt.draw()
		a = plt.waitforbuttonpress(0)   # this will wait for indefinite time
		print(a)
		plt.close()


# Main body
if __name__ == '__main__':
	main()
