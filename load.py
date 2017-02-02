import pandas as pd
from scipy.misc import imread
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

def load_data(driving_log, num_images=None, augment=True):
	'''
	load image and steering angle data and return it as numpy arrays
	'''
	driving_log_df = pd.read_csv(driving_log)[["center", "steering"]]

	feature_paths = driving_log_df["center"].values
	feature_paths = np.array([path[path.index("IMG/"):] for path in feature_paths])
	features = np.array([imread(path) for path in feature_paths])
	labels = driving_log_df["steering"].values

	if num_images:
		features, labels = shuffle(features, labels)
		features = features[:num_images]
		labels = labels[:num_images]

	if augment:
		turn_df = driving_log_df[np.absolute(driving_log_df["steering"]) > 0]
		turn_feature_paths = turn_df["center"].values
		turn_feature_paths = np.array([path[path.index("IMG/"):] for path in turn_feature_paths])
		turn_features = np.array([np.fliplr(imread(path)) for path in turn_feature_paths])
		turn_labels = turn_df["steering"].values
		turn_labels = np.negative(turn_labels)

		features = np.concatenate((features, turn_features), axis=0)
		labels = np.concatenate((labels, turn_labels), axis=0)

	return features, labels

if __name__ == "__main__":
	X_train, y_train = load_data("driving_log.csv", augment=False)
	
	print("X_train shape: {}".format(X_train.shape))
	print("y_train shape: {}".format(y_train.shape))

	plt.hist(y_train, bins=50)
	plt.title("Steering Angle Histogram")
	plt.savefig("hist.png")