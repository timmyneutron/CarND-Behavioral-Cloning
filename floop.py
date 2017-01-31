import numpy as np
from pandas import read_csv
from scipy.misc import imread

test_driving_log_df = read_csv("test_driving_log.csv")

test_image_paths = test_driving_log_df["center"].values

test_steering_angles = test_driving_log_df["steering"].values

test_image_paths = [path[path.index("IMG/"):] for path in test_image_paths]
X_test = np.array([imread(path) for path in test_image_paths])
y_test = test_steering_angles

for x in np.nditer(X_test[]):
	print(x)