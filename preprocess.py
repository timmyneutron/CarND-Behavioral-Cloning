import numpy as np
from pandas import read_csv
from scipy.misc import imread, imresize
import cv2
from load import load_data
import matplotlib.pyplot as plt
import pickle

### helper functions for preprocessing

def convert_rgb_to_yuv(rgb_array):
	'''converts rgb images to yuv'''
	yuv_array = np.empty_like(rgb_array, dtype=np.uint8)
	for i in range(len(rgb_array)):
		yuv_array[i] = cv2.cvtColor(rgb_array[i], cv2.COLOR_BGR2YUV)

	return yuv_array

def take_one_channel(img_array, channel_num):
	'''takes array of 3 channel images and returns one channel'''
	new_array = img_array[:, :, :, channel_num]
	new_array = np.expand_dims(new_array, 3)
	return new_array

def crop(img_array, top=0, bottom=0):
	'''crops rows off the top of images'''
	if bottom != 0:
		return img_array[:, top:-bottom, :, :]
	else:
		return img_array[:, top:, :, :]

def scale_images(img_array, scale_factor=1.0):
	'''scale image size'''
	samples, height, width, channels = img_array.shape
	new_height = int(height * scale_factor)
	new_width = int(width * scale_factor)

	new_img_array = np.empty((samples, new_height, new_width, channels),
							  dtype=np.uint8)
	for i in range(len(img_array)):
		new_img_array[i] = imresize(img_array[i], (new_height, new_width))

	return new_img_array

def gaussian_blur(img_array, kernel_size=3):
    '''apply a Gaussian blur to images'''
    new_img_array = np.squeeze(img_array)
    new_img_array = np.array([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) for img in img_array])
    
    if len(new_img_array.shape) == 3:
    	new_img_array = np.expand_dims(new_img_array, 3)
    return new_img_array

def region_of_interest(img_array):
	'''Applies an image mask. Only keeps the region of the image defined by
	the polygon formed from `vertices`. The rest of the image is set to black.
	'''

	img_array = np.squeeze(img_array)

	height = img_array.shape[1]
	width = img_array.shape[2]

	vertices = []

	vertices.append([0, height // 6])
	vertices.append([width // 2, 0])
	vertices.append([width // 2, 0])
	vertices.append([width, height // 6])
	vertices.append([width, height])
	vertices.append([0, height])

	vertices = np.array([vertices])

	mask_array = np.zeros_like(img_array)   
    
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img_array.shape) == 3:
		ignore_mask_color = 255
	else:
		channel_count = img_array.shape[3]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count

    # filling pixels inside the polygon defined by "vertices" with the fill color
	for i in range(len(mask_array)):
		cv2.fillPoly(mask_array[i], vertices, ignore_mask_color)
    
		#returning the image only where mask pixels are nonzero

		mask_array[i] = cv2.bitwise_and(img_array[i], mask_array[i])

	if len(mask_array.shape) == 3:
		mask_array = np.expand_dims(mask_array, 3)

	return mask_array

def canny(img_array, low_threshold=50, high_threshold=150):
	'''Applies the Canny transform to an array of images
	Note: input images must be only 1 channel'''
	new_img_array = np.squeeze(img_array)
	new_img_array = np.array([cv2.Canny(img, low_threshold, high_threshold) for img in img_array])
	new_img_array = np.expand_dims(new_img_array, 3)
	return new_img_array

def laplacian(img_array):
	'''Applies Laplacian gradient to an array of images
	Note: input images must be only 1 channel'''
	new_img_array = np.squeeze(img_array)
	new_img_array = np.array([cv2.Laplacian(img, cv2.CV_64F) for img in img_array])
	new_img_array = np.expand_dims(new_img_array, 3)

	return new_img_array

def color_canny(img_array, low_threshold=50, high_threshold=150):
	'''Applies Canny transform to each channel individually, then 
	compiles the channels into one image'''
	ch0_array = take_one_channel(img_array, 0)
	ch1_array = take_one_channel(img_array, 1)
	ch2_array = take_one_channel(img_array, 2)

	ch0_canny = canny(ch0_array, low_threshold=low_threshold, high_threshold=high_threshold)
	ch1_canny = canny(ch1_array, low_threshold=low_threshold, high_threshold=high_threshold)
	ch2_canny = canny(ch2_array, low_threshold=low_threshold, high_threshold=high_threshold)

	new_img_array = np.concatenate((ch0_canny, ch1_canny, ch2_canny), axis=3)

	return new_img_array

def get_preprocess_dict():
	'''Preprocessing specifications used by the model'''
	preprocess_dict = {
			"crop": {
				"crop": True,
				"top": 50,
				"bottom": 0
			},
			"convert_rgb_to_yuv": False,
			"take_one_channel": {
				"take": False,
				"channel_num": 0
			},
			"scale_images": {
				"scale": True,
				"scale_factor": 0.5
			},
			"gaussian_blur": {
				"blur": True,
				"kernel_size": 5
			},
			"laplacian": False,
			"canny_edge_detection": {
				"detect": False,
				"low_threshold": 100,
				"high_threshold": 150
			},
			"color_canny": {
				"detect": False,
				"low_threshold": 30,
				"high_threshold": 120
			}
		}
	return preprocess_dict

def preprocess(img_array, preprocess_dict):
	'''Takes in image data and preprocessing specifications, and applies
	the relevant transforms'''

	new_img_array = img_array

	if preprocess_dict["crop"]["crop"]:
		top = preprocess_dict["crop"]["top"]
		bottom = preprocess_dict["crop"]["bottom"]
		new_img_array = crop(img_array, top, bottom)

	if preprocess_dict["scale_images"]["scale"]:
		scale_factor = preprocess_dict["scale_images"]["scale_factor"]
		new_img_array = scale_images(new_img_array, scale_factor)

	if preprocess_dict["convert_rgb_to_yuv"]:
		new_img_array = convert_rgb_to_yuv(new_img_array)

	if preprocess_dict["take_one_channel"]["take"]:
		channel_num = preprocess_dict["take_one_channel"]["channel_num"]
		new_img_array = take_one_channel(new_img_array, channel_num)

	if preprocess_dict["gaussian_blur"]["blur"]:
		kernel_size = preprocess_dict["gaussian_blur"]["kernel_size"]
		new_img_array = gaussian_blur(new_img_array, kernel_size)

	if preprocess_dict["laplacian"]:
		new_img_array = laplacian(new_img_array)

	if preprocess_dict["canny_edge_detection"]["detect"]:
		low_threshold = preprocess_dict["canny_edge_detection"]["low_threshold"]
		high_threshold = preprocess_dict["canny_edge_detection"]["high_threshold"]
		new_img_array = canny(new_img_array, low_threshold, high_threshold)

	if preprocess_dict["color_canny"]["detect"]:
		low_threshold = preprocess_dict["color_canny"]["low_threshold"]
		high_threshold = preprocess_dict["color_canny"]["high_threshold"]
		new_img_array = color_canny(new_img_array, low_threshold, high_threshold)

	return new_img_array

if __name__ == "__main__":
	'''Plots sample images with specified preprocessing'''
	num_images = 4

	X_test, y_test = load_data("driving_log.csv", num_images=num_images)

	preprocess_dict = get_preprocess_dict()
	X_test_preprocessed = preprocess(X_test, preprocess_dict)

	fig = plt.figure(figsize=(20, 5))
	plt.title("Original and Processed Images")
	plt.axis("off")
	X_test_preprocessed = np.squeeze(X_test_preprocessed)

	for i in range(0, 4):
		fig.add_subplot(2, 4, i + 1)
		plt.imshow(X_test[i])
		fig.add_subplot(2, 4, i + 5)
		plt.imshow(X_test_preprocessed[i])


	plt.savefig("processed_images.png", bbox_inches='tight', format="png")

	# plt.show()

