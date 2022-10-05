from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    # grabh the spatial dimensions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # allocate memory for the output image, taking care to "pad" the orders of the input image
    # so the spatial size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # loop over the input image, "sliding" the kernel across each (x, y) - coordinate from
    # left-to-right and top-to-bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the *center* region of the current (x,y)-coordinates dim
            roi = image[y - pad : y + pad +1, x - pad : x + pad + 1]

            # perform the actual convolution by taking the element-wise multiplicaiton between the ROI
            # and the kernel, the summing the matrix
            k = (roi * K).sum()

            # store the convolve value in the output (x,y)-corodinate of the output image
            output[y-pad, x-pad] = k

    # store the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="girl.png", help="path to input image")
args = vars(ap.parse_args())

smallBlur = np.ones((7,7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# laplacian kernel used to detect edges
laplacian = np.array((
    [0, 1, 0 ],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int")

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss))

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, k) in kernelBank:
    convolveOutput = convolve(gray, k)
    opencvOutput = cv2.filter2D(gray, -1, k)

    cv2.imshow("Original", gray)
    cv2.imshow(f"{kernelName} - convolve", convolveOutput)
    cv2.imshow(f"{kernelName} - opencv", opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

