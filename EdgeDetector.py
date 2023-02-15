# import
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d

# read images
lena = cv2.imread('lena.png')
cameraMan = cv2.imread('cameraman.tif')

# grey images
greyLena = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)
greyCameraMan = cv2.cvtColor(cameraMan, cv2.COLOR_RGB2GRAY)


# our problem has 5 step

# step 1
# first is gaussian kernel for noise reduction "size: is the kernel size [must be odd] and sigma: is sigma"
def gaussianKernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    G = (1 / (2 * np.pi * sigma ** 2)) * (np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2))))
    return G


# This func which takes an image and a kernel and returns the convolution of them.
def myConvolve2d(image, kernel):
    # Flip the kernel
    fKernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (fKernel * image_padded[y: y + 3, x: x + 3]).sum()

    return output


# step 2
# now we use Sobel filtering for get the gradient intensity and edges direction of image matrix
def sobelFilter(image):
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    # Ix = cv2.filter2D(image,-1, sobelx)
    # Iy = cv2.filter2D(image,-1,sobely)

    Ix = myConvolve2d(image, sobelx)
    Iy = myConvolve2d(image, sobely)

    # Ixx = convolve2d(image[:, :], sobelx, mode="same", boundary="symm")
    # Iyy = convolve2d(image[:, :], sobely, mode="same", boundary="symm")

    gradient = abs(Ix) + abs(Iy)  # np.hypot(Ix, Iy)  # sqrt(Ix**2+Iy**2) the same
    gradient = gradient / gradient.max() * 255
    # cv2.imshow('myIx', Ix)
    # cv2.imshow('myIy', Iy)
    # immy = gradient.astype(np.uint8)
    # cv2.imshow('mygradient', immy)

    # gradientsci = abs(Ixx) + abs(Iyy)  # np.hypot(Ix, Iy)  # sqrt(Ix**2+Iy**2) the same
    # gradientsci = gradientsci / gradientsci.max() * 255
    # cv2.imshow('sciIx', Ixx)
    # cv2.imshow('sciIy', Iyy)
    # imgsss = gradientsci.astype(np.uint8)
    # cv2.imshow('scipygradient', imgsss)
    # cv2.waitKey()
    theta = np.arctan2(Ix, Iy)  # return radian

    return gradient, theta


# step 3
# now preform non max suppression to thin out the edges and it goes through all points in gradient intensity matrix
def nonMaxSuppression(nmsimage, degreeOfTheta):
    M, N = nmsimage.shape
    nms = np.zeros((M, N), dtype=np.float32)
    angle = degreeOfTheta * 180. / np.pi
    angle[angle < 0] += 180

    # scan neighbors for finding
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if ((0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180)):
                    q = nmsimage[i, j + 1]
                    r = nmsimage[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = nmsimage[i + 1, j - 1]
                    r = nmsimage[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = nmsimage[i + 1, j]
                    r = nmsimage[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = nmsimage[i - 1, j - 1]
                    r = nmsimage[i + 1, j + 1]

                if ((nmsimage[i, j] >= q) and (nmsimage[i, j] >= r)):
                    nms[i, j] = nmsimage[i, j]
                else:
                    nms[i, j] = 0

            except IndexError as e:
                pass

    return nms


# step 4
# Double threshold for finding weak and strong and non-relevant(for next step)
def dThreshold(dtimage, lowThresholdRatio, highThresholdRatio):
    highThreshold = dtimage.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = dtimage.shape
    resizedValue = np.zeros((M, N), dtype=np.float32)

    weak = np.float32(25)
    strong = np.float32(255)

    strongI, strongJ = np.where(dtimage >= highThreshold)
    weakI, weakJ = np.where((dtimage < highThreshold) & (dtimage >= lowThreshold))

    resizedValue[strongI, strongJ] = strong
    resizedValue[weakI, weakJ] = weak

    return resizedValue, weak, strong


    # step 5 here is continue of previous part and will search around every pixels and if exist any strong one near that
    # change the value of that to strong pixel
def hysteresis(hyimage, weak, strong=255):
    M, N = hyimage.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if hyimage[i, j] == weak:
                try:
                    if ((hyimage[i + 1, j - 1] == strong) or (hyimage[i + 1, j] == strong) or (
                            hyimage[i + 1, j + 1] == strong) or (hyimage[i, j - 1] == strong) or (
                            hyimage[i, j + 1] == strong) or (hyimage[i - 1, j - 1] == strong) or (
                            hyimage[i - 1, j] == strong) or (hyimage[i - 1, j + 1] == strong)):
                        hyimage[i, j] = strong
                    else:
                        hyimage[i, j] = 0
                except IndexError as e:
                    pass
    return hyimage


def inputs(images, gaussianSize, gaussianSigma, lowThresh, highThresh):
    gaussian = gaussianKernel(gaussianSize, gaussianSigma)
    blurredImage = ndimage.convolve(images, gaussian)
    gradientImage, degreeImage = sobelFilter(blurredImage)
    nonMaxSuppImage = nonMaxSuppression(gradientImage, degreeImage)
    dTImage, weakPx, strongPx = dThreshold(nonMaxSuppImage, lowThresh, highThresh)
    cannyImage = hysteresis(dTImage, weakPx, strongPx)
    return cannyImage, blurredImage


def main():
    cannyCameraMan, blurredCameraMan = inputs(greyCameraMan, 5, 1.4, 0.01, 0.15)
    cv2.imshow('ORG CameraMan', greyCameraMan)
    cv2.imshow('Blurred CameraMan', blurredCameraMan)
    cv2.imshow('Canny CameraMan', cannyCameraMan)
    cannyCmanOCV = cv2.Canny(blurredCameraMan, 100, 200)
    cv2.imshow('Canny OCV CameraMan', cannyCmanOCV)

    cannyLena, blurredLena = inputs(greyLena, 5, 1.4, 0.11, 0.19)
    cv2.imshow('ORG Lena', greyLena)
    cv2.imshow('Blurred Lena', blurredLena)
    cv2.imshow('Canny Lena', cannyLena)
    cannyLenaOCV = cv2.Canny(blurredLena, 50, 100)
    cv2.imshow('Canny Lena OCV', cannyLenaOCV)
    cv2.waitKey()


if __name__ == '__main__':
    main()
