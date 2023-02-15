# Canny Edge Detector
implementation of Canny's edge detection algorithm from scratch

The Canny edge detector is an edge detection operator that uses a multi stage algorithm to detect a wide range of edges in images. It was developed by John F. Canny in 1986. Canny also produced a computational theory of edge detection explaining why the technique works.

Canny edge detection is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed. It has been widely applied in various computer vision systems. Canny has found that the requirements for the application of edge detection on diverse vision systems are relatively similar.

# Steps:

At first I turn the image to grayscale
• Step 1 : First is gaussian kernel for noise reduction.

• Step 2 : then we use Sobel filtering for get the gradient intensity and edges direction of image matrix.

• Step 3 : Now preform non max suppression to thin out the edges and it goes through all points in gradient intensity matrix.

• Step 4 : Double threshold for finding weak and strong and non relevant(for next step).

• Step 5 : Here is continue of previous part and will search around every pixels and if it exists any of the strong one near them, that change the value of that to strong pixel.


# Notice

larger kernels would remove more noise from the image.
But they will also mean more undesirable artifacts as well. 

For an example, a 7 x 7 Gaussian would filter out more noise than a 3 x 3 Gaussian kernel... But they again, the 7 x 7 would blur out edges more. Its the same with many other filter kernels. So, you would have to compromise between the 2 and pick a size.
