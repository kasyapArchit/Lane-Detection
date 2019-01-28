import cv2
import numpy as np

img = cv2.imread("images/6.jpg",1)
#cv2.imshow("img", img)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)

# It helps better distinguish the lane markers
hls_img = convert_hls(img)
#cv2.imshow("hls", hls_img)

# Considering only the yellow and white colors(they repreent the lane markers)
wy_img = select_white_yellow(img)
#cv2.imshow("wyim",wy_img)

# Converting to single channel
gray_img = convert_gray_scale(wy_img)
#cv2.imshow("gray",gray_img)

# apply gaussian smoothing
gau_smth = apply_smoothing(gray_img, kernel_size=5)
#cv2.imshow("gau",gau_smth)

# Use canny edge detector to detect edges
edg_img = detect_edges(gau_smth)
#cv2.imshow("edg",edg_img)

# filter out the intrest regions
roi_img = select_region(edg_img)
#cv2.imshow("roi",roi_img)

# apply hough transform
hou_lines = hough_lines(roi_img)
for line in hou_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 5)
cv2.imshow("hou", img)

cv2.waitKey(0)
cv2.destroyAllWindows()