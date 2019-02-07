import cv2
import numpy as np

img = cv2.imread("images/4.jpg",1)
img = cv2.resize(img, (600, 400))
cv2.imshow("img", img)

# We change the color space to HSL as it helps better visualize the lane markers.
hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
cv2.imshow("hls", hls_img)

# Considering only the yellow and white colors(they repreent the lane markers)
# white color mask
lw_w = np.uint8([0, 200, 0])
up_w = np.uint8([255, 255, 255])
wt = cv2.inRange(hls_img, lw_w, up_w)
# yellow color mask
lw_y = np.uint8([10, 0, 100])
up_y = np.uint8([40, 255, 255])
yw = cv2.inRange(hls_img, lw_y, up_y)
# combine the mask
mask = cv2.bitwise_or(wt, yw)
wy_img = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow("wyim",wy_img)

# Now we convert the scale to grayscale so that we can get both the whote and yellow in same scale and it also helps to do gaussian smoothing
gray_img = cv2.cvtColor(wy_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray_img)

# apply gaussian smoothing
gau_smth = cv2.GaussianBlur(gray_img, (7, 7), 3)
cv2.imshow("gau",gau_smth)

# We use canny edge detector to detect the edges.
edg_img = cv2.Canny(gau_smth, 30, 100)
cv2.imshow("edg",edg_img)

# The canny edge detector returns the edges even for noises like trees, other cars. Since the road forms a region from the bottom of the image(a polygon region) we try to filter out this region so that we can get better accuracy in deciding the lane markers(as other edges don't generally occur in this region such as trees, passerby's)
# Thus we form the polygonal region below by making vertices for the region
rows, cols = edg_img.shape[:2]
bt_lt  = [cols*0.1, rows*0.95]
tp_lt     = [cols*0.4, rows*0.6]
bt_rt = [cols*0.9, rows*0.95]
tp_rt    = [cols*0.6, rows*0.6] 
vt = np.array([[bt_lt, tp_lt, tp_rt, bt_rt]], dtype=np.int32)
# now we filter out the edges only in this region
mask = np.zeros_like(edg_img)
if len(mask.shape)==2:
    cv2.fillPoly(mask, vt, 255)
else:
    cv2.fillPoly(mask, vt, (255,)*mask.shape[2])       

roi_img = cv2.bitwise_and(edg_img, mask)
cv2.imshow("roi",roi_img)

# apply hough transform
hou_lines = cv2.HoughLinesP(roi_img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)
for line in hou_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 4)
cv2.imshow("res", img)

cv2.waitKey(0)
# cv2.destroyAllWindows()