import cv2
from camera_utils import stitch_two_images, show_and_save_stitch
img_left = cv2.imread("images/stitching/left.jpg")
img_right = cv2.imread("images/stitching/right.jpg")
stitched_img, homography_matrix = stitch_two_images(img_left, img_right)

'''
Homography matrix gives how to warp image 1 so it aligns to image 2.
[[ 1.01806988e+00 -5.62649798e-03 -7.30793160e+02]
 [ 2.79057491e-02  9.88905932e-01  1.57362233e+02]
 [ 1.57738415e-05 -4.69213040e-06  1.00000000e+00]]

 Since image1 should be warped to left of image2, the cordinates in homography is negative (opencv cant create images with -ve coordinates).
 We translate the homography matrix to make it work on opencv.
'''
print(homography_matrix)
show_and_save_stitch(img_left, img_right, stitched_img, "outputs/stitching/stitched_figure.png")