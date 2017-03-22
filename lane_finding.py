# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

SHOW_IMAGE = True
SAVE_IMAGE = False

def show_image(img, title, gray=True):
     if gray:
         plt.imshow(img, cmap='gray')
     else:
         plt.imshow(img)
     plt.title(title)
     plt.show()
     
def save_image(img, filepath):
    cv2.imwrite('output_images/' + filepath, img)

def get_calibrate_matrix():
    nx = 9 # number of insider corners in x
    ny = 6 # number of insider corners in y
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # object points
    
    objpoints = [] # 3d points in world space
    imgpoints = [] # 2d points in image plane
    
    img_size = (0, 0)
    for i in range(1, 21):
        frame = 'camera_cal/calibration%s.jpg' % (i)
        img = cv2.imread(frame)
        if i == 1:
            img_size = (img.shape[1], img.shape[0])
         
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret:
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            objpoints.append(objp)
            imgpoints.append(corners)
            #if SHOW_IMAGE:
            #    show_image(img, 'Chessboard image %d with corners' % i)
            if SAVE_IMAGE:
                save_image(img, 'chessborad_%i.jpg')
        else:
            print('image %d cannot find chessboard' % i)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    if SHOW_IMAGE:
        print('original image vs undistort image:')
        print(img.shape)
        print(dst.shape)
        show_image(img, 'original image', False)
        show_image(dst, 'undistort image', False)
    if SAVE_IMAGE:
        save_image(dst, 'undistort.jpg')
    return dst

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img
    
def gaussian_smooth(img, kernel_size=5):
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    if SHOW_IMAGE:
        show_image(img, 'gaussian_smooth')
    return img
    
def gradient_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # x direction gradient
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # y direction gradient
    # magnitude    
    sobel = np.sqrt(sobelx **2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    magnitude_output = np.zeros_like(scaled_sobel)
    magnitude_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    if SHOW_IMAGE:
        show_image(magnitude_output, 'gradient_magnitude_thresh')
    # direction
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        direction_output = np.zeros_like(scaled_sobel)
        direction_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1
        if SHOW_IMAGE:
            show_image(direction_output, 'graident_direction_thresh')
    output = np.zeros_like(scaled_sobel)
    output[(magnitude_output > 0) & (direction_output > 0)] = 1
    if SHOW_IMAGE:
        show_image(output, 'gradient_thresh')
    return output

def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if SHOW_IMAGE:
        show_image(img, 'gray')
    return img

def color_thresh(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    s_binary = np.zeros_like(s)
    s_binary[(s >= thresh[0]) & (s <= thresh[1])] = 1
    if SHOW_IMAGE:
        show_image(s_binary, 'color_thresh')
    return s_binary    

def unwarp(img):
    area_of_interest = [[600,460],[725,460],[1050,650],[310,670]]
    offset1 = 200
    offset2 = 0
    offset3 = 0
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(area_of_interest)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    dst = np.float32([[offset1, offset3], 
                      [img_size[0]-offset1, offset3], 
                      [img_size[0]-offset1, img_size[1]-offset2], 
                      [offset1, img_size[1]-offset2]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    return warped, M, Minv

def find_position(image):
    # Find the position of the car from the center
    # It will show if the car is 'x' meters from the left or right
    position = image.shape[1]/2
    
    left_pts = left_line.pts.reshape((720, -1))
    right_pts = right_line.pts.reshape((720, -1))
    
    left  = np.min(left_pts[(left_pts[:,0] < position) & (left_pts[:,1] > 700)][:,0])
    right = np.max(right_pts[(right_pts[:,0] > position) & (right_pts[:,1] > 700)][:,0])
   
    center = (left + right)/2
    print(left, right)
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension    
    return round((position - center)*xm_per_pix, 2)
    
def plot_lanes(undist, warped, Minv):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts = np.hstack((left_line.pts, right_line.pts))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    image = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    deviation = find_position(undist)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: %s m" % int(left_line.curvature)
    text2 = "Car is %sm %s of center" % (abs(deviation), 'right' if deviation >0 else 'left')
    cv2.putText(image,text,(400,100), font, 1,(255,255,255),2)
    cv2.putText(image,text2,(400,150), font, 1,(255,255,255),2)
    
    plt.imshow(image)
    plt.show()
    return image
    

def find_lanes(img):
    #histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    #plt.plot(histogram)

    #histogram = np.sum(img, axis=0)
    #plt.plot(histogram)
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    if left_line.fit and right_line.fit:
        left_lane_inds = ((nonzerox > (left_line.fit[0]*(nonzeroy**2) + left_line.fit[1]*nonzeroy + left_line.fit[2] - margin)) & \
                          (nonzerox < (left_line.fit[0]*(nonzeroy**2) + left_line.fit[1]*nonzeroy + left_line.fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_line.fit[0]*(nonzeroy**2) + right_line.fit[1]*nonzeroy + right_line.fit[2] - margin)) & \
                           (nonzerox < (right_line.fit[0]*(nonzeroy**2) + right_line.fit[1]*nonzeroy + right_line.fit[2] + margin)))
    else:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            #print(win_y_low, win_y_high)
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    left_line.curvature = left_curverad
    right_line.curvature = right_curverad
    
    left_line.pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line.pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

def process_pipeline(image, mtx, dist):
    # undistort
    src = undistort_image(image, mtx, dist)
    img = np.copy(src)
    img = gray(img)
    img = gaussian_smooth(img, 5)
    # color thresh
    color_thresh_img = color_thresh(src, (160, 255))
    # gradient thresh
    gradient_thresh_img = gradient_thresh(img, 11, (40, 255), (.65, 1.05))
    output = np.zeros_like(img)
    output[(gradient_thresh_img > 0) | (color_thresh_img > 0)] = 1
    if SHOW_IMAGE:
        show_image(output, 'gradient and color thresh')
    
    # Masked area
    imshape = img.shape
    left_bottom = (100, imshape[0])
    right_bottom = (imshape[1]-20, imshape[0])
    apex1 = (610, 410)
    apex2 = (680, 410)
    inner_left_bottom = (310, imshape[0])
    inner_right_bottom = (1150, imshape[0])
    inner_apex1 = (700,480)
    inner_apex2 = (650,480)
    vertices = np.array([[left_bottom, apex1, apex2, \
                          right_bottom, inner_right_bottom, \
                          inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)
    #output = region_of_interest(output, vertices)
    if SHOW_IMAGE:
        show_image(output, 'region of interest')
        
    unwarp_img, M, Minv = unwarp(output)
    if SHOW_IMAGE:
        show_image(unwarp_img, 'unwarp')
    
    find_lanes(unwarp_img)
    dst = plot_lanes(src, output, Minv)
    return dst
        
def process_image(image):
    return process_pipeline(image, mtx, dist)

class Line():
    def __init__(self):
        self.fit = None
        self.pts = None
        self.curvature = 0

mtx, dist = get_calibrate_matrix()    
white_output = 'my.mp4'
clip1 = VideoFileClip("project_video.mp4")
left_line = Line()
right_line = Line()
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

#image = plt.imread("camera_cal/calibration1.jpg")
#show_image(image, 'original')
#undistored = cv2.undistort(image, mtx, dist, None, mtx)
#show_image(undistored, "undistored image")