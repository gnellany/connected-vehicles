import cv2
import numpy as np


def sift(img1,img2):
    im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, des1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, des2 = sift.detectAndCompute(im2Gray, None)

    # initialize Brute force matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1,des2)
    #sort the matches
    matches = sorted(matches, key= lambda match : match.distance)
    #get good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    #draw matches
    #matched_imge = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:30], None)
    # Extract location of good matches
    ###

    ###
    im1Reg = alin(img1,img2,matches,keypoints1,keypoints2)

    return im1Reg

def orb(img1,img2):

    im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, des1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, des2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches//not needed to run
    #imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
    im1Reg = alin(img1,img2,matches,keypoints1,keypoints2)

    return im1Reg


def alin(img1,img2,matches,keypoints1,keypoints2):

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = img2.shape
    im1Reg = cv2.warpPerspective(img1, h, (width, height))
    print(h)
    return im1Reg


#camera feed
video1,video2 = cv2.VideoCapture(0),cv2.VideoCapture(1)


MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.2


while True:
#read camera to while loop
    ret, image1 = video1.read()
    ret, image2 = video2.read()

    orb_img = orb(image1,image2)
    sift_img = sift(image1,image2)

    img = np.concatenate((image1,image2,orb_img,sift_img), axis=1)
    cv2.imshow("Matching Images", img)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break

cv2.destroyAllWindows()