"""
Find bounding box (open rectangle) using cv2.findContours and cv2.boundingRect.
"""

import cv2


# Normal routines
img = cv2.imread('square.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, 1)

# Remove some small noise if any.
dilate = cv2.dilate(thresh, None)
erode = cv2.erode(dilate, None)

# Find contours with cv2.RETR_CCOMP
contours, hierarchy = cv2.findContours(erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    # Check if it is an external contour and its area is more than 100
    if hierarchy[0, i, 3] == -1 and cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        m = cv2.moments(cnt)
        cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        cv2.circle(img, (int(cx), int(cy)), 3, 255, -1)

cv2.imshow('img', img)
cv2.imwrite('sofsqure.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
