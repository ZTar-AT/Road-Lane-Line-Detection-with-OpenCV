
import cv2  
import numpy as np
import matplotlib.pyplot as plt 


def nothing(x):
	pass

img = cv2.imread('example/1.png')
img = cv2.resize(img, (500,500))
rows,cols = img.shape[0],img.shape[1]

cv2.namedWindow('controls')

cv2.createTrackbar('thresh1','controls',30,500,nothing)
cv2.createTrackbar('thresh2','controls',150,500,nothing)
cv2.createTrackbar('min','controls',50,rows-200,nothing)
cv2.createTrackbar('max','controls',50,cols-200,nothing)
kernel=np.ones((5,5),np.uint8)


while(1):
 

	original=img.copy()
	Blur=img.copy()
	Blur=cv2.GaussianBlur(Blur,(3,3),1.4)



	thresh1= int(cv2.getTrackbarPos('thresh1','controls'))
	thresh2= int(cv2.getTrackbarPos('thresh2','controls'))
	min= int(cv2.getTrackbarPos('min','controls'))
	max= int(cv2.getTrackbarPos('max','controls'))



	gray = cv2.cvtColor( Blur, cv2.COLOR_BGR2GRAY)
	ret, img_threshold = cv2.threshold(gray, 0, 1, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
	

	
	closing=cv2.erode(cv2.dilate(img_threshold,kernel,iterations=1),kernel,iterations=1)
	#iterations คือการทำซ้ำรูปภาพเดิมตามจำนวนที่ใส่
	edges = cv2.Canny(closing,thresh1,thresh2)


	lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=min,maxLineGap=max)

	if (str(lines)!="None"):
		for i in range(len(lines)):
		    for x1,y1,x2,y2 in lines[i]:
		        
		        cv2.line( original, (x1,y1), (x2,y2), (0,0,255),2)
	
	cv2.imshow('Draw Line', cv2.resize(original,(600,600)))
	cv2.imshow('closing', cv2.resize(closing,(400,400)))
	cv2.imshow('Edges', cv2.resize(edges,(400,400)))
	
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
		

cv2.destroyAllWindows()