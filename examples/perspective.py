"""
Tests geomtric measurements over images.

---
Made by: muramena
"""
from phantom.measures import *
from matplotlib import pyplot as plt

# A few considerations when trying to find the height of an object
# In case of defining the plane manually:
    # You will need 12 points to define the space (two for each line, and two 
    # lines for the 3 axes X, Y and Z)
    # Know height, with the corrisponding base and top points of the reference
    # object. You can use this two points as reference for the vertical plane
    # Three more points to determine the object you want to estimate the height
    # Two on the top (creating a straight line), and one on the bottom
    # So, at least you'll need 15 points and a known height
# In case of detecting the plane automatically:
    # You will need five points: two for the base and the top of the reference
    # object and the other three are to determine the object you want to 
    # estimate the height. Two on the top (creating a straight line), and one
    # on the bottom. And you'll also need to specify the height of the 
    # reference object

# --------------------------------------------------------------------------- #
# EXAMPLE 1 - Defining the plane manually, and estimating the height

""" # Reading image
img = cv2.imread('shed.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Setting the points of the lines used later for the definition of the plane
# z11 and z12 are also the object with known height
known_height = 294.3
x11 = (569,285,1)
x12 = (747,301,1)
img = cv2.line(img,(int(x11[0]),int(x11[1])),(int(x12[0]),int(x12[1])),[255,255,0],thickness=2)
x21 = (573,468,1) 
x22 = (731,466,1)
img = cv2.line(img,(int(x21[0]),int(x21[1])),(int(x22[0]),int(x22[1])),[255,255,0],thickness=2)
y11 = (497,376,1)
y12 = (420,379,1)
img = cv2.line(img,(int(y11[0]),int(y11[1])),(int(y12[0]),int(y12[1])),[255,255,0],thickness=2)
y21 = (533,627,1)
y22 = (208,530,1)
img = cv2.line(img,(int(y21[0]),int(y21[1])),(int(y22[0]),int(y22[1])),[255,255,0],thickness=2)
z11 = (544,634,1)
z12 = (550,235,1)
img = cv2.line(img,(int(z11[0]),int(z11[1])),(int(z12[0]),int(z12[1])),[255,255,0],thickness=2)
z21 = (409,643,1)
z22 = (416,327,1)
img = cv2.line(img,(int(z21[0]),int(z21[1])),(int(z22[0]),int(z22[1])),[255,255,0],thickness=2)

# Points in the base and the top of the man. The two points on the top are to
# find the exact straight vertical line from the base one, according to the 
# definition of the plane.
top1 = (319,390,1)
top2 = (332,390,1)
base = (330,608,1)

# Defining the plane and estimating the height, while drawing the man's 
# reference points
plane, alfa = define_plane((x11,x12),(x21,x22),(y11,y12),(y21,y22),(z11,z12),(z21,z22),known_height)
Height = estimate_height(alfa,plane,top1,top2,base, img)

# Drawing the horizon and showing the result
img = cv2.line(img,(int(plane[0][0]),int(plane[0][1])),(int(plane[1][0]),int(plane[1][1])),[0,255,0],thickness=3)
plt.title(f'Estimated height: {Height:.2f}cm')
plt.suptitle('Plane defined manually (real height of the man: 180cm)')
plt.imshow(img)
plt.show() """

# --------------------------------------------------------------------------- #

# EXAMPLE 2 - Defining the plane manually, and estimating the height

""" # Reading the image
img = cv2.imread('phone.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Setting the points of the lines used later for the definition of the plane
x11 = (1260,2493,1)
x12 = (2031,2630,1)
img = cv2.line(img,(int(x11[0]),int(x11[1])),(int(x12[0]),int(x12[1])),[255,255,0],thickness=5)
x21 = (1017,3840,1) 
x22 = (1658,4032,1)
img = cv2.line(img,(int(x21[0]),int(x21[1])),(int(x22[0]),int(x22[1])),[255,255,0],thickness=5)
y11 = (2371,3466,1)
y12 = (2248,4060,1)
img = cv2.line(img,(int(y11[0]),int(y11[1])),(int(y12[0]),int(y12[1])),[255,255,0],thickness=5)
y21 = (830,3149,1)
y22 = (982,2992,1)
img = cv2.line(img,(int(y21[0]),int(y21[1])),(int(y22[0]),int(y22[1])),[255,255,0],thickness=5)
z11 = (2125,3045,1)
z12 = (2178,2718,1)
img = cv2.line(img,(int(z11[0]),int(z11[1])),(int(z12[0]),int(z12[1])),[255,255,0],thickness=5)
z21 = (1033,2301,1)
z22 = (1030,1650,1)
img = cv2.line(img,(int(z21[0]),int(z21[1])),(int(z22[0]),int(z22[1])),[255,255,0],thickness=5)
h1 = (520,3143,1)
h2 = (477,2636,1)
img = cv2.line(img,(int(h1[0]),int(h1[1])),(int(h2[0]),int(h2[1])),[0,255,0],thickness=5)

known_height = 11
plane,alfa = define_plane((x11,x12),(x21,x22),(y11,y12),(y21,y22),(z11,z12),(z21,z22),known_height,obj=(h1,h2))

# Base and top of the reference object
top1 = (2381,2427,1)
top2 = (2652,2471,1)
base = (2363,3226,1)

Height = estimate_height(alfa,plane,top1,top2,base,img)
plt.title(f'Estimated height: {Height:.2f}cm')
plt.suptitle('Plane defined manually (real height of the phone: 15)')
plt.imshow(img)
plt.show() """

# --------------------------------------------------------------------------- #

# EXAMPLE 3 - Automatic detection of plane, and estimation of height

# Reading image
img = cv2.imread('shed.jpg')

# Base, top and height of the reference object
z11 = (544,634,1)
z12 = (550,235,1)
height = 294.3

# Plane detection and definition
plane, alfa = detect_plane(img, (z11,z12), height)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.line(img,(int(z11[0]),int(z11[1])),(int(z12[0]),int(z12[1])),[255,255,0],thickness=2)

# Top and base points for the man
top1 = (319,392,1) 
top2 = (332,392,1)
base = (330,608,1)

# Estimating height
height = estimate_height(alfa,plane,top1,top2,base,img)

# Showing results
img = cv2.line(img,(int(plane[0][0]),int(plane[0][1])),(int(plane[1][0]),int(plane[1][1])),[0,255,0],thickness=3)
plt.imshow(img)
plt.title(f'Estimated height: {height:.2f}cm')
plt.suptitle('Plane defined automatically (real height of the man: 180cm)')
plt.show()
