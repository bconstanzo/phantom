"""
Functions and algorithms related to measuring things in images.
"""

# See:
# * Single View Metrology, Criminisi et al. (1999)

import cv2
import numpy as np
import math

def vanishing_points(x1,x2,y1,y2,z1,z2):
    """
    Calculates de vanishing points in an image. For this, it uses two straight
    lines in each direction.

    :param x1: Tuple of two points in a straight line, in the X plane
    :param x2: Tuple of two points in a straight line, in the X plane
    :param y1: Tuple of two points in a straight line, in the Y plane
    :param y2: Tuple of two points in a straight line, in the Y plane
    :param z1: Tuple of two points in a straight line, in the Z plane
    :param z2: Tuple of two points in a straight line, in the Z plane
    :return: List of three elements, each one being the vanishing point in the
        X, Y and Z planes.
    """
    
    xline1 = np.cross(x1[0], x1[1])
    xline2 = np.cross(x2[0], x2[1])
    v_x = np.cross(xline1,xline2)
    if v_x[2] != 0:
        v_x = v_x/v_x[2]

    yline1 = np.cross(y1[0], y1[1])
    yline2 = np.cross(y2[0], y2[1])
    v_y = np.cross(yline1,yline2)
    if v_y[2] != 0:
        v_y = v_y/v_y[2]

    zline1 = np.cross(z1[0], z1[1])
    zline2 = np.cross(z2[0], z2[1])
    v_z = np.cross(zline1,zline2)
    if v_z[2] != 0:
        v_z = v_z/v_z[2]

    return [v_x, v_y, v_z]

def define_plane(x1,x2,y1,y2,z1,z2,height,obj=None):
    """
    Defines the plane in an image, and the scale factor alfa
    :param x1: Tuple of two points in a straight line, in the X plane
    :param x2: Tuple of two points in a straight line, in the X plane
    :param y1: Tuple of two points in a straight line, in the Y plane
    :param y2: Tuple of two points in a straight line, in the Y plane
    :param z1: Tuple of two points in a straight line, in the Z plane
    :param z2: Tuple of two points in a straight line, in the Z plane
    :param height: Known height of the object in the image. His base and
        top points must be the same used in the parameter 'z1', or be
        specified in the parameter 'obj'
    :param obj: [optional] Tuple of two points in a straight line, in the Z
        plane, representing the base and top of the known height object
    :return: Returns two objects. The first one is the projection matrix
        (except for the third value, which is just 'v'), and the second one is
        the scale factor alfa
    """
    v_points = vanishing_points(x1,x2,y1,y2,z1,z2)
    v_line = np.cross(v_points[0],v_points[1])
    p4 = v_line/np.linalg.norm(v_line)
    p4 = p4/p4[2]
    if obj:
        alfa = -(np.linalg.norm(np.cross(obj[0],obj[1]))/(np.dot(p4,obj[0])*np.linalg.norm(np.cross(v_points[2], obj[1]))))/height
    else:    
        alfa = -(np.linalg.norm(np.cross(z1[0],z1[1]))/(np.dot(p4,z1[0])*np.linalg.norm(np.cross(v_points[2], z1[1]))))/height
    return [v_points[0], v_points[1], v_points[2], p4], alfa

def estimate_height(alfa, p, top_point_1, top_point_2, bottom_point, img=[]):
    """
    Estimates the height of an object in an image on the same plane (for
    example the grownd) as the object used to define the plane. Also, draws the
    intersection between the vanishing line of the bottom and the top of the
    object, with the vanishing line of the scene if the image is given.
    :param alfa: Scale factor alfa 
    :param p: Projection matrix (with 'v' as the third element)
    :param top_point_1: Point on the top of the object, creating a straight
        line with top_point_2
    :param top_point_2: Point on the top of the object, creating a straight
        line with top_point_1
    :param bottom_point: Point at the base of the object 
    :param img: [optional] Source image to be drawn on
    :return: Estimated height of the object
    """
    v_line = np.cross(p[0],p[1])
    top_line = np.cross(top_point_1,top_point_2)
    vanishing_point = np.cross(top_line,v_line)
    vanishing_point = vanishing_point/vanishing_point[2]

    top_point = np.cross(np.cross(bottom_point, p[2]), top_line)
    top_point = top_point/top_point[2]
    base_point = bottom_point

    Z = -(np.linalg.norm(np.cross(base_point,top_point))/((np.dot(p[3],base_point))*(np.linalg.norm(np.cross(p[2]*alfa, top_point)))))

    if len(img)>0:
        img = cv2.line(img,(int(vanishing_point[0]),int(vanishing_point[1])),(int(base_point[0]),int(base_point[1])),[0,0,255],thickness=5)
        img = cv2.line(img,(int(vanishing_point[0]),int(vanishing_point[1])),(int(top_point[0]),int(top_point[1])),[0,0,255],thickness=5)
        img = cv2.line(img,(int(base_point[0]),int(base_point[1])),(int(top_point[0]),int(top_point[1])), [255,0,0], thickness=5)
    return Z

def height_parallel_planes(p, zr, alfa, top_point_1, top_point_2, bottom, img=[]):
    """
    Estimates the distance between 2 planes, different from the plane used to
    define the projection matrix. Also, draws the intersection between the
    vanishing line of the bottom and the top of the object, with the vanishing
    line of the scene if the image is given.
    :param p: Projection matrix
    :param zr: Distance between the plane used as reference for the projection
        matrix, and the bottom plane of the other two
    :param alfa: Scale factor alfa 
    :param top_point_1: Point on the top of the object, creating a straight
        line with top_point_2
    :param top_point_2: Point on the top of the object, creating a straight
        line with top_point_1
    :param bottom_point: Point at the base of the object 
    :param img: [optional] Source image to be drawn on
    :return: Estimated height of the object
    """
    p2 = p
    p2[3] = zr*p[2]+p[3]
    
    top_line = np.cross(top_point_1,top_point_2)
    v_line = np.cross(p2[0],p2[1])
    vanishing_point = np.cross(top_line,v_line)
    vanishing_point = vanishing_point/vanishing_point[2]
    #lb = np.cross(pb,vanishing_point)

    top_point = np.cross(np.cross(bottom, p2[2]), top_line)
    top_point = top_point/top_point[2]
    base_point = bottom

    aux = np.dot(p2[3],base_point)/(1+np.dot((zr*p2[2]*alfa),p2[3]))
    Z = -np.linalg.norm(np.cross(base_point,top_point))/(aux*np.linalg.norm(np.cross(p2[2]*alfa, top_point)))
    
    if len(img)>0:
        img = cv2.line(img,(int(vanishing_point[0]),int(vanishing_point[1])),(int(base_point[0]),int(base_point[1])),[0,0,255],thickness=3)
        img = cv2.line(img,(int(vanishing_point[0]),int(vanishing_point[1])),(int(top_point[0]),int(top_point[1])),[0,0,255],thickness=3)
        img = cv2.line(img,(int(base_point[0]),int(base_point[1])),(int(top_point[0]),int(top_point[1])), [255,0,0], thickness=3)
    return Z

def calculate_angle(line_1, line_2):
    """
    Calculates the angle between two lines

    :param line_1: First line, defined like ((x1,y1,z1),(x2,y2,z2))
    :param line_2: Second line, defined like ((x1,y1,z1),(x2,y2,z2))
    :return: Angle of separation
    """
    cat_1 = abs(line_1[0][0]-line_1[1][0])
    cat_2 = abs(line_1[0][1]-line_1[1][1])
    A = math.sqrt(cat_1**2 + cat_2**2)

    cat_1 = abs(line_2[0][0]-line_2[1][0])
    cat_2 = abs(line_2[0][1]-line_2[1][1])
    B = math.sqrt(cat_1**2 + cat_2**2)

    p1 = line_1[0]
    p2 = (line_1[1][0] + line_2[0][0]-line_2[1][0], line_1[1][1] + line_2[0][1]-line_2[1][1])
    cat_1 = abs(p1[0]-p2[0])
    cat_2 = abs(p1[1]-p2[1])
    C = math.sqrt(cat_1**2 + cat_2**2)

    angle = math.degrees(math.acos((A**2 + B**2 - C**2)/(2*A*B)))
    return angle

def distance_line_point(p1, p2, p3):
    """
    Calculates the distance between a line and a point

    :param p1: Point on the line defined like (x,y)
    :param p2: Point on the line defined like (x,y)
    :param p3: Point outside the line defined like (x,y)
    :return: Distance between the line and the point
    """
    a = (p2[1]-p1[1])/(p2[0]-p1[0])
    b = a*p1[0]-p1[1]
    top = abs(a*p3[0]-1*p3[1]+b) 
    bottom = math.sqrt((a**2)+1)
    d = top/bottom
    return d

def detect_plane(img, obj, height):
    """
    Detects and defines the space on an image and a scale factor 'alfa', based
    on the detection of straight lines using Canny edge detector and Hough line
    transform, and the height of an object in said image.
    :param img: Image used as reference
    :param obj: Base and Top points of the reference object defined like 
        ((xbase, ybase, z), (xtop, ytop, z))
    :param height: Height of the object
    :return: Returns two objects. The first one is the projection matrix
        (except for the third value, which is just 'v'), and the second one is
        the scale factor alfa 
    """
    # Border and lines detection
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,350,400,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,None,minLineLength,maxLineGap)

    # Divinding lines in three different lists, according to their angle in
    # relation with a selected line (the selected line must be horizontal)
    vertical_lines = []
    horizontal_lines = []
    descarded_lines = []

    original = lines[0]
    horizontal_lines.append(original)
    p11 = (original[0][0],original[0][1],1)
    p12 = (original[0][2],original[0][3],1)
    selected_line = np.cross(p11,p12)
    selected_line = selected_line/selected_line[2]
    calculate_angle((p11, p12), (p11,p12))
    for second_line in lines[1:]:
        p21 = (second_line[0][0],second_line[0][1],1)
        p22 = (second_line[0][2],second_line[0][3],1)
        comparing_line = np.cross(p21,p22)
        comparing_line = comparing_line/comparing_line[2]
        angle = calculate_angle((p11,p12), (p21,p22))
        if angle<5: 
            horizontal_lines.append(second_line)
        elif angle > 40:
            vertical_lines.append(second_line)
        else:
            descarded_lines.append(second_line)

    #Calculating the intersection points between all the lines
    points = []
    for i,line in enumerate(horizontal_lines, start=1):
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        first_line = np.cross(first_point, second_point)
        for second_line in horizontal_lines[i:]:
            x0, y0, x1, y1 = second_line[0]
            first_point = (x0,y0,1)
            second_point = (x1,y1,1)
            second_line = np.cross(first_point, second_point)
            point = np.cross(first_line, second_line)
            point = point/point[2]
            point = point[0:2]
            points.append(point)

    median = np.median(points, axis=0)
    std = np.std(points, axis=0)

    # Filtering the points that are too far away, and reorganizing them in two 
    # lists, one with the X position and one with the Y position
    x_points = []
    y_points = []
    for point in points: # FALTA FILTRAR MEJOR PUNTOS LEJANOS
        if (point[0]>median[0]-std[0] and point[0]<median[0]+std[0] and
            point[1]>median[1]-std[1] and point[1]<median[1]+std[1]):
            x_points.append(point[0])
            y_points.append(point[1])

    # Least square fitting to estimate the horizon line
    mean_x = np.mean(x_points)
    mean_y = np.mean(y_points)
    n = len(x_points)
    top = 0
    bottom = 0
    for i in range(n):
        top += (x_points[i] - mean_x) * (y_points[i] - mean_y)
        bottom += (x_points[i] - mean_x) ** 2
    m = top / bottom
    b = mean_y - (m * mean_x)

    point_1 = (0,b,1)
    point_2 = (img.shape[1],(img.shape[1]*m)+b,1)
    p = (point_1,point_2)
    horizon = np.cross(point_1,point_2)

    # Intersecting every horizontal line with the horizon
    intersection_points = []
    for line in lines:
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        line = np.cross(first_point, second_point)
        point = np.cross(line, horizon)
        point = point/point[2]
        intersection_points.append(point)

    # Divinding the intersection points in two big groups, one on the left side
    intersection_1 = []
    intersection_2 = []
    mean = np.mean(intersection_points, axis=0)
    for point in intersection_points:
        if point[0]<mean[0]:
            intersection_1.append(point)
        else:
            intersection_2.append(point)

    intersection_1 = np.array(intersection_1)
    intersection_2 = np.array(intersection_2)
    x_lines = []
    y_lines = []

    # Divinding the horizontal line in two groups, according with the group 
    # that their intersection point with the horizon line landed
    for line in horizontal_lines:
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        aux = np.cross(first_point, second_point)
        point = np.cross(aux, horizon)
        point = point/point[2]
        if (point == intersection_1).all(axis=1).any():
            x_lines.append(line)
        else:
            y_lines.append(line)

    # Calculating the intersection point between the lines on the X group
    points = []
    for i,line in enumerate(x_lines, start=1):
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        first_line = np.cross(first_point, second_point)
        for second_line in x_lines[i:]:
            x0, y0, x1, y1 = second_line[0]
            first_point = (x0,y0,1)
            second_point = (x1,y1,1)
            second_line = np.cross(first_point, second_point)
            point = np.cross(first_line, second_line)
            if point[0]:
                point = point/point[2]
                point = point[0:2]
                points.append(point)

    # Calculating the distance between every X line, and the mean point of 
    # intersection with the horizon estimation
    mean = np.mean(points, axis=0)
    x_lines_filtered = []
    x_distances = []
    for line in x_lines:
        p1 = (line[0][0],line[0][1])
        p2 = (line[0][2],line[0][3])
        p3 = mean
        d = distance_line_point(p1,p2,p3)
        x_distances.append(d)

    # If the distance between the line and that point is too far, the line is 
    # discarted
    mean_d = np.mean(x_distances)
    for i,d in enumerate(x_distances, start=0):
        if d < mean_d*1.2:
            x_lines_filtered.append(x_lines[i])

    # Calculating the intersection point between the lines on the Y group
    points = []
    for i,line in enumerate(y_lines, start=1):
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        first_line = np.cross(first_point, second_point)
        for second_line in y_lines[i:]:
            x0, y0, x1, y1 = second_line[0]
            first_point = (x0,y0,1)
            second_point = (x1,y1,1)
            second_line = np.cross(first_point, second_point)
            point = np.cross(first_line, second_line)
            if point[0]:
                point = point/point[2]
                point = point[0:2]
                points.append(point)

    # Calculating the distance between every Y line, and the mean point of 
    # intersection with the horizon estimation
    mean = np.mean(points, axis=0)
    vanishing_y = mean
    std = np.std(points, axis=0)
    y_lines_filtered = []
    y_distances = []
    for line in y_lines:
        p1 = (line[0][0],line[0][1])
        p2 = (line[0][2],line[0][3])
        p3 = mean
        d = distance_line_point(p1,p2,p3)
        y_distances.append(d)

    # If the distance between the line and that point is too far, the line is 
    # discarted
    mean_d = np.mean(y_distances)
    for i,d in enumerate(y_distances, start=0):
        if d < mean_d*1.2:
            y_lines_filtered.append(y_lines[i])

    # Calculating the intersection between all the filtered Y lines to generate
    # a better estimation of the horizon
    points = []
    for i,line in enumerate(y_lines_filtered, start=1):
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        first_line = np.cross(first_point, second_point)
        for second_line in y_lines_filtered[i:]:
            x0, y0, x1, y1 = second_line[0]
            first_point = (x0,y0,1)
            second_point = (x1,y1,1)
            second_line = np.cross(first_point, second_point)
            point = np.cross(first_line, second_line)
            if point[0]:
                point = point/point[2]
                point = point[0:2]
                points.append(point)

    # Again, if the point is too far away from the rest, it's discarted
    median = np.median(points, axis=0)
    std = np.std(points, axis=0)
    y_points = []
    for point in points: # FALTA FILTRAR MEJOR PUNTOS LEJANOS
        if (point[0]>median[0]-std[0]*1.5 and point[0]<median[0]+std[0]*1.5 and
            point[1]>median[1]-std[1]*1.5 and point[1]<median[1]+std[1]*1.5):
            y_points.append(point)

    # The final vanishing point in the Y direction is the mean of the 
    # intersection of all the lines in the Y direction, after the filtering
    vanishing_y = np.mean(y_points, axis=0)

    # Calculating the intersection between all the filtered X lines to generate
    # a better estimation of the horizon
    points = []
    for i,line in enumerate(x_lines_filtered, start=1):
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        first_line = np.cross(first_point, second_point)
        for second_line in x_lines_filtered[i:]:
            x0, y0, x1, y1 = second_line[0]
            first_point = (x0,y0,1)
            second_point = (x1,y1,1)
            second_line = np.cross(first_point, second_point)
            point = np.cross(first_line, second_line)
            if point[0]:
                point = point/point[2]
                point = point[0:2]
                points.append(point)

    # Again, if the point is too far away from the rest, it's discarted
    median = np.median(points, axis=0)
    std = np.std(points, axis=0)
    x_points = []
    for point in points: # FALTA FILTRAR MEJOR PUNTOS LEJANOS
        if (point[0]>median[0]-std[0]*1.5 and point[0]<median[0]+std[0]*1.5 and
            point[1]>median[1]-std[1]*1.5 and point[1]<median[1]+std[1]*1.5):
            x_points.append(point)

    # The final vanishing point in the X direction is the mean of the 
    # intersection of all the lines in the X direction, after the filtering
    vanishing_x = np.mean(x_points, axis=0)

    # Now, doing the same thing as for the X and Y vanishing points,
    # but for the vertical (Z) one
    points = []
    for i,line in enumerate(vertical_lines, start=1):
        x0, y0, x1, y1 = line[0]
        first_point = (x0,y0,1)
        second_point = (x1,y1,1)
        first_line = np.cross(first_point, second_point)
        for second_line in vertical_lines[i:]:
            x0, y0, x1, y1 = second_line[0]
            first_point = (x0,y0,1)
            second_point = (x1,y1,1)
            second_line = np.cross(first_point, second_point)
            point = np.cross(first_line, second_line)
            if point[0]:
                point = point/point[2]
                point = point[0:2]
                points.append(point)

    median = np.median(points, axis=0)
    std = np.std(points, axis=0)
    z_points = []
    # In addition to discarding the intersection points that are too far away,
    # all the points that have a different sign than the median are also
    # discarded
    for point in points: # FALTA FILTRAR MEJOR PUNTOS LEJANOS
        if (point[1] * median[1] > 0 and 
            point[0]>median[0]-std[0]*1.5 and point[0]<median[0]+std[0]*1.5 and
            point[1]>median[1]-std[1]*1.5 and point[1]<median[1]+std[1]*1.5):
            z_points.append(point)

    # The mean between all the final points is the vertical vanishing point
    vanishing_z = np.mean(z_points, axis=0)

    # Calculating the real horizon line
    p1 = (vanishing_x[0],vanishing_x[1],1)
    p2 = (vanishing_y[0],vanishing_y[1],1)
    horizon = np.cross(p1,p2)

    # Calculating P4
    p4 = horizon/np.linalg.norm(horizon)
    p4 = p4/p4[2]

    # Calculating alfa
    alfa = -(np.linalg.norm(np.cross(obj[0],obj[1]))/(np.dot(p4,obj[0])*np.linalg.norm(np.cross(vanishing_z, obj[1]))))/height
    
    return [np.array([vanishing_x[0],vanishing_x[1],1]), np.array([vanishing_y[0],vanishing_y[1],1]), np.array([vanishing_z[0],vanishing_z[1],1]), p4], alfa
