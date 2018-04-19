# adaptive thresholding method
def adaptive_thresholding(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    adaptive_threshed = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    cv2.imshow("adaptive", adaptive_threshed)
    return adaptive_threshed

# Normalize the thresholded image to a binary image
def normalize(img):
    normalizeImg = np.zeros_like(img)
    normalizeImg[img == 255] = 1
    return normalizeImg

def get_center(coordinates, colInterval):
    SIZE_THRESHOLD = 30
    index = 0
    centers = []
    while coordinates[index] != None:
        if (len(coordinates) > SIZE_THRESHOLD):
            sums = [0,0]
            for i in coordinates[index]:
                (row, col) = i
                sums[0] = sums[0] + row
                sums[1] = sums[1] + col
            sums[0] = int(math.floor(sums[0] / len(coordinates[index])))
            sums[1] = int(math.floor(sums[1] / len(coordinates[index]))) + colInterval
            sums = switchRowCol(sums)
            centers.append(tuple(sums))
            index = index + 1
    return centers

# Switch the row and col for the drawing function
def switchRowCol(origCoor):
    col = origCoor[1]
    row = origCoor[0]
    return [col, row]

# add the coordinates of same label to "coordinates"
def addCoordinates(coorNum, label, labelCoor, startRow, coordinates):
    for index in range(0, coorNum):
        labelCoor[index][0] = labelCoor[index][0] + startRow
        #labelCoorT = switchRowCol(labelCoor[index])
        if coordinates[label-2] != None:
            coordinates[label-2].append(labelCoor[index])
        else:
            coordinates[label-2] = [labelCoor[index]]

# double raster for image segmentation
# returns the center coordinates of each of the segment
def double_raster(imgTakein, startRow, colInterval):
    # take in binary image; startRow is the start row of the current image slice
    img = normalize(imgTakein)
    cur_label=2
    coordinates = [None] * 50
    eq=[0] * len(img)*len(img[0])
    for row in range(0,len(img)):
        for col in range(0,len(img[row])):
            if(img[row][col]==1):
                if(row>0):
                    up=img[row-1][col]
                else:
                    up=0
                if(col>0):
                    left=img[row,col-1]
                else:
                    left=0
                if(up==0 and left==0):
                    img[row][col]=cur_label
                    cur_label=cur_label+1
                elif(up!=0 and left!=0):
                    img[row][col]=min(up,left)
                    if(up!=left):
                        eq[max(up,left)]=min(up,left)
                        a=min(up,left)
                        while(eq[a]!=0):
                            eq[max(up,left)]=eq[a]
                            a=eq[a]
                elif(up==0 or left==0):
                    img[row][col]=max(up,left)

    # changed nested for loop of the second sweep to below, faster for 5-6 second
    max_label = cur_label   # record the max label number
    labelPixNumber = [0] * max_label    # The number of pixels in each label
    coorAdded = False   # switch of whether the coordinates has been recorded
    for label in range(0, max_label):
        labelCoor = np.argwhere(img == label)   # get the coordinates of pixels with same label
        coorNum = len(labelCoor)
        labelPixNumber[label] = coorNum
        if (eq[label] != 0):
            eqLabel = eq[label]
            img = eqLabel * (img == label) + img
            # Add the number of pixels of the current label to the equiv label
            # and set the current label pixel number to 0
            labelPixNumber[eqLabel] = labelPixNumber[eqLabel] + labelPixNumber[label]
            labelPixNumber[label] = 0
            addCoordinates(coorNum, eqLabel, labelCoor, startRow, coordinates)
            coorAdded = True
        if not coorAdded:
            addCoordinates(coorNum, label, labelCoor, startRow, coordinates)
        coorAdded = False

    centers = get_center(coordinates, colInterval)
    # print("finished double raster for one slice of image")
    return centers

# Returns   1. Segment centors (including two different paths)
#           2. bool path diverge state
def row_segment_center(img, NUM_SEGS, colInterval):
    global doubleRasterTime
    # Segment the original image into 20 segments
    numSegs = NUM_SEGS
    numRows = img.shape[0]
    numCols = img.shape[1]
    rowInterval = numRows/numSegs
    segmentCenters = [None] * numSegs
    blockCenters = []
    startRow = 0
    for i in range(0, numSegs):
        imgSeg = img[startRow:startRow+rowInterval, 0:numCols]
        # Threshold imageSegments and calculate the centor of each segments
        coor = np.argwhere(imgSeg == 255)
        if len(coor) == 0:
            rmean = img.shape[0]/2
            cmean = img.shape[1]/2
        else:
            rmean=int(math.floor(np.mean(coor[:,0])))
            cmean=int(math.floor(np.mean(coor[:,1])))

        segmentCenters[i] = (cmean+colInterval, startRow+rmean)
        startRow = startRow + rowInterval   # update row
        doubleRasterStart = time.time()
        blockCenters.append(double_raster(imgSeg, startRow, colInterval))
        doubleRasterEnd = time.time()

        doubleRasterTime += doubleRasterEnd - doubleRasterStart

    return segmentCenters, blockCenters
