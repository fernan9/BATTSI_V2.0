# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 13:15:30 2022

Biological Activity Transformation Tool
Verison 2.0
Let's try to run a single script

@author: Rodrigo Perez
"""
import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

# Function to extract frames 
def FirstFrame(folder, vidName): 
    # Path to video file 
    vidObj = cv2.VideoCapture('\\'.join([folder,vidName]))
    # checks whether frames were extracted 
    success = 1
    while success:
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()
        # Saves the frames with frame-count
        fFrame = "firstFrame-{}.jpg".format(vidName[:-4])
        imageDir = folder +"\\BATTSI-{}\\{}".format(vidName[:-4],fFrame)
        print(imageDir)
        cv2.imwrite(imageDir, image)
        print(" - captured!\n")
        return imageDir

# Function to select a rectangle on image
def shape_selection(event, x, y, flags, param): 
    # grab references to the global variables 
    global ref_point, crop
    # if the left mouse button was clicked, record the starting 
    # (x, y) coordinates and indicate that cropping is being performed 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)]
    # check to see if the left mouse button was released 
    elif event == cv2.EVENT_LBUTTONUP: 
        # record the ending (x, y) coordinates and indicate that 
        # the cropping operation is finished 
        ref_point.append((x, y))
        # draw a rectangle around the region of interest 
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2) 
        cv2.imshow("BATTSI", image)

# Initiaize some variables        
fFrame=''
directory = os.getcwd()
isFile = False

print('''--------------------
### BATTSI- v2.0 ###
--------------------\n''')
print("#   SET UP   #")

videoName = ""

while isFile != True:
    videoName = str(input('-> Video file (include extension): '))
    path = '\\'.join([directory,videoName])
    isFile = os.path.isfile(path)
    if isFile == False:
        print(" - file not found... try again")
    else:
        print(" - file found!\n")

print(" - Directory setup...")
print(" - Initilizing log...")


#videoName = "HKDBerea826F4.mp4"
isExist = os.path.exists(directory  + "\\BATTSI-{}".format(videoName[:-4]) )
resultsDir = directory + "\\BATTSI-{}".format(videoName[:-4])
if not isExist:
    os.makedirs(resultsDir)
    print(" - Results directory for {} created".format(videoName))

else:
    print(" - Directory for {} results already exists".format(videoName))
    print(" - Results with new parameters will be created...")
    

print("\n - looking for first frame...")
videoImage = FirstFrame(directory, videoName)

sampleSize = int(input('-> Sample size: '))
#sampleSize = 2

print('''\n-----.
BATTSI will analyze {} samples in {}
------\n'''.format(sampleSize, videoName))
print('''#  DEFINE WELLS  #

------
TO draw one Region of Interest (ROI)
   * click-and-hold TOP RIGHT corner
    ** drag-while-holding BOTTOM LEFT corner
     *** release
------\n''')

# initialize the well column/row list
nombres = []
for i in range(sampleSize):
    a = 'well-%i' %(i+1)
    nombres.append(a)
#print(nombres)
# now let's initialize the list of reference point 
ref_point = [] 
crop = False
pozos = []

#videoImage = fFrame
image = cv2.imread(videoImage) 
clone = image.copy()
ventana = ""
cv2.namedWindow("BATTSI")
cv2.setMouseCallback("BATTSI", shape_selection)

# run for all cases
for i in nombres:
    #llave = cv2.waitKey(1) & 0xFF
    print("-> Draw ROI for {} <------> Press 'c' to confirm <-".format(i))
    
    # keep looping until the 'q' key is pressed 
    while True: 
        # display the image and wait for a keypress 
        cv2.imshow("BATTSI", image) 
        key = cv2.waitKey(1) & 0xFF
      
        # press 'r' to reset the window 
        if key == ord("r"): 
            image = clone.copy()
      
        # if the 'c' key is pressed, break from the loop 
        elif key == ord("c"): 
            break

    if len(ref_point) == 2: 
        pozos.append(ref_point)
        print("   - coordinates: {}\n\n     <-> TO continue, press 'c'".format(ref_point))
            
        cv2.waitKey(0)

# close all open windows 
cv2.destroyAllWindows()

print('''
------
      
#  CALIBRATE  #\n
------
''')
blurKernel = 1      # blur kernel can only be odd
dilationIter = 1    # dilation iterations
invalidCalibration = True

while invalidCalibration:
    calibrate = input('use DEFAULT or MANUAL calibration? (d/m):')
    
    if calibrate == 'm':
        print('''#  MANUAL CALIBRATION  #\n------\nSELECT parameters using toogle bar
        - k - kernel size for Gaussian blurring
        - i - number of dilation iterations\n\nPress C to accept parameters and CONTINUE\n------\n''')
                
        topR = [row[0] for row in pozos]
        botL = [row[1] for row in pozos]
        topR = np.asarray(topR)
        botL = np.asarray(botL)
        
        cap = cv2.VideoCapture(path)
        def trackbar_callback(value, idx):
            global dilationIter
            global blurKernel
            if idx == 0 :
                dilationIter = value
            else:
                if idx==1 and (value % 2) != 0:
                    blurKernel = value
            #print(blurKernel)
            #print(dilationIter)
        # check for correct file opening
        if(cap.isOpened()==False):
            print ("Error opening video stream or file")
        
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        
        # printing loop
        cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('transformed', cv2.WINDOW_NORMAL)
        cv2.namedWindow('barras', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('i','barras' , 3, 9, lambda v: trackbar_callback(v,0))
        cv2.createTrackbar('k', 'barras', 3, 9, lambda v: trackbar_callback(v,1))
        
        
        while(cap.isOpened()):
            ret,frame = cap.read()
        
            if ret == True:
                diff = cv2.absdiff(frame1, frame2)
                gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (blurKernel,blurKernel), 0)
                _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                dilated = cv2.dilate(thresh, None, iterations=dilationIter)
                be_ene = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                frame1 = frame2
                ret, frame2 = cap.read()
                # here is where we print many things
                # lets print a ROI with flies
                # and a ROI with the final mask up
                puntoA = topR[0][0]
                puntoB = topR[0][1]
                puntoC = botL[0][0]
                puntoD = botL[0][1]
                cropped1 = be_ene[puntoB:puntoD,puntoA:puntoC]
                cropped3 = dilated[puntoB:puntoD,puntoA:puntoC]
                #todes = np.concatenate((cropped1,cropped3), axis=1)
                #cv2.imshow('todes', todes)
                cv2.imshow('original', cropped1)
                cv2.imshow('transformed', cropped3)
                
                # press C on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('c'):
                    break
            else:
                break
            
        cap.release()
        cv2.destroyAllWindows()
        print('''------\nIgnore the warnings above!\n------\n - Parameters selected:''')
        print("   - Dilation Iteration:{}".format(dilationIter))
        print("   - Blur Kernel:{}".format(blurKernel))
        invalidCalibration = False
    elif calibrate == 'd' :
        blurKernel = 3      # blur kernel can only be odd
        dilationIter = 3    # dilation iterations
        
        print('''------\nDEFAULT parameters selected:''')
        print("   - Dilation Iteration:{}".format(dilationIter))
        print("   - Blur Kernel:{}".format(blurKernel))
        invalidCalibration = False
    else:
        print(' - Invalid calibration option')
        calibrate = input('use DEFAULT or MANUAL calibration? (d/m):')
        

print('''
------
      
#  TRANSFORMATION  #\n
------
''')
cap = cv2.VideoCapture(path)
cuadrosTot = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cuadros = np.ceil(cap.get(cv2.CAP_PROP_FPS))

salida = [[]]
l2 = []
i = 0
ret, frame1 = cap.read()
ret, frame2 = cap.read()

nombres = []
for i in range(sampleSize):
    a = '%i' %(i+1)
    nombres.append(a)
#
tamano = pozos
#
while cap.isOpened():
    if ret:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurKernel,blurKernel), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=dilationIter)

        frame1 = frame2
        ret, frame2 = cap.read()
        
        temp2 = []
        
        clone=dilated.copy()
        for j in range(len(pozos)):
            temp = clone[pozos[j][0][1]:pozos[j][1][1],pozos[j][0][0]:pozos[j][1][0]]
            actividad =  np.sum(temp)/(temp.shape[0]*temp.shape[1])
            temp2.append(actividad)
        salida.append(temp2)
        
        i+=1
        if i % 1000 == 0: print(" - {} % done".format(round(i/cuadrosTot*100,2)))
    elif i > cuadrosTot-1 :         # video length should be here
        print(" - Transformation complete")
        break
    else:
        print(" - Invalid frame at {} position".format(i))
        ret, frame2 = cap.read()
        i+=1

cv2.destroyAllWindows()
cap.release()

print("\n------\nVideo file: {}".format(videoName))
print("Frame rate: {}".format(cuadros))
print("Frame differences analyzed: {}".format(cuadrosTot))


print('''
------
      
#  RESULT SUMMARY  #\n
------
''')

print("\n - Plotting figure...")
# Plot of activity traces
df=pd.DataFrame(salida,columns=nombres)
df['seconds'] = df.index / cuadros
colnames = list(df.columns)
barras = df.plot(x='seconds', y=colnames[:-1], subplots = True)
datasetName = 'BATTSI-{}\\activity-{}-k{}-i{}.pdf'.format(videoName[:-4],videoName[:-4],blurKernel,dilationIter)
plt.savefig(datasetName)      # k=kernel i=iterations
print(" - Figure plotted!\n")

print("------")
print("# KNOCK DOWN ANALYSIS #")
print("------\n")

# Critical temperature analysis
knockdownCriteria = False
while(knockdownCriteria==False):
    print(" -(d)- Default criteria?")
    print("     - pixel intensity change (PIC) > 0 %")
    print(" -(m) Set threshold manualy?")
    print("     - Set a threshold PIC percentage (1-100)%")
    
    analysisOption = input(" - Use default (d) or manual (m) threshold?: ")
    
    if analysisOption == 'd':
        criteria = 0
        knockdownCriteria = True
    elif analysisOption == "m":
        manualThreshold = False
        while True:
            tmp = input(" - PIC percent threshold (1-100)% : ")
            if int(tmp) > 0 & int(tmp) <= 100:
                criteria = int(tmp)
                knockdownCriteria = True
                break
            else:
                print(" - Not a valid percentage, try again!")
    else:
        print(" - Not a valid option, try again!")
print("\n - Knockdown threshold set to: {}%".format(criteria))
print(" - Running analysis...")
# Criteria for determining 
knockDown = []

for column in df:
    s = df[column].tolist()    
    ls = [k for k, e in enumerate(s) if e > criteria]
    if len(ls) == 0:
        knockDown.append(0)
    else:
        knockDown.append(ls[(len(ls)-1)]/cuadros)

duracion = knockDown.pop()

print(" - Saving dataset...")

df.to_csv('BATTSI-{}\\DataFrame-{}-k{}-i{}.csv'.format(videoName[:-4],videoName[:-4],blurKernel,dilationIter))  # where to save it, usually as a csv

print(" -> Saved as CSV file in folder BATTSI-{}\n".format(videoName[:-4]))
#Then you can load it back using:
#df = pd.read_pickle(file_name)

print(" - Saving results table... (sample, knockdown time (seconds))")
resultados = {'Well':nombres,'CTmax':knockDown}
dfctmax = pd.DataFrame(resultados)
dfctmax.to_csv('BATTSI-{}\\result-{}-k{}-i{}.csv'.format(videoName[:-4],videoName[:-4],blurKernel,dilationIter))
print(" -> Saved as CSV file in folder BATTSI-{}\n".format(videoName[:-4]))

logName = "{}\\BATTSI-{}\\log-{}-i{}-k{}.txt".format(directory,videoName[:-4],videoName[:-4],dilationIter, blurKernel)

# print info to log file
log = open(logName, 'a')

log.write("--\nBATTSI v2.0 - patient pothos\n--\n - RUN begins\n")

log.write("Video file analyzed: {}\n".format(videoName))
log.write("Frame rate: {}\n".format(cuadros))
log.write("Number of frame differences transformed: {}\n--\n".format(cuadrosTot))
log.write("number of samples to be analyzed: {}\n".format(sampleSize))
if calibrate == "m":
    log.write("Calibration type selected: manual\n")
elif calibrate == 'd':
    log.write("Calibration type selected: default\n")
log.write("Iteration number for Dilation: {}\n".format(dilationIter))
log.write("Kernel size for Gaussian Blur: {}\n".format(blurKernel))
log.write("Knock down threshold criteria: {} %\n".format(criteria))
log.write("--\nRUN ends at {}\n--\n\n".format(datetime.now()))
log.close()

print(" - Run info logged at TXT file")
print("------\nBATTSI run finished\n------")
