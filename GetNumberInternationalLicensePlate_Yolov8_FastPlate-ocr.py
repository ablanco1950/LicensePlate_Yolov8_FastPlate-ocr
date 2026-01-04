# -*- coding: utf-8 -*-


"""
Author: Alfonso Blanco
 https://github.com/ankandrew/fast-plate-ocr

"""

dirname= "Test1"

from fast_plate_ocr import LicensePlateRecognizer

import cv2

import time
Ini=time.time()

dirnameYolo="best.pt"

from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
#print(class_list)

import numpy as np

X_resize=220
Y_resize=70

import os
import re

import imutils


#####################################################################
"""
Copied from https://gist.github.com/endolith/334196bac1cac45a4893#

other source:
    https://stackoverflow.com/questions/46084476/radon-transformation-in-python
"""

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft



try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
        
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency

           

def GetFastPlate_ocr(img):
    m = LicensePlateRecognizer('cct-xs-v1-global-model')
    #print(m.run('test_plate.png'))
    cv2.imwrite("gray.jpg",img)
    img_path = 'gray.jpg'
    text= m.run(img_path)
    return text[0], 0.0
#########################################################################
def FindLicenseNumber (gray, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################

    grayColor=gray
    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
   
    TotHits=0 
    
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    
    # en la mayoria de los casos no es necesaria rotacion
    # pero en algunos casos si (ver TestRadonWithWilburImage.py)
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
        print(License + " rotate "+ str(rotation))
        gray=imutils.rotate(gray,angle=rotation)
    
    
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    
    X_resize=x_resize
    Y_resize=y_resize
    #print("gray.shape " + str(gray.shape)) 
    Resize_xfactor=1.5
    Resize_yfactor=1.5
   
    TotHits=0

    text, Accuraccy = GetFastPlate_ocr(gray)

    #print(text)
    #print(RR)

    print( "DETECTED " + text)
   
    text = ''.join(char for char in text if char.isalnum())
    
    text=ProcessText(text)
    if ProcessText(text) != "":
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==License:
           print(text + "  Hit " )
           TotHits=TotHits+1
        else:
            print(License + " detected  as "+ text)
    
   
    ################################################################
    return TabLicensesFounded, ContLicensesFounded
   
 ########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 #if License != "PGMN112": continue
                 
                 image = cv2.imread(filepath)
                                
                           
                 images.append(image)
                 Licenses.append(License)
                 
                 Cont+=1
     
     return images, Licenses

def Detect_International_LicensePlate(Text):
    if len(Text) < 3 : return -1
    for i in range(len(Text)):
        if (Text[i] >= "0" and Text[i] <= "9" )   or (Text[i] >= "A" and Text[i] <= "Z" ):
            continue
        else: 
          return -1 
       
    return 1

def ProcessText(text):
  
    if len(text)  > 10:
        text=text[len(text)-10]
        if len(text)  > 9:
          text=text[len(text)-9]
        else:
            if len(text)  > 8:
              text=text[len(text)-8]
            else:
        
                if len(text)  > 7:
                   text=text[len(text)-7:] 
    if Detect_International_LicensePlate(text)== -1: 
       return ""
    else:
       return text

def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded


# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectLicenseWithYolov8 (img):
  
   TabcropLicense=[]
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   results = model.predict(img)
   for i in range(len(results)):
       # may be several plates in a frame
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       # Get Class name
       class_name = [class_list[z] for z in class_id]
       # Pack together for easy use
       sum_output = list(zip(class_name, confidence,xyxy))
       # Copy image, in case that we need original image for something
       out_image = img.copy()
       for run_output in sum_output :
           # Unpack
           #print(class_name)
           label, con, box = run_output
           if label == "vehicle":continue
           cropLicense=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           #cv2.imshow("Crop", cropLicense)
           #cv2.waitKey(0)
           TabcropLicense.append(cropLicense)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))
       
   return TabcropLicense, y,yMax,x,xMax


###########################################################
# MAIN
##########################################################

imagesComplete, Licenses=loadimagesRoboflow(dirname)

print("Number of imagenes : " + str(len(imagesComplete)))

print("Number of   licenses : " + str(len(Licenses)))

ContDetected=0
ContNoDetected=0
TotHits=0
TotFailures=0
with open( "LicenseResults.txt" ,"w") as  w:
    for i in range (len(imagesComplete)):
          
            gray=imagesComplete[i]
            
            License=Licenses[i]
            #gray1, gray = Preprocess.preprocess(gray)
            TabImgSelect, y, yMax, x, xMax =DetectLicenseWithYolov8(gray)

          
            
            if TabImgSelect==[]:
                print(License + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                ContDetected=ContDetected+1
                print(License + " DETECTED ")
            for x in range(len(TabImgSelect)):
               
                if len(TabImgSelect[x]) == 0: continue
                gray=TabImgSelect[x]
                #cv2.imshow('Frame', gray)           
                #cv2.waitKey()
                                
                x_off=3
                y_off=2
                
                x_resize=215
                y_resize=70
                
                Resize_xfactor=1.78
                Resize_yfactor=1.78
                
                ContLoop=0
                
                SwFounded=0
                
                BilateralOption=0
                               
                TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray, x_off, y_off,  License, x_resize, y_resize, \
                                       Resize_xfactor, Resize_yfactor, BilateralOption)
                           
                ymax=-1
                contmax=0
                licensemax=""
                             
                for z in range(len(TabLicensesFounded)):
                    if ContLicensesFounded[z] > contmax:
                        contmax=ContLicensesFounded[z]
                        licensemax=TabLicensesFounded[z]
                
                if licensemax == License:
                   print(License + " correctly recognized") 
                   TotHits+=1
                else:
                    print(License + " Detected but not correctly recognized")
                    TotFailures +=1
                print ("")  
                lineaw=[]
                lineaw.append(License) 
                lineaw.append(licensemax)
                lineaWrite =','.join(lineaw)
                lineaWrite=lineaWrite + "\n"
                w.write(lineaWrite)
                break # only one plate per image to verify plate with image name
             
              
print("")           
print("Total Hits = " + str(TotHits ) + " from " + str(len(imagesComplete)) + " images readed")

print("")

#print("Total Hits filtro nuevo= " + str(TabTotHitsFilter[0] ) + " from " + str(len(imagesComplete)) + " images readed")
#print("Total Failures filtro nuevo= " + str(TabTotFailuresFilter[0] ) + " from " + str(len(imagesComplete)) + " images readed")

print( " Time in seconds "+ str(time.time()-Ini))
