import math
import os
import natsort
import numpy as np
import datetime
import cv2 
from skimage.color import rgb2hsv
import skimage.measure

from color_equalisation import RGB_equalisation
from global_stretching_RGB import stretching
from hsvStretching import HSVStretching

from histogramDistributionLower import histogramStretching_Lower
from histogramDistributionUpper import histogramStretching_Upper
from rayleighDistribution import rayleighStretching
from rayleighDistributionLower import rayleighStretching_Lower
from rayleighDistributionUpper import rayleighStretching_Upper
from sceneRadiance import sceneRadianceRGB

e = np.e
esp = 2.2204e-16
np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/RayleighDistribution"
folder = "D:/Single-Underwater-Image-Enhancement-and-Color-Restoration-master/underwater-test-dataset/upload/DATABASE 1"
path = folder 
files = os.listdir(path)
files =  natsort.natsorted(files)
starttime = datetime.datetime.now()
c =1
avg_psnr=0
avg_mse=0
avg_entropy=0
for i in files:    
    img = cv2.cv2.imread("D:/Single-Underwater-Image-Enhancement-and-Color-Restoration-master/underwater-test-dataset/upload/DATABASE 1./"+str(c)+".jpg")
    height = len(img)
    width = len(img[0])
    sceneRadiance = RGB_equalisation(img, height, width)
    # sceneRadiance = stretching(img)
    sceneRadiance = stretching(sceneRadiance)
    sceneRadiance_Lower, sceneRadiance_Upper = rayleighStretching(sceneRadiance, height, width)

    sceneRadiance = (np.float64(sceneRadiance_Lower) + np.float64(sceneRadiance_Upper)) / 2

    # cv2.imwrite('OutputImages/' + prefix + 'Lower0.jpg', sceneRadiance_Lower)
    # cv2.imwrite('OutputImages/' + prefix + 'Upper0.jpg', sceneRadiance_Upper)

    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    cv2.cv2.imwrite( str(c)+'_out.jpg', sceneRadiance)
    entropy = skimage.measure.shannon_entropy(sceneRadiance)
    print(str(entropy) + '\t')
    mse = skimage.measure.compare_mse(img,sceneRadiance)
    
    psnr = skimage.measure.compare_psnr(img, sceneRadiance)
    print(str(mse) + '\t' + str(psnr)+'\n')
    c=c+1
    avg_entropy+=entropy
    avg_psnr+=psnr
    avg_mse+=mse 


endtime = datetime.datetime.now()
time = endtime-starttime
print('time := ',time )
print( '\n' )
print('avg_psnr := ', avg_psnr/len(files))
print('\n')
print('avg_mse := ',avg_mse/len(files))
print('\n')
print('avg_entropy := ', avg_entropy/len(files))
