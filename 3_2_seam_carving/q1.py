
import time
import cv2
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit

def low_energy_calc(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(frame)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)/255
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)/255
    ener = np.sqrt((sobelx**2+sobely**2)/(2))
    min_ener = ener.copy()
    for i in range(ener.shape[0]-2,-1,-1):
        for j in range(0, ener.shape[1]):
            if(j==0 or j==ener.shape[1]-1):
                if(j==0):
                    min_ener[i,j] = np.min([ener[i+1,j], ener[i+1,j+1]])
                else:
                    min_ener[i,j] = np.min([ener[i+1,j-1], ener[i+1,j]])
            else:
                min_ener[i,j] = np.min([ener[i+1,j-1], ener[i+1,j], ener[i+1,j+1]])
            min_ener[i,j] += ener[i,j] 
    return min_ener, ener

def least_seam(ener):
    start_pt = np.argmin(ener[0,:])
    seam = []
    seam.append(start_pt)
    for i in range(1,ener.shape[0]):
        if(start_pt==ener.shape[1]-1):
            update = np.argmin([ener[i,start_pt-1], ener[i,start_pt]])-1
        elif(start_pt==0):
            update = np.argmin([ener[i,start_pt], ener[i,start_pt+1]])
        else:
            update = np.argmin([ener[i,start_pt-1], ener[i,start_pt], ener[i,start_pt+1]])-1
        start_pt = start_pt + update
        seam.append(start_pt)
    return seam

def seam_cut(image, percent, ty):
    if(ty=="H"):
        im = cv2.transpose(image)
    else:
        im = image.copy()
    
    num = int(0.01*percent*im.shape[1])
    
    for i in range(num):
        print(i,num)
        ener, tot_ener = low_energy_calc(im)
        seam = least_seam(ener)
        im_list = im.tolist()
        for k in range(len(seam)):
            im_list[k].pop(seam[k])
        im = np.array(im_list, dtype= np.uint8)
    
    if(ty=="H"):
        final_im = cv2.transpose(im)
    else:
        final_im = im
    
    resize_im = cv2.resize(image, (final_im.shape[1], final_im.shape[0]))

    return final_im, image, resize_im 

def my_plot(seamed_img, img, resize_img, percent, im_name):
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Input Image")
    ax[1].imshow(cv2.cvtColor(seamed_img, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Seam _craved Image")
    ax[2].imshow(cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Resized Image")
    # plt.show()
    plt.savefig("../op/"+im_name+"_"+str(percent)+".jpg")

# image= cv2.imread("../images/bangalore-dancers.jpg")

# image= cv2.imread("../images/cat.jpg")
im_name = "uluru"
image= cv2.imread("../images/"+im_name+".jpg")
# print(image)
percent = 60
# final_im, image, resize_im = seam_cut(image, percent, "V")
final_im, image, resize_im = seam_cut(image, percent, "V")
my_plot(final_im, image, resize_im, percent, im_name)
