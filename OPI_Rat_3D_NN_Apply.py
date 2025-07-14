import model_3D_unet

import datetime

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Input, UpSampling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.utils import np_utils
from keras.utils import to_categorical


import numpy as np
from numpy import load
from numpy import ones
from numpy.random import randint
from numpy.random import uniform
from numpy import vstack
from skimage.transform import rotate, warp, resize
from skimage.io import imshow, show
from scipy.ndimage import zoom

import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
from matplotlib import pyplot

import SimpleITK as sitk
import nibabel as nib

from monai.transforms import (
    AsChannelFirstd,
    LoadImaged,
    Orientationd,
    )
from monai.data import write_nifti

import glob
from os import listdir, makedirs
from os.path import isfile, exists, join

from skimage.transform import resize
from skimage.io import imshow, show

import time

import warnings

K.clear_session()

def LoadNiftiImage(fileName):
        ##Loading Nifti T2w and Label Files:
        ##Setting up MONAI Nifti Images Loader:
        loader3D = LoadImaged(keys=["image"])#, reader="NibabelReader", image_only=True) #Loading in MRI ().nii) files and associated label (.label.nii) files
        channel_swap = AsChannelFirstd(keys=["image"])
        orientation = Orientationd(keys=["image"], axcodes="ALI") ##The default axis labels are Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S).
        warnings.filterwarnings("ignore")
        
        #Loading in Images:
        data_dict = loader3D(fileName) #Loads in nifti data for one animal (pair: T2w and Label)
        data_dict = channel_swap(data_dict) #Changes the last column (channel) to the first column for MONAI (want first column to be channel column)
        data_dict = orientation(data_dict) #Correctes the images from LR (x-axis) to TB (y-axis) orientation
        img_np = data_dict

        reader = sitk.ImageFileReader()
        reader.SetFileName((fileName['image']))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        img_itk = sitk.ReadImage(fileName['image'])
        
        return img_np, img_itk       
        
def main():

    threshold = Threshold #Threshold probability value for which pixels will be part of the label.

    # Load the nifti images
    MRI = img_Data_Dict['image']
    npI = MRI[0,:,:,:] - MRI[0,:,:,:].min()
    npI = npI / npI.max()  # Range from 0 to 1
    
    img_vol_shape = MRI.shape #All dimensions of the image volume, format:(channels,x,y,z)
    img_num_channels = img_vol_shape[0] #Number of channels from image
    img_xdim = img_vol_shape[1] #x dimension of image
    img_ydim = img_vol_shape[2] #y dimension of image
    img_zdim = img_vol_shape[3] #z dimension of image (or number of slices)  

    input_img = np.zeros((1,img_xdim, img_ydim, img_zdim, 1))
    input_img[0,:,:,:,0]=npI

    with tf.device('/cpu:0'):
        model = model_3D_unet.unet(input_size=(img_xdim, img_ydim, img_zdim, 1), numClasses=NumClasses) #numClasses = 15 for DIBs, numClasses = 10 for CACE Yr9

    model.load_weights(Model_FileName)

    outputSegmentation_np = np.zeros((img_xdim, img_ydim, img_zdim), dtype=np.float32)
  
    # Iterate and segment over each 2D slice:    
    # Segment the image using the trained network
    segmented_Img = model(input_img)

    # Combine the channels to get the label image
    outputSeg = np.zeros((1, img_xdim, img_ydim, img_zdim, 1), np.float32)

    for j in range(1, NumClasses):  # Skip background label
        tmp=np.array(segmented_Img[0, :, :, :, j])

        # Apply a threshold to the probability values (e.g. Threshold = 0.8)
        tmp[tmp < threshold]=0
        tmp[tmp != 0]=1

        thresh_num=np.str(threshold*100)

        # Multiply the image by the current image label to and add to the output discrete image
        outputSeg[:, :, :, :, 0]=outputSeg[0, :, :, :, 0] + j*tmp

    #if outputSeg.shape[1] != img_vol_shape[3]:
    #    # Resize to the original image size and convert back to gray scale
    #    outputSeg=resize(outputSeg,
    #                     (1, img_zdim, img_xdim, img_ydim, 1),
    #                     order=0, preserve_range=True, anti_aliasing=False)
    
    outputSegmentation_np = outputSeg[0,:,:,:,0]
    outputSegmentation_np = outputSegmentation_np.transpose(2,0,1)
    outputSegmentation_np = outputSegmentation_np[::-1, :, :] 

    # Export the segmentation as a SimpleITK nifti image
    numpySpacing = np.array(list(img_itk.GetSpacing()))

    outputSegmentation = sitk.GetImageFromArray(outputSegmentation_np, isVector=False)
    #outputSegmentation = sitk.PermuteAxes(outputSegmentation, [1,2,0])
    outputSegmentation.SetSpacing(numpySpacing)
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(Output_Directory +
                       imgFileName[:-4] + "_NN_Segmented" + thresh_num[0:2] + ".label.nii") #Add label on 7-19-22
    writer.Execute(outputSegmentation)
 
    if Export_Combined_Image == True:
        npI = MRI[0,:,:,:].transpose(2,0,1)
        npL = outputSegmentation_np
        s = npI.shape
        s = (s[0], s[1], 2*s[2])

        combinedNp = np.zeros(s)

        # For the combine image, scale each image to be a maximum of 255
        if npI.max() != 0:
            npI = npI / npI.max() * 255

        if npL.max() != 0:
            npL = npL / npL.max() * \
                255

        combinedNp[:, :, 1:int(s[2]/2)+1] = npI[::-1, :, :] #Swapping the z-axis from back-to-front to front-to-back for the TETS Mouse Dataset
        combinedNp[:, :, int(s[2]/2):] = npL[:, :, :]
      
        tmp = sitk.GetImageFromArray(combinedNp)

        writer.SetFileName(Combined_Directory +
                           imgFileName[:-4] + "_Combined" + thresh_num[0:2] + ".label.nii") #Add label on 7-19-22
        writer.Execute(tmp)

total_start_time = time.time() #Start time for NN learning
if __name__ == "__main__":
    Home_Directory = "V:/Datasets/CACE Yr8 DFP Rat Data/" #CACE Yr8 TETS Mice WBD Data/" #AD R21 WBD Data/" #CACE Yr8 DFP Rat Skull Strip Data/" #"D:/Google Drive/Research/Projects/Mouse Brain Segmentations/"
    MRI_Directory = Home_Directory + "MRI/20mm/3-15-23 Bias Correction 1 with Python 128x128x64/Test Images/" #"C:/Users/Valerie/Desktop/Datasets/AD R21 Data/3. Cropping to 128x128/Cropping to 128x128/" #CACE Yr8 TETS Mice Data/3. Cropping to 128x128/" #
    Model_FileName = "V:/Dissertation Neural Network Research/Whole Brain Delineation/CACE Yr8 DFP Rat Skull Strip Data/3D U-Net Results/3-15-23 WBD n_train=100 3D U-Net Model/Mar_15_23_WBD_3DUnet_Model_Final.hdf5" #"C:/Users/Valerie/Desktop/Dissertation Neural Network Research/Whole Brain Delineation/CACE Yr8 DFP Rat Skull Strip Data/New folder/3-15-23 WBD n_train=100 3D U-Net Model/Mar_15_23_WBD_3DUnet_Model_Final.hdf5"
    Output_Directory = Home_Directory + "WBD Labels/3D U-Net Whole Brain Labels 128x128x64/Test Images/" #"C:/Users/Valerie/Desktop/BF NN/Whole Brain Delineation/AD R21 WBD Data/Results/6-8-22 AD R21 WBD with OPI Rat 2D U-Net/" #CACE Yr8 TETS Mice WBD Data/Results/6-8-22 TETS Mice WBD with OPI Rat 2D U-Net/"
    Combined_Directory = Output_Directory + "Combined Images/" #"C:/Users/Valerie/Desktop/BF NN/Whole Brain Delineation/AD R21 WBD Data/Results/6-8-22 AD R21 WBD with OPI Rat 2D U-Net/Combined Images/" #CACE Yr8 TETS Mice WBD Data/Results/6-8-22 TETS Mice WBD with OPI Rat 2D U-Net/Combined Images/"

    #Create Output Directory if it does not exist:
    if exists(Output_Directory) == False:
        print("\nOutput directory does not exist...")
        makedirs(Output_Directory)
        print("\nOutput directory was created...",)

    #Create Combined Image Directory if it does not exist:
    if exists(Combined_Directory) == False:
            print("\nCombined image directory does not exist...")
            makedirs(Combined_Directory)
            print("\nCombined image directory was created...",)

    #Prints weight filename in python terminal log:
    print(f"\nModel Filename used: {Model_FileName}") #prints weight filename being used to create NN label image

    NumClasses = 2 #Number of label classes in the label image volume   

    MRI_FileNames = sorted(
            glob.glob(join(MRI_Directory, "*.nii"))
            )
    #List of MRI Directory + Filenames of MRI data to be processed
    Train_FileNames_Dicts = [
            {"image": image_name}
            for image_name in MRI_FileNames
            ] 
    
    MRIFilenames = [sub[len(MRI_Directory):] for sub in list(MRI_FileNames)] #String format to print the MRI filenames

    T =  [0.9]   #Threshold values to iterate through. Must have [] for FOR loop to work.
    
    for m in range(0,len(T)):
        Threshold = T[m]
        for n in range(0,len(Train_FileNames_Dicts)): 
            # Nifti file to segment
            start_time = time.time()

            #Printing image filename and threshold values being applied:
            print(f"\nImage to be Segmented: {MRIFilenames[n]}") #prints image filename being used to create NN label image
            print(f"Threshold Value: {T[m]}") #prints threshold value being used to create NN label image

            imgFileName = Train_FileNames_Dicts[n]['image'][len(MRI_Directory):]  #List of Filenames ONLY of MRI data to be processed
            img_Data_Dict, img_itk = LoadNiftiImage(Train_FileNames_Dicts[n])
            Export_Combined_Image = True     
            main()

            #Finishing Segmentation Notice:
            print("Segmentation Done.")

            #Time duration for each scan:
            end_time = time.time()
            hours, rem = divmod(end_time - start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print("\nScan Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                 
#Time Duration of NN
total_end_time = time.time()
total_hours, total_rem = divmod(total_end_time-total_start_time, 3600)
total_minutes, total_seconds = divmod(total_rem, 60)
print("Time Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(total_hours),int(total_minutes),total_seconds))
