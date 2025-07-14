import tensorflow as tf
import keras
from keras import backend as K
import datetime
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.pooling import MaxPooling3D, GlobalMaxPool3D
from keras.layers.merge import concatenate, add
from keras.layers import Input, MaxPooling3D, UpSampling3D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.utils import np_utils
from keras.utils import to_categorical

import model_3D_unet

from sklearn.utils.class_weight import compute_sample_weight

import pause as ps
import numpy as np
from numpy import load
from numpy import ones
from numpy.random import randint
from numpy.random import uniform

from skimage.transform import warp, rotate, resize
from skimage.io import imshow, show
from scipy.ndimage import zoom

import math

import json
from collections import defaultdict
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab

import torch
from monai.transforms import (
    AsChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand2DElastic,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandShiftIntensity,
    Spacingd,
)
from monai.data import NibabelReader
from monai.config import print_config
from monai.apps import download_and_extract
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import glob
from os import listdir, makedirs
from os.path import isfile, isdir, join, split, exists
from datetime import datetime

import warnings

class UNet_Segmentation(object):
    """
    docstring
    """

    def __init__(self, MRI_Directory, Label_Directory, Output_Directory, OutputModelFileName, JSONFilename, OutputHistoryFileName1, OutputHistoryFileName2):
        self.MRI_Directory = MRI_Directory #Path to MRI Directory
        self.Label_Directory = Label_Directory #Path to Label Directory
        self.Output_Directory = Output_Directory #Path to Output Directory
        self.JSONFilename = JSONFilename #Json Filename to save training values (Accuracy and Loss)
        self.OutputModelFileName = OutputModelFileName #Network Weights Filename to save trained weights by neural network
        self.OutputHistoryFileName1 = OutputHistoryFileName1 #Json Filename to save Accuracy Graph
        self.OutputHistoryFileName2 = OutputHistoryFileName2 #Json Filename to save Loss Graph

        check = isdir(self.Output_Directory)

        if check == False:
            raise ValueError('Directory for saving data does not exist or has a typo.')

        # Other hyperparameters:
        self.lossFunction = 'categorical_crossentropy' #Loss Function
        self.learning_rate = 0.00002 #Learning Rate
        self.ApplyClassWeights = True #Applying weights to labels during training that are based on volume of each label
        
        # Get the MRI and Label image filenames
        self.MRI_FileNames = sorted(
            glob.glob(join(self.MRI_Directory, "*.nii")) #Looks for .nii files.
            )
        self.Label_FileNames = sorted(
            glob.glob(join(self.Label_Directory, "*.label.nii")) #Looks for .label.nii files
            )
        #Create a dictionary containing all MRI and Label data paths and filenames, keys: "image" and "label"
        self.Train_FileNames_Dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.MRI_FileNames, self.Label_FileNames)
            ]
        
        self.MRI_FileNames = [sub[len(self.MRI_Directory):] for sub in list(self.MRI_FileNames)]
        self.Label_FileNames = [sub[len(self.Label_Directory):] for sub in list(self.Label_FileNames)]

        self.GetImageSize() #Collect Image dimension information

        # Initialize the 3D U-Net model on the CPU scope so that the model's weights are hosted on CPU memory
        # To avoid them being hosted on the GPU (which may be slower)
        with tf.device('/gpu:0'):
            self.model = model_3D_unet.unet(input_size=(
                self.img_xdim, self.img_ydim, self.img_zdim, 1), numClasses=self.numClasses)

        #Selection of Loss Function:
        self.model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1),
                           loss=self.lossFunction,
                           metrics=['accuracy', 'categorical_accuracy'])

        if PreTrainedWeightsAvailable==True:
            self.ImportModelWeights(self.PreTrainedWeightsFileName)
        
        self.model.summary(line_length = 115)

    def DataAugmentation(self, MRI, Label): #Outputs transformed MRI and Label data
        ##Random bilinear transformation setup:
        randaffine = RandAffine(
            prob = 0.3,
            rotate_range=(self.yawAngle, self.pitAngle, self.rotAngle), #in radians, see lines 91-93
            scale_range=(self.scaleVal_x,self.scaleVal_y,self.scaleVal_z), #scaling ratios in all three directions
            translate_range=(self.TranslationX, self.TranslationY, self.TranslationZ), # of voxel translations for each axis
            padding_mode="zeros", #Padding mode for outside grid values
            device=torch.device('cpu'), #device on which the tensor will be allocated
        )

        ##Randomly select a spatial axis and flip image:
        randflip = RandFlip(
            prob=0.5,
            spatial_axis = randint(3) #Number Range: [0,3); 0=Up/Down, 1=Left/Right, 2=Front/Back Flipping
        ) 

        ##Randomly adds gaussian noise to image:
        randnoise = RandGaussianNoise(
            prob=0.1,
            mean=self.noiseMean, #Mean of Gaussian Distribution
            std=self.noiseVal #Standard Deviation of Gaussian Distribution
        ) 

        ##Randomly offsets the intensity values of the image:
        randshiftintensity = RandShiftIntensity(
            prob=0.3,
            offsets=self.brightnessVal
        ) 
        
        ## Transformation:
        # convert both image and segmentation using bilineary interpolation mode
        n = np.random.randint(0,100000) #This allows for both the image and label to have the same transformations
                                        #peformed on them
        self.seed_n = n
        
        #Transforms the Image with rotations, scales, and translations (see above)
        randaffine.set_random_state(seed=n)
        New_MRI = randaffine(MRI, (self.img_xdim, self.img_ydim, self.img_zdim), mode="bilinear")
        randaffine.set_random_state(seed=n)
        New_Label = randaffine(Label, (self.img_xdim, self.img_ydim, self.img_zdim), mode="nearest")

        #Flips the images 50% of the time (50/50 for LR or UD Flip)
        randflip.set_random_state(seed=n)
        New_MRI = randflip(New_MRI.cpu().detach().numpy())
        randflip.set_random_state(seed=n)
        New_Label = randflip(New_Label.cpu().detach().numpy())

        #Adds Gaussian Noise 10% of the Time to just the T2wb 
        randnoise.set_random_state(seed=n)  
        New_MRI = randnoise(New_MRI)

        #Shifts Brightness Pixel Vaules 30% of the Time to just the T2w
        randshiftintensity.set_random_state(seed=n)  
        New_MRI = randshiftintensity(New_MRI)
        

        return New_MRI, New_Label
        
    def ImportModelWeights(self, filePath):
        # Load a previously trained model (must be the same size as this model)
        self.model.load_weights(filePath)

    def Load_Random_Image(self):
        # This function loads a random MRI (and corresponding label nifti file)
        # Then randomly selects a 2D image from the 3D volume
        # Optionally, data augmentation is applied to the image

        FileNdx = int(
            randint(0, min(len(self.MRI_FileNames), len(self.Label_FileNames)), 1)
            )

        # Optional: Only use the first image for training or debugging
        if self.UseOnlyFirstFile == True:
            FileNdx = 0

        # Load the nifti images
        Data_Dict = self.LoadNiftiImage(
            self.Train_FileNames_Dicts[FileNdx]
            )
        MRI, Label = Data_Dict['image'], Data_Dict['label']
                       
        # Re-number the values of the label image to be integers from one to number of channels (such as 1 to 15)
        vals = np.unique(Label)

        for i in range(0, len(vals)):
##            if vals[i] <= len(vals) & vals[i] != 0:
##                raise("Unique value is less than the number of unique values")
            Label[Label == vals[i]] = i

        New_MRI = np.zeros((1,self.img_xdim, self.img_ydim, self.img_zdim), np.float32)
        New_Label = np.zeros((1,self.label_xdim, self.label_ydim, self.label_zdim), np.float32)
        New_Label[0,:,:,:] = Label
        
        if self.Apply_Augmentation == True:
            New_MRI, New_Label = self.DataAugmentation(MRI, Label)

            # Apply a normalization to the MRI image to have the intensity values range from 0 to 1
            MRI_temp = New_MRI[0,:,:,:] - New_MRI[0,:,:,:].min()
            New_MRI[0,:,:,:] = MRI_temp / MRI_temp.max()  # Range from 0 to 1
    ##        New_MRI = (New_MRI - 0.5) * 2  # Range from -1 to 1
        else:
            # Apply a normalization to the MRI image to have the intensity values range from 0 to 1
            MRI_temp = MRI[0,:,:,:] - MRI[0,:,:,:].min()
            New_MRI[0,:,:,:] = MRI_temp / MRI_temp.max()  # Range from 0 to 1
    ##        New_MRI = (New_MRI - 0.5) * 2  # Range from -1 to 1

        return New_MRI, New_Label

    def generate_train_batch(self, batch_size):
        # This function is called during model fitting and returns a set of images and labels

        while 1:

            imgs = np.zeros((batch_size, self.img_xdim, self.img_ydim,
                             self.img_zdim), np.float32)
            labels = np.zeros((batch_size, self.label_xdim, self.label_ydim,
                               self.label_zdim), np.float32)
            classWeights = np.zeros((batch_size,self.numClasses))

            for i in range(0, batch_size):
                imgs[i, :, :, :], labels[i, :, :, :] = self.Load_Random_Image()

            #Determining class weights for each volume:
            if self.ApplyClassWeights == True:
                vals = np.unique(labels)
                for i in range(0, len(vals)):
                    if vals[i] <= len(vals) & vals[i] != 0: 
                        raise("Unique value is less than the number of unique values")
                    labels[labels == vals[i]] = i

                # Estimate the relative volumes of each label to apply a scaling to have a more equal class scaling
                self.classWeights = np.zeros(self.numClasses)
                for i in range(0, self.numClasses):
                    self.classWeights[i] = len(np.argwhere(labels == i)) #Number of voxels for each class label

                self.classWeights = self.classWeights / np.sum(self.classWeights)
                self.classWeights = 1 - self.classWeights
            else:
                self.classWeights = np.ones(self.numClasses)
            
            # This part may be a bit confusing
            # We need to have a copy of the image for each class label (which is equal to the number of segmentation labels)
            # For example, suppose batch_size = 10, img_height = 128, img_width = 128, and numClasses = 15
            # Size of imgs would then be [10, 128, 128, 59, 15] where the 4th dimension is just copies of the image

            batch_imgs = np.zeros(
                (batch_size, self.img_xdim, self.img_ydim, self.img_zdim, 1), np.float32)

            for i in range(0, batch_size):
                batch_imgs[i, :, :, :, 0] = imgs[i, :, :, :]

            # Similarly, we need to convert the label images to a binary class matrix which contains either 0 or 1
            batch_labels = np_utils.to_categorical(labels, num_classes=self.numClasses)

            # Lower the weight of the background label
            for i in range(0, self.numClasses):
                batch_labels[:, :, :, :, i] = self.classWeights[i] * batch_labels[:, :, :, :, i]

            yield batch_imgs, batch_labels

    def LoadNiftiImage(self, fileName):
        ##Loading Nifti T2w and Label Files:
        ##Setting up MONAI Nifti Images Loader:
        loader3D = LoadImaged(keys=["image", "label"])#, reader="NibabelReader", image_only=True) #Loading in MRI ().nii) files and associated label (.label.nii) files
        channel_swap = AsChannelFirstd(keys=["image", "label"])
        orientation = Orientationd(keys=["image", "label"], axcodes="ALI") ##The default axis labels are Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S).
        warnings.filterwarnings("ignore")

        #Loading in Images:
        data_dict = loader3D(fileName) #Loads in nifti data for one animal (pair: T2w and Label)
        data_dict = channel_swap(data_dict) #Changes the last column (channel) to the first column for MONAI (want first column to be channel column)
        data_dict = orientation(data_dict) #Correctes the images from LR (x-axis) to TB (y-axis) orientation
        img_np = data_dict
        
        return img_np

    def GetImageSize(self):
        # Load the first label image to get the image sizes
        Data_Dict = self.LoadNiftiImage(self.Train_FileNames_Dicts[0])
        MRI, Label = Data_Dict['image'], Data_Dict['label']
        
        self.img_vol_shape = MRI.shape #All dimensions of the image volume, format:(channels,x,y,z)
        self.img_num_channels = self.img_vol_shape[0] #Number of channels from image
        self.img_xdim = self.img_vol_shape[1] #x dimension of image
        self.img_ydim = self.img_vol_shape[2] #y dimension of image
        self.img_zdim = self.img_vol_shape[3] #z dimension of image (or number of slices)

        self.label_vol_shape = Label.shape #All dimensions of the label volume, format:(channels,x,y,z)
        self.label_num_channels = self.label_vol_shape[0] #Number of channels from label
        self.label_xdim = self.label_vol_shape[1] #x dimension of label
        self.label_ydim = self.label_vol_shape[2] #y dimension of label
        self.label_zdim = self.label_vol_shape[3] #z dimension of label (or number of slices)

        # Number of output segmentation labels
        self.numClasses = len(np.unique(Label))
        
        # Re-number the values of the label image to be integers from one to number of channels (such as 1 to 15)
        vals = np.unique(Label)

        if self.ApplyClassWeights == True:
            for i in range(0, len(vals)):
                if vals[i] <= len(vals) & vals[i] != 0: 
                   raise("Unique value is less than the number of unique values")
                Label[Label == vals[i]] = i

            # Estimate the relative volumes of each label to apply a scaling to have a more equal class scaling
            self.classWeights = np.zeros(self.numClasses)
            for i in range(0, self.numClasses):
                self.classWeights[i] = len(np.argwhere(Label == i)) #Number of voxels for each class label

            self.classWeights = self.classWeights / np.sum(self.classWeights)
            self.classWeights = 1 - self.classWeights
        else:
            self.classWeights = np.ones(self.numClasses)

    def Train(self):
        # Stop the training early if it has not improved the loss function in this number of epochs
        earlystopper = EarlyStopping(patience=5, verbose=1)

        gen_train = self.generate_train_batch(self.batch_size)

        history = {}

        for epoch in range(0, self.epochs):
            
            self.history = self.model.fit(gen_train, validation_data=next(gen_train), steps_per_epoch=self.steps_per_epoch,epochs=1,callbacks=[earlystopper]).history
            
            self.PlotExample(step=epoch) # Save an example segmentation figure to the disk

            self.OutputModelFileName_Epoch = self.OutputModelFileName[0:-5] + '_E' + str(epoch+1) + '.hdf5'
            
            self.model.save_weights(self.OutputModelFileName_Epoch) # Save the model weights to the disk

            if epoch == 0:
                history = self.history

            else:
                for key in history.keys():
                    history[key].append(self.history[key][0])
                    
            print(f'Epochs Completed: {epoch+1} out of {self.epochs}')
                        
        self.model.save_weights(self.OutputModelFileName[0:-5]+'_Final.hdf5')

        with open(self.JSONFilename, 'w') as f: 
            json.dump(history, f)

        x_start = len(history['accuracy'])
        x = np.arange(1,x_start+1)
        xlim = np.arange(0, x_start+1,10)
        xlim[0] = 1

        #Time Duration of NN
        end_time = time.time()
        hours, rem = divmod(end_time-start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        #Creating the x-axis data for the graphs of the NN training:
        x_end = len(history['accuracy'])
        x = np.arange(1,x_end+1)
        x1 = np.arange(1,x_end/2 + 1)
        xlim = np.arange(0, x_end+1,10)
        xlim[0] = 1

        #Calculating the figure size, based on the number of data points:
        x_tick_spacing = 0.5 #in inches
        w = max([6.4, x_tick_spacing*len(xlim)]) #Width of figures below, 6.4 is the default setting for the width of plt.figure().
        h = 4.8 #in, Height of figures below, 4.8 is the default setting for plt.figure().
        # summarize history for accuracy:
        plt.figure(1, figsize=[w,h])
        plt.plot(x, history['accuracy'])
        plt.plot(x, history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.xticks(xlim)
        plt.show(block=False)
        plt.savefig(self.OutputHistoryFileName1)
                    
        # summarize history for loss:
        plt.figure(2, figsize=[w,h])
        plt.plot(x, history['loss'])
        plt.plot(x, history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.xticks(xlim)
        plt.show(block=False)
        plt.savefig(self.OutputHistoryFileName2)

##        plt.show() #plots all figures at once with block=False


    def PlotExample(self, step=0):

        # Only plot up to 5 images or else they are too small to see well
        n_samples = np.min((self.batch_size, 5))
        generator = self.generate_train_batch(n_samples)

        imgs, labels = next(generator)
        PredictedLabels = self.model.predict(imgs)

        # Combine the channels to get the final label image
        outputSeg = np.zeros(
            (n_samples, PredictedLabels.shape[1], PredictedLabels.shape[2], PredictedLabels.shape[3], 1), np.float32)
        GT_Image = np.zeros(
            (n_samples, PredictedLabels.shape[1], PredictedLabels.shape[2], PredictedLabels.shape[3], 1), np.float32)

        for i in range(0, n_samples):
            for j in range(0, self.numClasses):
                outputSeg[i, :, :, :, 0] = outputSeg[i, :, :, :, 0] + \
                    j * PredictedLabels[i, :, :, :, j]

                GT_Image[i, :, :, :, 0] = GT_Image[i, :, :, :, 0] + \
                    j * labels[i, :, :, :, j]

        if np.allclose(outputSeg, np.zeros(outputSeg.shape))==True:
            raise NameError('3D U-Net Output Label is blank!')

        # imshow(outputSeg[i,:,:,0,0])
        # show()

        s = imgs.shape
        # Plot the MRI images
        for i in range(5):
            plt.subplot(3, 5, 1 + i)
            plt.axis('off')

            if len(s) == 3:
                plt.imshow(imgs[:,:, i*5 + 15], cmap='gray')
            else:
                plt.imshow(imgs[0, :, :, i*5 + 15,  0], cmap='gray')

        # Plot the predicted labels
        for i in range(5):
            plt.subplot(3, 5, 6 + i)
            plt.axis('off')

            if len(s) == 3:
                plt.imshow(GT_Image[:,:, i*5 + 15])
            else:
                plt.imshow(GT_Image[0, :, :, i*5 + 15,  0])

        # Plot ground truth label
        for i in range(5):
            plt.subplot(3, 5, 11 + i)
            plt.axis('off')

            if len(s) == 3:
                plt.imshow(outputSeg[:,:,i*5 + 15])
            else:
                plt.imshow(outputSeg[0, :, :, i*5 + 15,  0])

        # save plot to file
        OutputDirectory = split(self.OutputModelFileName)
        filename1 = OutputDirectory[0] + '/Plot_%03d.png' % (step+1)
        plt.savefig(filename1)
        plt.close()

        
start_time = time.time() #Start time for NN learning
if __name__ == "__main__":

################################# Path Variables to Declare ##############################################   
    # Free up RAM in case the model definition cells were run multiple times
    K.clear_session()

    Home_Directory = "C:/path/to/data/" 
    date = datetime.today().strftime('%#m-%d-%y')

    # Where are the MRI and label images located?
    MRI_Directory = Home_Directory + "Training Scans/" #Path to Training MRI Scans Folder
    Label_Directory = Home_Directory + "Training Labels/"  #Path to Training Label Scans Folder
    Output_Directory = Home_Directory + "Output Files/"+date+" OPI Rat 3D U-Net/" #Path to Output Folder; date incorporates the date the training was started into the output folder name

    # Pretrained Weights for Transfer Learning?
    PreTrainedWeightsAvailable==True #True if yes, False if no.
    
    # Location of Pre-Trained Weights, must be the same size as the model:
    if PreTrainedWeightsAvailable==True:
        PreTrainedWeightsFileName = Home_Directory + "path/to/pretrained_weights/weight_file.hdf5"
    
    # Output model weights, name should be a .hdf5 file
    OutputModelFileName = Output_Directory + date + "_RD_OPI_SS_3DUnet_Model.hdf5"
    
    # Output model history, name shoud be a .png file
    JSONFilename = Output_Directory + date + "training_data.json" #contains all data from during network model training
    OutputHistoryFileName1 = Output_Directory + date + "Model_Accuracy.png" #graph of accuracy during training
    OutputHistoryFileName2 = Output_Directory + date + "Model_Loss.png" #graph of loss during training

    #Create Output Directory if it does not exist:
    if exists(Output_Directory) == False:
        print("")
        print("Output directory does not exist...")
        makedirs(Output_Directory)
        print("Output directory was created...", "\n")
    
#########################################################################################################
    # Initiation of Unet file:
    U_Net = UNet_Segmentation(MRI_Directory,
                              Label_Directory,
                              Output_Directory, 
                              OutputModelFileName,
                              JSONFilename,
                              OutputHistoryFileName1,
                              OutputHistoryFileName2,)
 
    # If you do not want to use the GPU, uncomment the two lines below:
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
############################ Training Parameters to Declare (Optional) #################################   
    U_Net.Apply_Augmentation = True #Default: True
    U_Net.epochs = 300
    U_Net.batch_size = 1
    U_Net.steps_per_epoch = 20
    U_Net.learning_rate = 0.00002
    U_Net.UseOnlyFirstFile = False #Default: False, only utilize during code testing

    # Default values for data augmentation parameters
    # Set equal to 0 to skip that particular augmentation type
    U_Net.TranslationX = 40 #in voxels
    U_Net.TranslationY = 40 #in voxels
    U_Net.TranslationZ = 5 #in voxels (or slices)
    U_Net.rotAngle = np.pi/3 #Needs to be in radians!
    U_Net.pitAngle = np.pi/6 #Needs to be in radians!
    U_Net.yawAngle = np.pi/6 #Needs to be in radians!
    U_Net.brightnessVal = 0.5
    U_Net.scaleVal_x = 0.1
    U_Net.scaleVal_y = 0.1
    U_Net.scaleVal_z = 0.1
    U_Net.noiseMean = 1 
    U_Net.noiseVal = 0.25
#########################################################################################################
##Printing Training Parameter Information in Terminal for Posterity Purposes
##It is useful to save this output as a text file for posterity
# Print the parameters in the code output shell:
    print('Data Augmentation Parameters:')
    print(f"U_Net.TranslationX = {U_Net.TranslationX}")
    print(f"U_Net.TranslationY = {U_Net.TranslationY}")
    print(f"U_Net.TranslationY = {U_Net.TranslationZ}")
    print(f"U_Net.rotAngle = {U_Net.rotAngle}")
    print(f"U_Net.rotAngle = {U_Net.pitAngle}")
    print(f"U_Net.rotAngle = {U_Net.yawAngle}")
    print(f"U_Net.brightnessVal = {U_Net.brightnessVal}")
    print(f"U_Net.scaleVal = {U_Net.scaleVal_x}")
    print(f"U_Net.scaleVal = {U_Net.scaleVal_y}")
    print(f"U_Net.scaleVal = {U_Net.scaleVal_z}")
    print(f"U_Net.noiseMean = {U_Net.noiseMean}")
    print(f"U_Net.noiseVal = {U_Net.noiseVal}")
    print(f"U_Net.Apply_Augmentation = {U_Net.Apply_Augmentation}\n")

    print('3D U-Net Model Training Parameters:')
    print(f"U_Net.batch_size = {U_Net.batch_size}")
    print(f"U_Net.epochs = {U_Net.epochs}")
    print(f"U_Net.steps_per_epoch = {U_Net.steps_per_epoch}")
    print(f"U_Net.UseOnlyFirstFile = {U_Net.UseOnlyFirstFile}")
    print(f"U_Net.lossFunction = {U_Net.lossFunction}")
    print(f"U_Net.learning_rate = {U_Net.learning_rate}\n")

    print("Training Image Data:")
    print(*U_Net.MRI_FileNames, sep='\n')
    print("\nTraining Label Data:")
    print(*U_Net.Label_FileNames, sep='\n')

    # Load the first label image to get the image size and the number of distinct segmentation labels
    print("\nImage Information:")
    print(f"U_Net.img_xdim = {U_Net.img_xdim}") 
    print(f"U_Net.img_ydim = {U_Net.img_ydim}")  
    print(f"U_Net.img_zdim = {U_Net.img_zdim}")  
    print(f"U_Net.numClasses = {U_Net.numClasses}")
    print(f"U_Net.classWeights = {U_Net.classWeights}")
    print("------------------Parameter Summary End------------------\n")

    U_Net.Train()
