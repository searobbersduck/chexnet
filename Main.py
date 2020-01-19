import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer

import fire
import json

#-------------------------------------------------------------------------------- 

def main ():
    
    fire.Fire()
    # runTest()
    #runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    config_file = './config_dr_dr.json'
    with open(config_file,encoding='gb2312') as f:
        config = json.load(f)

    

    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = './database'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs
    # trBatchSize = 32
    # trMaxEpoch = 100
    trBatchSize = config['batch_size']
    trMaxEpoch = config['epoch']
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    # imgtransResize = 256
    # imgtransCrop = 224
    imgtransResize = config['scale']
    imgtransCrop = config['scale']
        
    pathModel = 'm-' + timestampLaunch + '.pth.tar'
    
    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, None)
    
    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    config_file = './config_dr_dr.json'
    with open(config_file,encoding='gb2312') as f:
        config = json.load(f)

    pathDirData = './database'
    pathFileTest = './dataset/test_1.txt'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 14
    # trBatchSize = 16
    # imgtransResize = 256
    imgtransCrop = 224
    trBatchSize = config['batch_size']
    imgtransResize = config['scale']
    # imgtransCrop = config['scale']
    
    pathModel = 'm-14012020-150303.pth.tar'
    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





