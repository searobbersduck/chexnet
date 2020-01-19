import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

# from DensenetModels import DenseNet121
# from DensenetModels import DenseNet169
# from DensenetModels import DenseNet201

from DensenetModelsLocal import DenseNet121
from DensenetModelsLocal import DenseNet169
from DensenetModelsLocal import DenseNet201

import fire
from glob import glob
import shutil
import json

#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(transCrop, nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
          
        # model = torch.nn.DataParallel(model).cuda()

        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        model = torch.nn.DataParallel(model).cuda()

        self.model = model.module.densenet121.features
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize([transCrop,transCrop]))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.model.cuda()
        output = self.model(input.cuda())
        
        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
        
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        img = heatmap * 0.5 + imgOriginal
            
        cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 


def genHeatmap(pathInputImages, outDir, model_weights):
    os.makedirs(outDir, exist_ok=True)

    config_file = './config_dr_dr.json'
    with open(config_file,encoding='gb2312') as f:
        config = json.load(f)
    # pathInputImage = 'test/00009285_000.png'
    # pathInputImage = pathInputImage
    # pathOutputImage = 'test/heatmap1.png'

    pathModel = model_weights

    nnArchitecture = 'DENSE-NET-121'
    nnClassCount = 14


    transCrop = config['scale']
    # transCrop = 224

    for pathInputImage in pathInputImages:
        print('====> processing {}'.format(pathInputImage))
        pathOutputImage = os.path.join(outDir, os.path.basename(pathInputImage).split('.')[0]+'_heatmap.jpg')
        h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
        h.generate(pathInputImage, pathOutputImage, transCrop)
        shutil.copyfile(pathInputImage, os.path.join(outDir, os.path.basename(pathInputImage)))

def genHeatmapByFolder(infolder, outfolder, model_weights):
    # infolder = '/data/zhangwd/data/xray/dr_deformable_1024/气胸/气胸_pos_train'
    images_list = glob(os.path.join(infolder, '*.png'))
    genHeatmap(images_list, outfolder, model_weights)

if __name__ == '__main__':
    fire.Fire()