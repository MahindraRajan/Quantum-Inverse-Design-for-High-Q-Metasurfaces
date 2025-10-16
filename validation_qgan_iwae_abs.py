# -*- coding: utf-8 -*-
from __future__ import print_function
from Utilities.ConvertImageToBinary import Binary
from pathlib import Path
import scipy
from scipy import ndimage
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import pennylane as qml
from models.models import IWAE, QuantumGenerator

#Location of Saved Generator and IWAE
genDir = "/dgxb_home/se21pphy004/Multiclass_Metasurface/final_generator_qgan_iwae_abs.pth"
vaeDir = "/dgxb_home/se21pphy004/Multiclass_Metasurface/pretrained_iwae_abs.pth"

#Location of Training Data
spectra_path = '/dgxb_home/se21pphy004/Multiclass_Metasurface/Training_Data/absorptionData_HybridGAN_1.csv'

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compare(i, factor, shift, results_folder, genDir, vaeDir, spectra_path):
            
    # Load Generator
    n_qubits, q_depth, n_generator = 9, 2, 1
    generator = QuantumGenerator(n_qubits, q_depth, n_generator).to(device)
    generator.load_state_dict(torch.load(genDir))
    generator.eval()

    # Load VAE
    nc, beta, num_samples = 3, 2.0, 5
    vae = IWAE(n_qubits=n_qubits, nc=nc, beta=beta, num_samples=num_samples).to(device)
    vae.load_state_dict(torch.load(vaeDir))
    vae.eval()
    
    ##Create Generator Input
    excelTestData = pd.read_csv(spectra_path, index_col = 0)
    
    #for z in range(shift):
        #excelTestData.insert(0,str(z),0)
    excelDataSpectra = excelTestData.iloc[:,:4]
    excelDataSpectra = excelDataSpectra.shift(shift, axis=1, fill_value=0)
    excelTestDataTensor = torch.tensor(factor*excelDataSpectra.values).type(torch.FloatTensor)
    testTensor = torch.Tensor()
    
    index = i
    tensor1 = torch.cat((excelTestDataTensor[index],torch.rand(5)))
    tensor2 = tensor1.unsqueeze(1)
    tensor3 = tensor2.permute(1,0)
    testTensor = torch.cat((testTensor,tensor3),0).to(device)
    
    fake_latent = generator(testTensor)
    fake_images = vae.decoder(fake_latent)
    img = fake_images.detach().cpu()
    img = img.permute(3,2,1,0)
    img = img.squeeze()
    img = img.numpy()

    excelTestDataNames = pd.read_csv(spectra_path)
    name = excelTestDataNames.iloc[index,0]
    print(name)
    img = (img + 1)/2
    
    im_size = 64
    pmax = 4.0
    tmax = 10.0
    emax = 5.0
    
    psum = 0.0
    pnum = 0.0
    tsum = 0.0
    tnum = 0.0
    esum = 0.0
    enum = 0.0
    
    for row in range(im_size):
        for col in range(im_size):
            if img[row][col][0] > 0.2 or img[row][col][1] > 0.2:
                if img[row][col][0] > img[row][col][1]:
                    psum += img[row][col][0]
                    pnum += 1
                else:
                    esum += img[row][col][1]
                    enum += 1
            if img[row][col][2] > 0.2:
                tsum += img[row][col][2]
                tnum += 1

    if pnum > 0:
        pAvg = psum / pnum
    else:
        pAvg = 0.0
        
    if tnum > 0:
        tAvg = tsum / tnum
    else:
        tAvg = 0.0
        
    if enum > 0:
        eAvg = esum / enum
    else:
        eAvg = 0.0
    
    pfake = pmax * pAvg
    tfake = tmax * tAvg
    efake = emax * eAvg
    
    preal = excelTestData.iloc[index, 5]
    treal = excelTestData.iloc[index, 6]
    ereal = excelTestData.iloc[index, 7]

    print(preal, treal, ereal)
            
    plt.imshow(img)
    plt.imsave(results_folder+ '/Results_qgan_iwae_64/' + str(i) + '-test.png',img)
    
    if pnum > enum:
        pindexfake = pfake
        pindexreal = preal
        classifier = 0
    else:
        pindexfake = efake
        pindexreal = ereal
        classifier = 1
    
    print("Fake Plasma/Index:", pindexfake)
    print("Real Plasma/Index:", pindexreal)
    print("Fake Thickness:", tfake)
    print("Real Thickness:", treal)
    
    return [pindexfake, pindexreal, tfake, treal, classifier]

#Pass Sspectra into Generator
indices = [0, 5, 6, 56]
# indices = []
# for index in range(0,63):
#     indices.append(1 * index)
results_folder = os.path.dirname(os.path.realpath('/dgxb_home/se21pphy004/Multiclass_Metasurface/'))
Path(results_folder+ '/Results_qgan_iwae_64').mkdir(parents=True, exist_ok=True) #ref: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
file = open(results_folder + '/Results_qgan_iwae_abs/properties.txt',"w")
file.write("Index FakePlasma/Index RealPlasma/Index FakeThickness RealThickness Class(MIM=0/DM=1)")
for i in indices:
    props = compare(i, 1, 0, results_folder, genDir, vaeDir, spectra_path)
    props.insert(0, i)
    row = ""
    for j in props:
        row += str(round(j, 2)) + " "
    file.write("\n" + row)
file.close()

#Convert Images to Black and White
im_size = 64
im_path = results_folder+ '/Results_qgan_iwae_64/*-test.png'
imgFolder = glob.glob(im_path)
imgFolder.sort()

for img in imgFolder:
    rgb = mpimg.imread(img)
    for row in range(im_size):
        for col in range(im_size):
            if rgb[row][col][0] > rgb[row][col][2] or rgb[row][col][1] > rgb[row][col][2]:
                rgb[row][col][0] = 0
                rgb[row][col][1] = 0
                rgb[row][col][2] = 0
            else:
                rgb[row][col][0] = 1
                rgb[row][col][1] = 1
                rgb[row][col][2] = 1
                
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    cv2.normalize(gray, gray, -1, 1, cv2.NORM_MINMAX)
    
    #Apply Gaussian Filter
    img_filter = scipy.ndimage.gaussian_filter(gray,sigma=0.75)
    ret, img_filter = cv2.threshold(img_filter,0.1,1,cv2.THRESH_BINARY) # 0 = black, 1 = white; everything under first number to black
    
    plt.imshow(img_filter, cmap = "gray")
    plt.imsave(img[:-4]+'-bw.png', img_filter, cmap = "gray")

#Convert B/W Images to Binary (for COMSOL)
Binary.convert(results_folder)