# -*- coding: utf-8 -*-
"""
Created on 17/Jan/2020
@author: Yelin Liu, Zhiquan(George) Zhou

"""

import numpy as np
import random
import skimage


class Mtkeras:

    """
    Summary:
        The Mtkeras is a testing tool for metamorphic testing cases generation and metamorphic relation validation.

        Mtkeras enables automated metamorphic testing by providing the users with an MR library for testing their ML models and applications. 
        The design of MTKeras is centered around two basic concepts: metamorphic relation input patterns (MRIPs) and metamorphic relation output patterns(MROPs).
        Mtkeras is extendable as it allows a user to plug in new MRIPs and MROPs and conﬁgure them into concrete MRs.

    Implementation:
        Firstly, import the script using the command "from Mtkeras import Mtkeras"
        Secondly, the user can perform MT in a simple and intuitive way by writing a single line of code in the following format: Mtkeras(<sourceTestSet>,<dataType>).<MRIPs>[.<MROP>].

    Args:
        - myTestSet: an array or ndarray that contains image data or other kind of data. Each pieces of data should be a seperate array, and all these array should be stored in one array, which is the myTestSet array.
        - dataType: a string that can represent the context of the software undertest, it can be:
            1. grayscaleImage
            2. colorImage
            2. text
        - model: an object. It is the neural network model undertest, if the Mtkeras is only used for test case generation, this argument can be omitted. The "model" argument is needed only when MROP is performed. 

    Returns:
        It will return a Mtkeras object, by calling different attributes, the returns will be different.
        - return a tranformed dataset(an array/ndarray), call the property ".myTestSet"
        - return a dataset of violating cases, call the property ".violatingCases"
    """
    def __init__(self, myTestSet, dataType='grayscaleImage', model=None):
        self.myTestSet = myTestSet
        self.myStartTestSet = myTestSet
        self.dataType = dataType
        self.violatingCases = []
        self.model = model

    '''
    Summary:
        The "permutative" MRIP: the user can shuffle the order of the data randomly in the dataset

    Args: 
        None

    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)

    '''
    def permutative(self):
        # shuffle the order of every sentence 
        if(self.dataType == 'text'):
            for j,ele in enumerate(self.myTestSet):
                random.shuffle(ele)
        return self
            

    '''
    Summary: 
        The "additive" MRIP: increase (or decrease) numerical values by a constant for each pieces of data in the dataset

    Args:
        - n_additive: integer, the constant that is used to change each pieces of data

    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def additive(self, n_additive):
        # add a constant to every pixel in a picture
        if(self.dataType=='grayscaleImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            matrix = np.ones((picx,picy))*n_additive
            self.myTestSet += matrix
            return self


    '''
    Summary:
        The "brightness" MRIP: for images, this MRIP can adjust the image's brightness. It transforms the input image pixelwise according to the equation O = I**gamma after scaling each pixel to the range 0 to 1

    Args:
        - gamma(optional): float, non negative real number, the default value is 1
        - gain(optional): float, the constant multiplier. Default value is 1
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def brightness(self, gamma=1, gain=1):
        # adjust the brightness of the picture
        for index,ele in enumerate(self.myTestSet):
            self.myTestSet[index] = skimage.exposure.adjust_gamma(ele, gamma, gain)
            return self
    

    '''
    Summary:
        the "multiplicative" MRIP: multiply numerical values by a constant for each pieces of data in the dataset

    Args:
        - n_mul: integer, the constant used for multiplying the dataset
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def multiplicative(self, n_mul):
        # multiple every pixel by a constant
        if(self.dataType=='grayscaleImage'):
            self.myTestSet = self.myTestSet * n_mul
            return self
    

    '''
    Summary:
        the "invertive" MRIP: invert the element in the dataset, in the dataType of an image, the invertive version of the image will be a color-flipped picture

    Args:
        None
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def invertive(self):
        # invert the order of the text sequence
        if(self.dataType=='text'):
            for i in range(len(self.myTestSet)):
                self.myTestSet[i].reverse()
            return self

    
    '''
    Summary:
        the "noise" MRIP: create one or more noise points in a dataset

    Args:
        - n_noise: integer, n_noise>=0, the number of the noise point that is added to the dataset 
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def noise(self, n_noise):
        # add random noise point into a picture
        if(self.dataType=='grayscaleImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            noise = np.zeros((picx, picy))
            for i in range(n_noise):
                xaxis = np.random.randint(0, picx)
                yaxis = np.random.randint(0, picy)
                noise[xaxis][yaxis] = np.random.randint(0,255)
            self.myTestSet = self.myTestSet + noise
            return self
        elif(self.dataType=='colorImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            noise = np.zeros((picx, picy, 3))
            for i in range(n_noise):
                xaxis = np.random.randint(0, picx)
                yaxis = np.random.randint(0, picy)
                noise[xaxis][yaxis][np.random.randint(0,3)] = np.random.randint(0,255)
            self.myTestSet = self.myTestSet + noise
            return self
        # add random word into a text
        elif(self.dataType=='text'):
            for i in range(n_noise):
                self.myTestSet[np.random.randint(0, self.myTestSet.shape[0])].insert(0,4)
            return self


    '''
    Summary:
        the "fliph" MRIP: flip the data horizontally

    Args:
        None
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def fliph(self):
        # flip the picture horizontally
        self.myTestSet = np.fliplr(self.myTestSet)
        return self

    
    '''
    Summary:
        the "flipv" MRIP: flip the data vertically

    Args:
        None
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def flipv(self):
        # flip the picture vertically
        self.myTestSet = np.flipud(self.myTestSet)
        return self


    '''
    Summary:
        the "rotation" MRIP: rotate the data, 

    Args:
        -n_deg: float, specify the degree that the image rotate
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
    '''
    def rotate(self, n_deg):
        # rotate the picture to a certain degree
        for index,ele in enumerate(self.myTestSet):
            self.myTestSet[index] = skimage.transform.rotate(ele, n_deg,preserve_range=True)
        return self


    '''
    Summary:
        the "equal" MROP: the output of the first dataset should be equal to the second test dataset

    Args:
        none
        need to specify the argument "model"
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(an array/ndarray)
        - call the property ".violatingCases" to return a dataset of violating cases

    Outputs:
        the number of the violation cases will be printed
    '''
    def equal(self):
        if(self.dataType=='grayscaleImage'):
            self.myTestSet = self.myTestSet.reshape(10000,784)/255
            myStartTestSet = self.myStartTestSet.reshape(10000,784)/255
            predict1 = self.model.predict_classes(self.myTestSet)
            predict2 = self.model.predict_classes(myStartTestSet)
            for index in range(len(predict1)):
                if(predict1[index] != predict2[index]):
                    self.violatingCases.append(self.myStartTestSet[index])
            print("There are {num} violations of MROP equal.".format(num=len(violatingCases)))
            return self
