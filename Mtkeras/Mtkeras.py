# -*- coding: utf-8 -*-
"""
Created on 17/Jan/2020
@author: Yelin Liu, Zhiquan(George) Zhou

"""

import numpy as np
import random
# to deal with image manipulation
import skimage
from skimage.color import rgb2gray
import cv2
# to deal with search engine test
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time


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
            3. text
            4. searchTerm
        - model: an object. It is the neural network model undertest, if the Mtkeras is only used for test case generation, this argument can be omitted. The "model" argument is needed only when MROP is performed. 

    Returns:
        It will return a Mtkeras object, by calling different attributes, the returns will be different.
        - return a tranformed dataset(a list), call the property ".myTestSet"
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
        - call the property ".myTestSet" to return a tranformed dataset(a list)

    '''

    def permutative(self):
        # shuffle the order of every sentence
        if(self.dataType == 'text'):
            for j, ele in enumerate(self.myTestSet):
                random.shuffle(ele)
        return self

    '''
    Summary: 
        The "additive" MRIP: increase (or decrease) numerical values by a constant for each pieces of data in the dataset

    Args:
        - n_additive: integer, the constant that is used to change each pieces of data

    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(a list)
    '''

    def additive(self, n_additive):
        # add a constant to every pixel in a picture
        if(self.dataType == 'grayscaleImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            matrix = np.ones((picx, picy))*n_additive
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
        - call the property ".myTestSet" to return a tranformed dataset(a list)
    '''

    def brightness(self, gamma=1, gain=1):
        # adjust the brightness of the picture
        for index, ele in enumerate(self.myTestSet):
            self.myTestSet[index] = skimage.exposure.adjust_gamma(
                ele, gamma, gain)
            return self

    '''
    Summary:
        the "multiplicative" MRIP: multiply numerical values by a constant for each pieces of data in the dataset

    Args:
        - n_mul: integer, the constant used for multiplying the dataset
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(a list)
    '''

    def multiplicative(self, n_mul):
        # multiple every pixel by a constant
        if(self.dataType == 'grayscaleImage'):
            self.myTestSet = self.myTestSet * n_mul
            return self

    '''
    Summary:
        the "invertive" MRIP: invert the element in the dataset, in the dataType of an image, the invertive version of the image will be a color-flipped picture

    Args:
        None
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(a list)
    '''

    def invertive(self):
        # invert the order of the text sequence
        if(self.dataType == 'text'):
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
        - call the property ".myTestSet" to return a tranformed dataset(a list)
    '''

    def noise(self, n_noise=0):
        # add random noise point into a picture
        if(self.dataType == 'grayscaleImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            noise = np.zeros((picx, picy))
            for i in range(n_noise):
                xaxis = np.random.randint(0, picx)
                yaxis = np.random.randint(0, picy)
                noise[xaxis][yaxis] = np.random.randint(0, 255)
            self.myTestSet = self.myTestSet + noise
            return self
        elif(self.dataType == 'colorImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            noise = np.zeros((picx, picy, 3))
            for i in range(n_noise):
                xaxis = np.random.randint(0, picx)
                yaxis = np.random.randint(0, picy)
                noise[xaxis][yaxis][np.random.randint(
                    0, 3)] = np.random.randint(0, 255)
            self.myTestSet = self.myTestSet + noise
            return self
        # add random word into a text in the context of sentiment analysis
        elif(self.dataType == 'text'):
            for i in range(n_noise):
                self.myTestSet[np.random.randint(
                    0, self.myTestSet.shape[0])].insert(0, 4)
            return self
        # add a space in the search term when testing a search engine
        elif(self.dataType == 'searchTerm'):
            for i in self.myTestSet:
                i = " " + i
            return self

    '''
    Summary:
        the "fliph" MRIP: flip the data horizontally

    Args:
        None
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(a list)
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
        - call the property ".myTestSet" to return a tranformed dataset(a list)
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
        - call the property ".myTestSet" to return a tranformed dataset(a list)
    '''

    def rotate(self, n_deg):
        # rotate the picture to a certain degree
        for index, ele in enumerate(self.myTestSet):
            self.myTestSet[index] = skimage.transform.rotate(
                ele, n_deg, preserve_range=True)
        return self

    '''
    Summary:
        the "NoREC" MRIP: generate an optimized version of the search query

    Args:
        None

    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return an optimized query
    '''

    def NoREC(self):
        # change a query to a optimized version query
        if(self.dataType == 'SQL'): 
            arr = self.myStartTestSet.split(' ')
            index = arr.index('FROM')
            str1 = " ".join(arr[index:])
            str2 = " ".join(arr[1:index])
            self.myTestSet = arr[0] + " * " + str1 + " WHERE " + str2
            return self

    '''
    Summary:
        the "equal" MROP: the source and follow-up outputs must contain the same items and in the same order.

    Args:
        none
        need to specify the argument "model"
    
    Returns:
        - a Mtkeras Object
        - call the property ".myTestSet" to return a tranformed dataset(a list)
        - call the property ".violatingCases" to return a dataset of violating cases

    Outputs:
        the number of the violation cases will be printed
    '''

    def equality(self, params=None):
        if(self.dataType == 'grayscaleImage'):
            index_one = self.myTestSet.shape[0]
            index_two = self.myTestSet.shape[1]
            index_three = self.myTestSet.shape[2]

            self.myTestSet = self.myTestSet.reshape(
                index_one, index_two * index_three)/255
            self.myStartTestSet = self.myStartTestSet.reshape(
                index_one, index_two * index_three)/255

            predict1 = self.model.predict_classes(self.myTestSet)
            predict2 = self.model.predict_classes(self.myStartTestSet)
            for index in range(len(predict1)):
                if(predict1[index] != predict2[index]):
                    self.violatingCases.append(index)

        elif(self.dataType == 'colorImage'):
            sourceOutput = []
            followUpOutput = []
            for ele in self.myTestSet:
                followUpOutput.append(process_img(ele).tolist())
            for ele in self.myStartTestSet:
                sourceOutput.append(process_img(ele).tolist())

            predict1 = self.model.predict_classes(np.array(sourceOutput))
            predict2 = self.model.predict_classes(np.array(followUpOutput))

            for index in range(len(predict1)):
                if(predict1[index] != predict2[index]):
                    self.violatingCases.append(index)

        elif(self.dataType == 'searchTerm'):
            sourceOutput = test_search_engine(self.myStartTestSet, **params)
            followUpOutput = test_search_engine(self.myTestSet, **params)
            for index in range(len(sourceOutput)):
                if(sourceOutput[index] != followUpOutput[index]):
                    self.violatingCases.append(index)

        elif(self.dataType == 'query'):
            self.violatingCases.append(index)

        print("There are {num} violations of MROP equality.".format(
            num=len(self.violatingCases)))
        return self


'''
Summary: a helper function, for image preprocessing before prediction
'''


def process_img(img):
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    img = img.reshape(32, 32, 1)
    return img


'''
Summary: a helper function, conducting search engine test using selenium

Args:
    - searchTerm: the searchTerm that is input to the search engine
    - params: {
        chromeLocation: the local path of the chrome.exe
        website_name: the website under test
        search_bar_id: the search bar's id
        result_xpath: the element displays result, the user have to find its xpath
    }
    
Returns:
    - a Mtkeras Object
    - call the property ".myTestSet" to return a tranformed dataset(a list)
    - call the property ".violatingCases" to return a dataset of violating cases
'''


def test_search_engine(searchTerms, **params):
    options = webdriver.ChromeOptions()
    # 此步骤很重要，设置为开发者模式，防止被各大网站识别出来使用了Selenium
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    # 禁止加载图片，减少网络负担
    options.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2})
    # options.binary_location = r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
    options.binary_location = params["chromeLocation"]
    driver = webdriver.Chrome(options=options)
    script = '''
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
    })
    '''
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument", {"source": script})
    output = []
    for searchTerm in searchTerms:
        driver.delete_all_cookies()
        driver.get(params["website_name"])
        elem = driver.find_element_by_id(params["search_bar_id"])
        elem.clear()
        elem.send_keys(searchTerm)
        elem.submit()
        driver.implicitly_wait(3)
        try:
            resultString = driver.find_element_by_xpath(
                params["result_xpath"]).get_attribute("innerText")
            driver.implicitly_wait(3)
            result = int(resultString.strip().replace(',', ''))
        except Exception as e:
            result = 0
        output.append(result)
        print("The result for " + searchTerm + " is " + str(result))
    return output


class Mtkeras_mrip:

    """
    Summary:
        The MRIP class, the user can input a source test case and it will output a follow-up test case which can be transformed by multiple MRIPs
        Note: this library is only used when the user want to seperately use the mrop library

    Implementation:
        use one line of code: Mtkeras_mrip(<sourceTestSet>,<dataType>).<MRIPs>

    Args:
        - myTestSet: an array or ndarray that contains image data or other kind of data. Each pieces of data should be a seperate array, and all these array should be stored in one array, which is the myTestSet array.
        - dataType: a string that can represent the context of the software undertest, it can be:
            1. grayscaleImage
            2. colorImage
            2. text

    Returns:
        - call the property ".myTestSet", it will return a MRIP tranformed dataset(an array/ndarray).
    """

    def __init__(self, myTestSet, dataType='grayscaleImage'):
        self.myTestSet = myTestSet
        self.myStartTestSet = myTestSet
        self.dataType = dataType
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
            for j, ele in enumerate(self.myTestSet):
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
        if(self.dataType == 'grayscaleImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            matrix = np.ones((picx, picy))*n_additive
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
        for index, ele in enumerate(self.myTestSet):
            self.myTestSet[index] = skimage.exposure.adjust_gamma(
                ele, gamma, gain)
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
        if(self.dataType == 'grayscaleImage'):
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
        if(self.dataType == 'text'):
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
        if(self.dataType == 'grayscaleImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            noise = np.zeros((picx, picy))
            for i in range(n_noise):
                xaxis = np.random.randint(0, picx)
                yaxis = np.random.randint(0, picy)
                noise[xaxis][yaxis] = np.random.randint(0, 255)
            self.myTestSet = self.myTestSet + noise
            return self
        elif(self.dataType == 'colorImage'):
            picx = self.myTestSet.shape[1]
            picy = self.myTestSet.shape[2]
            noise = np.zeros((picx, picy, 3))
            for i in range(n_noise):
                xaxis = np.random.randint(0, picx)
                yaxis = np.random.randint(0, picy)
                noise[xaxis][yaxis][np.random.randint(
                    0, 3)] = np.random.randint(0, 255)
            self.myTestSet = self.myTestSet + noise
            return self
        # add random word into a text
        elif(self.dataType == 'text'):
            for i in range(n_noise):
                self.myTestSet[np.random.randint(
                    0, self.myTestSet.shape[0])].insert(0, 4)
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
        for index, ele in enumerate(self.myTestSet):
            self.myTestSet[index] = skimage.transform.rotate(
                ele, n_deg, preserve_range=True)
        return self


class Mtkeras_mrop:

    """
    Summary:
        The MROP library, the user can fill the source test outputs and the follow-up test outputs, then check whether it violates the MROP defined
        Note: this library is only used when the user want to seperately use the mrop library

    Implementation:
        use one line of code: Mtkeras_mrop(<sourceTestOutput>,<followUpTestOutput>).<MROPs>

    Args:
        - sourceTestOutput: the source test output, must be a list, each list item represents a source test case output
        - followUpTestOutput: the follow-up test output, must be a list, each list item represents a follow-up test case output

    Returns:
        it will print out the total number of the violating cases,
        the function will return a list of the indexes of those violating cases.
    """

    def __init__(self, sourceTestOutput, followUpTestOutput):
        self.sourceTestOutput = sourceTestOutput
        self.followUpTestOutuput = followUpTestOutput
        self.violatingCaseIndex = []
        self.count = 0

    '''
    Summary:
        This pattern represents those relations where the source and follow-up outputs include the same items although not necessarily in the same order.

    Args:
        None

    Returns:
        the function will return an array contains the indexes of the violating cases, which can be used for searching the test cases in the source testset.
    '''

    def equivalence(self):
        for i in range(len(self.sourceTestOutput)):
            if(set(self.sourceTestOutput[i]) != set(self.followUpTestOutuput[i])):
                self.count += 1
                self.violatingCaseIndex.append(i)
        print("There are {} cases violates the MROP equivalence.".format(self.count))
        return self.violatingCaseIndex

    '''
    Summary:
        This pattern represents those relations where the source and follow-up outputs must contain the same items and in the same order.

    Args:
        None

    Returns:
        the function will return an array contains the indexes of the violating cases, which can be used for searching the test cases in the source testset.
    '''

    def equality(self):
        for i in range(len(self.sourceTestOutput)):
            if(self.sourceTestOutput[i] != self.followUpTestOutuput[i]):
                self.count += 1
                self.violatingCaseIndex.append(i)
        print("There are {} cases violates the MROP equality.".format(self.count))
        return self.violatingCaseIndex

    '''
    Summary:
        This pattern groups those relations where the follow-up outputs should be subsets (or strict subsets) of the source output and subsets among them.

    Args:
        None

    Returns:
        the function will return an array contains the indexes of the violating cases, which can be used for searching the test cases in the source testset.
    '''

    def subset(self):
        for i in range(len(self.sourceTestOutput)):
            if(isinstance(self.sourceTestOutput[i], list)):
                if(set(self.sourceTestOutput[i]) < set(self.followUpTestOutuput[i])):
                    self.count += 1
                    self.violatingCaseIndex.append(i)
            elif(isinstance(self.sourceTestOutput[i], int)):
                if(self.sourceTestOutput[i] < self.followUpTestOutuput[i]):
                    self.count += 1
                    self.violatingCaseIndex.append(i)
        print("There are {} cases violates the MROP subset.".format(self.count))
        return self.violatingCaseIndex

    '''
    Summary:
        This pattern represents those relations where the intersection among the source and follow-up outputs should be empty

    Args:
        None

    Returns:
        the function will return an array contains the indexes of the violating cases, which can be used for searching the test cases in the source testset.
    '''

    def disjoint(self):
        for i in range(len(self.sourceTestOutput)):
            if(set(self.sourceTestOutput[i]) & set(self.followUpTestOutuput[i])):
                self.count += 1
                self.violatingCaseIndex.append(i)
        print("There are {} cases violates the MROP subset.".format(self.count))
        return self.violatingCaseIndex

    '''
    Summary:
        This pattern includes those relations where the union of the follow-up outputs should contain the same items as the source output

    Args:
        anotherFollowUpTestOutput: the another follow-up test output which is attempted to complete the first follow up test output compared with the source test output

    Returns:
        the function will return an array contains the indexes of the violating cases, which can be used for searching the test cases in the source testset.
    '''

    def complete(self, anotherFollowUpTestOutput):
        for i in range(len(self.sourceTestOutput)):
            if(self.sourceTestOutput[i] != self.followUpTestOutuput[i] + anotherFollowUpTestOutput[i]):
                self.count += 1
                self.violatingCaseIndex.append(i)
        print("There are {} cases violates the MROP complete.".format(self.count))
        return self.violatingCaseIndex

    '''
    Summary:
        This pattern includes those metamorphic relations where the source output and the follow-up output should differ in a specific set of items

    Args:
        - differSet: a set of items that the follow up output should differ from the source output

    Returns:
        the function will return an array contains the indexes of the violating cases, which can be used for searching the test cases in the source testset.
    '''

    def difference(self, differSet):
        for i in range(len(self.sourceTestOutput)):
            isEqual = list(
                set(self.followUpTestOutuput[i])-set(self.sourceTestOutput[i])) == differSet[i]
            if(not isEqual):
                self.count += 1
                self.violatingCaseIndex.append(i)
        print("There are {} cases violates the MROP difference.".format(self.count))
        return self.violatingCaseIndex
