#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 21:39:20 2017

@author: changing
"""
from PIL import Image
import numpy as np
import glob
import pandas as pd
import re

from sklearn.linear_model import LogisticRegression

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
#%%
#You need to change the root and image_path when running the program
#on another machine
root= "Dataset_A/"
image_path = "Dataset_A/data"
train_data = root+"train.txt"
val_data = root+"val.txt"
test_data = root+"test.txt"
img_addr = image_path+"/20586908_6c613a14b80a8591_MG_R_CC_ANON.png"

model_name = "EM_CNN_1"
input_shape = (256,256,1)
batch_size = 32
epochs = 100

#%%

def get_image_address(image_path):
    #Input: image_addr_list, only the number
    #Image_path: the folder which stores the images
    image_name_list = []
    for filename in glob.glob(image_path+'/*.png'):
        image_name_list.append(filename)
    
    #Return all the image_name in the folder
    return image_name_list

#%%
def gen_image_addr_label(txt_file,image_path,test_data=False):
    #generate the according image address to the txt_file
    #get the content of the data
    txt_image_label_list = []
    data = pd.read_csv(txt_file, header = None)
    #data['name'] represents the name of the image
    data['name'] = data[0].apply(lambda x: x.split('\t')[0])
    #data['label'] represents the label of the image
    if test_data==False:
        data['label'] = data[0].apply(lambda x: x.split('\t')[1])
    
    #change the name of the image to list
    txt_image_name_list = list(data['name'])
    if test_data==False:
        txt_image_label_list = list(data['label'])
    #this is the list to save the image address
    txt_image_address_list = []
    
    #get the whole image_name and convert it to string
    image_name_list = get_image_address(image_path)
    image_name_string = "\n".join(image_name_list)
    
    #for all the name we get the address
    #then append the address to the txt_image_address_list
    for image_addr in txt_image_name_list:
        pattern = r'.*%s.*'%image_addr
        image_name = re.findall(pattern,image_name_string)[0]
        txt_image_address_list.append(image_name)
    
    return txt_image_address_list, txt_image_label_list
#%%
# ###############   EM   ####################
# to generate the patches and the labels
# this function will use other functions
def gen_patches_labels(txt_file,image_path):
    #first we get the image_address_list
    shape = (256,256)
    txt_image_address_list,txt_image_label_list = gen_image_addr_label(txt_file,image_path)
    #we define a image_patches to save all the patches of the images
    whole_image_patches = []
    #we define a image_labels to save all the initial labels of the patches
    whole_patch_labels = []
    count = 0 
    #for each address in the list:
    
    for image_address in txt_image_address_list:
        one_image_patches = extract_image(image_address,shape)
        for p in one_image_patches:
            whole_image_patches.append(p)
            
        #get the images labels according to the number of patches
        #if there are n patches in the certain image which labeled as 1
        #the image_lables will return [1,....,1] with length n
        image_labels = [txt_image_label_list[count] for i in range(0, len(one_image_patches))]
        for j in image_labels:
            whole_patch_labels.append(j)
        #the counter is used to trace the index of of image labels
        count+=1
        print("process %d-th image"%count)
        
    whole_patch_labels = np.array(whole_patch_labels)
    whole_image_patches = np.array(whole_image_patches)
    whole_image_patches = whole_image_patches.reshape( whole_image_patches.shape[0],
                                                    whole_image_patches.shape[1],
                                                    whole_image_patches.shape[2],
                                                    1)
    
    
    #here we generate the X,y for the EM procedure
    #which is that, we train the CNN and using EM algorithm to improve the 
    #accuracy of the CNN model
    #the dimension of the whole_image_patches is 
    # [#patches total, shape1, shape2, 1]
    #the dimension of the whole_patch_labels is
    # [#patches total, ]
    return whole_image_patches, whole_patch_labels
#%%
def gen_CNN_patches(txt_file,image_path):
    shape = (256,256)
    txt_image_address_list,txt_image_label_list = gen_image_addr_label(txt_file,image_path)
    #we define a image_patches to save all the patches of the images
    CNN_image_patches = []
    CNN_patch_labels = np.array(txt_image_label_list)
    
    for image_address in txt_image_address_list:
        one_image_patches = extract_image(image_address,shape)
        CNN_image_patches.append(one_image_patches)
        
    return CNN_image_patches, CNN_patch_labels
    
#%%
def gen_test_patches(txt_file,image_path):
    #first we get the image_address_list
    shape = (256,256)
    txt_image_address_list,txt_image_label_list = gen_image_addr_label(txt_file,image_path,True)
    #we define a image_patches to save all the patches of the images
    image_patches = []
    #we define a image_labels to save all the initial labels of the patches
    patch_labels = []
    count = 0 
    #for each address in the list:
    
    for image_address in txt_image_address_list:
        one_image_patches = extract_image(image_address,shape)
        image_patches.append(one_image_patches)
        
    return image_patches
#%%
def extract_image(image_addr,shape):
    img = Image.open(image_addr)
    img = np.asarray(img)
    #patches = extract_patches_2d(img, shape)
    #patch = [img[shape[0]*i:shape[0]*(i+1);shape[1]*i:shape[1]*(i+1) for i in range(0,int())]
    if img.shape[0]==4084:
        black_edge = np.zeros((12,img.shape[1]))
        img = np.concatenate((img, black_edge),axis=0)
        
    #print(img.shape)
    #print(img.shape[0]/shape[0])
    #print(img.shape[1]/shape[1])
    
    patches = [img[shape[0]*i:shape[0]*(i+1),shape[1]*j:shape[1]*(j+1)] for i in range(0,
                   int(img.shape[0]/shape[0])) for j in range(0, 
                      int(img.shape[1]/shape[1]))  ]
    
    patches_output = [pat for pat in patches if pat.mean()>40]

    return patches_output
#%%
def EM_pro(X,y,input_shape, iter=50,thres_err=0.1):
    
    y_pre = y
    for i in range(0, iter):
        
        print("--------------------------------------------------")
        print(">>>>>>>>>>>>>>>>EM Procedure: the %d th iteration"%i)    
        print("--------------------------------------------------")
        
        model = load_model(root+model_name)
        model.fit(X, y_pre, epochs=epochs, batch_size=batch_size)
        y_now = model.predict(X)
        y_now = (y_now>0.3).astype(np.int32)
        predict_error = np.mean(np.absolute(y_pre-y_now))
        print("----------------------------------------")
        print(">>>>The predict error is:", predict_error)
        print("----------------------------------------")
        model.save(root+model_name)
        del model
        if predict_error<thres_err:
            break
        y_pre = y_now
        
    print("final iter:",iter)




#This function is to initialize the model
def init_EM_CNN(input_shape):
    X_input = Input(input_shape)
    
    # Conv2D(filters, kernel_size, strides=(1, 1), padding='valid')
    # INPUT 256X256X1
    X = Conv2D(40, (10,10), strides = (2,2), name = 'conv1',padding='valid')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = 'max_pool1')(X) 
    #  (256-10)/2 + 1 = 124X124X40
    # OUTPUT 62X62X40
    
    # INPUT 62X62X40
    X = Conv2D(120, (5,5), strides = (1,1), name = 'conv2')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = 'max_pool2')(X) 
    # OUTPUT  29X29X120
    
    # INPUT 29X29X120
    X = Conv2D(160, (4,4), strides = (1,1), name = 'conv3')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    # OUTPUT 26X26X160
    
    # INPUT 26X26X160
    X = Conv2D(200, (3,3), strides = (1,1), name = 'conv4')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name = 'max_pool4')(X) 
    # OUTPUT  12X12X200
    
    # INPUT 12X12X200
    X = Flatten()(X)
    X = Dense(320, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(320, activation='relu',name='fc2')(X)
    X = Dropout(0.5)(X)
    X = Dense(40, activation='relu',name='fc3')(X)
    X = Dense(1, activation='sigmoid', name = 'output')(X)
    
    model = Model(inputs = X_input, outputs = X, name='EM_CNN')
    return model
      
def train_model(X,y,model):
    train_datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=True)  
    
    train_datagen.fit(X)

    train_generator =  train_datagen.flow(X, y,
                                          batch_size=batch_size)
    
    model.fit_generator( 
            train_generator,
            epochs = epochs,      
            steps_per_epoch = len(X)/batch_size,
            validation_data = (val_data, val_label_oh)  
            )
    return model
    



#%%
EM_CNN = init_EM_CNN(input_shape)
EM_CNN.compile(optimizer='sgd', loss='binary_crossentropy', 
                  metrics=['accuracy'])
EM_CNN.save(root+model_name)
del EM_CNN
X,y = gen_patches_labels(train_data,image_path)
EM_pro(X,y,input_shape, iter=100,thres_err=0.1)

CNN_X, CNN_y = gen_CNN_patches(train_data,image_path)
# Here CNN_X is a list of images
# and each image is a list of patches
# So CNN_X is with a shape of [ #of images, ?, 256, 256, 1]
# CNN_Y is the label for the images
predict = []
EM_CNN = load_model(root+model_name)
for i in range(len(CNN_X)):
    pre = EM_CNN.predict(CNN_X)
    pre = (pre>0.3).astype(np.int32)
    predict.append(pre)
# Predict is the 2-dimensional matrix
# Now we calculate the mean of the list in the predict
predict_x = np.array([ i.mean() for i in predict ])
# we should use predict_x as the data for logistic regression
# and CNN_y is the label for logistic regression
#%%
# logistic regression part
# input: predcit_x, CNN_y
# it is a one-input one-output training

predict_x_ = 1 - predict_x
X = np.hstack((predict_x, predict_x_))

log_cls = LogisticRegression()
log_cls.fit(X, CNN_y)

#%%
#val_patches, val_labels = gen_CNN_patches(val_data,image_path)
#val_pre = []
#for i in range(len(val_patches)):
#    predict_one = EM_CNN.predcit(i)
#    predict_one = (predict_one>0.3).astype(np.int32)
#    val_pre.append(pre.mean())
# val_pre is a list with length n
# just use the logistic model to evaluate the val_pre and val_labels


#%%
test_patches = gen_test_patches(test_data,image_path)
test_pre = []
for i in range(len(test_patches)):
    predict_one = EM_CNN.predcit(i)
    predict_one = (predict_one>0.3).astype(np.int32)
    test_pre.append(pre.mean())
    
# test_pre is a list with length n
# just put the test_pre into the logistic regression and get the prediciton

test_predict = log_cls.predict(test_pre)
test_predict = pd.DataFrame(test_predict)
test_predict.to_csv('result.csv')

        
        
        
        

#%%
        
        
        
        
        