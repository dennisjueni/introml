import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import tensorflow
from keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB1
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier



def get_vector(image_path, model):
        img = image.load_img(image_path, target_size=(224, 224))
        input_arr = image.img_to_array(img)
        input_arr = np.array([input_arr])
        return model.predict(input_arr)



sorted_img = sorted(glob.glob('food' + '/*.jpg'))


e1_model = EfficientNetB1(weights='imagenet', include_top=False, pooling='avg')
e1_feature_list = [get_vector(item, e1_model) for item in sorted_img]


size = 20
pca = PCA(n_components=size, random_state=2)

e1_fla = [item.flatten() for item in e1_feature_list]
li_1 = pca.fit_transform(np.array(e1_fla))


file = open('train_triplets.txt')

df = pd.DataFrame()

temp_list = []

counter = 0

for line in file:
    
    words = line.split()
    
    index_x = int(words[0])
    index_y1 = int(words[1])
    index_y2 = int(words[2])
    
    x = li_1[index_x]
    y1 = li_1[index_y1]
    y2 = li_1[index_y2]
    
    
    if counter % 2 == 0:
        temp_list.append(np.concatenate((x, y1, y2), axis=None).reshape(3*size))
    else:
        temp_list.append(np.concatenate((x, y2, y1), axis=None).reshape(3*size))
    
    counter += 1

X = pd.DataFrame(temp_list)

y = pd.DataFrame(np.resize([1,0], 59515))

testFile = open('test_triplets.txt')

test_X = pd.DataFrame()

temp_list = []

for line in testFile:
    
    words = line.split()
    
    index_x = int(words[0])
    index_y1 = int(words[1])
    index_y2 = int(words[2])
    
    x = li_1[index_x]
    y1 = li_1[index_y1]
    y2 = li_1[index_y2]
    
    temp_list.append(np.concatenate((x, y1, y2), axis=None).reshape(3*size))

test_X = pd.DataFrame(temp_list)


clf = MLPClassifier(early_stopping=True, random_state=69, verbose=True, tol= 1e-5, alpha=0.1).fit(X, np.ravel(y))
test_y = clf.predict(test_X)

np.savetxt('answer.txt', test_y, fmt='%d')
