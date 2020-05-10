import cv2
import numpy as np
import os
import scipy

train_path = 'Images/Train/'  # Folder Names are Parasitized and Uninfected
# Train Folder contains side view gun images, front view gun images, non-gun images!
training_names = os.listdir(train_path)

print(training_names)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
#
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
#
    
# training_names = ['Side','Front']
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1


des_list = []

# TODO : Descriptor Types
brisk = cv2.BRISK_create(30)
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=30,scoreType=cv2.ORB_FAST_SCORE)
sift = cv2.xfeatures2d.SIFT_create()

des_front_list = []
des_side_list = []

for i in range(len(image_paths)):
    im = cv2.imread(image_paths[i])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (512, 512))    # Resize Image
    kpts, des = brisk.detectAndCompute(resized_image, None)

    if image_classes[i] == 0:
        des_side_list.append((image_paths[i], des))
    
    if image_classes[i] == 1:
        des_front_list.append((image_paths[i], des)) 

front_descriptors = des_front_list[0][1]
side_descriptors = des_side_list[0][1]

# Front Descriptors
for image_path, descriptor in des_front_list[1:]:
    front_descriptors = np.vstack((front_descriptors, descriptor)) 

# Side Descriptors
for image_path, descriptor in des_side_list[1:]:
    side_descriptors = np.vstack((side_descriptors, descriptor)) 
 
#kmeans works only on float, so convert integers to float   
descriptors_front_float = front_descriptors.astype(float)  
descriptors_side_float = side_descriptors.astype(float)  





# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

# TODO : 'k' no of clusters - HYPERPARAMETER
k = 200  #k means with 100 clusters gives lower accuracy for the aeroplane example

# FRONT
voc_front, variance_front = kmeans(descriptors_front_float, k, 1) 
front_features = np.zeros((len(des_front_list), k), "float32")

for i in range(len(des_front_list)):
    words, distance = vq(des_front_list[i][1],voc_front)
    for w in words:
        front_features[i][w] += 1


# SIDE
voc_side, variance_side = kmeans(descriptors_side_float, k, 1) 
side_features = np.zeros((len(des_side_list), k), "float32")

for i in range(len(des_side_list)):
    words, distance = vq(des_side_list[i][1],voc_side)
    for w in words:
        side_features[i][w] += 1

# AVG FEATURES
side_features_avg = np.mean(side_features,axis=0)
front_features_avg = np.mean(front_features,axis=0)

print(side_features)

lst_front=[]
lst_side=[]
count = 0

for item in side_features:
    # des = des.reshape((200))
    dist = scipy.spatial.distance.cityblock(side_features_avg, item)
    #print(dist)
    lst_side.append(dist)
    
lst_side.sort()
# print(lst_side)
print("\n\n")
# SET SIDE FEATURES THRESHOLD
side_features_threshold = 50.0

for item in front_features:
    # des = des.reshape((200))
    dist = scipy.spatial.distance.cityblock(front_features_avg, item)
    lst_front.append(dist)

lst_front.sort()
# print(lst_front)
# print("\n\n")

# SET FRONT FEATURES THRESHOLD
front_features_threshold = 70.0

# print(len(side_features))
# print(len(front_features))
# print("\n\n")

match_ct=0
for item in side_features:
    # des = des.reshape((200))
    dist = scipy.spatial.distance.cityblock(front_features_avg, item)
    # print(dist)
    if dist<580:
        match_ct+=1
    lst_side.append(dist)

# print("ct ia",match_ct)

# with open("features1.pickle","wb") as file:
#     pickle.dump((side_features_avg,front_features_avg,side_features_threshold,front_features_threshold),file)

# print(np.mean(lst_front))
# print(np.mean(lst_side))