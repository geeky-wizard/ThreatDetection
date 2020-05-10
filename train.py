import cv2
import numpy as np
import os
import scipy
from scipy.cluster.vq import kmeans, vq
import joblib
from sklearn.preprocessing import StandardScaler

train_path = 'Images/Train/'

# TODO : Descriptor Types
# Hyperparameter
brisk = cv2.BRISK_create(30)
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=30,scoreType=cv2.ORB_FAST_SCORE)
sift = cv2.xfeatures2d.SIFT_create()

DESCRIPTOR_TYPE = brisk

# No of Clusters
# Hyperparameter
k = 200

OutputPickleFile = "bovw.pkl"

def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

def readingImages(class_id,training_name):
    print("Training for Class ID : ",class_id)
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths=class_path
    image_classes=[class_id]*len(class_path)
    class_id+=1
    return [image_paths,image_classes]

def trainClass(image_paths,image_classes):
    des_list = []
    for image_path in image_paths:
        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (512, 512))
        kpts, des = DESCRIPTOR_TYPE.detectAndCompute(resized_image, None)
        des_list.append((image_path, des))   

    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  

    descriptors_float = descriptors.astype(float)  

    return [des_list,descriptors_float]

def calcFeatures(descriptors,des_list,image_paths):
    voc, variance = kmeans(descriptors, k, 1) 
    im_features = np.zeros((len(image_paths), k), "float32")

    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
    return im_features

def getAvgOfFeatures(all_features):
    all_features_avg = []
    for features in all_features:
        all_features_avg.append(np.mean(features,axis=0))
    return all_features_avg

def main():
    print("Training Started\n")
    # Train will contain Front and Side View Images
    training_names = os.listdir(train_path)
    print(training_names)    

    class_id = 0
    training_info = []
    # training_info will contain list of tuples : (image_paths,image_classes) 

    for training_name in training_names:
        [image_paths,image_classes] = readingImages(class_id,training_name)
        training_info.append((image_paths,image_classes))
        class_id += 1

    all_descriptors_stack = []
    all_descriptors_list = []

    for image_paths,image_classes in training_info:
        [des_list,des_stack] = trainClass(image_paths,image_classes)
        all_descriptors_list.append(des_list)
        all_descriptors_stack.append(des_stack)

    # print(len(all_descriptors_list[0]),len(all_descriptors_list[1]))
    # print(len(all_descriptors_stack[0]),len(all_descriptors_stack[1]))

    all_features = []
    for i in range(len(training_info)):
        all_features.append(calcFeatures(all_descriptors_stack[i],all_descriptors_list[i],training_info[i][0]))

    all_features_avgVector = getAvgOfFeatures(all_features)

    # print(all_features_avgVector[0])
    # print(all_features_avgVector[1]
    all_distances_euc=[[],[]]
    all_distances_cb=[[],[]]
    for i in range(len(all_features)):
        for img_feature in all_features[i]:
            cb_dist = scipy.spatial.distance.cityblock(all_features_avgVector[i], img_feature)
            all_distances_cb[i].append(cb_dist)

            euc_dist = scipy.spatial.distance.euclidean(all_features_avgVector[i], img_feature)
            all_distances_euc[i].append(euc_dist)

        # all_distances_euc[i].sort()
        # all_distances_cb[i].sort()


    # City Block Distance
    # print(all_distances_cb[0])
    # print("\n\n")
    # print(all_distances_cb[1])

    # print("\n\n\n")

    # for info in training_info[1][0]:
    #     print(info)

    # Euclidean Distance
    # print(all_distances_euc[0])
    # print("\n\n")
    # print(all_distances_euc[1])

    all_thresholds = [280,380] # city block

    print(all_features_avgVector[0])
    print("\n\n")
    print(all_features_avgVector[1])

    joblib.dump((all_features_avgVector, all_thresholds, training_names),OutputPickleFile, compress=3) 

    print("\nTraining Done.\nPickle File saved as ",OutputPickleFile)

if __name__ == "__main__":
    main()