import cv2
import numpy as np
import os
import scipy
from scipy.cluster.vq import kmeans, vq
import joblib
from sklearn.preprocessing import StandardScaler

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


test_path = 'Images/Train/'

def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

def readingImages(class_id,training_name):
    print("Testing for Class ID : ",class_id)
    dir = os.path.join(test_path, training_name)
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

def compareImages(all_features,all_features_avgVector,all_thresholds,testing_names,test_info):
  
    print("Thresholds ",all_thresholds)

    for i in range(len(all_features)):
        matches = [0,0]
        print("\nTest Folder : ",testing_names[i])
        print("No.of Images : ",len(all_features[i]))
        for j in range(len(all_features[i])):
            #print("Image : ",j+1)
            print("Info : ",test_info[i][0][j])
            side_dist = scipy.spatial.distance.cityblock(all_features_avgVector[0],all_features[i][j])
            # print("Side Dist : ",side_dist)
            if side_dist<all_thresholds[0]:
                matches[0]+=1

            front_dist = scipy.spatial.distance.cityblock(all_features_avgVector[1],all_features[i][j])
            print("Front Dist : ",front_dist)
            if front_dist<all_thresholds[1]:
                print("Match")
                matches[1]+=1

        print("Side : GunsDetected - ",matches[0])
        print("Front : GunsDetected - ",matches[1])

def main():
    print("Classifier Started\n")

    # Training info
    all_features_avgVector, all_thresholds, training_names = joblib.load("bovw.pkl")
    print(training_names)
    # Train will contain Front and Side View Images
    testing_names = os.listdir(test_path)
    print(testing_names)     

    class_id = 0
    test_info = []
    # training_info will contain list of tuples : (image_paths,image_classes) 

    for testing_name in testing_names:
        [image_paths,image_classes] = readingImages(class_id,testing_name)
        test_info.append((image_paths,image_classes))
        class_id += 1

    all_descriptors_stack = []
    all_descriptors_list = []

    for image_paths,image_classes in test_info:
        [des_list,des_stack] = trainClass(image_paths,image_classes) 
        all_descriptors_list.append(des_list)
        all_descriptors_stack.append(des_stack)

    all_features = []
    for i in range(len(test_info)):
        all_features.append(calcFeatures(all_descriptors_stack[i],all_descriptors_list[i],test_info[i][0]))

    # print(len(all_features[0]))
    # print(len(all_features[1]))

    compareImages(all_features,all_features_avgVector,all_thresholds,testing_names,test_info)


if __name__ == "__main__":
    main()