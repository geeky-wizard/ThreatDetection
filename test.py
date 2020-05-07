import cv2
import numpy as np
import os
import scipy
from scipy.cluster.vq import kmeans, vq
import joblib

all_features_avgVector, all_thresholds, training_names, k = joblib.load("bovw.pkl")

test_path = 'Images/Test'

# TODO : Descriptor Types
# Hyperparameter
brisk = cv2.BRISK_create(30)
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=30,scoreType=cv2.ORB_FAST_SCORE)
sift = cv2.xfeatures2d.SIFT_create()

DESCRIPTOR_TYPE = brisk

# HyperParameter
# window_size = [(32,32),(64,64),(128,128)]
window_size = [(128,128)]
dims = (512,512)

def main():
    images = [os.path.join(test_path,f) for f in os.listdir(test_path)]

    for w in window_size:
        print("Window : ",w)
        for image in images:
            print("Reading img : ",image)
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray, dims)

            height, width= resized_image.shape
            window_number = 1
    
            y=0
            while y<height:
                if(y+w[1]>dims[1]):
                    break  
                x=0
                while x<width:
                    if (x+w[0])>dims[0]:
                        break
                    print(window_number," : ",x,",",y)
                    window_number+=1
                    imgWindow = resized_image[y:y+w[1], x:x+w[0]]
                    
                    kp, des = DESCRIPTOR_TYPE.detectAndCompute(imgWindow, None)
                    
                    # matches = bf.match(des1,des2)
                    # cv2.imshow("Window Image", imgWindow)
                    # cv2.waitKey(0)   
                    # cv2.destroyAllWindows()
                    
                    x = x + w[0]
                y = y+w[1]

if __name__ == "__main__":
    main()