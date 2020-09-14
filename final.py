import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

class Container():
     def __init__(self, name = ""):
        self.name = name
        self.features = []
        self.labels = []
class Visual_BOW():
    def __init__(self, k=20, dictionary_size=50):
        self.k = k  # number of SIFT features to extract from every image
        self.dictionary_size = dictionary_size  # size of your "visual dictionary" (k in k-means)
        self.n_tests = 10  # how many times to re-run the same algorithm (to obtain average accuracy)
        self.containers = []
    def extract_sift_features(self):
        '''
        To Do:
            - load/read the Caltech-101 dataset
            - go through all the images and extract "k" SIFT features from every image
            - divide the data into training/testing (70% of images should go to the training set, 30% to testing)
        Useful:
            k: number of SIFT features to extract from every image
        Output:
            train_features: list/array of size n_images_train x k x feature_dim
            train_labels: list/array of size n_images_train
            test_features: list/array of size n_images_test x k x feature_dim
            test_labels: list/array of size n_images_test
        '''
        features = []
        labels = []
        self.containers = []
        #img = cv2.imread('home.jpg')
        # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(self.k)
        # kp = sift.detect(gray,None)

        for file in os.listdir('101_ObjectCategories'):
            self.containers.append(Container(file))
            for i in os.listdir('101_ObjectCategories/' + file):
                img = cv2.imread('101_ObjectCategories/' + file + "/" + i)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                kp = sift.detect(gray,None)
                kp,des = sift.compute(gray,kp)
                # kp, des = sift.detectAndCompute(grey, None)
                features.append(des)
                labels.append(file)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42, shuffle=True)
        return train_features, train_labels, test_features, test_labels

    def create_dictionary(self, features):
        '''
        To Do:
            - go through the list of features
            - flatten it to be of size (n_images x k) x feature_dim (from 3D to 2D)
            - use k-means algorithm to group features into "dictionary_size" groups
        Useful:
            dictionary_size: size of your "visual dictionary" (k in k-means)
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            kmeans: trained k-means object (algorithm trained on the flattened feature list)
        '''
        # counter = 0
        # for file in os.listdir('101_ObjectCategories'):
        #     counter = counter + 1
        
        flatten = []
        # features: [[[2,3],[4,5]], [[2,3],[4,5]]]
        # items: [[2,3],[4,5]]
        # item: [2,3]
        for items in features:
            for item in items:
#                print("item size: ", item)
                flatten.append(item)
        flatten = np.asarray(flatten)
        kmeans = KMeans(n_clusters=self.dictionary_size, random_state=0).fit(flatten)
        return kmeans

    def convert_features_using_dictionary(self, kmeans, features):
        '''
        To Do:
            - go through the list of features (images)
            - for every image go through "k" SIFT features that describes it
            - every image will be described by a single vector of length "dictionary_size"
            and every entry in that vector will indicate how many times a SIFT feature from a particular
            "visual group" (one of "dictionary_size") appears in the image. Non-appearing features are set to zeros.
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            features_new: list/array of size n_images x dictionary_size
        '''
        features_new = []
        for images in features:
            if images is None:
                pass
            else:
                empty_list = [0] * self.dictionary_size # initialize vector
#                print("image size: ", images)
#                print("array size: ", np.asarray(images))
                temp_feature_new_list = kmeans.predict(np.asarray(images))
                for feature in range(len(temp_feature_new_list)):
                    empty_list[temp_feature_new_list[feature]] += 1 # temp_feature_new_list[feature] means nth attributes
                features_new.append(empty_list)
        return features_new

    def train_svm(self, inputs, labels):
        '''
        To Do:
            - train an SVM classifier using the data
            - return the trained object
        Input:
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
        '''
        clf = svm.SVC()
        clf.fit(inputs, labels)
        return clf

    def test_svm(self, clf, inputs, labels):
        '''
        To Do:
            - test the previously trained SVM classifier using the data
            - calculate the accuracy of your model
        Input:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            accuracy: percent of correctly predicted samples
        '''
        temp = clf.predict(inputs)
        counter = 0
        correct_counter = 0
#        print("len label", len(labels))
#        print("len temp", len(temp))
        for i in range(len(temp)):
#            print("nice:", temp[i]," ", labels[i])
            if temp[i] == labels[i]:
                correct_counter += 1
            counter += 1
        
        accuracy = correct_counter/counter
        return accuracy

    def save_plot(self, features, labels):
        '''
        To Do:
            - perform PCA on your features
            - use only 2 first Principle Components to visualize the data (scatter plot)
            - color-code the data according to the ground truth label
            - save the plot
        Input:
            features: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        '''
        pca = PCA(n_components=2)
        pca.fit(np.asarray(features))
        features_new = pca.transform(features)
        
        number = 110
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, number)] # get 110 different colors from color map, we will need to use 101 of them
        
        for i in labels: # iterate the labels and add to each class
            index = (np.asarray(labels).squeeze() == i) # use ground truth label
            found = False
            for j in range(len(self.containers)):
                if i == self.containers[j].name:
                    found = True
                    self.containers[j].features.append(features_new[index, 0].tolist())
                    self.containers[j].labels.append(features_new[index, 1].tolist())
            if found == False:
                print("Not Found ", self.containers[j].name, i) # should not be reached. if so means i not in any of 101 labels
        color_counter = 0
        
        for item in self.containers: # plot once a class. And assign same color for same class
#            print("#",color_counter,"now drawing with color: ", colors[color_counter])
            x = item.features
            y = item.labels
            plt.plot(x,y,'.',color=colors[color_counter])
            color_counter += 1
        
#            x, y = features_new[index, 0], features_new[index, 1]
#            plt.plot(x,y,'.')
#            plt.legend()
#        label_list = []
#        feature_list = []
#        for item in labels:
#            if (item in label_list):
#                pass
#            else:
#                label_list.append(item)
#                feature_list.append()
        n_th_test = 0
        while(os.path.exists("result_k=20_dictionary_size=50_"+str(n_th_test)+".png")):
            n_th_test += 1
        save_path = "result_k=20_dictionary_size=50_"+str(n_th_test)+".png"
        plt.savefig(save_path)
        plt.cla()
        

############################################################################
################## DO NOT MODIFY ANYTHING BELOW THIS LINE ##################
############################################################################

    def algorithm(self):
        # This is the main function used to run the program
        # DO NOT MODIFY THIS FUNCTION
        accuracy = 0.0
        for i in range(self.n_tests):
            train_features, train_labels, test_features, test_labels = self.extract_sift_features()
            kmeans = self.create_dictionary(train_features)
            train_features_new = self.convert_features_using_dictionary(kmeans, train_features)
            classifier = self.train_svm(train_features_new, train_labels)
            test_features_new = self.convert_features_using_dictionary(kmeans, test_features)
            accuracy += self.test_svm(classifier, test_features_new, test_labels)
            self.save_plot(test_features_new, test_labels)
        accuracy /= self.n_tests
        return accuracy

if __name__ == "__main__":
    alg = Visual_BOW(k=20, dictionary_size=50)
    accuracy = alg.algorithm()
    print("Final accuracy of the model is:", accuracy)

