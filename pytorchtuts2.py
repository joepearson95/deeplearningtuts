import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# State if data is being built each time - used for bigger datasets
REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50 # making sure all images are the same size
    CATS = "PetImages/Cat" # these two strings are filepaths
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1} # similar to ohe
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        # Loop the folders
        for label in self.LABELS:
            print(label)
            # loop the images with progress bar, convert to greyscale and resize for simplicity
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # training data of the images in OHE
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    # if cat, add 1. otherwise, add 1 to dog
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                    else:
                        self.catcount += 1
                except Exception as e:
                    # some images aren't good, will cause error.
                    #print(str(e))
                    pass
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)

# If data is being rebuilt, make said training data
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

# Load the newly creataed training_data file and allow (some reason it caused an error)
training_data =  np.load("training_data.npy", allow_pickle=True)
print(len(training_data))
print(training_data[0])
# after printing some data, show the image in greyscale
plt.imshow(training_data[0][0], cmap="gray")
plt.show()
