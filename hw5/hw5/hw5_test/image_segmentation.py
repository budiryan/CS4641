from PIL import Image
import sys
import numpy as np
import random
from scipy import spatial


class ImageSegmenter:
    def __init__(self, image, image_output, K):
        self.image = image
        self.pixels = self.image.load()
        self.image_output = image_output
        self.result_image = image.copy()
        self.K = K
        self.size_x, self.size_y = self.image.size
        self.pixels_vector = []
        self.epsilon = 0.001

        for i in range(self.size_x):    # for every pixel:
            for j in range(self.size_y):
                self.pixels_vector.append(np.concatenate((np.array(self.pixels[i, j]), np.array([i, j]))))

        self.pixels_vector = np.array(self.pixels_vector, dtype=np.float64)

        self.maximums = []
        for i in range(self.pixels_vector.shape[1]):
            self.maximums.append(self.pixels_vector[:, i].max())
            self.pixels_vector[:, i] /= self.pixels_vector[:, i].max()

        self.centroids = [random.choice(list(enumerate(self.pixels_vector)))[1] for _ in range(self.K)]
        self.centroids = np.array(self.centroids)
        self.assignments = self.pixels_vector.copy()

    def assign_vector(self):
        for index_pixel, i in enumerate(self.pixels_vector):    # for every pixel:
            min_dist = -1
            for index, centroid in enumerate(self.centroids):
                if (min_dist == -1) or (np.linalg.norm(centroid - self.pixels_vector[index_pixel]) < min_dist):
                    min_dist = np.linalg.norm(self.pixels_vector[index_pixel] - centroid)
                    self.assignments[index_pixel] = centroid

    def compute_average(self):
        for index, centroid in enumerate(self.centroids):
            temp_array = []
            for index_pixel, i in enumerate(self.pixels_vector):    # for every pixel:
                if (centroid == self.assignments[index_pixel]).all():
                    temp_array.append(self.pixels_vector[index_pixel])
            if not temp_array:
                new_centroids[index] = random.choice(list(enumerate(self.pixels_vector)))[0]
            else:
                temp_array = np.array(temp_array)
                self.centroids[index] = np.average(temp_array, axis=0)


    def run_k_mean(self):
        # This function will run k-mean on the assignment vector
        # temp_assignment = self.assignment_vector.copy()
        iteration = 0
        print 'starting the kmean algorithm...'

        while True:
            temp_centroids = self.centroids.copy()
            self.assign_vector()
            self.compute_average()
            iteration += 1
            print 'iteration: ', iteration
            diff = np.average(np.absolute(temp_centroids - self.centroids), axis=0)
            print "diff norm:", np.linalg.norm(diff)
            if np.linalg.norm(diff) < self.epsilon:
                break
        print 'k mean algorithm finished!'

    def modify_new_image_and_save(self):
        print 'modifying the original image...'
        pixels = self.result_image.load()
        print 'colors are: ', self.centroids[:, 0:3] * self.maximums[0:3]
        counter = 0
        for i in range(self.size_x):    # for every pixel:
            for j in range(self.size_y):
                self.assignments[counter] = self.assignments[counter] * self.maximums
                pixels[i, j] = tuple(self.assignments[counter][0:3].astype(int))
                counter += 1

        self.result_image.save(self.image_output)
        self.result_image.show()


if __name__ == '__main__':
    try:
        rubbish, K, inputImageFilename, outputImageFilename = sys.argv
    except ValueError:
        print 'Please input correct format: K inputImageFilename outputImageFilename'
        sys.exit()

img = Image.open(inputImageFilename)

new_image = ImageSegmenter(img, outputImageFilename, int(K))
new_image.run_k_mean()
new_image.modify_new_image_and_save()
