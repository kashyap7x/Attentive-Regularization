import numpy as np
import cv2
import dlib
import h5py
detector = dlib.get_frontal_face_detector()

class imageProcessor:
    def __init__(self,threshold):
        self._threshold=threshold

    def __call__(self,filename):
        global detector
        im = cv2.imread(filename)
        im = np.array(im).astype(np.uint8)
        if len(im.shape) == 2:
            im = np.reshape(im, (im.shape[0], im.shape[1], 1))
            im = np.repeat(im, 3, axis=2)
        grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rects = detector(grayImage, 0)
        if (len(rects)==0):
            x1 = 70
            y1 = 70
            x2 = 180
            y2 = 180
            #rectangleImage = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.imshow('DetectedFace', rectangleImage)
            #cv2.waitKey(1)
        else:
            detectedFace = rects[0]
            x1 = detectedFace.left()
            y1 = detectedFace.top()
            x2 = detectedFace.right()
            y2 = detectedFace.bottom()
            if y1 < 0:
                y1 = 0
            if y2 > im.shape[0]:
                y2 = im.shape[0]
            if x1 < 0:
                x1 = 0
            if x2 > im.shape[1]:
                x2 = im.shape[1]
            #rectangleImage = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.imshow('DetectedFace', rectangleImage)
            #cv2.waitKey(1)
        #print str(x1) + "," + str(y1) + " " + str(x2) + "," + str(y2)
        croppedImage = im[int(y1):int(y2), int(x1):int(x2)]
        im = cv2.resize(croppedImage, (96, 96))
        return im

def read_pair_numbers(line):
    numbers = line.strip('\n').split('\t')
    num_folds = int(numbers[0])
    num_pos_pairs = int(numbers[1])
    num_neg_pairs = num_pos_pairs
    return num_folds, num_pos_pairs, num_neg_pairs

folds = []

with open('pairs.txt') as f:
    first_line = f.readline()
    num_folds, num_pos_pairs, num_neg_pairs = read_pair_numbers(first_line)
    for i_fold in range(num_folds):
        positives = []
        for i_pos_pair in range(num_pos_pairs):
            line = f.readline()
            tokens = line.strip('\n').split('\t')
            person = tokens[0]
            path1 = 'lfw-deepfunneled/%s/%s_%04d.jpg' % (person, person, int(tokens[1]))
            path2 = 'lfw-deepfunneled/%s/%s_%04d.jpg' % (person, person, int(tokens[2]))
            positives.append([path1, path2])
        negatives = []
        for i_neg_pair in range(num_neg_pairs):
            line = f.readline()
            tokens = line.strip('\n').split('\t')
            path1 = 'lfw-deepfunneled/%s/%s_%04d.jpg' % (tokens[0], tokens[0], int(tokens[1]))
            path2 = 'lfw-deepfunneled/%s/%s_%04d.jpg' % (tokens[2], tokens[2], int(tokens[3]))
            negatives.append([path1, path2])
        folds.append({'positives': positives, 'negatives': negatives})

proc = imageProcessor(128)

for f in range(10):
    imageFileList = (folds[f]['positives'])
    X1p = np.ndarray((len(imageFileList), 96, 96, 3), dtype=float)
    X2p = np.ndarray((len(imageFileList), 96, 96, 3), dtype=float)
    Yp = np.ndarray(len(imageFileList))
    for i in range(len(imageFileList)):
        X1p[i] = proc(imageFileList[i][0])
        X2p[i] = proc(imageFileList[i][1])
        Yp[i] = 1
    imageFileList = (folds[f]['negatives'])
    X1n = np.ndarray((len(imageFileList), 96, 96, 3), dtype=float)
    X2n = np.ndarray((len(imageFileList), 96, 96, 3), dtype=float)
    Yn = np.ndarray(len(imageFileList))
    for i in range(len(imageFileList)):
        X1n[i] = proc(imageFileList[i][0])
        X2n[i] = proc(imageFileList[i][1])
        Yn[i] = 0
    X1 = np.vstack([X1p, X1n])
    X2 = np.vstack([X2p, X2n])
    Y = np.concatenate([Yp, Yn])
    dataFile = h5py.File('fold'+ str(f) + '.hdf5', 'w')
    dataFile.create_dataset("X1", data=X1)
    dataFile.create_dataset("X2", data=X2)
    dataFile.create_dataset("Y", data=Y)
    dataFile.close()
