import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def noisy(image, noise_type="gauss", args={}):
    if noise_type == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 126
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy = np.minimum(noisy, 255)
        noisy = np.maximum(noisy, 0)
        return noisy
    elif noise_type == "cells":
        noisy = np.copy(image)
        row,col,ch= image.shape
        cell_count = 32
        if "cell_count" in args:
            cell_count = args["cell_count"]

        for i in range(cell_count):
            cell_x = np.random.randint(row)
            cell_y = np.random.randint(col)
            for channel in range(ch):
                noisy[cell_x][cell_y][channel] = np.random.randint(255)
        
        return noisy
    # TODO: the noise types below assume pixels in [0,1], but they are in [0,255].
    # Check them and correct them. (not pressing, currently working only with gauss and cells noise)
    elif noise_type == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    
def noise_data(x_test, y_test, noise_type="gauss", args = {}):
    K = len(x_test)
    noisy_imgs = []
    correct_labels = []
    for i in range(K):
        img = x_test[i]
        img_noise = noisy(img, noise_type, args)
        noisy_imgs.append(img_noise)
    return noisy_imgs

def check_noise_robustness(model, x_test, y_test, noise_type="gauss", args = {}):
    noisy_imgs = noise_data(x_test, y_test, noise_type, args)
    noisy_pred = model.predict(np.array(noisy_imgs))
    noisy_labels = np.argmax(noisy_pred, axis=1)
    correct_labels = np.argmax(y_test, axis = 1)
    agreements = noisy_labels == correct_labels
    accuracy = agreements.sum() / len(x_test)
    return accuracy, agreements, noisy_imgs

def check_noise_robustness_multiple_rounds(model, sample_x, sample_y, steps = 5, noise_type="gauss", verbose = True, args = {}):
    # TODO: add early stopping if, after 3 rounds, empirical robustness decreased by only 0.05 % or so
    K = len(sample_x)
    robustness_progress = []
    saved_noisy_imgs = [[]] * K
    if verbose:
        print("Step", 0)
#     print("X")
    accuracy, agreements, noisy_imgs = check_noise_robustness(model, sample_x, sample_y, noise_type, args)
#     print("Y")
    robustness_progress.append(np.sum(agreements)/len(agreements))
    for i in range(K):
        if(agreements[i] == False and saved_noisy_imgs[i] == []):
            saved_noisy_imgs[i] = noisy_imgs[i]
#     print("Z")
    for i in range(steps - 1):
        print(i + 1, "/",steps)
        if verbose:
            clear_output(wait=True)
#             print("Step", i + 1)
#             import pdb
#             pdb.set_trace()
            print("Previous robustness: ",np.sum(agreements)/len(agreements))
#             print("Step", i + 1)
#             plt.plot(robustness_progress)
#             plt.show()
        accuracy_local, agreements_local, noisy_imgs_local = check_noise_robustness(model, sample_x, sample_y, noise_type, args)
        agreements = np.logical_and(agreements, agreements_local)
        robustness_progress.append(np.sum(agreements)/len(agreements))
        for i in range(K):
            if(agreements[i] == False and saved_noisy_imgs[i] == []):
                saved_noisy_imgs[i] = noisy_imgs[i]
    if verbose:
        clear_output(wait=True)
    return agreements, saved_noisy_imgs