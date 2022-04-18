from cmath import pi
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

x = np.load('image.npz')["arr_0"]
y = pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

x_trained, x_test, y_trained, y_test = train_test_split(x, y, train_size = 3500, test_size = 500)
x_trained_scaled = x_trained/255
x_test_scaled = x_test/255

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_trained_scaled, y_trained)

def get_prediction(img):
    pil = Image.open(img)
    img_bw = pil.convert("L")
    img_bw_resized = img_bw.resize((22, 30), Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resized, pixel_filter)
    img_bw_resized_inverted_scaled = np.clip(img_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(img_bw_resized)
    img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel

    sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 660)
    predict = clf.predict(sample)

    return predict[0]