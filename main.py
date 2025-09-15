import torch
import pandas as pd
import numpy as np
import cv2

TRAIN_PATH = '/home/meheraj/Develop/mnist/data/train.csv'
TEST_PATH = '/home/meheraj/Develop/mnist/data/test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

img = train_df.loc[1].values[1:].reshape(28, 28).astype(np.uint8)
cv2.imshow('img', img)
if cv2.waitKey(0) == 27:  # Wait for ESC key to exit
    cv2.destroyAllWindows()