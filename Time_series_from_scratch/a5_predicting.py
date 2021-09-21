import pickle
import pandas                as pd
import numpy                 as np
import matplotlib.pyplot     as plt
from sklearn.model_selection import train_test_split


with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)


