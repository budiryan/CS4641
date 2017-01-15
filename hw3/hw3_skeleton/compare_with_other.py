import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

other_df = pd.read_csv('peer_review/kenta.txt', header=None)
budi_df = pd.read_csv('predictions-BestClassifier.dat', header=None)


other_y = np.array(other_df[[0]])
budi_y = np.array(budi_df[[0]])


print accuracy_score(other_y, budi_y)
