import pickle
from model import model_test
import pandas as pd
import numpy as np

df = pd.DataFrame({'player_identifier': [1,1],
                   'date_registered': ['2017-12-03','2017-12-03'],
                   'marketing_channel': ['Facebook', 'Google'],
                   'day_after_registration': [1,7],
                   'prata_spent': [182.72,182.72]})
               

print(model_test(df))

