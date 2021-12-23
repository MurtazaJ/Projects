import eli5
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd


def permutation_importance(x,y):
    model = RandomForestClassifier().fit(x,y)
    perm  = PermutationImportance(model).fit(x,y)
    return eli5.show_weights(perm, feature_names = x.columns.tolist())


def mi_score(x,y, std = True):
    if std:
        x = pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)
        score = pd.DataFrame(mutual_info_classif(x, y, discrete_features = False), index = x.columns, columns = ['mi_score']).sort_values('mi_score', ascending = False )
        plot_mi(score)
        return score


def plot_mi(score):
    socre = score.sort_values('mi_score', ascending = True)
    plt.barh(score.index, score.mi_score)
    plt.title('mutual info classif on x feats')
    return


