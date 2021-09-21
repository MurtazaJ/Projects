import pandas as pd
import numpy as np

# If you want to reduce the rows
def pairing(df, seq_length=3):
    x = []
    y = []

    for i in range(0, (df.shape[0] - (seq_length+1)), seq_length+1):
        seq = np.zeros((seq_length, df.shape[1]))
        for j in range(seq_length):
            seq[j] = df.values[i+j]
        x.append(seq.flatten())
        y.append( df['meantemp'][i + seq_length] )

    return np.array(x), np.array(y)

# If you want to enhance the rows
def data_enhancement(df, percentage):
    df_copy = df.copy()
        
    def enhance(df_copy):
        for temp_month in df_copy['month'].unique():
            meantemp_std = df_copy[df_copy['month'] == temp_month]['meantemp'].std()/10
            # print(meantemp.shape)
            for i in df_copy[df_copy['month']==temp_month].index :
                if np.random.randint(2) ==1:
                    df_copy['meantemp'].values[i] += meantemp_std
                else:
                    df_copy['meantemp'].values[i] -= meantemp_std

        return df_copy
    
    gen = enhance(df_copy)
    gen = gen.sample(int(gen.shape[0]*(percentage/100)))
    x= pd.concat((df_copy , gen), axis=0)
    return x 

