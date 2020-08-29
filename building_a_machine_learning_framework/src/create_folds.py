import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop = True) #shuffle the data

    kf = model_selection.StratifiedKfold(n_splits=5,shuffle=False, random_state=42)

    for fold,(train_idx, val_idx) in enumerate(kf.n_splits(X=df,y=df.target.values)):
        