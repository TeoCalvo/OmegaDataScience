# %% 

import pandas as pd
from sklearn import tree
from feature_engine.encoding import OneHotEncoder

# %%

df_celula = pd.read_csv("../data/celulas.csv",
                        sep=";")

df_celula

# %%

features = ["nucleos", "caudas", "cor", "membrana"]

# onehot da lib features engine
onehot = OneHotEncoder(variables=["cor", "membrana"])
onehot.fit(df_celula[features])

# %%

# Transformando o dado original
df_fit = onehot.transform( df_celula[features] )
df_fit

# %%

# Treinando o modelo de machine learning

target = "classe"
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit( df_fit, df_celula[target] )

# %%

# Salvando o algoritmo
model = pd.Series(
        {"model":clf_tree,
         "onehot": onehot,
         "features": features,
         "target": target} )

model.to_pickle("../models/celulas_tree_onehot.pkl")

model