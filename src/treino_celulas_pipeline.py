# %%

import pandas as pd

from sklearn import pipeline
from sklearn import tree

from feature_engine.encoding import OneHotEncoder

# %%

df_celula = pd.read_csv("../data/celulas.csv",
                        sep=";")

df_celula

# %%

features = ["nucleos", "caudas", "cor", "membrana"]
target = 'classe'

# Nosso objeto de onehot
onehot = OneHotEncoder(variables=["cor", "membrana"])

# Nosso objeto de modelo
clf_tree = tree.DecisionTreeClassifier()

# Nosso pipeline com todos objetos
model_pipeline = pipeline.Pipeline( steps= [("Onehot", onehot),
                                            ("Tree", clf_tree)] )

# Ajustando o modelo
model_pipeline.fit(df_celula[features], df_celula[target])

# %%

model = pd.Series( {"model": model_pipeline,
                   "features": features} )

model.to_pickle("../models/celulas_tree_pipeline.pkl")