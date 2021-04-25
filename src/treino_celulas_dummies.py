# %% 

import pandas as pd
from sklearn import pipeline
from sklearn import tree

# %%

df_celula = pd.read_csv("../data/celulas.csv",
                        sep=";")

df_celula

# %%

# Como criar variáveis dummies?

# Cria variáveis dummies
df_dummys = pd.get_dummies(df_celula[["cor", "membrana"]])

# Cruza dataset anterior com as dummies
df_tentacao = pd.concat( [df_celula, df_dummys], axis=1 )

# Remove as variáveis originais
del df_tentacao['cor']
del df_tentacao["membrana"]

# Novo dataset para modelagem
df_tentacao

# %%

features = ["nucleos",
            "caudas",
             "cor_Clara",
             "cor_Escura",
             "membrana_Fina",
             "membrana_Grossa"]

target = "classe"

clf_tree = tree.DecisionTreeClassifier()

clf_tree.fit( df_tentacao[features], df_tentacao[target] )

# %%

# Salvando o algoritmo

model = pd.Series(
        {"model":clf_tree,
         "features": features,
         "target": target} )

model.to_pickle("../models/celulas_tree_dummies.pkl")

model
