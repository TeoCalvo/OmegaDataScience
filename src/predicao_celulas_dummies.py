# %%
import pandas as pd

model = pd.read_pickle("../models/celulas_tree_dummies.pkl")

model

## %%

# Pegando info do usuário
nucleos = int(input("Entre com a qtde de núcleos da celula: "))
caudas = int(input("Entre com a qtde de caudas da celula: "))
cor = input("Entre com o tipo de cor da celula: ")
membrana = input("Entre com o tipo de membrana da celula: ")

data = pd.DataFrame( {"nucleos":[nucleos],
                     "caudas":[caudas],
                     "cor":[cor],
                     "membrana":[membrana]} )

data

# %%

# Tentando prever:

# model["model"].predict( data )

# %%

df_new = pd.get_dummies( data[["cor", "membrana"]] )

df_full = pd.concat( [data, df_new], axis=1 )

for f in model['features']:
    if f not in df_full.columns:
        df_full[f] = 0

df_full = df_full[model['features']]

df_full

## %%

pred = model["model"].predict( df_full )[0]

print(f"A célula e do tipo: {pred}")