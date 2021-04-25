# %%
import pandas as pd

model = pd.read_pickle("../models/celulas_tree_onehot.pkl")

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

df_full = model['onehot'].transform(data)
pred = model["model"].predict( df_full )[0]

print(f"A célula e do tipo: {pred}")