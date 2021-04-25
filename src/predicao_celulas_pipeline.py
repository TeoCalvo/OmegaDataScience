# %%
import pandas as pd

model = pd.read_pickle("../models/celulas_tree_pipeline.pkl")

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

# %%
pred = model["model"].predict(data[model["features"]])[0]

print(f"A célula e do tipo: {pred}")