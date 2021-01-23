import pandas as pd
from kmodes import kmodes

#Carregar e transformar os dados em dummies
df = pd.read_csv("C:/Users/milen/Desktop/Case_-_Cred.csv", sep=';', decimal=',')
df.drop(columns=["Atualizado em", "StoneCode", "Descredenciado"], inplace=True)
df_dummy = pd.get_dummies(df)

x = df_dummy.reset_index().values

km = kmodes.KModes(n_clusters=2, init='Huang', n_init=5, verbose=0)
clusters = km.fit_predict(x)
df_dummy['clusters'] = clusters

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(2)

# Transformar a dummy em duas colunas no PCA
plot_columns = pca.fit_transform(df_dummy)

# Plotar os grupos 
plt.scatter(x=plot_columns[:,1], y=plot_columns[:,0], c=df_dummy["clusters"], s=30)
plt.show()

# Verificar as modas
km.cluster_centroids_
