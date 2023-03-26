import pandas as pd

print("There are", 97452, "lines of data")

x = pd.read_csv("data/xTrain.csv", delimiter=',')
y = pd.read_csv("data/yTrain.csv", names=['y'])
df = pd.concat([x, y], axis=1)

print("\nNumber of different cows:", len(x['idCow'].unique()))
print("\nDuration of the study in hours:", len(x['data_hour'].unique())+23)

missing_values = x.groupby('idCow').agg(lambda x: x.isnull().sum())

print("\nNumber of missing values for each cow in hours:")
print(missing_values[['all0', 'rest0', 'eat0']])

# Compter le nombre de lignes où il manque une donnée sur au moins une colonne
nb_lignes_manquantes = df.isna().any(axis=1).sum()
print(f"\nThere are {nb_lignes_manquantes} lines with at least one data missing")

# Compter le nombre de fois où y vaut 1.0 pour chaque vache
count_y_by_cow = df[df['y'] == 1.0].groupby('idCow').size()

print("\nNumber of hours during each cow has been sick:")
print(count_y_by_cow)