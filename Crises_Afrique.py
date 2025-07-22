import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Affichage de mon jeu de donnee
African_crises = pd.read_csv('African_crises_dataset.csv')
African_crises

# Affichage des informations de base de mon dataset
basic_info = {
    "shape": African_crises.shape,
    "columns": African_crises.dtypes.to_dict(),
    "missing_values": African_crises.isnull().sum(),
    "duplicated_rows": African_crises.duplicated().sum(),
    "descriptive_stats": African_crises.describe()
}

basic_info

# Rapport de profilage
# Exploration des données avec ydata-Profiling
!pip install ydata-profiling
from ydata_profiling import ProfileReport

# Generer le rapport de profilage
profile = ProfileReport(African_crises, title="Rapport de Profiling ydata")

profile.to_notebook_iframe()

# Gestion des valeurs aberrantes de mon dataset
def detect_outliers_iqr_all_columns(data):
    outliers_count = {}  # Dictionnaire pour stocker le nombre d'outliers par colonne
    numeric_cols = data.select_dtypes(include='number').columns  # Colonnes numériques uniquement
    
    for column in numeric_cols:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers_count[column] = outliers.shape[0]  # Nombre de valeurs aberrantes pour cette colonne

    return outliers_count

# Utilisation :
outliers_par_colonne = detect_outliers_iqr_all_columns(African_crises)

# Affichage :
for col, nb in outliers_par_colonne.items():
    print(f"{col} : {nb} valeurs aberrantes détectées")

# Encodage des variables categorielles de mon dataset
# One-hot encoding de la colonne 'country'
df_encoded = pd.get_dummies(African_crises, columns=['country'], drop_first=True)

# Encodage binaire de 'banking_crisis'
df_encoded['banking_crisis'] = df_encoded['banking_crisis'].map({'no_crisis': 0, 'crisis': 1})

# Liste des colonnes à convertir en numérique
cols_to_convert = [
    'currency_crises',
    'inflation_crises',
    'systemic_crisis',
    'country_code',
    'domestic_debt_in_default',
    'sovereign_external_debt_default',
    'independence'
]

# Conversion
for col in cols_to_convert:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
# Affichage du nouveau dataset encodé 
 print(df_encoded.head())

# Variable cible et caracteristiques
X = df_encoded.drop(columns=['systemic_crisis'])
y = df_encoded['systemic_crisis']

# Separation du dataset en ensemble d'entrainement et de test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrainement du modele avec Random Forest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation des performances du modele
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualisation des donnees d'entrainement et de test
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Réel', marker='o')
plt.plot(y_pred[:100], label='Prédit', marker='x')
plt.title('Comparaison entre valeurs réelles et prédites')
plt.xlabel('Échantillons')
plt.ylabel('Consommation d\'énergie')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
