import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_excel('data_tigaraksa.xlsx')

def get_index(element):
    return data.columns.get_loc(element)

# Extract features
feature = data.iloc[:, [get_index("Berat"), get_index("Tinggi"), get_index("Usia"), get_index("JK"), get_index("Status Gizi")]].copy()

# Encode categorical variables
feature["JK"] = LabelEncoder().fit_transform(feature["JK"].astype("category"))
feature["Status Gizi"] = LabelEncoder().fit_transform(feature["Status Gizi"].astype("category"))

# Normalize data
label = feature["Status Gizi"]
feature_1 = feature.drop(columns=['Status Gizi'])
scaler = MinMaxScaler()
feature_norm = pd.DataFrame(scaler.fit_transform(feature_1), columns=feature_1.columns)
feature_norm["Status Gizi"] = label

# Handle imbalanced data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(feature_norm.drop(columns=['Status Gizi']), feature_norm['Status Gizi'])

# Split the dataset into training and testing
X = feature[['Berat', 'Tinggi', 'Usia','JK']] #atribut
y = feature['Status Gizi'] #target
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)

# print(df_train.head())

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Save the best models
models = {
    'RandomForest': rfc,
    'DecisionTree': tree
}

with open('models.pkl', 'wb') as file:
    pickle.dump(models, file)

print("Models have been trained and saved successfully.")
