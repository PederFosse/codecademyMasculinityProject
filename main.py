import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

survey = pd.read_csv('masculinity.csv')
pd.options.display.max_columns = None
pd.options.display.max_rows = None

cols_to_map2 = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
                "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",
                "q0007_0010", "q0007_0011"]

cols_to_map = ['q0007_0001', 'q0007_0002', 'q0007_0003', 'q0007_0004', 'q0007_0005', 'q0007_0008', 'q0007_0009']

for col in cols_to_map:
    survey[col] = survey[col].map({"Never, and not open to it": 0, "Never, but open to it": 1, "Rarely": 2,
                                   "Sometimes": 3, "Often": 4})

# create a model
model = KMeans(n_clusters=2)

rows_to_cluster = survey.dropna(subset=cols_to_map)
model.fit(
    rows_to_cluster[cols_to_map])

cluster_zero_indices = []
cluster_one_indices = []

for index, label in enumerate(model.labels_):
    if label == 1:
        cluster_one_indices.append(index)
    else:
        cluster_zero_indices.append(index)

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

print(cluster_zero_df['educ4'].value_counts() / len(cluster_zero_df))
print("\n")
print(cluster_one_df['educ4'].value_counts() / len(cluster_zero_df))

# seems like points get clustered / separated by education..
