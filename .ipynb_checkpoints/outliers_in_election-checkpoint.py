import pandas as pd
import numpy as np
from scipy.spatial import distance
import folium
import haversine as hs

def geodesic_distance(lat1, lon1, lat2, lon2):
    return hs.haversine((lat1, lon1), (lat2, lon2)).km

# Load the data
df = pd.read_csv("outliers_detection.csv")
print (df)

# Convert Latitude and Longitude columns to float type
df['Latitude'] = df['Latitude'].apply(lambda x: float(x))
df['Longitude'] = df['Longitude'].apply(lambda x: float(x))

from geopy.distance import geodesic

# Define the proximity radius (1 km)
proximity_radius = 1000  # in meters

# Create a function to group polling units into clusters based on their proximity
def cluster_polling_units(df):
    clusters = {}
    for index, row in df.iterrows():
        lat1, lon1 = row['Latitude'], row['Longitude']
        for other_index, other_row in df.iterrows():
            if index != other_index:
                lat2, lon2 = other_row['Latitude'], other_row['Longitude']
                distance_between_points = geodesic((lat1, lon1), (lat2, lon2)).meters
                if distance_between_points <= proximity_radius:
                    cluster_id = f"Cluster {index}"
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(other_index)
    return clusters

# Group polling units into clusters
clusters = cluster_polling_units(df)

# Calculate outlier scores for each party within each cluster
for cluster_id, unit_indices in clusters.items():
    cluster_df = df.iloc[unit_indices]
    for party in ['Party A', 'Party B', 'Party C']:
        party_votes = cluster_df[party]
        mean_vote = np.mean(party_votes)
        std_deviation = np.std(party_votes)
        outlier_scores = (party_votes - mean_vote) / std_deviation
        for index, score in zip(unit_indices, outlier_scores):
            df.loc[index, f'outlier_score_{party}'] = score

# Record the outlier scores for each party and unit
outlier_df = df[['unit', 'outlier_score_Party A', 'outlier_score_Party B', 
                 'outlier_score_Party C']].copy()

# Sort the outlier scores for each party and record the top 3 outliers
outlier_df.sort_values(by='outlier_score_Party A', ascending=False).head(3).to_csv('party_a_outliers.csv')
outlier_df.sort_values(by='outlier_score_Party B', ascending=False).head(3).to_csv('party_b_outliers.csv')
outlier_df.sort_values(by='outlier_score_Party C', ascending=False).head(3).to_csv('party_c_outliers.csv')

# Highlight the top 3 outliers and their closest units
top_3_outliers_party_a = pd.read_csv('party_a_outliers.csv').head(3)
top_3_outliers_party_b = pd.read_csv('party_b_outliers.csv').head(3)
top_3_outliers_party_c = pd.read_csv('party_c_outliers.csv').head(3)

print("Top 3 Outliers for Party A:")
print(top_3_outliers_party_a)
print("\nTop 3 Outliers for Party B:")
print(top_3_outliers_party_b)
print("\nTop 3 Outliers for Party C:")
print(top_3_outliers_party_c)


# Add geospatial data to the dataset
df['latitude'] = df['latitude'].apply(lambda x: float(x))
df['longitude'] = df['longitude'].apply(lambda x: float(x))

import folium
m = folium.Map(location=[5.7600, 6.4774], zoom_start=10)

for index, row in df.iterrows():
    lat, lon = row['latitude'], row['longitude']
    folium.CircleMarker([lat, lon], radius=5).add_to(m)

m.save('outlier_map.html')

print()