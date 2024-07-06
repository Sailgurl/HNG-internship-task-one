import pandas as pd
import numpy as np
from scipy.spatial import distance
import folium



# Read the CSV file
df = pd.read_csv("outliers_detection.csv")
print (df)

# Extract latitude and longitude columns
lats = df['Latitude']
lons = df['Longitude']

# Create a Folium map
m = folium.Map(location=[5.7600, 6.4774], zoom_start=10)

# Add markers to the map
for lat, lon in zip(lats, lons):
    folium.CircleMarker([lat, lon], radius=5).add_to(m)

# Save the map to an HTML file
m.save('outlier_map.html')