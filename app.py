import streamlit as st
import pandas as pd
import folium
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load our data

@st.cache(persist = True, suppress_st_warning = True)
def load_data():
    dallas_df = pd.read_csv('./Data/dallas_combined_df.csv')
    houston_df = pd.read_csv('./Data/houston_combined_df.csv')
    dallas_top_10_venues = pd.read_csv('./Data/dallas_top_10_venues.csv')
    houston_top_10_venues = pd.read_csv('./Data/houston_top_10_venues.csv')
    return dallas_df, houston_df, dallas_top_10_venues, houston_top_10_venues

def find_similar_neighorhoods(zip_code, dallas_df, houston_df, dallas_top_10, houston_top_10):
    # Check to see what Metroplex the user inputted zip code is located
    if zip_code in houston_df['Zip_Code'].values:
        # Pull the zip codes' features
        zip_code_data = houston_df[houston_df['Zip_Code'] == zip_code]
        # Append to other Metroplex dataframe
        combined_df = dallas_df.append(zip_code_data, sort = False).reset_index(drop = True)
        combined_df = combined_df.replace(np.nan, 0)
        training_df = combined_df.drop(columns = ['Neighborhood', 'Zip_Code','lat', 'lng'])
        top_10_venues = dallas_top_10
    elif zip_code in dallas_df['Zip_Code'].values:
        # Pull the zip codes' feature
        zip_code_data = dallas_df[dallas_df['Zip_Code'] == zip_code]
        # Append to other Metroplex dataframe
        combined_df = houston_df.append(zip_code_data, sort = False).reset_index(drop = True)
        combined_df = combined_df.replace(np.nan, 0)
        # Drop data not used for training
        training_df = combined_df.drop(columns = ['Neighborhood', 'Zip_Code','lat', 'lng'])
        top_10_venues = houston_top_10
    else:
        return st.markdown('Please Enter A Valid Zip Code')
        
    # Scale data
    scaler = StandardScaler()
    training_data_scaled = scaler.fit_transform(training_df)
    kmeans = KMeans(n_clusters = 20, random_state = 0)
    kmeans.fit(training_data_scaled)
    # Put cluster labels into dataframe
    combined_df.insert(0, 'Cluster Labels', kmeans.labels_)
    # Find cluster where user inputted zip code is located
    neighborhood_cluster = combined_df[combined_df['Cluster Labels'] == combined_df.iloc[-1]['Cluster Labels']]
    neighborhood_cluster = neighborhood_cluster.drop(columns = 'Cluster Labels')
    neighborhood_cluster = neighborhood_cluster.loc[:,:'Doctorate Degree']
    neighborhood_df = pd.concat([neighborhood_cluster.set_index('Neighborhood'), top_10_venues.set_index('Neighborhood')], axis = 1, sort = False).dropna()
    neighborhood_df = neighborhood_df.drop(columns = ['lat', 'lng'])
    education_columns = ['Less Than High School', 'High School Graduate', "Associate's Degree", "Bachelor's Degree", "Master's Degree", 'Professional Degree', 'Doctorate Degree']
    for education in education_columns:
        neighborhood_df[education] = (neighborhood_df[education] * 100).round(2).astype(str) + '%'
    format_dict = {'median_home_value': '${0:,.0f}', 'median_household_income': '${0:,.0f}'}
    neighborhood_df = neighborhood_df.style.format(format_dict)
    return neighborhood_cluster, neighborhood_df


def cluster_map(neighborhood_cluster, token):
    # Drop user zip code from cluster
    cluster_df = neighborhood_cluster.iloc[:-1]
    if len(cluster_df) < 1:
        return st.markdown('No Similiar Neighborhoods')
    # Find latitude for cluster center
    cluster_lat = cluster_df['lat'].sum() / len(cluster_df['lat'])
    # Find Longitude for cluster center
    cluster_lng = cluster_df['lng'].sum() / len(cluster_df['lng'])
    # Create map object
    map_clusters = folium.Map(location=[cluster_lat, cluster_lng], tiles ='https://api.mapbox.com/styles/v1/josh-miller/ckabacn800o2d1ipjlx88fksf/tiles/256/{z}/{x}/{y}@2x?' + f'access_token={token}',
                          zoom_start=9, attr = 'None')
    # Add circle markers for our neighborhood locations
    for lat, lon, poi in zip(neighborhood_cluster['lat'], neighborhood_cluster['lng'], neighborhood_cluster['Neighborhood']):
        tooltip_label = folium.Tooltip(str(poi))
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            color = '#0c325b',
            fill = True,
            fill_opacity=0.9,
            tooltip=tooltip_label).add_to(map_clusters)
    return st.markdown(map_clusters._repr_html_(), unsafe_allow_html=True)





def main():
    dallas_df, houston_df, dallas_top_10, houston_top_10 = load_data()
    token = os.getenv('token')
    st.header('Similiar Areas of DFW and Houston')
    st.image('./Data/Houston-vs-Dallas-direct-flights-to-India.png')
    st.markdown('The current job market has substantially contracted in the last few months due to the economic decline caused by the novel coronavirus(COVID-19).  Many people have lost their jobs or were already looking to change jobs.  With the job prospects looking thinner by the day, job seekers will start considering relocation to expand their opportunities for employment. The transition of uprooting your family and moving to a new area can only add to the stress of changing jobs. The goal of this analysis is to help locate a neighborhood that is similar to the neighborhood that one is currently living in.  This analysis focuses on those transitioning between the Greater Houston and Greater Dallas markets.  The neighborhoods will be clustered using the *k-means* algorithm based on a variety of features of each neighborhood. The features used for clustering will include types of shops located near the neighborhood, population, median household income, education, and median home value.')
    zip_code = st.number_input('Input Your Zip Code', value = 77511)
    neighborhood_cluster, neighborhood_df = find_similar_neighorhoods(zip_code, dallas_df, houston_df, dallas_top_10, houston_top_10)
    cluster_map(neighborhood_cluster, token)
    st.dataframe(neighborhood_df)


if __name__ == "__main__":
    main()