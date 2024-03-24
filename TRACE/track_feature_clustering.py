import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import RandomForestClassifier
import umap.umap_ as umap
import plotly.express as px
from collections import Counter


class TrackFeatureClustering:
    def __init__(self, file_path_old_tracks):
        self.df = pd.read_csv(file_path_old_tracks, sep='\t')  # track_metadata_and_features.txt
        self.embedding = None
        self.columns_to_subset = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness',
                                  'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

    def cleanup_track_data(self):
        """
        Clean up track_metadata_and_features.txt by keeping only release date and first contributing artist(s),
        and removing tracks with tempo greater than 150.
        """
        self.df['release_date'] = self.df['release_date'].astype(str).str[:4]  # Extract release year
        self.df['artists'] = self.df['artists'].astype(str).str.split(", ").str[0]  # Extract first artist
        self.df = self.df[self.df['tempo'] <= 150]  # Remove tracks with tempo > 150

    def cluster_features(self):
        """
        Cluster features using UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction and
        HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) for clustering.
        HDBSCAN can effectively distinguish between clusters and noise points, which is of great help to increase
        cluster confidence.
        """
        features = self.df.loc[:, self.columns_to_subset]

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)  # Standardize features

        reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
        self.embedding = reducer.fit_transform(scaled_features)  # Reduce dimensionality

        clusters = HDBSCAN(min_samples=10).fit_predict(self.embedding)  # Perform clustering
        self.df['clusters'] = clusters.astype(str)  # Assign cluster labels
        self.df.to_csv('../data/track_metadata_and_features_clustered.txt', sep='\t', header=True, index=False)

    def visualize_umap(self, columns_to_visualize):
        """
        Visualize UMAP embeddings colored by specified columns.

        :param columns_to_visualize: List of columns to visualize.
        """
        for column in columns_to_visualize:
            fig = px.scatter(
                self.embedding, x=0, y=1,
                color=self.df[column], labels={'color': column}
            )

            # Show & save the figure
            fig.show()
            fig.write_image(f"../output/tfc_UMAP_{column}.pdf")

    def train_classifier(self, df_new_tracks):
        """
        Train a random forest classifier to predict clusters for new tracks.

        :param df_new_tracks: DataFrame containing new track features.
        """
        # Train classifier on old track data
        rfc = RandomForestClassifier()
        features_old = self.df.loc[:, self.columns_to_subset]
        clusters_old = self.df['clusters']
        rfc.fit(features_old.values, clusters_old)

        # Predict clusters for new tracks
        df_new_tracks.drop_duplicates(subset=['uri'], keep=False, inplace=True)  # Remove duplicate URIs
        new_features = df_new_tracks.loc[:, self.columns_to_subset]
        new_scaled_features = StandardScaler().fit_transform(new_features)
        predicted_labels = rfc.predict(new_scaled_features)
        df_new_tracks['clusters'] = predicted_labels

        # Save predicted clusters
        df_new_tracks.to_csv('../data/new_track_features_clustered.txt', sep='\t', header=True, index=False)

    def get_latest_liked_tracks_and_clusters(self, spotify_data, df_new_track_clusters):
        """
        Get the latest liked tracks, based on the Spotify data, and get respective clusters.

        :param spotify_data: Instance of SpotifyData containing music_dict.
        :param df_new_track_clusters: DataFrame containing new track features and predicted clusters.
        :return: Dictionary containing the matching new tracks for each cluster.
        """
        # Retrieve the URIs of the latest 150 tracks from the Spotify data
        uris = [value['uri'] for key, value in list(spotify_data.music_dict.items())[-150:]]
        df = self.df

        # Create a mask to filter the dataframe based on the URIs obtained earlier
        mask = df['uri'].isin(uris)
        filtered_df = df[mask][-30:]  # Select only the last 30 rows

        # Count the occurrences of each cluster label in the filtered dataframe
        counts = Counter(list(filtered_df['clusters']))
        filtered_counts = {k: v for k, v in counts.items() if k != '-1'}  # Remove the noise cluster (-1)
        sorted_values = sorted(filtered_counts, key=filtered_counts.get, reverse=True)  # Sort based on count

        # Extract unique sorted cluster labels (removing duplicates)
        unique_sorted_clusters = list(dict.fromkeys(sorted_values))

        # Filter new track dataset based on my latest liked clusters
        newest_matching_tracks = {}
        for cluster in unique_sorted_clusters:
            uris = df_new_track_clusters.loc[df_new_track_clusters['clusters'] == int(cluster), 'uri'].tolist()
            newest_matching_tracks[cluster] = uris

        return newest_matching_tracks
