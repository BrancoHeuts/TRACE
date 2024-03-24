from spotify_data import SpotifyData
from track_metadata_merger import TrackMetadataMerger
import pandas as pd
from visualize_track_data import VisualizeTrackData
from track_feature_clustering import TrackFeatureClustering

LOCAL_DIR = "D:/Music/01_Library/Current_collection/BRNC/BRNC_collection"
SPOTIFY_DATA = "https://open.spotify.com/playlist/1DinCFd7XnmeGsx2HB2Rku?si=b6d9d23a22fc4a37"

# Load Spotify playlist data
sd = SpotifyData(SPOTIFY_DATA)

# Extract track metadata and audio features
tmm = TrackMetadataMerger(local_dir=LOCAL_DIR, spotify_data=sd)
tmm.extract_track_metadata()  # Creates track_metadata.txt
df = pd.read_csv('../data/track_metadata.txt', sep='\t')
tmm.extract_track_features(df=df)
tmm.concat_track_metadata()  # Creates track_metadata_and_features.txt

# Visualize track metadata and audio features
vtd = VisualizeTrackData('../data/track_metadata_and_features.txt')
vtd.create_track_features_histogram()
vtd.create_track_metadata_histograms(metadata_to_visualize=['artists', 'release_date', 'genre', 'record_label'])
vtd.create_track_metadata_histogram_per_year(metadata='record_label', year='2022')

# Track clustering
tfc = TrackFeatureClustering("../data/track_metadata_and_features.txt")
tfc.cleanup_track_data()
tfc.cluster_features()
columns_to_visualize = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
                        'genre', 'record_label', 'artists', 'release_date', 'clusters']
tfc.visualize_umap(columns_to_visualize=columns_to_visualize)

# Find features for new tracks
new_data = pd.read_csv("../data/1001tracklist_tracks.csv", sep='\t')
tmm.extract_track_features(df=new_data)
tmm.concat_track_metadata(only_features=True)

# Predict clusters
new_data = pd.read_csv("../data/new_track_features.txt", sep='\t')
tfc.train_classifier(df_new_tracks=new_data)

# Extract the new (predicted) tracks from the latest liked clusters
df_new_track_clusters = pd.read_csv("../data/new_track_features_clustered.txt", sep='\t')
newest_matching_tracks = tfc.get_latest_liked_tracks_and_clusters(spotify_data=sd,
                                                                  df_new_track_clusters=df_new_track_clusters)

# Create a Spotify playlists (limit to 50 tracks per playlist)
for cluster, uri_list in newest_matching_tracks.items():
    print(f"Cluster: {cluster}, count: {len(uri_list)}")
    if uri_list:
        # The predicted ones
        playlist = sd.user_playlist_create(user='brancoheuts', name=f'predicted playlist {cluster}', public=False)
        sd.playlist_add_items(playlist_id=playlist['id'], items=newest_matching_tracks[cluster][:50])

        # The old tracks in that cluster (for testing purposes)
        df_old_track_clusters = pd.read_csv("../data/track_metadata_and_features_clustered.txt", sep='\t')
        cluster_uri_values = df_old_track_clusters.loc[df_old_track_clusters['clusters'] == int(cluster), 'uri'].tolist()
        playlist = sd.user_playlist_create(user='brancoheuts', name=f'cluster {cluster} playlist', public=False)
        sd.playlist_add_items(playlist_id=playlist['id'], items=cluster_uri_values[:50])
