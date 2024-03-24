import pandas as pd
from music_tag import load_file
from os import listdir, path, makedirs
from pathlib import Path
import time
from spotipy import SpotifyException
from glob import glob
from shutil import move


class TrackMetadataMerger:
    def __init__(self, local_dir, spotify_data):
        self.local_dir = local_dir
        self.sd = spotify_data

    def extract_track_metadata(self):
        """
        Extracts ID3 tags (metadata) from tracks that are present in both the local music library and Spotify playlist.

        Dataframe is stored in a tab delimited txt file.
        """
        # Find all MP3 files in the local directory
        mp3_files = [file for file in listdir(self.local_dir) if file.endswith('.mp3')]

        # Initialize DataFrame to store track metadata
        track_metadata_df = pd.DataFrame(columns=[
            'uri',  # Uniform Resource Indicator (=unique ID)
            'artists',
            'track_title',
            'release_date',
            'release_date_precision',
            'genre',
            'record_label',
        ])

        # Iterate over each MP3 file
        for file in mp3_files:
            path_to_file = path.join(self.local_dir, file)
            track_name = Path(path_to_file).stem
            id3_object = load_file(path_to_file)  # music_tag object

            # Extract record label from comment tag: /* {LIBRARY_NAM} / {TRACK_QUALIFIER} / {RECORD_LABEL} */
            comment = str(id3_object['comment'])
            comment_parts = [s.strip("/* ").strip(" */") for s in comment.split(' / ')]
            record_label = comment_parts[2]

            # If track is found in Spotify playlist, extract track metadata and ID3 tags, add to DataFrame
            if track_name in self.sd.music_dict:
                track_metadata = [
                    self.sd.music_dict[track_name]['uri'],
                    self.sd.music_dict[track_name]['artists'],
                    self.sd.music_dict[track_name]['tracktitle'],
                    self.sd.music_dict[track_name]['release_date'],
                    self.sd.music_dict[track_name]['release_date_precision'],
                    str(id3_object['genre']),
                    record_label,
                ]
                track_metadata_df.loc[len(track_metadata_df)] = track_metadata

        # Save track metadata to a tab-delimited file
        track_metadata_df.to_csv('../data/track_metadata.txt', sep='\t', header=True, index=False)

        # Save copy of df before processing batches in extract_track_features()
        track_metadata_df.to_csv('../data/track_metadata_copy.txt', sep='\t', header=True, index=False)

    def extract_track_features(self, df):
        """
        Collects track audio features from Spotify. Spotify's track audio features are a set of quantitative metrics
        that describe various aspects of a musical track's audio characteristics. For example, 'Danceability' is a
        measure of how suitable a track is for dancing, based on factors like rhythm stability, beat strength,
        and overall tempo.

        A rate limitation has been applied, as Spotify doesn't like a high retry rate. patience is required for
        larger datasets.

        Dataframe is stored in a tab delimited txt file.

        :param df: Dataframe from file generated by extract_track_metadata()
        """

        # Extract uris from metadata_df
        uris = [track for track in df['uri']]

        # Spotify API limits
        sleep_time = 300
        batch_size = 100
        batches = [uris[i:i + batch_size] for i in range(0, len(uris), batch_size)]

        # Process batches
        batch_n = 1
        feature_file_path = f'../data/track_features_batch_{batch_n}.txt'
        while path.exists(feature_file_path):  # If batch already exists, continue from there
            batch_n += 1
            feature_file_path = f'../data/track_features_batch_{batch_n}.txt'
        if not path.exists(feature_file_path):
            feature_file_path = f'../data/track_features_batch_{batch_n}.txt'

        for batch in batches:
            print(f"Batch number: {batch_n}/{len(batches)}")
            print(f"Time remaining: {(((len(batches) + 1) - batch_n) * sleep_time) / 60} minutes")
            track_features_df = pd.DataFrame(columns=[
                'uri',
                'danceability',
                'energy',
                'key',
                'loudness',
                'mode',
                'speechiness',
                'acousticness',
                'instrumentalness',
                'liveness',
                'valence',
                'tempo',
                'duration_ms',
            ])
            try:
                audio_features_list = self.sd.audio_features(tracks=batch)  # Get audio feature information

                # Extract features from each track
                for features_dict in audio_features_list:
                    track_features = [
                        features_dict['uri'],
                        features_dict['danceability'],
                        features_dict['energy'],
                        features_dict['key'],
                        features_dict['loudness'],
                        features_dict['mode'],
                        features_dict['speechiness'],
                        features_dict['acousticness'],
                        features_dict['instrumentalness'],
                        features_dict['liveness'],
                        features_dict['valence'],
                        features_dict['tempo'],
                        features_dict['duration_ms'],
                    ]
                    track_features_df.loc[len(track_features_df)] = track_features

                # Save batch to file
                track_features_df.to_csv(feature_file_path, sep='\t', header=True, index=False)
                batch_n += 1
                feature_file_path = f'../data/track_features_batch_{batch_n}.txt'

                # If processing is stopped, save which tracks have not been processed yet.
                metadata_df = df[~df['uri'].isin(batch)]
                metadata_df.to_csv('../data/track_metadata.txt', sep='\t', header=True, index=False)

            except SpotifyException as e:
                if e.http_status == 429:
                    print("TAKE A BREAK FOR 1.5 HOURS, retry value too high")
                    break

            # Rate limit
            time.sleep(sleep_time)

    def concat_track_metadata(self, only_features=None):
        """
        Concatenates track feature dataframes (batches) into one, as well as merging metadata information (if present).

        :param only_features: Enables option to concat only feature data batches (recommended for new tracks)
        """
        # Find all batch files
        try:
            matching_files = glob('../data/track_features_batch_*.txt')
            num_matching_files = len(matching_files)
        except:
            print("Batch files have been moved, they are not in data directory anymore.")

        batch_dfs = []
        # Load each batch file into a DataFrame
        for i in range(1, num_matching_files + 1):
            file_path = f'../data/track_features_batch_{i}.txt'
            df = pd.read_csv(file_path, sep='\t')
            batch_dfs.append(df)

        # Concatenate all DataFrames
        merged_batch_df = pd.concat(batch_dfs, ignore_index=True)
        merged_batch_df = merged_batch_df.drop_duplicates()

        if only_features:
            merged_batch_df.to_csv('../data/new_track_features.txt', sep='\t', header=True, index=False)
        else:
            merged_batch_df.to_csv('../data/track_features.txt', sep='\t', header=True, index=False)

        # Merge metadata and features by default
        if only_features is None:
            df_metadata = pd.read_csv('../data/track_metadata_copy.txt', sep='\t')
            df_features = pd.read_csv('../data/track_features.txt', sep='\t')
            merged_df = pd.merge(df_metadata, df_features, on='uri', how='inner')
            merged_df.to_csv('../data/track_metadata_and_features.txt', sep='\t', header=True, index=False)

        # Create new directory for batch files
        existing_directories = [d for d in listdir('../batches/') if d.startswith('track_feature_batches_')]
        if existing_directories:
            latest_count = max([int(d.split('_')[-1]) for d in existing_directories])
        else:
            latest_count = 0

        new_directory = f'../batches/track_feature_batches_{latest_count + 1}'
        makedirs(new_directory)

        # Move batch files to new directory
        for file_path in matching_files:
            move(file_path, path.join(new_directory, path.basename(file_path)))