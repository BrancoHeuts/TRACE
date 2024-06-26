# TRACE: TRack Analysis for Customized Exploration


## Description
TRACE (TRack Analysis for Customized Exploration) is a project aimed at leveraging audio track features from Spotify to gain insights into my listening profile and predict new matching tracks to create personalized playlists. It uses a dimensionality reduction technique (UMAP: Uniform Manifold Approximation and Projection) and clustering algorithm (HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise) to analyze audio features and metadata extracted from my personal music libraries (local MP3 files and Spotify playlist), and predicts new matching tracks based on a Random Forest Classifier.


## Background
Single-cell RNA-seq data contains information about each cell and its thousands of genes. Direct visualization of this data is challenging due to its high-dimensional nature. However, by using dimensionality reduction techniques such as UMAP (Uniform Manifold Approximation and Projection), the data can be condensed into a more manageable 2D or 3D space, which allows visualization.

This concept sparked my interest when I discovered Spotify's audio track features. Similar to genes providing a glimpse into a cell’s state and biological function, these audio features offer insights into a song's musical characteristics. I wondered whether I could leverage these features to:

1. Gain insights into my own listening profile.
2. Cluster tracks based on their similarity.
3. Predict whether new tracks match my listening profile (clusters), essentially creating my own take on Spotify's "Discover Weekly" playlists.


## Scripts
Here, I present a project which I can structure into five parts:
1. spotify_data.py
   - Uses a lightweight Python library for the Spotify Web API, allowing access to my personal Spotify data. 
2. track_metadata_merger.py 
   - Extracts ID3 tags (metadata) from MP3 files in my personal (local) music library, extracts the audio features using Spotify’s API, and merges this data into one file.

3. visualize_track_data.py 
   - Creates histograms using feature data and metadata to examine my own listening profile.

4. track_feature_clustering.py 
   - Cluster tracks based on features using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise), visualizes features or metadata using UMAPs, trains a random forest classifier to predict to which clusters new tracks will belong, and determines which clusters I am interested in most recently.
 
5. main.py
   - Executes functionality of previous 4 scripts, and adds new tracks that are predicted to match my recent listening profile to a new personal ‘Weekly Discover’ Spotify playlist.

## Download and Installation

To download and install the TRACE package from GitHub, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/brancoheuts/TRACE.git
    ```


2. Navigate to the project directory:

    ```bash
    cd TRACE
    ```

3. Install the package using pip:

    ```bash
    pip install .
    ```

   This will install all the necessary dependencies specified in the `setup.py` file.


4. Navigate to the package directory:

    ```bash
    cd TRACE
    ```


## Spotify API Initialization

Before running the script, initialize the Spotify API by providing your credentials in an .env file:
1. Create an ".env" file
2. Add your own credentials (CLIENT_ID & CLIENT_SECRET):

    ```plaintext
    # environment variables
    # Spotify API credentials
    REDIRECT_URI="http://example.com"
    CLIENT_ID=your_client_id
    CLIENT_SECRET=your_client_secret
    ```
&nbsp;
#### Here is how:

1. Go to your [Spotify for Developers Dashboard](https://developer.spotify.com/dashboard/).
2. Log in to your Spotify account.
3. Create an App.
4. Go to 'Edit Settings'.
5. Add 'http://example.com' to 'Redirect URIs'.
   - Click 'Add'.
6. Copy and paste 'Client ID' and 'Client Secret' to .env.
7. Run `main.py`.
   - If Spotify authentication was successful, you should see a Spotify page asking you to agree to their terms.
8. Click 'Agree'.
9. Copy the entire URL and paste it into the prompt of your IDE.
10. Close and restart your IDE.


## Usage

#### Run main.py
  
 ```bash
 python main.py
 ```

## Audio Features

**Danceability**
- Describes the suitability of a track for dancing, considering factors like tempo, rhythm, beat strength, and regularity.

**Energy**
- Measures the perceptual intensity and activity level of a track, encompassing elements such as dynamic range, perceived loudness, and timbre.

**Key**
- Provides an estimate of the overall key of the track, represented using standard Pitch Class notation.

**Loudness**
- Represents the overall volume or loudness of a track in decibels (dB), averaged across the entire duration.

**Mode (not used for clustering and prediction)**
- Indicates whether the track's melodic content is derived from a major or minor scale.

**Speechiness**
- Detects the presence of spoken words in a track, ranging from purely speech-like recordings to those with a mix of music and speech.

**Acousticness**
- Indicates the likelihood of a track being acoustic, with 1.0 representing high confidence in acoustic instrumentation.

**Instrumentalness**
- Predicts the presence of vocals in a track, with values closer to 1.0 indicating a higher likelihood of instrumental content.

**Liveness**
- Detects the probability of a live audience presence in the recording, with higher values suggesting a live performance.

**Valence**
- Measures the musical positivity conveyed by a track, with higher values indicating a more positive or upbeat mood.

**Tempo**
- Provides an estimated tempo of the track in beats per minute (BPM), reflecting its speed or pace.

**Duration**
- Specifies the duration of the song in milliseconds.


