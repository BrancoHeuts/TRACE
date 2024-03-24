import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


class VisualizeTrackData:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, sep='\t')  # track_metadata_and_features.txt

    def create_track_features_histogram(self):
        """
        Creates a histogram to visualize the distribution of track features (after scaling).
        The histogram is facetted by feature, with each facet representing a different track feature.

        The resulting histogram is saved as a PDF file.
        """
        # Subset the DataFrame based on features
        features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
        df_subset = self.df.loc[:, features]

        # Check for NaN values
        print("NaN values:\n", df_subset.isna().sum())

        # Scale the data (between 0 and 1)
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_subset), columns=df_subset.columns)

        # Melt the DataFrame to have a 'variable' column for faceting
        df_melted = df_scaled.melt(var_name='variable', value_name='value')

        # Create a histogram with scaled count on y-axis
        fig = px.histogram(df_melted, x='value', facet_col='variable', facet_col_wrap=4,
                           labels={'value': 'Scaled Value'}, title='', facet_col_spacing=0.05)

        # Update layout and annotations
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True))
        fig.update_xaxes(title='')
        fig.update_yaxes(title='', matches=None)
        fig.update_layout(
            autosize=True,
            width=1000,
            height=700,
            margin=dict(l=20, r=20, t=70, b=70),
            showlegend=False
        )
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=0.5,
            yref='paper',
            y=-0.07,
            text='Scaled feature scores',
            font=dict(
                size=14
            ),
        )
        fig.add_annotation(
            showarrow=False,
            xanchor='center',
            xref='paper',
            x=-0.045,
            yanchor='middle',
            yref='paper',
            y=0.5,
            textangle=-90,
            text='Track count',
            font=dict(
                size=14
            ),
        )

        # Customize the color scale
        fig.update_traces(marker_color='#3498db')

        # Show & save the figure
        fig.show()
        fig.write_image("../output/vtd_Track_Feature_histogram.pdf")

    def create_track_metadata_histograms(self, metadata_to_visualize):
        """
        Creates histograms to visualize the distribution of specified track metadata.

        :param metadata_to_visualize: List of metadata columns to visualize.
        """
        # Subset DataFrame with specified metadata column(s)
        df = self.df.loc[:, metadata_to_visualize]

        # Keep only release year
        if 'release_date' in df and not df['release_date'].empty:
            df['release_date'] = df['release_date'].astype(str).str[:4]

        # Keep only first contributing artist(s)
        if 'artists' in df and not df['artists'].empty:
            df['artists'] = df['artists'].astype(str).str.split(", ").str[0]

        # Iterate over columns to visualize
        for column in metadata_to_visualize:
            counts = df[column].value_counts().reset_index().rename(columns={'index': column, column: 'count'})
            print(f"Total count {column}: {len(counts)}")
            counts = counts.head(30)  # Show only top counts
            counts.columns = [column, 'count']

            # Create barchart
            fig = px.bar(counts, x=column, y='count', title=f'Histogram of {column}',
                         labels={column: column, 'count': 'Count'})

            # Show & save the figure
            fig.show()
            fig.write_image(f"../output/vtd_Track_metadata_{column}_histogram.pdf")

    def create_track_metadata_histogram_per_year(self, metadata, year):
        """
        Creates a histogram to visualize the distribution of specified track metadata for a given year.

        :param metadata: The metadata column to visualize.
        :param year: The year for which the histogram is created.
        """
        # Filter DataFrame for specified year
        df = self.df
        df['release_date'] = df['release_date'].astype(str).str[:4]
        df = df[df['release_date'] == year]

        # Count values for specified ID3 tag and limit to top 30
        counts = df[metadata].value_counts().reset_index().rename(columns={'index': metadata, metadata: metadata})
        print(f"Total count {metadata}: {len(counts)}")
        counts = counts.head(30)  # Top counts

        # Create bar plot
        fig = px.bar(counts, x=metadata, y='count', title=f'Histogram of {metadata} - {year}',
                     labels={metadata: metadata, 'count': 'Count'})

        # Show & save the figure
        fig.show()
        fig.write_image(f"../output/vtd_Track_Metadata_{metadata}_in_{year}_histogram.pdf")
