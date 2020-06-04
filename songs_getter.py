# https://blog.prokulski.science/index.php/2020/03/24/muzyka-i-data-science/
# https://spotipy.readthedocs.io/en/2.9.0/#

import spotipy
from spotipy import util
import yaml
from typing import List
import re
import sys
import pandas as pd
from pathlib import Path


def get_token(credentials_file_path: str, scp: str = None) -> str:
    '''
    Get Spotify user token
    :param credentials_file_path: path to yaml file with credentials
    :param scp: Spotify Authorization Scopes, by default scopes from file
    :return: Spotify API token
    '''
    with open(credentials_file_path) as file:
        credentials = yaml.load(file, Loader=yaml.FullLoader)
        username = credentials['username']
        client_id = credentials['client_id']
        client_secret = credentials['client_secret']
        redirect_uri = credentials['redirect_uri']
        scope = credentials['scope'] if scp is None else scp
    return util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)


def get_playlists(playlists: List, keywords: List) -> List:
    '''
    Get playlists whose name contains any of the keywords
    :param playlists: 'items' from paging object with playlists returned by Spotify
    :param keywords: list of words, playlist names should contain at least one of them to be returned
    :return: list if tuples (playlist_id, playlist_name) of valid playlists
    '''
    keywords = ['.*' + word + '.*' for word in keywords]
    valid_playlists = []
    words = '|'.join(keywords)
    pattern = re.compile(words)
    for playlist in playlists:
        if pattern.match(playlist['name'].lower()):
            valid_playlists.append((playlist['id'], playlist['name']))
    return valid_playlists


def get_playlist_songs(playlist_id, sp):
    playlist = sp.playlist(playlist_id)
    tracks = []
    for track in playlist['tracks']['items']:
        if track['track'] is not None:
            track_data = []
            track_data.append(track['track']['id'])
            track_data.append(track['track']['name'])
            track_data.append(track['track']['artists'][0]['name'])
            track_data.append(track['track']['album']['name'])
            tracks.append(track_data)
    return tracks


def get_genre_songs(playlists: List, sp):
    tracks = []
    for playlist in playlists:
        playlist_tracks = get_playlist_songs(playlist[0], sp)
        tracks.extend(playlist_tracks)
    return tracks


def save_tracks(file_name: str, tracks: List):
    path = Path("data/titles/" + file_name + '.csv')
    tracks_df = pd.DataFrame(tracks, columns=['id', 'name', 'artist', 'album'])
    tracks_df.drop_duplicates(subset='id', inplace=True)
    tracks_df.to_csv(path, header=True, index=False)
    print(tracks_df.shape)


def get_user(credentials_file_path: str) -> str:
    '''
    get spotify user id
    :param credentials_file_path: path to yaml file with credentials
    :return: spotify user id
    '''
    with open(credentials_file_path) as file:
        credentials = yaml.load(file, Loader=yaml.FullLoader)
        return credentials['username']


if __name__ == '__main__':
    # input arguments genre, *keywords
    user = get_user('credentials.yaml')
    token = get_token('credentials.yaml')
    sp = spotipy.Spotify(auth=token)
    my_playlists = sp.user_playlists(user)

    valid_playlists = get_playlists(my_playlists['items'], sys.argv[2:])
    tracks = get_genre_songs(valid_playlists, sp)
    save_tracks(sys.argv[1], tracks)
