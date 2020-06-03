#https://blog.prokulski.science/index.php/2020/03/24/muzyka-i-data-science/
#https://spotipy.readthedocs.io/en/2.9.0/#

import spotipy
from spotipy import util
import yaml
from typing import List
import re


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


def get_playlists(playlists:List, keywords:List) -> List:
    '''
    Get playlists whose name contains any of the keywords
    :param playlists: 'items' from paging object with playlists returned by Spotify
    :param keywords: list of words, playlist names should contain at least one of them to be returned
    :return: list if tuples (playlist_id, playlist_name) of valid playlists
    '''
    keywords = ['.*'+word+'.*' for word in keywords]
    valid_playlists = []
    words = '|'.join(keywords)
    print(words)
    pattern = re.compile(words)
    for playlist in playlists:
        if pattern.match(playlist['name'].lower()):
            valid_playlists.append((playlist['id'],playlist['name']))
    return valid_playlists


def get_user(credentials_file_path: str) -> str:
    '''
    get spotify user id
    :param credentials_file_path: path to yaml file with credentials
    :return: spotify user id
    '''
    with open(credentials_file_path) as file:
        credentials= yaml.load(file, Loader=yaml.FullLoader)
        return credentials['username']


if __name__ == '__main__':
    user = get_user('credentials.yaml')
    token = get_token('credentials.yaml')
    sp = spotipy.Spotify(auth=token)
    my_playlists = sp.user_playlists(user)
    rap_playlists = get_playlists(my_playlists['items'], ['rap', 'hip'])
    print(rap_playlists)



