#https://blog.prokulski.science/index.php/2020/03/24/muzyka-i-data-science/
#https://spotipy.readthedocs.io/en/2.9.0/#

import spotipy
from spotipy import util
import pandas as pd
import yaml


def get_token(credentials_file_path:str, scp:str = None) -> str:
    with open(credentials_file_path) as file:
        credentials= yaml.load(file, Loader=yaml.FullLoader)
        username = credentials['username']
        client_id = credentials['client_id']
        client_secret = credentials['client_secret']
        redirect_uri = credentials['redirect_uri']
        scope = credentials['scope'] if scp is None else scp
    return util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)


if __name__ == '__main__':

    token = get_token('credentials.yaml')
    sp = spotipy.Spotify(auth=token)
