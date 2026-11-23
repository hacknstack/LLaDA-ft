"""Renewable Drive API credentials read privately; never print credential values."""
import configparser
import json
from pathlib import Path


def store():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from drive_checkpoint_upload import DriveCheckpointStore
    c=configparser.ConfigParser();c.read('/root/.config/research-rclone/rclone.conf')
    s=c['researchdrive'];token=json.loads(s['token'])
    credentials=Credentials(None,refresh_token=token['refresh_token'],client_id=s['client_id'],client_secret=s['client_secret'],
                    token_uri='https://oauth2.googleapis.com/token',scopes=['https://www.googleapis.com/auth/drive'])
    return DriveCheckpointStore(build('drive','v3',credentials=credentials,cache_discovery=False))
