import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

client_id = 'your_client_id'
client_secret = 'your_client_secret'

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)


def search_playlists(keyword, limit=10):
    """Tìm playlist theo từ khóa"""
    results = sp.search(q=keyword, type='playlist', limit=limit)
    print("Kết quả tìm kiếm:", results)  

    playlists = []
    for item in results.get('playlists', {}).get('items', []):
        if item:
            playlists.append({
                "name": item.get('name', 'Unknown'),
                "id": item.get('id', 'Unknown'),
                "url": item.get('external_urls', {}).get('spotify', 'Unknown')
            })
    return playlists


def get_unique_tracks_from_playlist(playlist_id, genre_label, limit=100):
    """Lấy bài hát không trùng từ 1 playlist"""
    offset = 0
    tracks = []
    batch_size = 50

    while offset < limit * 2:
        results = sp.playlist_items(playlist_id, limit=batch_size, offset=offset)
        items = results.get('items', [])
        if not items:
            break
        
        for item in items:
            track = item.get('track')
            if track:
                track_info = {
                    "track_id": track.get('id'),
                    "name": track.get('name'),
                    "uri": track.get('uri'),
                    "genres": genre_label
                }
                tracks.append(track_info)
        
        offset += batch_size
        time.sleep(0.3)

    df = pd.DataFrame(tracks)
    df_unique = df.drop_duplicates(subset='track_id').head(limit)
    return df_unique

found_playlists = search_playlists("tình yêu", limit=12)

print("\nPlaylists found:")
for pl in found_playlists:
    print(f"- {pl['name']}: {pl['url']}")

all_tracks = []
for pl in found_playlists:
    print(f"\nĐang lấy bài hát từ playlist: {pl['name']}")
    df_tracks = get_unique_tracks_from_playlist(pl['id'], "love", limit=100)
    all_tracks.append(df_tracks)

df_all = pd.concat(all_tracks).drop_duplicates(subset='track_id').reset_index(drop=True)

print(df_all)

df_all.to_csv("luvv_auto_collected.csv", index=False)
