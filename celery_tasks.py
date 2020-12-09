from celery import Task, Celery
from pymongo import MongoClient
import os
from pathlib import Path
import pickle
from MultVAE import MultVAE, CSRDataset
import torch
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pylast


class RecommendationTask(Task):
    def __init__(self):
        def load_als_model(path):
            with open(path, "rb") as file:
                return pickle.load(file)

        def load_vae_model(path, conf):
            checkpoint = torch.load(path)
            vae_model = MultVAE(conf, CSRDataset(2817819), torch.device("cpu"))
            vae_model.load_state_dict(checkpoint["model_state_dict"])
            vae_model.eval()

            return vae_model

        def load_song_encodings(path):
            with open(path, "rb") as file:
                return pickle.load(file)

        vae_model_conf = {
            "enc_dims": [200],
            "dropout": 0.5,
            "anneal_cap": 1,
            "total_anneal_steps": 10000,
            "learning_rate": 1e-3,
        }

        self.als_model = load_als_model(Path(os.getenv("ALS_MODEL_DIR")))
        self.song_encodings = load_song_encodings(Path(os.getenv("SONG_ENCODINGS_DIR")))
        self.vae_model = load_vae_model(Path(os.getenv("VAE_MODEL_DIR")), vae_model_conf)
        self.mongo = MongoClient(os.getenv("LOCAL_MONGO_CLIENT"))
        self.spotify = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=os.getenv("SPOTIFY_KEY"), client_secret=os.getenv("SPOTIFY_SECRET")
            ),
            requests_timeout=20,
        )
        self.network = pylast.LastFMNetwork(
            api_key=os.getenv("LFM_KEY"), api_secret=os.getenv("LFM_SECRET")
        )

        print("Done loading models into celery worker.")


# def make_celery(app):
#     celery = Celery(
#         app.import_name,
#         backend=app.config['CELERY_BACKEND'],
#         broker=app.config['CELERY_BROKER']
#     )
#     celery.conf.update(app.config)
#
#     class ContextTask(celery.Task):
#         def __call__(self, *args, **kwargs):
#             with app.app_context():
#                 return self.run(*args, **kwargs)
#
#     celery.Task = ContextTask
#     return celery
