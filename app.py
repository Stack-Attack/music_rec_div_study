from flask import Flask, render_template, send_from_directory, request
import pickle
from MultVAE import MultVAE, CSRDataset
from rec_tools import get_user_tracks, encode_user_tracks, get_als_recs, get_vae_recs, get_vector_div, process_rec_list
import json
from pymongo import MongoClient
import pylast
import torch
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

N = 1000

ALS_MODEL_DIR = Path(os.getenv("ALS_MODEL_DIR"))
VAE_MODEL_DIR = Path(os.getenv("VAE_MODEL_DIR"))

VAE_MODEL_CONF = {
    "enc_dims": [200],
    "dropout": 0.5,
    "anneal_cap": 1,
    "total_anneal_steps": 10000,
    "learning_rate": 1e-3
}

client = MongoClient(os.getenv("MONGO_CLIENT"))
db = client.LFM_TRACKS

network = pylast.LastFMNetwork(api_key=os.getenv("LFM_KEY"), api_secret=os.getenv("LFM_SECRET"))


def load_als_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_vae_model(path, conf):
    checkpoint = torch.load(path)
    vae_model = MultVAE(conf, CSRDataset(2817819), torch.device('cpu'))
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    vae_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    vae_model.eval()

    return vae_model


als_model = load_als_model(ALS_MODEL_DIR)
#vae_model = load_vae_model(VAE_MODEL_DIR, VAE_MODEL_CONF)

app = Flask(__name__)


@app.route('/')
def landing():
    return render_template('verification.html')


@app.route('/generate_recs', methods=['POST'])
def generate_recs():
    username = request.form['username']

    LEs = get_user_tracks(username, db, network, verbose=True, refresh=False)
    LEs, _, _ = encode_user_tracks(LEs, db, verbose=True)

    als_recs = process_rec_list(get_als_recs(LEs, als_model, n=N), db, verbose=True)
    vae_recs = process_rec_list(get_als_recs(LEs, als_model, n=N), db, verbose=True)

    rec_lists = [als_recs, vae_recs]

    return render_template('study.html', rec_lists=rec_lists, username=username)


if __name__ == '__main__':
    app.config.update(
        TEMPLATES_AUTO_RELOAD=True
    )

    app.run()
