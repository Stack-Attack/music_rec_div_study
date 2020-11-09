from flask import Flask, render_template, request, session, jsonify
from flask_executor import Executor
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
import gdown

load_dotenv()

N = 20

client = MongoClient(os.getenv("MONGO_CLIENT"))
db = client.LFM_TRACKS

network = pylast.LastFMNetwork(api_key=os.getenv("LFM_KEY"), api_secret=os.getenv("LFM_SECRET"))

app = Flask(__name__)
executor = Executor(app)
app.config['EXECUTOR_MAX_WORKERS'] = None
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET")

if int(os.getenv("PRODUCTION")):
    gdown.download(os.getenv("ALS_MODEL_REMOTE"), 'als_model.pkl')
    # gdown.download(os.getenv("VAE_MODEL_REMOTE"), 'vae_model.pt')

ALS_MODEL_DIR = Path(os.getenv("ALS_MODEL_DIR"))
VAE_MODEL_DIR = Path(os.getenv("VAE_MODEL_DIR"))

VAE_MODEL_CONF = {
    "enc_dims": [200],
    "dropout": 0.5,
    "anneal_cap": 1,
    "total_anneal_steps": 10000,
    "learning_rate": 1e-3
}


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


# vae_model = load_vae_model(VAE_MODEL_DIR, VAE_MODEL_CONF)


@app.route('/')
def landing():
    return render_template('verification.html')


def generate_recs(username):
    LEs = get_user_tracks(username, db, network, verbose=True, refresh=False)
    LEs, _, _ = encode_user_tracks(LEs, db, verbose=True)
    als_recs = process_rec_list(get_als_recs(LEs, als_model, n=N), db, verbose=True)
    vae_recs = process_rec_list(get_als_recs(LEs, als_model, n=N), db, verbose=True)
    rec_lists = [als_recs, vae_recs]

    return rec_lists


@app.route('/request_recs', methods=['POST'])
def request_recs():
    # Include authorization here
    username = request.form['username']
    executor.submit_stored(username, generate_recs, username)
    session['username'] = username

    return render_template('loading.html')


@app.route('/check_recs')
def check_recs():
    if not executor.futures.done(session['username']):
        return jsonify({'status': executor.futures._state(session['username'])})

    return jsonify({'status': 'finished'})


@app.route('/rec_lists')
def show_rec_lists():
    future = executor.futures.pop(session['username'])

    return render_template('study.html', rec_lists=future.result(), username=session['username'])


if __name__ == '__main__':
    app.run()
