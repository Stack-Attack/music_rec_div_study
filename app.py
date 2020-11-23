from flask import Flask, render_template, request, session, jsonify
from flask_executor import Executor
import pickle
from MultVAE import MultVAE, CSRDataset
from rec_tools import get_user_tracks, encode_user_tracks, get_als_recs, get_vae_recs, get_vector_div, process_rec_list, get_div_recs, get_genre_hash, get_filtered_recs
import json
from pymongo import MongoClient
import pylast
import torch
from dotenv import load_dotenv
from pathlib import Path
import os
import gdown
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

N = 1000

client = MongoClient(os.getenv("MONGO_CLIENT"))
db = client.LFM_TRACKS

network = pylast.LastFMNetwork(api_key=os.getenv("LFM_KEY"), api_secret=os.getenv("LFM_SECRET"))

spotify = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_KEY"),
        client_secret=os.getenv("SPOTIFY_SECRET")
    ),
    requests_timeout=20
)

app = Flask(__name__)
executor = Executor(app)
app.config['EXECUTOR_MAX_WORKERS'] = None
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET")

if int(os.getenv("PRODUCTION")):
    gdown.download(os.getenv("ALS_MODEL_REMOTE"), 'als_model.pkl')
    gdown.download(os.getenv("SONG_ENCODINGS_REMOTE"), 'song_encodings.pkl')
    # gdown.download(os.getenv("VAE_MODEL_REMOTE"), 'vae_model.pt')


ALS_MODEL_DIR = Path(os.getenv("ALS_MODEL_DIR"))
VAE_MODEL_DIR = Path(os.getenv("VAE_MODEL_DIR"))
SONG_ENCODINGS_DIR = Path(os.getenv("SONG_ENCODINGS_DIR"))

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
    vae_model.eval()

    return vae_model


def load_song_encodings(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


als_model = load_als_model(ALS_MODEL_DIR)
song_encodings = load_song_encodings(SONG_ENCODINGS_DIR)
#vae_model = load_vae_model(VAE_MODEL_DIR, VAE_MODEL_CONF)

print('Done loading models into ram.')


@app.route('/')
def landing():
    return render_template('verification.html')


def generate_recs(id, username, region, refresh_recs=False, refresh_LEs=False):
    user_data = db.users.find_one({'id': id}, {'recs': 1})

    rec_lists = {}
    if refresh_recs or 'recs' not in user_data:
        LEs = get_user_tracks(username, db, network, verbose=True, refresh=refresh_LEs)
        LEs, _, _ = encode_user_tracks(LEs, song_encodings, verbose=True)
        genre_dict = get_genre_hash(LEs, id, db, verbose=True)

        als_rec_idx, als_rec_rank = get_als_recs(LEs, als_model, n=N)
        rec_lists['als'] = process_rec_list(als_rec_idx, als_rec_rank, db, spotify, region, verbose=True)

        als_max_div_idx, als_max_div_rank, als_max_div_spot = get_div_recs(als_rec_idx, als_rec_rank, als_model.item_factors, spotify, db, region, n=10)
        rec_lists['als_max_div'] = process_rec_list(als_max_div_idx, als_max_div_rank, db, spotify, region, spotify_ids=als_max_div_spot, verbose=True)

        als_filt_idx, als_filt_rank, als_filter = get_filtered_recs(als_rec_idx, als_rec_rank, genre_dict, db, als_model.item_factors, spotify, region)
        als_filt_div_idx, als_filt_div_rank, als_filt_div_spot = get_div_recs(als_filt_idx, als_filt_rank, als_model.item_factors, spotify, db, region, n=10)
        rec_lists['als_filt_div'] = process_rec_list(als_filt_div_idx, als_filt_div_rank, db, spotify, region, spotify_ids=als_filt_div_spot, verbose=True)

        vae_rec_idx, vae_rec_rank = get_als_recs(LEs, als_model, n=N)
        rec_lists['vae'] = process_rec_list(vae_rec_idx, vae_rec_rank, db, spotify, region, verbose=True)

        vae_max_div_idx, vae_max_div_rank, vae_max_div_spot = get_div_recs(vae_rec_idx, vae_rec_rank, als_model.item_factors, spotify, db, region, n=10)
        rec_lists['vae_max_div'] = process_rec_list(vae_max_div_idx, vae_max_div_rank, db, spotify, region, spotify_ids=vae_max_div_spot, verbose=True)

        vae_filt_idx, vae_filt_rank, vae_filter = get_filtered_recs(vae_rec_idx, vae_rec_rank, genre_dict, db, als_model.item_factors, spotify, region)
        vae_filt_div_idx, vae_filt_div_rank, vae_filt_div_spot = get_div_recs(vae_filt_idx, vae_filt_rank, als_model.item_factors, spotify, db, region, n=10)
        rec_lists['vae_filt_div'] = process_rec_list(vae_filt_div_idx, vae_filt_div_rank, db, spotify, region, spotify_ids=vae_filt_div_spot, verbose=True)

        db.users.update_one(
            {'id': id},
            {'$set': {
                'recs': rec_lists,
                'als_filter': als_filter,
                'vae_filter': vae_filter
            }},
            upsert=True
        )

    else:
        rec_lists = user_data['recs']

    return rec_lists


@app.route('/request_recs', methods=['POST'])
def request_recs():
    # Include authorization here
    if db.users.find_one({'id': request.form['participant_id']}, {'_id': 1}) is None:
        return render_template('verification.html')

    session['id'] = request.form['participant_id']
    session['region'] = request.form['region']

    try:
        executor.submit_stored(
            session['id'],
            generate_recs, session['id'],
            request.form['username'],
            refresh_recs=True,
            refresh_LEs=False,
            region=session['region']
        )
    except ValueError:
        pass

    user_data = db.users.find_one({'id': session['id']}, {'intro_survey': 1})
    if 'intro_survey' in user_data:
        return render_template('loading.html')

    return render_template('intro_survey.html')


@app.route('/recieve_form', methods=['POST'])
def recieve_form():
    if request.form['formID'] == 'intro_survey':
        db.users.update_one(
            {'id': session['id']},
            {'$set': {request.form['formID']: request.form}},
            upsert=True
        )

        return render_template('loading.html')

    else:
        return render_template('intro_survey.html')


@app.route('/check_recs')
def check_recs():
    if not executor.futures.done(session['id']):
        return jsonify({'status': executor.futures._state(session['id'])})

    session['rec_lists'] = executor.futures.pop(session['id']).result()

    return jsonify({'status': 'finished'})


@app.route('/rec_lists/', methods=['GET', 'POST'])
def show_rec_lists():
    if 'sequence' not in session:
        cursor = db.latin.find()
        min_count = float(9999999)
        for doc in cursor:
            if doc['count'] <= min_count:
                min_count = doc['count']
                session['group'] = doc['group']
                session['sequence'] = doc['sequence']

        db.users.update_one(
            {'id': session['id']},
            {'$set': {
                'group': session['group'],
                'sequence': session['sequence']
            }},
            upsert=True
        )

        db.latin.update_one(
            {'group': session['group']},
            {'$inc': {
                'count': 1
            }}
        )

    if 'list_count' in session and session['list_count'] == 4:
        return render_template('finished.html')

    elif request.method == 'POST':
        if int(request.form['list_count']) == session['list_count']:
            db.users.update_one(
                {'id': session['id']},
                {'$set': {request.form['list_key']: request.form}},
                upsert=True
            )
            session['list_count'] += 1

    elif request.method == 'GET':
        if 'list_count' not in session:
            session['list_count'] = 0

    return render_template('study.html',
                           rec_list=session['rec_lists'][list(session['rec_lists'].keys())[session['sequence'][session['list_count']]]],
                           list_count=session['list_count'],
                           list_key=list(session['rec_lists'].keys())[session['sequence'][session['list_count']]]
                           )


if __name__ == '__main__':
    app.run()
