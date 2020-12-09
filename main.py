from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from flask_executor import Executor
import pickle
from MultVAE import MultVAE, CSRDataset
from rec_tools import (
    get_user_tracks,
    encode_user_tracks,
    get_als_recs,
    get_vae_recs,
    process_rec_list,
    get_div_recs,
    get_genre_hash,
    get_filtered_recs,
)
from flask_pymongo import PyMongo
import pylast
import torch
from dotenv import load_dotenv
from pathlib import Path
import os
import gdown
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime

load_dotenv()

N = 1000
KEY_LIST = ["als", "als_filt_div", "als_max_div", "vae", "vae_filt_div", "vae_max_div"]

app = Flask(__name__)
executor = Executor(app)
app.config["EXECUTOR_MAX_WORKERS"] = None
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET")
app.config["MONGO_URI"] = os.getenv("MONGO_CLIENT")
mongo = PyMongo(app)

network = pylast.LastFMNetwork(
    api_key=os.getenv("LFM_KEY"), api_secret=os.getenv("LFM_SECRET")
)

spotify = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_KEY"), client_secret=os.getenv("SPOTIFY_SECRET")
    ),
    requests_timeout=20,
)

if int(os.getenv("PRODUCTION")):
    gdown.download(os.getenv("ALS_MODEL_REMOTE"), "als_model.pkl")
    gdown.download(os.getenv("SONG_ENCODINGS_REMOTE"), "song_encodings.pkl")
    gdown.download(os.getenv("VAE_MODEL_REMOTE"), "vae_model.pt")
    ALS_MODEL_DIR = Path("als_model.pkl")
    VAE_MODEL_DIR = Path("vae_model.pkl")
    SONG_ENCODINGS_DIR = Path("song_encodings.pkl")
else:
    ALS_MODEL_DIR = Path(os.getenv("ALS_MODEL_DIR"))
    VAE_MODEL_DIR = Path(os.getenv("VAE_MODEL_DIR"))
    SONG_ENCODINGS_DIR = Path(os.getenv("SONG_ENCODINGS_DIR"))

VAE_MODEL_CONF = {
    "enc_dims": [200],
    "dropout": 0.5,
    "anneal_cap": 1,
    "total_anneal_steps": 10000,
    "learning_rate": 1e-3,
}


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


als_model = load_als_model(ALS_MODEL_DIR)
song_encodings = load_song_encodings(SONG_ENCODINGS_DIR)
vae_model = load_vae_model(VAE_MODEL_DIR, VAE_MODEL_CONF)

print("Done loading models into ram.")


@app.route("/")
def landing():
    return render_template("login.html")


@app.route("/consent")
def consent():
    return render_template("consent_form.html")


def generate_recs(id, username, region, verbose=False, refresh_recs=False, refresh_LEs=False):
    user_data = mongo.db.users.find_one({"id": id}, {"recs": 1})

    rec_lists = {}
    if refresh_recs or refresh_LEs or "recs" not in user_data:
        LEs = get_user_tracks(
            id, username, mongo.db, network, verbose=verbose, refresh=refresh_LEs
        )

        if LEs is None:
            return 'Not enough listening events'

        LEs, rec_lists['known_track_count'], rec_lists['unknown_track_count'], rec_lists['unknown_LE_count'] = encode_user_tracks(LEs, song_encodings, verbose=verbose)
        genre_dict = get_genre_hash(LEs, id, mongo.db, verbose=verbose)

        als_rec_idx, als_rec_rank = get_als_recs(LEs, als_model, n=N)
        rec_lists["als"], rec_lists["als_removed"] = process_rec_list(
            als_rec_idx, als_rec_rank, mongo.db, spotify, region, verbose=verbose
        )

        als_max_div_idx, als_max_div_rank, als_max_div_spot, rec_lists["als_max_div_removed"] = get_div_recs(
            als_rec_idx,
            als_rec_rank,
            als_model.item_factors,
            spotify,
            mongo.db,
            region,
            n=10,
        )
        rec_lists["als_max_div"], _ = process_rec_list(
            als_max_div_idx,
            als_max_div_rank,
            mongo.db,
            spotify,
            region,
            spotify_ids=als_max_div_spot,
            verbose=verbose,
        )

        als_filt_idx, als_filt_rank, als_filter = get_filtered_recs(
            als_rec_idx,
            als_rec_rank,
            genre_dict,
            mongo.db,
            als_model.item_factors,
            spotify,
            region,
            verbose
        )
        als_filt_div_idx, als_filt_div_rank, als_filt_div_spot, rec_lists["als_filt_div_removed"] = get_div_recs(
            als_filt_idx,
            als_filt_rank,
            als_model.item_factors,
            spotify,
            mongo.db,
            region,
            n=10,
        )
        rec_lists["als_filt_div"], _ = process_rec_list(
            als_filt_div_idx,
            als_filt_div_rank,
            mongo.db,
            spotify,
            region,
            spotify_ids=als_filt_div_spot,
            verbose=verbose,
        )

        vae_rec_idx, vae_rec_rank = get_vae_recs(LEs, vae_model, n=N)
        rec_lists["vae"], rec_lists["vae_removed"] = process_rec_list(
            vae_rec_idx, vae_rec_rank, mongo.db, spotify, region, verbose=verbose
        )

        vae_max_div_idx, vae_max_div_rank, vae_max_div_spot, rec_lists["vae_max_div_removed"] = get_div_recs(
            vae_rec_idx,
            vae_rec_rank,
            als_model.item_factors,
            spotify,
            mongo.db,
            region,
            n=10,
        )
        rec_lists["vae_max_div"], _ = process_rec_list(
            vae_max_div_idx,
            vae_max_div_rank,
            mongo.db,
            spotify,
            region,
            spotify_ids=vae_max_div_spot,
            verbose=verbose,
        )

        vae_filt_idx, vae_filt_rank, vae_filter = get_filtered_recs(
            vae_rec_idx,
            vae_rec_rank,
            genre_dict,
            mongo.db,
            als_model.item_factors,
            spotify,
            region,
            verbose
        )
        vae_filt_div_idx, vae_filt_div_rank, vae_filt_div_spot, rec_lists["vae_filt_div_removed"] = get_div_recs(
            vae_filt_idx,
            vae_filt_rank,
            als_model.item_factors,
            spotify,
            mongo.db,
            region,
            n=10,
        )
        rec_lists["vae_filt_div"], _ = process_rec_list(
            vae_filt_div_idx,
            vae_filt_div_rank,
            mongo.db,
            spotify,
            region,
            spotify_ids=vae_filt_div_spot,
            verbose=verbose,
        )

        mongo.db.users.update_one(
            {"id": id},
            {
                "$set": {
                    "recs": rec_lists,
                    "als_filter": als_filter,
                    "vae_filter": vae_filter,
                }
            },
            upsert=True,
        )

    else:
        rec_lists = user_data["recs"]

    return rec_lists


@app.route("/request_recs", methods=["POST"])
def request_recs():
    # Only proceed id the id is in the database already
    if (
        mongo.db.users.find_one({"id": request.form["participant_id"]}, {"_id": 1})
        is None
    ):
        return render_template("login.html")

    if "id" in session and session["id"] != request.form["participant_id"]:
        session.clear()
    session["id"] = request.form["participant_id"]
    session["region"] = request.form["region"]

    if executor.futures.done(session["id"]) is None:
        executor.submit_stored(
            session["id"],
            generate_recs,
            session["id"],
            request.form["username"],
            session["region"],
            refresh_recs=False,
            refresh_LEs=False,
            verbose=False
        )

    user_data = mongo.db.users.find_one({"id": session["id"]}, {"intro_survey": 1})
    if "intro_survey" in user_data:
        return render_template("loading.html")
    else:
        mongo.db.users.update_one(
            {"id": session["id"]},
            {
                "$set": {
                    'intro_survey_start': datetime.utcnow(),
                }
            },
            upsert=True,
        )

    return render_template("intro_survey.html")


@app.route("/recieve_form", methods=["POST"])
def recieve_form():
    if request.form["formID"] == "intro_survey":
        mongo.db.users.update_one(
            {"id": session["id"]},
            {"$set": {
                request.form["formID"]: request.form,
                'intro_survey_end': datetime.utcnow()
            }},
            upsert=True,
        )

        return render_template("loading.html")

    elif request.form["formID"] == "consent_form":
        if (
            request.form["q3_iHave"] == "Yes"
            and request.form["q4_iAgree"] == "Yes"
            and request.form["q6_withFull"] == "I agree to participate."
        ):
            mongo.db.users.update_one(
                {"id": request.form["q8_whatIs"]},
                {
                    "$set": {
                        "consent_form": request.form
                    }
                },
                upsert=True,
            )
            return render_template("login.html")
        else:
            return render_template("consent_form.html")

    elif request.form['formID'] == "verify_ownership":
        artist_data = mongo.db.LEs.find_one({"id": session["id"]}, {"artists": 1})

        if session['v_attempts'] > 0:
            if request.form['artist_name'].lower() in artist_data['artists']:
                mongo.db.users.update_one(
                    {"id": session["id"]},
                    {"$set": {"verified": True}},
                    upsert=True,
                )
                return redirect(url_for('show_rec_lists'))
            else:
                session['v_attempts'] -= 1

        return redirect(url_for('verify_ownership'))

    else:
        return render_template("intro_survey.html")


@app.route("/check_recs")
def check_recs():
    if not executor.futures.done(session["id"]):
        if executor.futures.done(session["id"]) is None:
            if 'rec_lists' in session:
                return jsonify({'status': 'verify'})
            else:
                return jsonify({'status': 'retry'})
        else:
            return jsonify({"status": executor.futures._state(session["id"])})

    result = executor.futures.pop(session["id"]).result()
    if result == 'Not enough listening events':
        return jsonify({"status": result})
    else:
        session["rec_lists"] = result

    return jsonify({'status': 'verify'})


@app.route("/verify_ownership/")
def verify_ownership():
    verified_data = mongo.db.users.find_one({'id': session['id']}, {'verified': 1})

    if 'verified' in verified_data:
        return redirect(url_for('show_rec_lists'))
    elif 'v_attempts' not in session:
        session['v_attempts'] = 3

    return render_template("verify_ownership.html", v_attempts=session['v_attempts'])


@app.route("/rec_lists/", methods=["GET", "POST"])
def show_rec_lists():
    if 'rec_lists' not in session:
        rec_data = mongo.db.users.find_one({'id': session['id']}, {'recs': 1})
        session['rec_lists'] = rec_data['recs']

    # Check if sequence is already in cookies, if not pull it from the database or initialize it
    if "sequence" not in session:
        sequence_data = mongo.db.users.find_one({"id": session["id"]}, {"sequence": 1})
        if "sequence" in sequence_data:
            session["sequence"] = sequence_data["sequence"]
        else:
            cursor = mongo.db.latin.find()
            min_count = float(9999999)
            for doc in cursor:
                if doc["count"] <= min_count:
                    min_count = doc["count"]
                    session["group"] = doc["group"]
                    session["sequence"] = doc["sequence"]

            mongo.db.users.update_one(
                {"id": session["id"]},
                {"$set": {"group": session["group"], "sequence": session["sequence"]}},
                upsert=True,
            )

            mongo.db.latin.update_one(
                {"group": session["group"]}, {"$inc": {"count": 1}}
            )

    # Check if list_count is already in cookies, if not pull it from the database or initialize it
    if "list_count" not in session:
        list_count_data = mongo.db.users.find_one(
            {"id": session["id"]}, {"list_count": 1}
        )

        if "list_count" in list_count_data:
            session["list_count"] = list_count_data["list_count"]
        else:
            session["list_count"] = 0

    if request.method == "POST":
        if int(request.form["list_count"]) == session["list_count"]:
            session["list_count"] += 1
            mongo.db.users.update_one(
                {"id": session["id"]},
                {
                    "$set": {
                        request.form["list_key"]: request.form,
                        request.form["list_key"] + '_end': datetime.utcnow(),
                        "list_count": session["list_count"],
                    }
                },
                upsert=True,
            )

    if session["list_count"] >= 6:
        return render_template("finished.html")
    else:
        mongo.db.users.update_one(
            {"id": session["id"]},
            {
                "$set": {
                    KEY_LIST[session["sequence"][session["list_count"]]] + '_start': datetime.utcnow(),
                }
            },
            upsert=True,
        )

    return render_template(
        "study.html",
        rec_list=session["rec_lists"][
            KEY_LIST[session["sequence"][session["list_count"]]]
        ],
        list_count=session["list_count"],
        list_key=KEY_LIST[session["sequence"][session["list_count"]]],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
