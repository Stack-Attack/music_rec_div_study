import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from scipy import sparse
from numba import jit
from tqdm import tqdm
import time
from datetime import datetime

@jit(nopython=True)
def get_div_fast(v1_features, v2_features):
    div_vec = []
    for feature_vector in v1_features:
        sub = v2_features - feature_vector
        div = 0

        for vector in sub:
            dist = np.linalg.norm(vector)
            div += dist

        div_vec.append(div)

    return div_vec


def get_vector_div(v1_features, v2_features, norm=False, scaler=False):
    """
    Returns a list of diversity values for each track by summing pairwise euclidean distance.
    :param v1: A list of track features which you would like to obtain the diversity of
    :param v2: A list of track features which you would like to measure diversity against
    :param norm: Bool which determines whether or not each sample vector is l2 normalized before the calculations.
    :return:
    """
    if scaler is not False:
        v1_features = scaler.transform(v1_features)
        v2_features = scaler.transform(v2_features)

    if norm is not False:
        v1_features = normalize(v1_features, norm=norm)
        v2_features = normalize(v2_features, norm=norm)

    return np.array(get_div_fast(v1_features, v2_features))


def get_div_recs(
    rec_idx,
    rec_rank,
    latent_features,
    spotify,
    db,
    region,
    n=10,
    norm="l2",
    scaler=StandardScaler(),
):
    if scaler is not False:
        scaler.fit(latent_features)

    # Calculate the ILD for each track as a numpy array

    rec_idx_remainder = rec_idx
    rec_rank_remainder = rec_rank

    div_rec_idx = []
    div_rec_rank = []
    no_data = 0

    while not div_rec_idx:
        div_vec = get_vector_div(
            latent_features[rec_idx_remainder],
            latent_features[rec_idx_remainder],
            norm,
            scaler,
        )
        # Find the index of the maximally diverse track and add it's encoding to the recommendation list remove it from
        max_div_idx = np.argmax(div_vec)
        div_rec_idx.append(rec_idx_remainder[max_div_idx])
        div_rec_rank.append(rec_rank_remainder[max_div_idx])

        rec_idx_remainder = np.delete(rec_idx_remainder, max_div_idx)
        rec_rank_remainder = np.delete(rec_rank_remainder, max_div_idx)

        track_data = db.tracks.find_one({"encoding": int(div_rec_idx[-1])})

        if track_data["spotify"]["id"]:
            spotify_id = get_fresh_spotify_id(
                track_data["spotify"]["id"], spotify, region
            )
        else:
            spotify_id = False

        if spotify_id:
            div_rec_spotify_ids = [spotify_id]
        else:
            div_rec_idx.pop(-1)
            div_rec_rank.pop(-1)
            no_data += 1

    # Continuously add the track maximally different from the list so far
    while len(div_rec_idx) <= n:
        div_vec = get_vector_div(
            latent_features[rec_idx_remainder], latent_features[div_rec_idx], norm
        )

        max_div_idx = np.argmax(div_vec)
        div_rec_idx.append(rec_idx_remainder[max_div_idx])
        div_rec_rank.append(rec_rank_remainder[max_div_idx])

        rec_idx_remainder = np.delete(rec_idx_remainder, max_div_idx)
        rec_rank_remainder = np.delete(rec_rank_remainder, max_div_idx)

        track_data = db.tracks.find_one({"encoding": int(div_rec_idx[-1])})

        if track_data["spotify"]["id"]:
            spotify_id = get_fresh_spotify_id(
                track_data["spotify"]["id"], spotify, region
            )
        else:
            spotify_id = False

        if spotify_id:
            div_rec_spotify_ids.append(spotify_id)
        else:
            div_rec_idx.pop(-1)
            div_rec_rank.pop(-1)
            no_data += 1

    return div_rec_idx, div_rec_rank, div_rec_spotify_ids, no_data


def get_user_tracks(
    id,
    username,
    db,
    network,
    refresh=False,
    time_from=1590969600,
    limit=None,
    verbose=False,
    min_count=50
):
    user_data = db.LEs.find_one({"id": id}, {'LEs': 1})

    if user_data and not refresh:
        LEs = user_data["LEs"]
        if verbose:
            print("Reading cached LEs")

    else:
        if verbose:
            print("Fetching LE's")

        tries = 3
        while tries > 0:
            try:
                user = network.get_user(username)
                result = user.get_recent_tracks(time_from=time_from, limit=limit)
                break
            except Exception as e:
                if str(e) == "Operation failed - Most likely the backend service failed. Please try again.":
                    tries -= 1
                    time.sleep(10)
                else:
                    db.logs.insert_one({'exception': str(e), 'id': id, 'username': username, 'time': datetime.utcnow()})
                    return None

            if tries == 0:
                return None

        LEs = {}
        artists = {}
        for LE in result:
            track = (LE.track.artist.name, LE.track.title)
            date = LE.timestamp

            LEs[date] = track
            if LE.track.artist.name[0] != '$':
                artists[LE.track.artist.name.lower()] = 1

        if len(LEs) >= min_count:
            db.LEs.update_one({"id": id},
                                {"$set": {
                                    "LEs": LEs,
                                    "LE_count": len(LEs),
                                    "artists": artists
                                }}, upsert=True)
        else:
            return None

    if verbose:
        print(len(LEs), "listening events.")

    return LEs


def encode_user_tracks(LEs, encodings, verbose=False):
    user_vector = np.zeros(len(encodings), dtype=np.int64)

    unknown = set()
    unknown_LE_count = 0
    known = set()

    for track in tqdm(LEs.values(), total=len(LEs), disable=not verbose):
        if tuple(track) in encodings:
            user_vector[int(encodings[tuple(track)])] += 1

            if tuple(track) not in known:
                known.add(tuple(track))
        else:
            unknown.add(tuple(track))
            unknown_LE_count += 1

    if verbose:
        print(unknown_LE_count, " listening events unknown.")
        print(len(known), "tracks known.")
        print(len(unknown), "tracks unknown.")

    return user_vector, len(known), len(unknown), unknown_LE_count


def get_genre_hash(LEs, id, db, refresh=False, verbose=False):
    genre_data = db.LEs.find_one({"id": id}, {"genres": 1})

    if "genres" not in genre_data or refresh:
        genre_dict = {}
        for encoding in tqdm(LEs.nonzero()[0], disable=not verbose):
            track_data = db.tracks.find_one(
                {"encoding": int(encoding)}, {"spotify.genres": 1}
            )
            track_genres = track_data["spotify"]["genres"]
            for genre in track_genres:
                if genre in genre_dict:
                    genre_dict[genre] += 1
                else:
                    genre_dict[genre] = 1

        db.LEs.update_one({"id": id}, {"$set": {"genres": genre_dict}}, upsert=True)

    else:
        genre_dict = genre_data["genres"]

    return genre_dict


def get_als_recs(user_data, als_model, n=500):
    recs = als_model.recommend(
        0,
        sparse.csr_matrix(user_data),
        n,
        recalculate_user=True,
        filter_already_liked_items=True,
    )

    return np.array([rec[0] for rec in recs]), np.array(range(len(recs)))


def get_vae_recs(user_data, vae_model, n=500):
    raw_recs = vae_model.forward(sparse.csr_matrix(user_data))
    _, rec_idx = raw_recs.topk(n)

    return rec_idx[0].detach().numpy(), np.array(range(rec_idx.shape[1]))


def get_filtered_recs(
    rec_idx, rec_rank, genre_dict, db, latent_features, spotify, region, verbose=True
):
    # Retrieve the most diverse track in the entire recommendation list
    max_div_encodings, _, _, _ = get_div_recs(
        rec_idx, rec_rank, latent_features, spotify, db, region, n=0
    )
    max_div_track = db.tracks.find_one(
        {"encoding": int(max_div_encodings[0])}, {"spotify.genres": 1}
    )

    # Set the genre threshold (number of times a genre must occur in the users listening history) to be one greater than
    # the most diverse tracks known genres. This ensures a different filtered list than non-filtered list
    genre_threshold = float("inf")
    threshold_genre = ""
    other_genres = max_div_track["spotify"]["genres"]
    for genre in max_div_track["spotify"]["genres"]:
        if genre in genre_dict and genre_dict[genre] <= genre_threshold:
            genre_threshold = genre_dict[genre]
            threshold_genre = genre
    if genre_threshold == float("inf"):
        genre_threshold = 0

    # Generate new recommendation list which only contains tracks tagged with common genres in the users listening history
    filter_rec_idx, filter_rec_rank = [], []
    for encoding, rank in tqdm(
        zip(rec_idx, rec_rank), total=len(rec_idx), disable=not verbose
    ):
        track_data = db.tracks.find_one(
            {"encoding": int(encoding)}, {"spotify.genres": 1}
        )
        track_genres = track_data["spotify"]["genres"]

        for i, genre in enumerate(track_genres):
            if genre not in genre_dict or genre_dict[genre] <= genre_threshold:
                break
            if i == len(track_genres) - 1:
                filter_rec_idx.append(encoding)
                filter_rec_rank.append(rank)

    # How many tracks were removed through filtering
    filter_count = len(rec_idx) - len(filter_rec_idx)

    if verbose:
        print(filter_count, " tracks removed by filter.")
        print(
            "Genre threshold set at ",
            genre_threshold,
            " for ",
            threshold_genre,
            " out of ",
            other_genres,
        )

    return (
        filter_rec_idx,
        filter_rec_rank,
        {"genre_threshold": genre_threshold, "filter_count": filter_count},
    )


def get_fresh_spotify_id(old_id, spotify, region):
    result = spotify.tracks([old_id], market=region)

    if len(result["tracks"]) > 0 and result["tracks"][0]["is_playable"]:
        return result["tracks"][0]["id"]
    else:
        return False


def process_rec_list(
    rec_idx, rec_rank, db, spotify, region, spotify_ids=False, n=10, verbose=False
):
    rec_list = []
    no_data = 0
    i = 0
    while len(rec_list) < n:
        result = db.tracks.find_one({"encoding": int(rec_idx[i])})

        song = {
            "artist": result["track"][0],
            "track": result["track"][1],
            "rel": float(rec_rank[i]),
            "pop": result["popularity"],
        }

        if spotify_ids:
            song["spotify"] = spotify_ids[i]
        elif result["spotify"]["id"]:
            song["spotify"] = get_fresh_spotify_id(
                result["spotify"]["id"], spotify, region
            )
        else:
            song["spotify"] = False

        if song["spotify"]:
            rec_list.append(song)
        else:
            no_data += 1
        i += 1

    if verbose:
        print(no_data, "songs removed due to lack of metadata.")

    return rec_list, no_data
