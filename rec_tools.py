import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from scipy import sparse
from numba import jit
from tqdm import tqdm


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


def get_div_recs(rec_idx, rec_rank, latent_features, spotify, db, region, n=10, norm='l2', scaler=StandardScaler()):
    if scaler is not False:
        scaler.fit(latent_features)

    # Calculate the ILD for each track as a numpy array
    div_vec = get_vector_div(latent_features[rec_idx], latent_features[rec_idx], norm, scaler)

    div_rec_idx = []
    div_rec_rank = []

    while not div_rec_idx:
    # Find the index of the maximally diverse track and add it's encoding to the recommendation list remove it from
        max_div_idx = np.argmax(div_vec)
        div_rec_idx.append(rec_idx[max_div_idx])
        div_rec_rank.append(rec_rank[max_div_idx])

        rec_idx_remainder = np.delete(rec_idx, max_div_idx)
        rec_rank_remainder = np.delete(rec_rank, max_div_idx)

        decoding = db.tracks.find_one({'encoding': int(div_rec_idx[-1])})['track']
        spotify_id = get_fresh_spotify_id(decoding[0], decoding[1], spotify, region)

        if spotify_id:
            div_rec_spotify_ids = [spotify_id]
        else:
            div_rec_idx.pop(-1)

    # Continuously add the track maximally different from the list so far
    while len(div_rec_idx) <= n:
        div_vec = get_vector_div(latent_features[rec_idx_remainder], latent_features[div_rec_idx], norm)

        max_div_idx = np.argmax(div_vec)
        div_rec_idx.append(rec_idx_remainder[max_div_idx])
        div_rec_rank.append(rec_rank_remainder[max_div_idx])

        rec_idx_remainder = np.delete(rec_idx_remainder, max_div_idx)
        rec_rank_remainder = np.delete(rec_rank_remainder, max_div_idx)

        decoding = db.tracks.find_one({'encoding': int(div_rec_idx[-1])})['track']
        spotify_id = get_fresh_spotify_id(decoding[0], decoding[1], spotify, region)

        if spotify_id:
            div_rec_spotify_ids.append(spotify_id)
        else:
            div_rec_idx.pop(-1)

    return div_rec_idx, div_rec_rank, div_rec_spotify_ids


def get_user_tracks(username, db, network, refresh=False, time_from=1590969600, limit=None, verbose=False):
    user_data = db.users.find_one({'username': username})

    if user_data and not refresh:
        LEs = user_data['LEs']
        if verbose:
            print('Reading cached LEs')

    else:
        if verbose:
            print("Fetching LE's")
        user = network.get_user(username)
        result = user.get_recent_tracks(time_from=time_from, limit=limit)

        LEs = {}
        for LE in result:
            track = (LE.track.artist.name, LE.track.title)
            date = LE.timestamp

            LEs[date] = track

        db.users.update_one(
            {'username': username},
            {'$set': {'LEs': LEs}},
            upsert=True
        )
    if verbose:
        print(len(LEs), 'listening events.')

    return LEs


def encode_user_tracks(LEs, encodings, verbose=False):
    user_vector = np.zeros(
        len(encodings)
        , dtype=np.int64)

    unknown = set()
    unknown_LE_count = 0
    known = set()

    for track in tqdm(LEs.values(), disable=not verbose):
        if tuple(track) in encodings:
            user_vector[int(encodings[tuple(track)])] += 1

            if tuple(track) not in known:
                known.add(tuple(track))
        else:
            unknown.add(tuple(track))
            unknown_LE_count += 1

    if verbose:
        print(unknown_LE_count, ' listening events unknown.')
        print(len(known), "tracks known.")
        print(len(unknown), "tracks unknown.")

    return user_vector, known, unknown


def get_als_recs(user_data, als_model, n=500):
    recs = als_model.recommend(
        0,
        sparse.csr_matrix(user_data),
        n,
        recalculate_user=True,
        filter_already_liked_items=True
    )

    return np.array([ rec[0] for rec in recs ]), np.array(range(len(recs)))


def get_vae_recs(user_data, vae_model, n=500):
    raw_recs = vae_model.forward(sparse.csr_matrix(user_data))
    _, rec_idx = raw_recs.topk(n)

    return rec_idx[0].detach().numpy(), np.array(range(rec_idx.shape[1]))


def get_fresh_spotify_id(artist, song, spotify, region):
    q = 'artist:' + artist + ' track:' + song.replace("'", "")
    result = spotify.search(q.encode(encoding='UTF-8'), limit=1, type='track', market=region)

    if len(result['tracks']['items']) > 0 and result['tracks']['items'][0]['is_playable']:
        return result['tracks']['items'][0]['id']
    else:
        return False


def process_rec_list(rec_idx, rec_rank, db, spotify, region, spotify_ids=False, n=10, verbose=False):
    rec_list = []
    no_data = 0
    i = 0
    while len(rec_list) < n:
        result = db.tracks.find_one({'encoding': int(rec_idx[i])})

        song = {
            'artist': result['track'][0],
            'track': result['track'][1],
            'rel': float(rec_rank[i]),
            'pop': result['popularity'],
            'tags': [ tag['tag'] for tag in result['lfm_tags'] ],
            'genres': result['spotify']['genres'],
            'spotify': get_fresh_spotify_id(result['track'][0], result['track'][1], spotify, region)
        }

        if spotify_ids:
            song['spotify'] = spotify_ids[i]
        else:
            song['spotify'] = get_fresh_spotify_id(result['track'][0], result['track'][1], spotify, region)

        if song['spotify']:
            rec_list.append(song)
        else:
            no_data += 1
        i += 1

    if verbose:
        print(no_data, "songs removed due to lack of metadata.")

    return rec_list
