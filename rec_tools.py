import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse
from numba import jit
from tqdm import tqdm

@jit(nopython=True)
def get_div_fast(v1_features, v2_features):
    div_vec = []
    for feature_vector in v2_features:
        sub = v1_features - feature_vector
        div = 0

        for i, vector in enumerate(sub):
            dist = np.linalg.norm(vector)
            div += dist

        div_vec.append(div)

    return div_vec


def get_vector_div(v1, v2, norm=False):
    if norm is not False:
        v1 = normalize(v1, norm=norm)
        v2 = normalize(v2, norm=norm)

    return get_div_fast(v1, v2)


def get_user_tracks(username, db, network, refresh=False, time_from=1577836800, limit=None, verbose=False):
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

    return recs


def get_vae_recs(user_data, vae_model, n=500):
    raw_recs = vae_model.forward(sparse.csr_matrix(user_data))
    rec_rel, rec_idx = raw_recs.topk(n)

    return list(zip(rec_idx[0].detach().numpy(), rec_rel[0].detach().numpy()))


def process_rec_list(recs, db, verbose=False):
    rec_list = []
    no_data = 0
    for encoding, rel in recs:
        result = db.tracks.find_one({'encoding': int(encoding)})

        song = {
            'artist': result['track'][0],
            'track': result['track'][1],
            'rel': float(rel),
            'tags': [ tag['tag'] for tag in result['lfm_tags'] ],
            'genres': result['spotify']['genres'],
            'spotify': result['spotify']['id']
        }

        if song['spotify']:
            rec_list.append(song)
        else:
            no_data += 1

    if verbose:
        print(no_data, "songs removed due to lack of metadata.")

    return rec_list
