let Y_MARGINS = 30 + 80
let Y_START = 50
let NUM_RANKS = 10

let known_songs = 0

const SPOTIFY_URL = 'https://open.spotify.com/embed/track/';

let QUESTIONS = [
    'Please answer the following questions about the list you just viewed as a whole.',
    'The list of recommended music was diverse.',
    'The list of recommended music portrayed the breadth of my music interests.',
    'I was satisfied with the list of recommended music.',
    'I was satisfied with each individual recommendation.',
    'The list of music portrayed a wide range of genres.',
    'The list of music was too diverse.',
    'The list of music was not diverse enough.'
]

function get_preview_element(id, top, load=true){
    let preview = document.createElement('div');
    preview.className = 'preview';
    preview.id = id
    preview.style.opacity = '0';
    preview.style.top = top.toString().concat('px');

    let spotify = document.createElement('iframe');

    spotify.className = "song_preview";

    let likert_form = document.createElement('div');
    likert_form.className = 'likert_form';
    likert_form.style.display = 'flex'

    /*
    let familiar = document.createElement('a');
    familiar.innerHTML = "Song Known";
    familiar.className = 'known_button';
    familiar.onclick = () => remove(id);
    */

    let familiar_label = document.createElement('label');
    familiar_label.innerHTML = "Track<br/>Known<br/>"
    let familiar = document.createElement('input');
    familiar.type = 'checkbox';
    familiar.className = 'radio_check'
    familiar.name = id.concat('_track_known')
    familiar.value = 'yes'
    familiar_label.appendChild(familiar)

    let artist_known_label = document.createElement('label');
    artist_known_label.innerHTML = "Artist<br/>Known<br/>"
    let artist_known = document.createElement('input');
    artist_known.type = 'checkbox';
    artist_known.className = 'radio_check'
    artist_known.name = id.concat('_artist_known')
    artist_known.value = 'yes'
    artist_known_label.appendChild(artist_known)

    let l1_label = document.createElement('label');
    l1_label.innerHTML = "Strongly<br/>Dislike<br/>"
    let l1 = document.createElement('input');
    l1.type = 'radio';
    l1.name = id;
    l1.value = 'sd'
    l1.className = 'radio_check'
    l1.required = true
    l1_label.appendChild(l1);

    let l2_label = document.createElement('label');
    l2_label.innerHTML = "Dislike<br/>";
    let l2 = document.createElement('input');
    l2.type = 'radio';
    l2.name = id;
    l2.value = 'd'
    l2.className = 'radio_check'
    l2.required = true
    l2_label.appendChild(l2);

    let l3_label = document.createElement('label');
    l3_label.innerHTML = "Undecided<br/>";
    let l3 = document.createElement('input');
    l3.type = 'radio';
    l3.name = id;
    l3.value = 'u'
    l3.className = 'radio_check'
    l3.required = true
    l3_label.appendChild(l3);

    let l4_label = document.createElement('label');
    l4_label.innerHTML = "Like<br/>";
    let l4 = document.createElement('input');
    l4.name = id
    l4.type = 'radio';
    l4.value = 'l'
    l4.className = 'radio_check'
    l4.required = true
    l4_label.appendChild(l4);

    let l5_label = document.createElement('label');
    l5_label.innerHTML = "Strongly Like<br/>";
    let l5 = document.createElement('input');
    l5.name = id
    l5.type = 'radio';
    l5.value = 'sl'
    l5.className = 'radio_check'
    l5.required = true
    l5_label.appendChild(l5);

    likert_form.appendChild(familiar_label)
    likert_form.appendChild(artist_known_label);
    likert_form.appendChild(l1_label);
    likert_form.appendChild(l2_label);
    likert_form.appendChild(l3_label);
    likert_form.appendChild(l4_label);
    likert_form.appendChild(l5_label);

    //preview.appendChild(familiar);
    preview.appendChild(spotify);
    preview.appendChild(likert_form);

    if (load) {
        let promise = new Promise((resolve, reject) => {
            spotify.onload = () => resolve();
            spotify.onerror = reject;
        });

        spotify.src = SPOTIFY_URL.concat(id);

        return{'preview': preview, 'iframe': spotify, 'promise': promise}
    }

    return{'preview': preview, 'iframe': spotify}
}

function get_likert_question(question, top, likert){
    let container = document.createElement('div');
    container.className = 'preview';
    container.id = question
    container.style.opacity = '1';
    container.style.top = (top + 100).toString().concat('px');

    let question_label = document.createElement('div');
    question_label.className = "question_label";
    question_label.innerText = question

    container.appendChild(question_label);

    if (likert === true) {
        let likert_form = document.createElement('div');
        likert_form.className = 'likert_form';
        likert_form.style.display = 'flex'

        let l1_label = document.createElement('label');
        l1_label.innerHTML = "Strongly<br/>Dislike<br/>"
        let l1 = document.createElement('input');
        l1.type = 'radio';
        l1.name = question
        l1.value = 'sd'
        l1.className = 'radio_check'
        l1.required = true
        l1_label.appendChild(l1);

        let l2_label = document.createElement('label');
        l2_label.innerHTML = "Dislike<br/>";
        let l2 = document.createElement('input');
        l2.type = 'radio';
        l2.name = question
        l2.value = 'd'
        l2.className = 'radio_check'
        l2.required = true
        l2_label.appendChild(l2);

        let l3_label = document.createElement('label');
        l3_label.innerHTML = "Like<br/>";
        let l3 = document.createElement('input');
        l3.type = 'radio';
        l3.name = question
        l3.value = 'u'
        l3.className = 'radio_check'
        l3.required = true
        l3_label.appendChild(l3);

        let l4_label = document.createElement('label');
        l4_label.innerHTML = "Strongly Like<br/>";
        let l4 = document.createElement('input');
        l4.name = question
        l4.type = 'radio';
        l4.value = 'a'
        l4.className = 'radio_check'
        l4.required = true
        l4_label.appendChild(l4);

        let l5_label = document.createElement('label');
        l5_label.innerHTML = "Strongly Like<br/>";
        let l5 = document.createElement('input');
        l5.name = question
        l5.type = 'radio';
        l5.value = 'sa'
        l5.className = 'radio_check'
        l5.required = true
        l5_label.appendChild(l5);

        likert_form.appendChild(l1_label);
        likert_form.appendChild(l2_label);
        likert_form.appendChild(l3_label);
        likert_form.appendChild(l4_label);
        likert_form.appendChild(l5_label);

        container.appendChild(likert_form);
    }

    return{'question': container}
}

async function init(songs){
    //compute or get ranks
    loading()

    let element = document.createElement('input')
    element.type = 'submit'
    element.value = 'Continue'
    element.className = 'preview_screen'
    element.style.top = (Y_START + ((QUESTIONS.length + NUM_RANKS)*Y_MARGINS)).toString().concat('px')
    element.id = 'next_page'
    document.getElementById('rec_form').appendChild(element)

    for (let i = 0; i<songs.length; i++){
        let container = document.getElementById('rec_form')
        if (i < NUM_RANKS + 1) {
            let preview_object = get_preview_element(songs[i]['spotify'], (Y_START + (NUM_RANKS * Y_MARGINS)))
            container.appendChild(preview_object.preview)
            songs[i]['element'] = preview_object
            await preview_object.promise
        }
        else{
            let preview_object = get_preview_element(songs[i]['spotify'], (Y_START + (NUM_RANKS * Y_MARGINS)), load=false)
            container.appendChild(preview_object.preview)
        }
    }

    let likert_question;
    for (let i = 0; i < QUESTIONS.length; i++) {
        let container = document.getElementById('rec_form')
        if (i === 0) {
            likert_question = get_likert_question(QUESTIONS[i], (Y_START + (NUM_RANKS * Y_MARGINS)) + 100 * i, false)
        } else {
            likert_question = get_likert_question(QUESTIONS[i], (Y_START + (NUM_RANKS * Y_MARGINS)) + 100 * i, true)
        }
        container.appendChild(likert_question.question)
    }

    update_display(songs)
    not_loading()
}

function update_display(songs) {
    for (let i = 0; i < songs.length; i++) {
        let element = songs[i]['element']['preview']
            if (i < NUM_RANKS) {
                element.style.opacity = '1'
                let new_top = (Y_MARGINS*i)+Y_START
                element.style.display = 'flex'
                element.style.top = new_top.toString().concat('px')
            }
            else {
                element.style.top = (Y_START + (NUM_RANKS*Y_MARGINS)).toString().concat('px')
                element.style.opacity = '0.0'
            }

            if (i === NUM_RANKS + 1){
                let iframe = songs[i]['element']['iframe']
                if (iframe.src === "") {
                    iframe.src = SPOTIFY_URL.concat(element.id);
                }
            }
        }
}

function remove(id, songs){
    for (let i = 0; i < songs.length; i++) {
        if (songs[i]['spotify'] === id) {
            let element = songs[i]['element']['preview']

            document.getElementById('rec_form').removeChild(element)
            songs.splice(i, 1)
            known_songs += 1

            if (NUM_RANKS > Object.keys(songs).length) {
                NUM_RANKS--
            }

            let button = document.getElementById('next_page')
            button.style.top = (Y_START + ((QUESTIONS.length + NUM_RANKS)*Y_MARGINS)).toString().concat('px')
        }
    }
    update_display(songs)
}

function loading(){
    document.getElementById('loading').style.display = "flex";
    document.getElementById('content').style.display = "none";
}

function not_loading(){
    document.getElementById('loading').style.display = "none";
    document.getElementById('content').style.display = "flex";

}