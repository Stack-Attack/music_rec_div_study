let Y_MARGINS = 30 + 80
let Y_START = 50
let NUM_RANKS = 10
let COMPLETION_REQ = 20

let rec_lists = []
let username = undefined
let known_songs = 0
let phase = 0
let songs = []
let completion_count = 0
let curr_list = 0

const SPOTIFY_URL = 'https://open.spotify.com/embed/track/';

function get_preview_element(id, top, load=true){
    let preview = document.createElement('div');
    preview.className = 'preview';
    preview.id = id
    preview.style.opacity = '0';
    preview.style.top = top.toString().concat('px');

    let spotify = document.createElement('iframe');

    spotify.className = "song_preview";

    let likert_form = document.createElement('form');
    likert_form.className = 'likert_form';
    likert_form.style.display = 'flex'

    let familiar = document.createElement('a');
    familiar.innerHTML = "Well Known";
    familiar.className = 'known_button';
    familiar.onclick = () => remove(id);


    let artist_known_label = document.createElement('label');
    artist_known_label.innerHTML = "Artist<br/>Known<br/>"
    let artist_known = document.createElement('input');
    artist_known.type = 'checkbox';
    artist_known.className = 'radio_check'
    artist_known.name = 'artist_known'
    artist_known.value = 'yes'
    artist_known_label.appendChild(artist_known)

    let l1_label = document.createElement('label');
    l1_label.innerHTML = "Strongly<br/>Dislike<br/>"
    let l1 = document.createElement('input');
    l1.type = 'radio';
    l1.name = 'likert';
    l1.value = 'sd'
    l1.className = 'radio_check'
    l1_label.appendChild(l1);

    let l2_label = document.createElement('label');
    l2_label.innerHTML = "Dislike<br/>";
    let l2 = document.createElement('input');
    l2.type = 'radio';
    l2.name = 'likert';
    l2.value = 'd'
    l2.className = 'radio_check'
    l2_label.appendChild(l2);

    let l3_label = document.createElement('label');
    l3_label.innerHTML = "Like<br/>";
    let l3 = document.createElement('input');
    l3.type = 'radio';
    l3.name = 'likert';
    l3.value = 'l'
    l3.className = 'radio_check'
    l3_label.appendChild(l3);

    let l4_label = document.createElement('label');
    l4_label.innerHTML = "Strongly Like<br/>";
    let l4 = document.createElement('input');
    l4.name = 'likert'
    l4.type = 'radio';
    l4.value = 'sl'
    l4.className = 'radio_check'
    l4_label.appendChild(l4);

    likert_form.appendChild(artist_known_label);
    likert_form.appendChild(l1_label);
    likert_form.appendChild(l2_label);
    likert_form.appendChild(l3_label);
    likert_form.appendChild(l4_label);

    preview.appendChild(familiar);
    preview.appendChild(spotify);
    preview.appendChild(likert_form);

    let form_data = new FormData(likert_form)

    if (load) {
        let promise = new Promise((resolve, reject) => {
            spotify.onload = () => resolve();
            spotify.onerror = reject;
        });

        spotify.src = SPOTIFY_URL.concat(id);

        return{'preview': preview, 'iframe': spotify, 'promise': promise, 'form_data': form_data}
    }

    return{'preview': preview, 'iframe': spotify, 'form_data': form_data}
}

async function init(){
    //compute or get ranks
    loading()

    //Full list phase
    if (phase === 2){
        //Give entire survey
        alert('Survey time!')
    }
    else {
        songs = rec_lists[curr_list]
    }


    let element = document.createElement('button')
    element.type = 'button'
    element.innerText = 'Continue'
    element.className = 'preview_screen'
    element.style.top = (Y_START + (songs.length*Y_MARGINS)).toString().concat('px')
    element.onclick = next_page
    element.id = 'next_page'
    document.getElementById('preview_container').appendChild(element)

    for (let i = 0; i<songs.length; i++){
        let container = document.getElementById('preview_container')
        if (i < NUM_RANKS + 1) {
            let preview_object = get_preview_element(songs[i]['spotify'], (Y_START + (songs.length * Y_MARGINS)))
            container.appendChild(preview_object.preview)
            songs[i]['element'] = preview_object
            await preview_object.promise
        }
        else{
            let preview_object = get_preview_element(songs[i]['spotify'], (Y_START + (songs.length * Y_MARGINS)), load=false)
            container.appendChild(preview_object.preview)
            songs[i]['element'] = preview_object
        }
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

function remove(id){
    if (phase%2 === 0) {
        for (let i = 0; i < songs.length; i++) {
            if (songs[i]['spotify'] === id) {
                let element = songs[i]['element']['preview']

                document.getElementById('preview_container').removeChild(element)
                rec_lists[phase].splice(i, 1)
                songs = rec_lists[curr_list]
                known_songs += 1

                if (NUM_RANKS > Object.keys(songs).length) {
                    NUM_RANKS--
                }

                let button = document.getElementById('next_page')
                button.style.top = (Y_START + (NUM_RANKS * Y_MARGINS)).toString().concat('px')
            }
        }
        update_display(songs)
    }
    else{
        let node = document.getElementById('preview_container')
        node.innerHTML = ""
        init()
    }
}

//MAKE SURE DATA IS SAVED HERE
function next_page(){
    //Clear the DOM of elements
    let node = document.getElementById('preview_container')
    node.innerHTML = ""

    //Move from list to single items
    if (phase%2 === 0){
        //phase == 2
        curr_list += 1
        init()
    }
    //Move from single items
    else{
        //To list
        if (completion_count >= COMPLETION_REQ-1 || rec_lists[phase].length < 1){
            console.log("moving on")
            phase += 1
            init()
        }
        //To next single item
        else{
            completion_count += 1
            init()
        }
    }
}

function loading(){
    document.getElementById('loading').style.display = "flex";
    document.getElementById('content').style.display = "none";
}

function not_loading(){
    document.getElementById('loading').style.display = "none";
    document.getElementById('content').style.display = "flex";

}