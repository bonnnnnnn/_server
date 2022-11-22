from flask import Flask, request, jsonify
import werkzeug
import uuid

app = Flask(__name__)

@app.route('/upload', methods=["POST", "GET"])
def upload():
    if(request.method == "POST"):
        songname = request.args.get('songname')
        userName = request.args.get('userName')
        index = request.args.get('index')

        newfilename = userName+"-"+songname+"-"+str(uuid.uuid4())+".wav"

        vocalpath = "uploadSound/"+newfilename
        instrupath = "instrumentsfolder/"+songname+".wav"
        newfilepath = "completesong/"+newfilename
        
        jsonpath = "jsonfolder/"
        modelpath = "modelfolder/modeled2.h5"
        ori_filename = songname+".wav"

        soundFile = request.files['sound']
        user_filename = werkzeug.utils.secure_filename(soundFile.filename)
        soundFile.save("uploadSound/" + newfilename)

        mergesongandinstruments(vocalpath, instrupath, newfilepath)

        diff("original_vocal/" + ori_filename, "uploadSound/" + newfilename)
        stars = predict(jsonpath+newfilename[:-4]+".json", modelpath)

        upload2firebase(userName=userName, songname=songname, filename=newfilename, stars=stars, index=index)

        return jsonify({
            "message": "sound uploaded success"
        })
    # elif(request.method == "GET"):
    #     songname = request.args.get('songname')
    #     username = request.args.get('username')
    #     return jsonify({
    #         "songname": songname,
    #         "username": username
    #     })

import json
import numpy as np
from tensorflow import keras

def predict(jsonpath, modelpath):
    with open(jsonpath, "r") as fp:
        data = json.load(fp)
    mfcc = np.array(data["mfcc"])
    chroma = np.array(data["chroma"])
    zerocros = np.array(data["zerocros"])
    multi_cols = []
    for i in range(mfcc.shape[0]):
        join_cols = np.hstack((mfcc[i], chroma[i], zerocros[i]))
        multi_cols.append(join_cols.tolist())
    X = np.array(multi_cols)
    #X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
    #y = X[np.newaxis, ...]
    #X = X[np.newaxis, ...]
    model = keras.models.load_model(modelpath)
    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    print(X.shape)
    print(mfcc.shape)
    print(chroma.shape)
    print(zerocros.shape)
    prediction = model.predict(X)
    predicted = np.argmax(prediction, axis=1)
    print(predicted)
    sum = 0
    for i in predicted:
        sum = sum+i
    print(jsonpath[11:len(jsonpath)-5]+" = "+str(sum/X.shape[0]))
    return int(sum/X.shape[0])


import librosa
import math

def diff(file1path, file2path):
    data = {
        "mfcc" : [],
        "chroma" : [],
        "zerocros" : []
    }
    y1, sr1 = librosa.load(file1path)
    y2, sr2 = librosa.load(file2path) 

    TRACK_DURATION = int(min(librosa.get_duration(y2),librosa.get_duration(y1)))
    TRACK_SAMPLE = TRACK_DURATION*22050
    numsegment = math.ceil(TRACK_DURATION/4)
    num_mfcc_vectors_per_segment = math.ceil((5*22050) / 512)

    for i in range (numsegment) :

        start = i*22050*4
        stop = start+22050*5
        if stop > TRACK_SAMPLE :
            stop = TRACK_SAMPLE

        mfcc1 = librosa.feature.mfcc(y1[start:stop], 22050, n_mfcc=13, n_fft=2048, hop_length=512)
        chroma1 = librosa.feature.chroma_stft(y=y1[start:stop], sr=22050)
        zerocros1 = librosa.feature.zero_crossing_rate(y1[start:stop])

        mfcc2 = librosa.feature.mfcc(y2[start:stop], 22050, n_mfcc=13, n_fft=2048, hop_length=512)
        chroma2 = librosa.feature.chroma_stft(y=y2[start:stop], sr=22050)
        zerocros2 = librosa.feature.zero_crossing_rate(y2[start:stop])

        mfcc = mfcc1-mfcc2
        chroma = chroma1-chroma2
        zerocros = zerocros1-zerocros2

        mfcc = mfcc.T
        chroma = chroma.T
        zerocros = zerocros.T

        if len(mfcc) == num_mfcc_vectors_per_segment :
          data["mfcc"].append(mfcc.tolist())
          data["chroma"].append(chroma.tolist())
          data["zerocros"].append(zerocros.tolist())

        else :
          diff = 216 - len(mfcc)
          fix_mfcc = [0]*13
          fix_chroma = [0]*12
          fix_zerocros = [0]

          temp_data_mfcc = mfcc.tolist()
          temp_data_chroma = chroma.tolist()
          temp_data_zerocros = zerocros.tolist()

          for k in range(diff) :
            temp_data_mfcc.append(fix_mfcc)
            temp_data_chroma.append(fix_chroma)
            temp_data_zerocros.append(fix_zerocros)

          data["mfcc"].append(temp_data_mfcc)
          data["chroma"].append(temp_data_chroma)
          data["zerocros"].append(temp_data_zerocros)
    
    with open("jsonfolder/"+file2path[12:-4]+".json", "w") as fp:
        json.dump(data, fp, indent=4)

from firebase_admin import credentials
from firebase_admin import db, storage
import firebase_admin

import soundfile as sf
import datetime

def upload2firebase(filename, userName, songname, stars, index):

    newfilepath = "completesong/"+filename
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(newfilepath)
    blob.make_public()

    print("your file url", blob.public_url)

    ref = db.reference()

    songs_ref = ref.child('Songs')
    
    new_songs_ref = songs_ref.push()
    new_songs_ref.set({
        'title': songname,
        'stars': stars,
        'date': str(datetime.datetime.now()),
        'img': loadimg(int(index)),
        'song_url': blob.public_url,
        'userName': userName
    })

    user_ref = ref.child('Users').order_by_child('userName').equal_to(userName).get()
    key = ""
    for temp_key in user_ref :
        key = temp_key
    songs_count = int(len(songs_ref.order_by_child('userName').equal_to(userName).get())) + 1
    total_stars = ref.child('Users/'+key+'/stars').get()+stars
    user1_ref = ref.child('Users/'+key)
    user1_ref.update({
        'songs_count' : songs_count,
        'stars' : total_stars,
        'rating' : total_stars/songs_count
    })

    # users_ref = ref.child(songname+'/'+username+'/songlist')
    # users_ref.set({
    #     "7RING": {
    #         'star(s)': stars,
    #         'time': str(datetime.datetime.now()),
    #         'url': blob.public_url
    #     }
    # })

    # songs_ref = ref.child('songs/'+songname)
    # new_songs_ref = songs_ref.push()
    # new_songs_ref.set({
    #     'star(s)': stars,
    #     'time': str(datetime.datetime.now()),
    #     'url': blob.public_url
    # })

    # songs_ref = ref.child('Songs')
    # songs_ref.update({
    #     'count' : str(len(songs_ref.order_by_key().get()))
    # })
    # new_songs_ref = songs_ref.push()
    # new_songs_ref.set({
    #     'color': '',
    #     'date': datetime.datetime.now(),
    #     'description': '',
    #     'duration': '',
    #     'img': '',
    #     'song_url': '',
    #     'userName': ''
    # })
    
    # song_id = new_songs_ref.key

    # users_ref = ref.child('Users/'+"username"+"/songname")
    # new_users_ref = users_ref.push()
    # users_ref.update({
    #     'count' : str(len(users_ref.order_by_key().get()))
    # })
    # users_ref.push({
    #     "path":"songs/"+songname+"/"+song_id+"/url"
    # })

    # ref = db.reference('Songs/songname')
    # snapshot = ref.order_by_key().get()
    # for key, val in snapshot.items():
    #     print(val['star(s)'])

def loadimg(index):
    with open("songs_json.json", "r") as fp:
        data = json.load(fp)
    songs = data["Songs"]
    pics = songs[index]["img"]
    return pics

def mergesongandinstruments(vocalpath, instrupath, newfilepath):
    instru, sr = librosa.load(instrupath)
    vocal, sr = librosa.load(vocalpath)
    if(len(instru)>len(vocal)) :
        completesong = vocal + instru[:len(vocal)]
    else :
        completesong = vocal[:len(instru)] + instru
    sf.write(newfilepath, completesong, 22050)

if __name__ == "__main__":

    setting = {
        'databaseURL': 'https://karaoke-real-one-default-rtdb.asia-southeast1.firebasedatabase.app/',
        'storageBucket': 'karaoke-real-one.appspot.com'
    }
    cred = credentials.Certificate("karaoke-real-one-firebase-adminsdk-cktez-2ab5bdc0ad.json")
    firebase_admin.initialize_app(cred, setting)

    app.run(debug=True,host="0.0.0.0", port=4000)