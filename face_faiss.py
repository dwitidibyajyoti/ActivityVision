# face_faiss.py

import os
import numpy as np
import faiss
import face_recognition

ENCODINGS_FILE = "face_index.index"
NAMES_FILE = "face_names.npy"


def load_face_data(known_faces_folder="known_faces"):
    encodings = []
    names = []

    for filename in os.listdir(known_faces_folder):
        path = os.path.join(known_faces_folder, filename)
        image = face_recognition.load_image_file(path)
        face_enc = face_recognition.face_encodings(image)
        if face_enc:
            encodings.append(face_enc[0])
            names.append(os.path.splitext(filename)[0])

    return np.array(encodings).astype("float32"), names


def build_faiss_index(encodings):
    index = faiss.IndexFlatL2(128)
    index.add(encodings)
    return index


def save_index(index, names):
    faiss.write_index(index, ENCODINGS_FILE)
    np.save(NAMES_FILE, names)


def load_index():
    index = faiss.read_index(ENCODINGS_FILE)
    names = np.load(NAMES_FILE, allow_pickle=True)
    return index, names


def match_face(encoding, index, names, threshold=0.6):
    face = np.array([encoding]).astype('float32')
    distance, result = index.search(face, 1)
    if distance[0][0] < threshold:
        return names[result[0][0]]
    return "Unknown"
