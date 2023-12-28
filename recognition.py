import face_recognition
import io
import requests
import os
import statistics
import pickle

FILE_NAME_ENCODING_SET = "encoding_set.dat"

def load_image(url):
    online_file = io.BytesIO(requests.get(url).content)
    return face_recognition.load_image_file(online_file)

def get_face_locations(image):
    return face_recognition.face_locations(image)

def load_single_encoding_from_url(url):
    image = load_image(url)
    face_locations = get_face_locations(image)

    if len(face_locations) == 1:
        # set large jitters to get accurate encodings as source data
        encoding = face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=50)[0]
        print(f'encoding= {encoding}')
        return encoding
    else:
        return None

def get_encodings(file_name):
    print(f'loading file= {file_name}')
    encodings = []

    with open(file_name, 'r') as file:

        for line in file:
            print(f'loading url= {line}')
            encoding = load_single_encoding_from_url(line)
            if encoding is not None:
                encodings.append(encoding)

    return encodings

def get_encoding_set():
    encoding_set = {}

    files = os.listdir('source_data/')

    # make key value pair of source data file name and encoding value
    # single file can contain multiple URLs as source data
    for file in files:
        encoding_set[file] = get_encodings('source_data/' + file)

    return encoding_set

def save_encoding_set():
    encoding_set = get_encoding_set()
    
    with open(FILE_NAME_ENCODING_SET, 'wb') as file:
        pickle.dump(encoding_set, file)

def load_encoding_set():
    retVal = None
    with open(FILE_NAME_ENCODING_SET, 'rb') as file:
        retVal = pickle.load(file)

    return retVal

def test_image(url, encoding_set, tolerance):
    retVal = None

    test_image = load_image(url)
    face_locations = get_face_locations(test_image)
    test_encodings = face_recognition.face_encodings(test_image, known_face_locations=face_locations, model="large")

    for x in range(len(face_locations)):
        test_encoding = test_encodings[x]
        print(f'face location= {face_locations[x]}')

        for encoding_key in encoding_set:
            results = face_recognition.face_distance(encoding_set[encoding_key], test_encoding)
            mean = statistics.mean(results)

            print(f'key={encoding_key} mean={mean}')

            if mean < tolerance:
                retVal = encoding_key

    return retVal

def find(url, tolerance=0.35):
    retVal = None

    encoding_set = load_encoding_set()

    if encoding_set is not None:
        retVal = test_image(url, encoding_set, tolerance)
    
    return retVal
