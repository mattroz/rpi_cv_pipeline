import io
import os
import socket
import struct
import cv2
import numpy as np
import time

import face_recognition

from argparse import ArgumentParser
from PIL import Image


RESIZE_FACTOR = 2
FRAME_SKIP_RATIO = 2


def get_face_encodings(known_face_dir='./faces'):

    if not os.path.isdir(known_face_dir):
        raise FileNotFoundError

    known_face_encodings = []
    known_face_names = []

    for face_dir in os.listdir(known_face_dir):
        if not os.path.isdir(os.path.join(known_face_dir, face_dir)):
            continue
        
        print("Parsing face: ", face_dir)
        known_face_names.append(face_dir)
        face_encodings = np.zeros((128,1))

        for file in os.listdir(os.path.join(known_face_dir, face_dir)):
            if not (file.endswith('jpeg') or file.endswith('jpg')):
                continue
            print("\t", file)
            image = face_recognition.load_image_file(os.path.join(known_face_dir, face_dir, file))
            encoding = face_recognition.face_encodings(image, num_jitters=5)
            
            if encoding == []:
                continue

            face_encoding = encoding[0]
            face_encoding = np.reshape(face_encoding, (-1,1))
            face_encodings = np.concatenate((face_encodings, face_encoding), axis=1)

        known_face_encoding = list(np.mean(face_encodings[:, 1:], axis=1))
        known_face_encodings.append(known_face_encoding)

    print("Found {0} faces".format(len(known_face_encodings)))

    return known_face_encodings


def get_detection(frame, process_this_frame, face_attributes):
    # Grab a single frame of video
    #ret, frame = video_capture.read()

    face_locations, face_encodings, face_names = face_attributes

    # Only process every other frame of video to save time
    if process_this_frame == 0:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=1./RESIZE_FACTOR, fy=1./RESIZE_FACTOR)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            print(matches)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= RESIZE_FACTOR
        right *= RESIZE_FACTOR
        bottom *= RESIZE_FACTOR
        left *= RESIZE_FACTOR

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, [face_locations, face_encodings, face_names]


def main():

    args = ArgumentParser()
    args.add_argument('--faces_dir', type=str, default='./faces')
    args = args.parse_args()

    process_this_frame = 0

    face_locations = []
    face_names = []
    face_encodings = get_face_encodings(args.faces_dir)

    face_attributes = [face_locations, face_encodings, face_names]

    print('Ready to receive...')
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 8001))
    server_socket.listen(0)
    connection = server_socket.accept()[0].makefile('rb')

    try:
        while True:
            # Read the length of the image as a 32-bit unsigned int. If the
            # length is zero, quit the loop
            start = time.time()
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            # Construct a stream to hold the image data and read the image
            # data from the connection
            image_stream = io.BytesIO()
            image_stream.write(connection.read(image_len))
            # Rewind the stream, open it as an image with PIL and do some
            # processing on it
            image_stream.seek(0)
            #image = Image.open(image_stream)
            image = cv2.imdecode(np.fromstring(image_stream.read(), np.uint8), 1)
            # image = cv2.resize(image,(800,600), interpolation=cv2.INTER_CUBIC)
            image, face_attributes = get_detection(image, process_this_frame, face_attributes)
            process_this_frame = process_this_frame + 1 if process_this_frame < FRAME_SKIP_RATIO else 0

            cv2.imshow('frame', image)
            if cv2.waitKey(1) and 0xFF==ord('q'):
                break
    finally:
        connection.close()
        server_socket.close()


if __name__ == "__main__":
    main()
