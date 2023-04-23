"""
Person of Interest : The Machine

By Jo-dan
"""
import csv
import logging
import os
import queue
import sys
import threading
import time
import timeit

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from deepface.commons import functions
from PIL import Image

# my functions
import faceframes
from voicecontrol import get_mp3, get_nato, get_speech

# ==============================OPTIONS====================================== #
# =========================================================================== #
# webcam number
Camera_Number = input("Camera Number (Usually 0) >>> ")
vc = cv2.VideoCapture(int(Camera_Number))
# vc = cv2.VideoCapture("spedup.mp4")

# paths
face_database = 'facebase'

log_file = 'Machine_log.log'
subject_types = ['ADMIN', 'ANALOG', 'THREAT', 'UNKNOWN', 'USER']

# image borders
top_border = 150
side_border = 250

# colours
admin_colour = (255, 000, 000)
analog_colour = admin_colour
user_colour = (58, 238, 247)
unknown_colour = (000, 000, 255)
threat_colour = (000, 000, 255)
back_colour = (255, 255, 255)

# font of text on video
font = cv2.FONT_HERSHEY_SIMPLEX

# =========================================================================== #
# =========================================================================== #

# Set Print to flush


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)


# load subject database
logging.basicConfig(filename=log_file, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Initialised')

with open('subjects.csv', "r") as subjects:
    reader = csv.reader(subjects)
    subject_name = []
    subject_type = []
    for row in reader:
        if len(row) == 0 or len(row[0]) == 0:
            continue
        subject_name.append(row[1])
        if len(row[2]) == 0 or row[2].upper() not in subject_types:
            subject_type.append('UNKNOWN')
        else:
            subject_type.append(row[2])
    subject_name = [x.upper() for x in subject_name]
    subject_type = [x.upper() for x in subject_type]
    logging.info('CSV Read')
    subject_type[0] = "UNKNOWN"
    subject_name[0] = "UNKNOWN"


def rewrite_csv():
    """updates csv using subject_name and subject_type"""
    with open('subjects.csv', "w", newline='') as subjects:
        writer = csv.writer(subjects)
        for x in range(len(subject_type)):
            if x == 0:
                writer.writerow(['Subject No.', 'Name',
                                 'Type (ADMIN/USER/THREAT/ANALOG/UNKNOWN)'])
            else:
                row = [x, subject_name[x], subject_type[x]]
                writer.writerow(row)


# shape_type = raw_input("(b)oxes, (c)circles, poi (o)verlay,\
# samaritan (so)overlay, (p)oi or (s)amaritan? >>> ")
shape_type = 'o'
logging.info('   Run in "{}" Mode. \n'.format(shape_type))

if shape_type == 'p' or shape_type == 'o':
    admin_colour = (58, 238, 247)
    analog_colour = (58, 238, 247)
    user_colour = (243, 124, 13)
    unknown_colour = (254, 254, 254)
    threat_colour = (000, 000, 255)
    back_colour = (000, 000, 000)

q = queue.Queue()
q2 = queue.Queue()

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def facerec():
    """ Face recognition and video stream"""

    model_name = "VGG-Face"
    detector_backend = "opencv"
    distance_metric = "cosine"
    pivot_img_size = 112  # face recognition result image

    # find custom values for this input set
    target_size = functions.find_target_size(model_name=model_name)

    nbr_replacement = []
    nbr_old = [-1]
    nbr_predicted = 0
    display_infobox = True
    display_status = False
    present = 'unknown'
    exitprog = False
    accesstext = False
    starttime = int(timeit.default_timer())

    while True:
        # read frame by frame
        ret, frame_nobord = vc.read()

        try:
            frame = cv2.copyMakeBorder(frame_nobord, top_border, top_border,
                                       side_border, side_border, cv2.BORDER_CONSTANT,
                                       (0, 0, 0, 0))
        except:
            print(
                "No camera stream found, exit the program and try another camera number")
            break

        try:
            resolution_x = frame_nobord.shape[1]
        except:
            print("Error: Invalid Camera selection, try a different number")
            quit()
        resolution_y = frame_nobord.shape[0]

        admin_present = False
        user_present = False
        unknown_present = False
        threat_present = False
        analog_present = False

        try:
            # just extract the regions to highlight in webcam
            face_objs = DeepFace.extract_faces(
                img_path=frame_nobord,
                target_size=target_size,
                detector_backend=detector_backend,
                enforce_detection=False,
                grayscale=True
            )
            faces = []
            for face_obj in face_objs:
                facial_area = face_obj["facial_area"]
                if facial_area["w"] < (resolution_x - 5):
                    faces.append(
                        (
                            facial_area["x"],
                            facial_area["y"],
                            facial_area["w"],
                            facial_area["h"],
                        )
                    )
        except:  # to avoid exception if no face detected
            faces = []

        for (x, y, w, h) in faces:
            if w < 130:  # discard small detected faces
                continue

            detected_face = frame_nobord[int(y): int(
                y + h), int(x): int(x + w)]  # crop detected face

            dfs = DeepFace.find(
                img_path=detected_face,
                db_path=face_database,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
                silent=True,
            )

            nbr_predicted = 0
            confidence = 0

            if len(dfs) > 0:
                # directly access 1st item because custom face is extracted already
                df = dfs[0]

                if df.shape[0] > 0:
                    candidate = df.iloc[0]
                    label = candidate["identity"]
                    # --------------------
                    label = os.path.normpath(label).split(os.path.sep)
                    nbr_predicted = int(label[-2])
                    confidence = candidate[f"{model_name}_{distance_metric}"]

                # else:
                #    continue

            x = x + side_border
            y = y + top_border

            # strings for stream
#            subtxt = "Subject: {}".format(nbr_predicted)
#            nametxt = "Name: {}".format(subject_name[nbr_predicted])
#            typetxt = "Type: {}".format(subject_type[nbr_predicted])

            # Text on stream
            if subject_type[nbr_predicted] == 'ADMIN':
                all_colour = admin_colour
                admin_present = True
            elif subject_type[nbr_predicted] == 'USER':
                all_colour = user_colour
                user_present = True
            elif subject_type[nbr_predicted] == 'UNKNOWN':
                all_colour = unknown_colour
                unknown_present = True
            elif subject_type[nbr_predicted] == "THREAT":
                all_colour = threat_colour
                threat_present = True
            elif subject_type[nbr_predicted] == "ANALOG":
                all_colour = analog_colour
                analog_present = True

            if shape_type == 'o':
                frame = faceframes.poi_image(frame, x, y, w, h,
                                             subject_type[nbr_predicted])
                if display_infobox:
                    frame = faceframes.poi_infobox(frame, x+w+30, y+int(h*.5-50), nbr_predicted,
                                                   subject_name[nbr_predicted], subject_type[nbr_predicted])


#                subco = (x-20, y+h+45)
#                nameco = (x-20, y+h+70)
#                typeco = (x-20, y+h+95)
#            elif shape_type == 'c':
#                cv2.circle(frame, (x+int(round(.5*w)), y+int(round(.5*h))),
#                           int(round(.6*h)), all_colour, 4)
#                subco = (x+w+30, y+int(round(.5*h))-25)
#                nameco = (x+w+30, y+int(round(.5*h)))
#                typeco = (x+w+30, y+int(round(.5*h))+25)
#            elif shape_type == 'p':
#                faceframes.poi_box(frame, x, y, w, h,
#                                   subject_type[nbr_predicted])
#                subco = (x, y+h+25)
#                nameco = (x, y+h+50)
#                typeco = (x, y+h+75)
#            elif shape_type == 's':
#                faceframes.sam_circle(frame, x, y, w, h,
#                                      subject_type[nbr_predicted])
#                subco = (x+w+30, y+int(round(.5*h))-25)
#                nameco = (x+w+30, y+int(round(.5*h)))
#                typeco = (x+w+30, y+int(round(.5*h))+25)
#
#            elif shape_type == 'so':
#                frame = faceframes.samaritan_image(frame, x, y, w, h,
#                                                   subject_type[nbr_predicted])
#                subco = (x+w+30, y+int(round(.5*h))-25)
#                nameco = (x+w+30, y+int(round(.5*h)))
#                typeco = (x+w+30, y+int(round(.5*h))+25)
#            else:
#                cv2.rectangle(frame, (x, y), (x+w, y+h), all_colour, 2)
#                subco = (x, y+h+25)
#                nameco = (x, y+h+50)
#                typeco = (x, y+h+75)
#            if not display_infobox:
#                cv2.putText(frame, subtxt, subco, font, .7, back_colour, 3)
#                cv2.putText(frame, subtxt, subco, font, .7, all_colour, 2)
#                cv2.putText(frame, nametxt, nameco, font, .7, back_colour, 3)
#                cv2.putText(frame, nametxt, nameco, font, .7, all_colour, 2)
#                cv2.putText(frame, typetxt, typeco, font, .7, back_colour, 3)
#                cv2.putText(frame, typetxt, typeco, font, .7, all_colour, 2)

            if nbr_predicted not in nbr_old:
                if nbr_predicted != 0:
                    # print "Recognized as {} ({}). (Confidence : {})".format(nbr_predicted,
                    #                                                        subject_name[nbr_predicted], conf)
                    #                data={"value1":subject_type[nbr_predicted],
                    #                      "value2":subject_name[nbr_predicted],
                    #                      "value3":str(Camera_Number)})
                    logging.info('      Subject {} recognised:  {} \n'.format(nbr_predicted,
                                                                              subject_type[nbr_predicted]))
                else:
                    # print "Unrecognised face"
                    logging.info('      Unrecognised face detected\n')

                # recognp = normal_subject_path(nbr_predicted)
                # cv2.imshow("Recognised as...", recognp)
                # oldnp = normal_subject_path(nbr_old)
                # cv2.imshow("Previous", oldnp)
                nbr_replacement.append(nbr_predicted)
                nbr_old = list(nbr_replacement)
        if len(nbr_old) != 0 and len(faces) == 0:
            # print 'No face in frame.'
            del nbr_old[:]
        del nbr_replacement[:]

        if threat_present:
            if accesstext:
                cv2.putText(frame, 'THREAT DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'THREAT DETECTED', (5, 25),
                            font, 1, threat_colour, 2)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, threat_colour, 2)
            present = 'threat'
        elif analog_present:
            if accesstext:
                cv2.putText(frame, 'ANALOG INTERFACE DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ANALOG INTERFACE DETECTED', (5, 25),
                            font, 1, analog_colour, 2)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, analog_colour, 2)
            present = 'analog'
        elif admin_present:
            if accesstext:
                cv2.putText(frame, 'ADMIN DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ADMIN DETECTED', (5, 25),
                            font, 1, admin_colour, 2)
                cv2.putText(frame, 'ACCESS: GRANTED', (5, 55),
                            font, 1, admin_colour, 2)
            present = 'admin'
        elif user_present:
            if accesstext:
                cv2.putText(frame, 'USER DETECTED', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: RESTRICTED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'USER DETECTED', (5, 25),
                            font, 1, user_colour, 2)
                cv2.putText(frame, 'ACCESS: RESTRICTED', (5, 55),
                            font, 1, user_colour, 2)
            present = 'user'
        elif unknown_present:
            if accesstext:
                cv2.putText(frame, 'UNKNOWN USER', (5, 25),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, back_colour, 5)
                cv2.putText(frame, 'UNKNOWN USER', (5, 25),
                            font, 1, unknown_colour, 2)
                cv2.putText(frame, 'ACCESS: DENIED', (5, 55),
                            font, 1, unknown_colour, 2)
            present = 'unknown'

        vcheight, vcwidth = frame.shape[:2]
        cv2.putText(frame, 'Camera ' + str(Camera_Number),
                    (0, vcheight - 10), font, 1, (0, 0, 0), 4)
        cv2.putText(frame, 'Camera ' + str(Camera_Number),
                    (0, vcheight - 10), font, 1, (255, 255, 255), 1)

        stoptime = int(timeit.default_timer())
        uptimesec = stoptime - starttime
        if uptimesec > 59:
            mins, secs = divmod(round(uptimesec), 60)
            if mins > 59:
                hrs, mins = divmod(mins, 60)
                if hrs >= 24:
                    days, hrs = divmod(hrs, 24)
                    if days != 1:
                        uptime = "{} DAYS, {} HOURS".format(
                            int(days), int(hrs))
                    else:
                        uptime = "1 DAY, {} HOURS".format(int(hrs))
                else:
                    if hrs != 1:
                        uptime = "{} HOURS, {} MINUTES".format(
                            int(hrs), int(mins))
                    else:
                        uptime = "1 HOUR, {} MINUTES".format(int(mins))
            else:
                if mins != 1:
                    uptime = "{} MINUTES, {} SECONDS".format(
                        int(mins), int(secs))
                else:
                    uptime = "1 MINUTE, {} SECONDS".format(int(secs))
        else:
            uptime = "{} SECONDS".format(int(uptimesec))

        if display_status:
            frame = faceframes.poi_statusbox(
                frame, 0, vcheight - 150, uptime, len(faces))

        q.put(present)
        if not q2.empty():
            queuein = q2.get(block=False)
            if queuein == 'info':
                if not display_infobox:
                    display_infobox = True
                else:
                    display_infobox = False
            elif queuein == 'status':
                if not display_status:
                    display_status = True
                else:
                    display_status = False

            elif queuein == 'exit':
                print('exiting')
                exitprog = True

        cv2.imshow('stream', frame)
        wait = cv2.waitKey(1)
        if wait == 27 or exitprog:
            vc.release()
            cv2.destroyAllWindows()
            break


def commands():
    """ Console and voice command system"""
    # commandlist = ['info', 'set', 'names', 'train', 'voice', 'exit']
    # asset_types = ['ADMIN', 'ANALOG', 'USER', 'UNKNOWN', 'THREAT']
    vocal_input = False
    print("What are your commands?")
    while True:
        if vocal_input:
            user_input = get_speech()
        else:
            user_input = input('>>> ')
        if not q.empty():
            while not q.empty:
                q.get()
            time.sleep(.001)
            present = q.get(block=False)
            if 'exit' in user_input:
                q2.put('exit')
                break
            if present == 'threat':
                print('Threat detected. Taking precautions. Shutdown imminent')
                q2.put('exit')
                break
            elif present == 'analog' or present == 'admin':
                if user_input == 'info':
                    q2.put('info')
                elif user_input == 'status':
                    q2.put('status')
                elif 'set' in user_input:
                    set_comm = user_input.replace('set ', "").split(' as ')
                    if set_comm[1].upper() in subject_types:
                        try:
                            int_set_comm = int(set_comm[0])
                            subject_type[int(set_comm[0])
                                         ] = set_comm[1].upper()
                            print("Subject {} ({}) set as {}".format(set_comm[0], subject_name[int_set_comm],
                                                                     subject_type[int(set_comm[0])]))
                        except ValueError:
                            try:
                                upper_name = set_comm[0].upper()
                                subject_type[subject_name.index(
                                    upper_name)] = set_comm[1].upper()
                                print("Subject {} ({}) set as {}".format(subject_name.index(upper_name),
                                                                         upper_name, subject_type[subject_name.index(upper_name)]))
                            except ValueError as e:
                                print(str(e))
                                print("Name not found")
                    else:
                        print("Invalid designation")
                elif 'names' in user_input:
                    if vocal_input:
                        namelist = ""
                        for name in subject_name[1:]:
                            namelist += get_nato(name) + ";"
                        get_mp3(namelist[:len(namelist) - 1])
                    else:
                        print(subject_name[1:])
                        # print str(subject_name[1:]).replace(',', ';').replace("[", "").replace("]","").replace("'","")
                elif 'voice' in user_input:
                    if not vocal_input:
                        get_mp3('Can you hear me?')
                        confirmation = get_speech()
                        for yes in ['yes', 'absolutely', 'yeah']:
                            if yes in confirmation:
                                vocal_input = True
                                get_mp3('good ; analog interface enabled')
                        if not vocal_input:
                            get_mp3(
                                'analog interface not detected ; voice commands disabled')
                    else:
                        get_mp3('analog interface disabled')
                        vocal_input = False
                else:
                    print('Unknown command')
            elif present == 'user':
                print("Unauthorized user or command unknown.")
            elif present == 'unknown':
                print('Unknown subject detected. Access Denied')
        else:
            print("No face detected")


recog = threading.Thread(target=facerec)
recog.setDaemon(True)
recog.start()
commands()
recog.join()
rewrite_csv()

# Delete the representation of the database to be rebuilt next run if changes are made
filelist = os.listdir(face_database)
for item in filelist:
    if item.endswith(".pkl"):
        os.remove(os.path.join(face_database, item))

timenow = time.strftime("%d/%m/%Y") + ' - ' + time.strftime("%I:%M:%S")
logging.info('Program Terminated at {}. \n'.format(timenow))
print('\n\n.......\nGoodbye \n.......')
time.sleep(2.5)
