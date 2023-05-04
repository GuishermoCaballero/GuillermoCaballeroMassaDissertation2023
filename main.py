import sys

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

from ultralytics import YOLO


from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette

import os

from typing import List
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import cv2
from sklearn.cluster import DBSCAN
import uuid

import streamlit as st
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from activity_recognition import *
from format_logs_functions import *

import random
import datetime

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]




# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
        detections: Detections,
        tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


def filter_by_class(classes, dictionary, results):
    results_final = []

    # Get id giving class name
    classes_ids = []
    for i in classes:
        classes_ids.append(list(dictionary.keys())[list(dictionary.values()).index(i)])

    #     print('Id de las clases elegidas:')
    #     print(classes_ids)

    results_classes = results.boxes.cls.cpu().numpy().astype(int)

    positions_to_delete = [i for i in range(len(results_classes)) if results_classes[i] not in classes_ids]

    #     print('Posiciones que no son esas clases:')
    #     print(positions_to_delete)

    xyxy = np.delete(results.boxes.xyxy.cpu().numpy(), positions_to_delete, 0)
    confidence = np.delete(results.boxes.conf.cpu().numpy(), positions_to_delete)
    class_id = np.delete(results.boxes.cls.cpu().numpy().astype(int), positions_to_delete)

    return [xyxy, confidence, class_id]



def get_cluster_labels(X, distance, min_samples):
    db = DBSCAN(eps=distance, min_samples=min_samples).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return labels, n_clusters_, n_noise_


def filter_results_by_tracker(results, class_id, tracker_id):
    final_results = []
    final_tracker_id = []
    final_classes = []

    for i in range(len(tracker_id)):
        if isinstance(tracker_id[i], (int, float)) and tracker_id[i] is not None:
            final_results.append(results[i])
            final_tracker_id.append(tracker_id[i])
            final_classes.append(class_id[i])

    return [final_results, final_tracker_id, final_classes]


def filter_results_by_class_id(results, class_id, tracker_id):
    final_results = []
    final_tracker_id = []
    final_classes = []

    for i in range(len(class_id)):
        if class_id[i] == 0 and isinstance(tracker_id[i], (int, float)) and tracker_id[i] is not None:
            final_results.append(results[i])
            final_tracker_id.append(tracker_id[i])
            final_classes.append(class_id[i])

    return [final_results, final_tracker_id, final_classes]


def detect_groups(frame, coordinates, tracker_id, moment):

    simple_coords = []

    for coordinate in coordinates:
        x = (coordinate[0] + coordinate[2]) / 2
        y = (coordinate[1] + coordinate[3]) / 2
        simple_coords.append([x, y])

    if not simple_coords:
        return frame

    labels, _, _ = get_cluster_labels(np.array(simple_coords), 100, 2)

    global last_frame
    global this_frame
    global ever_ran

    # if (moment == 0):
    if ever_ran == False:
        ever_ran = True

        last_frame = Record(tracker_id, labels)
        last_frame = remove_none_elements(last_frame)
        create_new_members_and_groups(last_frame)

    else:

        this_frame = Record(tracker_id, labels)
        this_frame = remove_none_elements(this_frame)
        # if len(last_frame.labels_array) > 1:
        check_changes(last_frame, this_frame)

    for idx, group in enumerate(groups):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, group.__str__(), (10, frame.shape[0] - 20 - (idx * 30)), font, 0.5, colors[idx], 1,
                    cv2.LINE_AA)

    for idx, coords in enumerate(simple_coords):
        frame = cv2.circle(frame, (int(coords[0]), int(coords[1])), radius=5, color=colors[labels[idx]], thickness=-1)

    cv2.putText(frame, f'Frame: {moment}', (10, 30), font, 0.5, (70, 235, 52), 1, cv2.LINE_AA)

    return frame


def check_changes(last_frame, this_frame):
    # Both Records get sorted to be compared later on
    if (last_frame.labels_array != [] and last_frame.trackers_array != []):
        if len(last_frame.labels_array) > 1:
            last_frame = sort_arrays(last_frame)

    if (this_frame.labels_array != [] and this_frame.trackers_array != []):
        if len(last_frame.labels_array) > 1:
            this_frame = sort_arrays(this_frame)

    new_elements = getNewElements(last_frame, this_frame)
    existing_elements = getExistingElements(last_frame, this_frame)
    missing_elements = getMissingElements(last_frame, this_frame)

    handle_new_elements(new_elements)
    handle_existing_elements(existing_elements)
    handle_missing_elements(missing_elements)

    delete_empty_groups()
    next_frame()


def write_log(text, frame, code):
    objeto = {"code": code, "message": text, "frame": frame, "frame_rate":frame_rate}
    store_object(objeto, video_name)

    f = open("Log.txt", "a")
    f.write(f"{text} - frame: {frame} \n")
    f.close()

    new_message = Message(text, frame)
    global messages
    messages.append(new_message)


class Record:
    def __init__(self, trackers_array, labels_array):
        self.trackers_array = trackers_array
        self.labels_array = labels_array

    def sort(self):
        self = sort_arrays(self)

    def __str__(self):
        return f"id: {self.trackers_array}. Label: {self.labels_array}."


class Message:
    def __init__(self, text, frame):
        self.text = text
        self.frame = frame

    def __str__(self):
        return f"{self.text}. Frame: {self.frame}."


class Member:
    def __init__(self, tracker_id, group, since_frame):
        self.tracker_id = tracker_id
        self.group = group
        self.since_frame = since_frame
        self.last_group = None

    def moveToGroup(self, new_group, since_frame):
        self.last_group = self.group
        self.group = new_group
        self.since_frame = since_frame

    def __str__(self):
        return f"{self.tracker_id}. In Group: {self.group}. SinceFrame: {self.since_frame}. LastGroup: {self.last_group}."


def create_member(tracker, group=None):
    new_member = Member(tracker, group, moment)
    members.append(new_member)

    log = f'NEW ELEMENT HAS BEEN SPOTED: {new_member.tracker_id}'
    write_log(log, moment, 0)
    return new_member


def findMemberById(id):
    result = None
    for member in members:
        if (member.tracker_id == id):
            result = member
    return result


def delete_member_by_id(member_id):
    global members

    for i in range(len(members)):
        if members[i].tracker_id == member_id:
            del members[i]
            log = f'MEMBER {member_id} IS NO LONGER DETECTED'
            write_log(log, moment,0)
            return True
    return False


# ================================================================================================================================================================ #
class Group:
    def __init__(self, members, name, label, frame_n):

        self.id = uuid.uuid4()
        self.name = name
        self.label = label
        self.members = members
        self.members_old = []
        self.members_new = []
        self.created_at_frame = frame_n

    def addMember(self, members):
        for member in members:
            self.members.append(member)

    def addNewMember(self, members):
        for member in members:
            self.members_new.append(member)

    def addOldMember(self, members):
        for member in members:
            self.members_old.append(member)

    def removeMember(self, members):
        for member in members:
            if (member in self.members):
                self.members.remove(member)

                log = f'MEMBER {member} HAS LEFT GROUP {self.label}'
                write_log(log, moment,0)

        if member in self.members_new:
            self.members_new.remove(member)

    def __str__(self):
        return f"Label: {self.label}. Members: {self.members}. Old: {self.members_old}. New: {self.members_new}"


#         return f"ID: {self.id}. {self.name}. Label: {self.label}. Members: {self.members}. OldMembers: {self.members_old}. NewMembers: {self.members_new}"


def create_group(label):
    new_group = Group([], f'Group {int(label)}', label, moment)
    groups.append(new_group)

    log = f'NEW GROUP HAS BEEN CREATED: {new_group.label}'
    write_log(log, moment,0)

    return new_group


def findGroupById(id):
    result = None
    for group in groups:
        if (str(group.id) == id):
            result = group

    return result


def findGroupByLabel(label):
    result = None
    for group in groups:
        if (group.label == label):
            result = group

    return result


def getAllGroupLabels():
    labels = []
    for group in groups:
        labels.append(group.label)

    return labels


# ======================================================================
def change_member_to_group(tracker_id, label_new_group):
    member = findMemberById(tracker_id)

    last_group = findGroupById(str(member.group))
    new_group = findGroupByLabel(label_new_group)

    last_group.removeMember([int(tracker_id)])
    new_group.addMember([tracker_id])

    # Checks if it was an old member, so it doesn´t put them in new members
    if (tracker_id not in new_group.members_old):
        new_group.addNewMember([tracker_id])

        if (label_new_group == -1):
            log = f'ELEMENT: {tracker_id} IT´S BY ITSELF'

        else:
            log = f'ELEMENT: {tracker_id} MOVED TO GROUP WITH LABEL: {label_new_group}'

        write_log(log, moment,0)

    else:
        log = f'ELEMENT: {tracker_id} WENT BACK TO OLD GROUP: {label_new_group}'
        write_log(log, moment,0)

    member.moveToGroup(new_group.id, moment)


def remove_none_elements(record):
    new_trackers_array = []
    new_labels_array = []

    for i in range(len(record.trackers_array)):
        if record.trackers_array[i] is not None:
            new_trackers_array.append(record.trackers_array[i])
            new_labels_array.append(record.labels_array[i])

    return Record(new_trackers_array, new_labels_array)


def getExistingElements(last_frame, this_frame):
    """
    Given two objects, returns an object with the existing trackers and their corresponding labels.
    """
    # Extract the trackers and labels from each object
    trackers1, labels1 = last_frame.trackers_array, last_frame.labels_array
    trackers2, labels2 = this_frame.trackers_array, this_frame.labels_array

    # Find the new trackers in obj2 that are in obj1
    new_trackers = []
    new_labels = []
    for i, tracker in enumerate(trackers2):
        if tracker in trackers1:
            new_trackers.append(tracker)
            new_labels.append(labels2[i])

    # Return an object with the new trackers and their corresponding labels
    new_record = Record(new_trackers, new_labels)
    return new_record


def getNewElements(last_frame, this_frame):
    """
    Given two objects, returns an object with the new trackers and their corresponding labels.
    """
    # Extract the trackers and labels from each object
    trackers1, labels1 = last_frame.trackers_array, last_frame.labels_array
    trackers2, labels2 = this_frame.trackers_array, this_frame.labels_array

    # Find the new trackers in obj2 that are not in obj1
    new_trackers = []
    new_labels = []
    for i, tracker in enumerate(trackers2):
        if tracker not in trackers1:
            new_trackers.append(tracker)
            new_labels.append(labels2[i])

    # Return an object with the new trackers and their corresponding labels
    new_record = Record(new_trackers, new_labels)
    return new_record


def getMissingElements(last_frame, this_frame):
    last_frame_trackers = last_frame.trackers_array
    current_trackers = this_frame.trackers_array

    missing_trackers = find_missing_ids(last_frame_trackers, current_trackers)

    missing_trackers_array = []
    missing_labels_array = []

    for index, tracker in enumerate(last_frame.trackers_array):
        if (tracker in missing_trackers):
            missing_trackers_array.append(last_frame.trackers_array[index])
            missing_labels_array.append(last_frame.labels_array[index])

    new_record = Record(missing_trackers_array, missing_labels_array)
    return new_record


def handle_new_elements(new_elements):
    all_labels = getAllGroupLabels()

    new_trackers, new_labels = new_elements.trackers_array, new_elements.labels_array

    just_created = []

    for index, label in enumerate(new_labels):

        if (label in all_labels):

            group = findGroupByLabel(label)

            new_member = create_member(new_trackers[index], group.id)

            change_member_to_group(new_member.tracker_id, group.label)
            # if(label in just_created):
            #   group.addOldMember([new_member.tracker_id])
            # else:
            #   group.addNewMember([new_member.tracker_id])

        else:

            group = create_group(label)

            new_member = create_member(new_trackers[index], group.id)

            change_member_to_group(new_member.tracker_id, group.label)
            just_created.append(label)

        all_labels = getAllGroupLabels()


def handle_missing_elements(missing_elements):
    missing_trackers, missing_labels = missing_elements.trackers_array, missing_elements.labels_array

    for index, tracker in enumerate(missing_trackers):
        member = findMemberById(tracker)
        group = findGroupById(str(member.group))
        group.removeMember([member.tracker_id])
        delete_member_by_id(member.tracker_id)


def handle_existing_elements(existing_elements):
    global last_frame
    if len(last_frame.labels_array) > 1:
        last_frame = sort_arrays(last_frame)

    last_frame_trackers, last_frame_labels = last_frame.trackers_array, last_frame.labels_array
    existing_trackers, existing_labels = existing_elements.trackers_array, existing_elements.labels_array

    all_labels = getAllGroupLabels()
    for index, label in enumerate(existing_labels):
        if (label not in all_labels):
            create_group(label)
            all_labels = getAllGroupLabels()

    old_existing_trackers = []
    old_existing_labels = []
    for i, tracker in enumerate(last_frame_trackers):
        if tracker in existing_trackers:
            old_existing_trackers.append(tracker)
            old_existing_labels.append(last_frame_labels[i])

    old_existing_elements = Record(old_existing_trackers, old_existing_labels)
    regroupExistingElements(old_existing_elements, existing_elements)


def regroupExistingElements(last_frame, this_frame):
    changed_indexes = find_indexes_of_different_values(last_frame.labels_array, this_frame.labels_array)

    changed_trackers = []
    for index in changed_indexes:
        changed_trackers.append(this_frame.trackers_array[int(index)])

    for changed_index in changed_indexes:
        change_member_to_group(this_frame.trackers_array[changed_index], this_frame.labels_array[changed_index])


def next_frame():
    global last_frame
    last_frame = this_frame


def delete_empty_groups():
    global groups

    for group in groups:
        if (len(group.members) == 0 and group.label != -1):
            # print('Group with label: ', group.label, ' got empty, sad and deleted')
            log = 'Group with label: ', group.label, ' got empty, sad and deleted'
            write_log(log, moment,0)
            groups.remove(group)
        elif (len(group.members) < 2 and group.label != -1):
            print('ERROR OR GROUP with label: ', group.label)


def sort_arrays(object):
    # Zip the two arrays together

    combined = list(zip(object.trackers_array, object.labels_array))

    # Sort the combined list based on the values in the first array
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Unzip the sorted list into two separate arrays
    sorted_ids, sorted_labels = zip(*sorted_combined)

    return Record(sorted_ids, sorted_labels)


def find_missing_ids(first_ids, second_ids):
    missing_ids = []
    for id in first_ids:
        if id not in second_ids:
            missing_ids.append(id)
    return missing_ids


def find_indexes_of_different_values(arr1, arr2):
    # Create an empty list to store the indexes of different values
    diff_indexes = []

    # Iterate through the arrays and check for differences
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            diff_indexes.append(i)

    return diff_indexes


# Function that runs at the beggining
def create_new_members_and_groups(record):
    tracker_id = record.trackers_array
    labels = record.labels_array

    # CREATED A MEMBER PER TRACKER AND A GROUP PER LABEL
    for index, tracker in enumerate(tracker_id):
        create_member(tracker)

    import numpy as np
    for idx, label in enumerate(np.unique(labels)):
        if (label != -1):  # As group -1 is created by default
            create_group(label)

    # AT THE BEGGINING THEY MUST BE POPULATED WITH THE POSITIONS
    for idx, label in enumerate(np.unique(labels)):
        group = findGroupByLabel(label)
        indexes_in_group = [i for i, j in enumerate(labels) if j == label]

        for index_in_group in indexes_in_group:
            id = tracker_id[index_in_group]
            group.addMember([id])
            group.addOldMember([id])
            member = findMemberById(id)
            member.moveToGroup(group.id, moment)


def show_messages(frame):
    img = frame

    # Define margin parameters
    margin_width = 450  # in pixels
    margin_color = (255, 255, 255)  # white color

    # Create margin by adding a blank white image on the right
    margin = np.zeros((img.shape[0], margin_width, 3), dtype=np.uint8)
    margin[:] = margin_color
    img_with_margin = np.concatenate((img, margin), axis=1)

    # Define array of messages to display on the margin

    global messages
    messages_copy = messages.copy()
    messages_copy.reverse()

    # Display messages using putText()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (0, 0, 0)  # black color
    line_spacing = 20  # in pixels

    y = 20  # initial y-coordinate for first message
    for message in messages_copy:
        cv2.putText(img_with_margin, message.__str__(), (img.shape[1] + 10, y),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_spacing

    return img_with_margin


def write_log_for_activities(detect_activities,activity_dictionary, moment):
    for key in activity_dictionary:
        if(activity_dictionary[key] in detect_activities):
            write_log(f"#{key} was detected {activity_dictionary[key]}", moment, 2)


def write_log_for_objects(detect_objects, clases_detected,  dictionary,moment):
    for clas in clases_detected:
        class_name = dictionary[clas]
        if(class_name in detect_objects):
            write_log(f"#{class_name} was detected ", moment, 1)


def create_pickle_file(filename, data):
    folder_name = "logs"
    filename = filename + ".pickle"
    file_path = os.path.join(folder_name, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {filename} already exists and has been deleted.")

    with open(file_path, 'wb') as f:
        try:
            pickle.dump(data, f)
            print(f"File {filename} has been created successfully.")
        except:
            raise ValueError(f"Unable to dump data into file {filename}")


def store_object(obj, filename):
    log_dir = "logs"
    filename = filename + ".pickle"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    filepath = os.path.join(log_dir, filename)

    # If the file already exists, load the existing object(s) from it
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            existing_objs = pickle.load(f)
    else:
        existing_objs = []

    # Append the new object to the existing ones
    existing_objs.append(obj)

    # Store the updated list of objects back in the file
    with open(filepath, 'wb') as f:
        pickle.dump(existing_objs, f)

def detect_activity(tracker_id, xyxy, clases, shaped, keypoints_with_scores, confidence):

    global model_activity
    global actions

    activities_labels = []

    bodies = filter_bodies(shaped, confidence)
    keypoints_with_scores = filter_bodies(keypoints_with_scores, confidence)

    tracker_id, xyxy = filter_arrays_for_poses(tracker_id, xyxy, clases)

    global sequences

    box_index_body_dict = calculate_body_inside_box(bodies, xyxy, tracker_id, confidence)

    dictionary_to_return = {}

    for key in box_index_body_dict:

        if key not in sequences and key != None:
            sequences[key] = []

        elif ((key in sequences) and (key not in box_index_body_dict)):
            del sequences[key]

        if key not in dictionary_to_return and key != None:
            dictionary_to_return[key] = ""

        points = keypoints_with_scores[box_index_body_dict[key]].reshape((1, 17, 3))
        points = points.reshape((51))

        if key is not None:

            sequences[key].insert(0, points)

            if len(sequences[key]) > 30:
                sequences[key] = sequences[key][:30]

            if (len(sequences[key]) == 30):
                res = model_activity.predict(np.expand_dims(sequences[key], axis=0))[0]
                activities_labels.append(actions[np.argmax(res)])
                dictionary_to_return[key] = actions[np.argmax(res)]

    return dictionary_to_return


def run_analyzer( video_title, group_tracking, activity_recognition, detect_dangerous_objects, draw_skeletons, write_logs ):


    global video_name
    video_name = video_title
    progress_bar = st.progress(0)

    # global moment

    activities_to_detect = ['box']
    objects_to_detect = ['knife']

    create_pickle_file(video_title, [])
    # Load ByteTrack
    @dataclass(frozen=True)
    class BYTETrackerArgs:
        track_thresh: float = 0.25
        track_buffer: int = 30
        match_thresh: float = 0.8
        aspect_ratio_thresh: float = 3.0
        min_box_area: float = 1.0
        mot20: bool = False

    # Load YOLO
    model = YOLO('yolov8x.pt')
    model.fuse()

    # Load Movenet
    model_movenet = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    movenet = model_movenet.signatures['serving_default']

    #Model that predicts activities
    # DATA_PATH = os.path.join('ACTIONS_DATA')
    global actions
    actions = ['hello', 'box', 'stand']

    actions = np.array(actions)

    global model_activity
    model_activity = Sequential()
    model_activity.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 51)))
    model_activity.add(LSTM(128, return_sequences=True, activation='relu'))
    model_activity.add(LSTM(64, return_sequences=False, activation='relu'))
    model_activity.add(Dense(64, activation='relu'))
    model_activity.add(Dense(32, activation='relu'))
    model_activity.add(Dense(actions.shape[0], activation="softmax"))

    model_activity.load_weights("action_new.h5")

    CLASS_NAMES_DICT = model.model.names

    byte_tracker = BYTETracker(BYTETrackerArgs())

    video_path = os.path.join('.', 'data', f'{video_title}.mp4')
    video_out_path = os.path.join('.', 'out', f'{video_title}-out.mp4')

    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=1, text_scale=1)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_cod = cv2.VideoWriter_fourcc(*'H264')



    global frame_rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    cap_out = cv2.VideoWriter(video_out_path, vid_cod, frame_rate,
                              (frame.shape[1] + 450, frame.shape[0]))

    global ever_ran
    ever_ran = False

    global moment
    moment = 0  # To know the exact moment of events
    global groups
    groups = []
    global members
    members = []
    global messages
    messages = []
    global sequences
    sequences = {}
    global sequence
    sequence = []

    # Elements without Group will be a group too:
    lonely_group = Group([], 'Loners', -1, 0)
    # global groups
    groups.append(lonely_group)

    while ret:

        # Yolo Predictions
        results = model(frame)[0]

        results_filtered = filter_by_class( ["person", "cup", 'knife'], model.model.names, results)

        detections = Detections(
            xyxy=results_filtered[0],
            confidence=results_filtered[1],
            class_id=results_filtered[2]
        )

        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        if detect_dangerous_objects:
            write_log_for_objects(objects_to_detect, results_filtered[2], model.model.names,moment)

        # MoveNet Predictions
        # Resize
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160, 256)
        input_img = tf.cast(img, dtype=tf.int32)

        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))

        if draw_skeletons:
            loop_thought_people(frame, keypoints_with_scores, EDGES, 0.3, moment)


        new_tracker_id = []
        new_results_filtered0 = []
        new_results_filtered2 = []

        for i in range(len(tracker_id)):
            if isinstance(tracker_id[i], (int, float)):
                new_tracker_id.append(tracker_id[i])
                new_results_filtered0.append(results_filtered[0][i])
                new_results_filtered2.append(results_filtered[2][i])

        activity_dictionary = {}
        if activity_recognition:
            activity_dictionary = detect_activity(new_tracker_id, new_results_filtered0, new_results_filtered2, shaped, keypoints_with_scores, 0.3)

            write_log_for_activities(activities_to_detect, activity_dictionary, moment)


        if tracker_id != []:
            labels = [
                #         f"#{tracker_id}{CLASS_NAMES_DICT[class_id]}"
                f"#{tracker_id}{CLASS_NAMES_DICT[class_id]}" + f" ({activity_dictionary[tracker_id]})" if tracker_id in activity_dictionary else f"#{tracker_id}{CLASS_NAMES_DICT[class_id]}"
                for _, confidence, class_id, tracker_id
                in detections
            ]


            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)


        if tracker_id != []:
            filtered_by_class = filter_results_by_class_id(results_filtered[0], results_filtered[2], tracker_id)
            if group_tracking:
                frame = detect_groups(frame, filtered_by_class[0], filtered_by_class[1], moment)

        if write_logs:
            frame = show_messages(frame)

        cv2.imshow('Detections', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        cv2.waitKey(25)

        cap_out.write(frame)
        ret, frame = cap.read()

        moment += 1
        progress_bar.progress(int(moment/total_frames*100))

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # Global variables
    HOME = os.getcwd()
    sys.path.append(f"{HOME}/ByteTrack")



    # Web content
    st.title("AI Enhanced CCTV Analyzer App")
    st.text("Let´s analyze your CCTV videos and show a report ")
    videos = st.file_uploader(" ", type=["mp4"])

    if "detect_dangerous_objects" not in st.session_state:
        st.session_state.detect_dangerous_objects = False
    if "group_tracking" not in st.session_state:
        st.session_state.group_tracking = False
    if "activity_recognition" not in st.session_state:
        st.session_state.activity_recognition = False
    if "draw_skeletons" not in st.session_state:
        st.session_state.draw_skeletons = False
    if "write_logs" not in st.session_state:
        st.session_state.write_logs = False



    if videos:
        st.title("")
        st.text(videos)
        st.text(st.text(os.path.abspath(videos.name)))

        # st.text(type(st.text(os.path.abspath(videos.name))))

        if st.button(f"Preview {videos.name}"):
            st.video(videos)

        st.title("")
        st.subheader("Settings")
        with st.expander("See seetings"):
            st.session_state.detect_dangerous_objects = st.checkbox("Detect dangerous objects")
            st.session_state.group_tracking = st.checkbox("Group Tracking")
            st.session_state.activity_recognition = st.checkbox("Activity Recognition")
            st.session_state.draw_skeletons = st.checkbox("Draw Skeletons on Frame")
            st.session_state.write_logs = st.checkbox("Write logs on the side")


        st.title("")
        if st.button(f"Analyse {videos.name}"):

            with st.spinner('Wait for it...'):

                run_analyzer(str(videos.name[:-4]), st.session_state.group_tracking, st.session_state.activity_recognition, st.session_state.detect_dangerous_objects, st.session_state.draw_skeletons, st.session_state.write_logs)

            st.success(f'Done! The output is available at {HOME}/out/{videos.name}')
            st.balloons()

        st.title("")
        file_path = os.path.abspath(f"{HOME}\out\{videos.name[:-4]}-out.mp4")
        if os.path.exists(file_path):
            st.subheader("Output")
            st.video(file_path, format="video/mp4", start_time=0)

            output_logs_route = os.path.join("logs", f"{videos.name[:-4]}.pickle")

            if os.path.exists(output_logs_route):
                with open(output_logs_route, 'rb') as f:
                    try:
                        output_logs = pickle.load(f)
                    except:
                        raise ValueError(f"Unable to load data from file {videos.name}")

                grouping_logs = [item for item in output_logs if item['code'] == 0]
                object_detection_logs = [item for item in output_logs if item['code'] == 1]
                activity_logs = [item for item in output_logs if item['code'] == 2]

                activity_logs = format_logs(activity_logs, 30)
                object_detection_logs = format_logs(object_detection_logs, 30)
                # grouping_logs = format_logs(grouping_logs, 30)

                # for log in grouping_logs:
                #     st.text(log)

                with st.expander("See logs"):
                    tab1, tab2, tab3 = st.tabs(["Group Tracking", "Object Detection", "Activity Recognition"])

                    with tab1:
                        st.header("Group Tracking")
                        for log in grouping_logs:
                            st.text(log)

                            # start_time = datetime.datetime.strptime(log['from'], '%M:%S.%f')
                            # start_second = start_time.second + start_time.microsecond / 1000000.0
                            # start_second = int(start_second)
                            #
                            # if st.button("Show Video", key=log):
                            #     st.video(file_path, format="video/mp4", start_time=start_second)
                            st.title("")

                    with tab2:
                        st.header("Object Detection")
                        for log in object_detection_logs:
                            st.text(log)

                            start_time = datetime.datetime.strptime(log['from'], '%M:%S.%f')
                            start_second = start_time.second + start_time.microsecond / 1000000.0
                            start_second = int(start_second)

                            if st.button("Show Video", key=log):
                                st.video(file_path, format="video/mp4", start_time=start_second)
                            st.title("")

                    with tab3:
                        st.header("Activity Recognition")
                        for log in activity_logs:
                            st.text(log)

                            start_time = datetime.datetime.strptime(log['from'], '%M:%S.%f')
                            start_second = start_time.second + start_time.microsecond / 1000000.0
                            start_second = int(start_second)

                            if st.button("Show Video", key=log):
                                st.video(file_path, format="video/mp4", start_time=start_second)
                            st.title("")
