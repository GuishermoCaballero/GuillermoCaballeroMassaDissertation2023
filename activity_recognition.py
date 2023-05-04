import cv2
import numpy as np
import random


def loop_thought_people(frame, keypoints_with_scores, edges, confidence_threshold, moment):
    count_people(frame, keypoints_with_scores, confidence_threshold)
    for index, person in enumerate(keypoints_with_scores):
        draw_connections(frame, person, edges, confidence_threshold, index)
        draw_keypoints(frame, person, confidence_threshold, index)


def count_people(frame, keypoints_with_scores, confidence_threshold):
    number_of_people = 0

    for index, person in enumerate(keypoints_with_scores):
        third_values = person[:, 2]  # get the third value of each array
        average_third_value = np.mean(third_values)  # calculate the average
        if (average_third_value > confidence_threshold):
            number_of_people += 1

    cv2.putText(frame, f'People detected by MoveNet: {int(number_of_people)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (70, 235, 52), 1, cv2.LINE_AA)


def draw_keypoints(frame, keypoints, confidence_threshold, index):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for idx, kp in enumerate(shaped):
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
            if (idx == 0):
                cv2.putText(frame, f'{int(index)}', (int(kx), int(ky)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1,
                            cv2.LINE_AA)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# import random
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]


def draw_connections(frame, keypoints, edges, confidence_threshold, index):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[index], 4)


###########################################################################################


def filter_bodies(people, confidence_threshold):
    filtered_indexes = []
    filtered_points = []

    for index, person in enumerate(people):  # keypoints_with_scores
        third_values = person[:, 2]  # get the third value of each array
        average_third_value = np.mean(third_values)  # calculate the average
        if average_third_value > confidence_threshold:
            filtered_indexes.append(index)

    for filtered_index in filtered_indexes:
        filtered_points.append(people[filtered_index])

    return np.array(filtered_points)


def calculate_body_inside_box(bodies, xyxy, trackers, confidence_threshold):
    overall_indexes = {}
    for index_box, box in enumerate(xyxy):
        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        scores_array = []
        for index_body, body in enumerate(bodies):
            body_count = 0
            for point in body:
                X = point[1]
                Y = point[0]
                if point[2] < confidence_threshold:
                    body_count = body_count
                elif x1 <= X <= x2 and y1 <= Y <= y2:
                    body_count += 1
                else:
                    body_count = body_count

            scores_array.append(body_count)

        if sum(scores_array) > 0:
            max_index = scores_array.index(max(scores_array))
            overall_indexes[trackers[index_box]] = max_index

    return overall_indexes


def filter_arrays_for_poses(tracks, xyxy, clases):

    filtered_tracks = []
    filtered_xyxy = []
    for i, c in enumerate(clases):
        if c == 0:
            filtered_tracks.append(tracks[i])
            filtered_xyxy.append(xyxy[i])
    return filtered_tracks, filtered_xyxy
