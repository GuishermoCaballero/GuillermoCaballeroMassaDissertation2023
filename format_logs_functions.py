def format_time(frame_number, frame_rate):
    # calculate the time in seconds
    time_in_seconds = int(frame_number / frame_rate)
    # calculate the remaining frames after whole seconds
    frames = int(frame_number % frame_rate)
    # convert the seconds to hours, minutes, and seconds
    hours = int(time_in_seconds / 3600)
    minutes = int((time_in_seconds % 3600) / 60)
    seconds = int(time_in_seconds % 60)
    # format the time as a string
    time_string = "{:02d}:{:02d}".format( minutes, seconds)
    # append the remaining frames to the string
    time_string += ".{:02d}".format(frames)
    return time_string


def format_logs(detections, distance_in_frames):
    output = []
    current_group = None

    for detection in detections:
        if current_group is None:
            current_group = {"code": detection["code"], "message": detection["message"], "from": detection["frame"],
                             "until": detection["frame"], "frame_rate": detection["frame_rate"]}
        else:
            if detection["frame"] <= current_group["until"] + distance_in_frames:
                current_group["until"] = detection["frame"]
            else:
                output.append(current_group)
                current_group = {"code": detection["code"], "message": detection["message"], "from": detection["frame"],
                                 "until": detection["frame"], "frame_rate": detection["frame_rate"]}

    if current_group is not None:
        output.append(current_group)

    for detection in output:
        detection["from"] = format_time(detection["from"], detection["frame_rate"])
        detection["until"] = format_time(detection["until"], detection["frame_rate"])
        del detection['frame_rate']

    return output