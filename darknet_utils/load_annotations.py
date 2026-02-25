"""
Functions for loading and viewing annotated videos.

E6792 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""

import json
import os
import cv2
import csv
import matplotlib.pyplot as plt
from PIL import Image

ANNOTATION_PATH = './annotations/'
VIDEO_PATH = './videos/'

FRAME_STEP = 6
THICKNESS = 1

CLASSES = ['car', 'truck', 'bus', 'face', 'license', 'person']
VEHICLES = ['car', 'truck', 'bus', 'license']
VEHICLES_NO_LP = ['car', 'truck', 'bus']
PERSON = ['person', 'face']

def load_annotation_objects(annotation_filename):
    """
    Returns dictionary representation of annotation objects
    """
    with open(annotation_filename) as json_file:
        return json.load(json_file)['annotations']['track']


def get_objects_in_frame(objects, current_frame_num, include_outside=False):
    """
    Returns all the object frames in the current frame.
    If include_outside=True, objects with outside property set
    will be returned.
    """
    frame_objects = []
    for obj in objects:
        if obj['_label'] in CLASSES:
            for object_frame in obj['box']:
                frame_num = int(int(object_frame['_frame']))
                if frame_num == current_frame_num:
                    attributes = object_frame['attribute']
                    occlusion = None
                    for attribute in attributes:
                        if obj['_label'] in VEHICLES:
                            if attribute['_name'] == 'license_id':
                                id_ = attribute['__text']
                        elif obj['_label'] in PERSON:
                            if attribute['_name'] == 'face_id':
                                id_ = attribute['__text']
                        if attribute['_name'] == 'occluded':
                            occlusion = attribute['__text']
                    if include_outside:
                        frame_objects.append(object_frame)
                    else:
                        if object_frame['_outside'] == '0':
                            
                            frame_objects.append((object_frame, obj['_label'], id_, occlusion))
    return frame_objects


def get_annotation_filename(video_filename, annotation_path):
    """
    Given a video filename, returns the corresponding annotation
    filename.
    """
    for annotation_filename in os.listdir(annotation_path):
        name = video_filename.split('.')[0]
        if name in annotation_filename:
            return annotation_filename
    return None


def get_frame_bboxes(frame_objects):
    """
    Return bboxes for given frame
    """
    bboxes = []
    for frame_object, label, id_, occlusion in frame_objects:
        bbox = get_bbox_coords(frame_object)
        bboxes.append((bbox, label, id_, occlusion))
    return bboxes


def get_bbox_coords(frame):
    """
    Returns the top left and bottom right coordinates of the bounding box
    in the given "frame"
    """
    xtl = int(float(frame['_xtl']))  # extract coordinates from json format
    ytl = int(float(frame['_ytl']))
    xbr = int(float(frame['_xbr']))
    ybr = int(float(frame['_ybr']))
    return (xtl, ytl), (xbr, ybr)


def draw_bboxes(frame, bboxes):
    """
    Draw bboxes on provided frame
    """
    for bbox in bboxes:
        frame = draw_bbox(frame, bbox)
    return frame


def draw_bbox(frame, bbox):
    """
    Draw single bbox on frame
    """
    return cv2.rectangle(frame, bbox[0], bbox[1], (255,0,0), THICKNESS)


def display_frame(frame):
    """
    Display provided frame using matplotlib
    """
    plt.figure(num=1, figsize=(25, 25))
    plt.imshow(frame)
    plt.show()


def get_annotations(video_path, videos_folder='./videos', annotations_folder='./labels_json', max_frames=None, max_videos=None, display=False):
    """
    Illustrates annotations for corresponding video frames
    """
    video_num = 0

    for video_filename in os.listdir(videos_folder):
        if max_videos is not None:
            if video_num >= max_videos:
                return

        annotation_filename = get_annotation_filename(video_filename, annotations_folder)
        if annotation_filename is None:
            print('Annotation file not found for video: ' + video_filename)
            return
        annotation_path = os.path.join(annotations_folder, annotation_filename)
        annotation_objects = load_annotation_objects(annotation_path)
        print("Reading video")
        video = cv2.VideoCapture(os.path.join(video_path, video_filename))
        print(os.path.join(video_path, video_filename))
        return
        frame_num = 0

        while True:

            if max_frames is not None:

                if frame_num / FRAME_STEP >= max_frames:
                    break

            return_value, frame = video.read()
            print(frame)
            if return_value:
                if frame_num % FRAME_STEP == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_objects = get_objects_in_frame(annotation_objects, frame_num)
                    print(frame_objects)
#                     frame_bboxes = get_frame_bboxes(frame_objects)

#                     if display:
#                         frame = draw_bboxes(frame, frame_bboxes)
#                         display_frame(frame)

                frame_num += 1


def make_dataset(max_frames=None, max_vids=None, labels_folder='./labels_json/', videos_folder='./videos/'):

    license_plate_lables = open('license_plate_labels.csv', 'w')
    face_labels = open('face_labels.csv', 'w')
    lp_writer = csv.writer(license_plate_lables)
    lp_writer.writerow(['filename', '((x_tl, y_tl), (x_br, y_br))', 'occlusion'])
    face_writer = csv.writer(face_labels)
    face_writer.writerow(['filename', '((x_tl, y_tl), (x_br, y_br))', 'occlusion'])
    vehicle_num = 0
    person_num = 0

    vids = 0
    for video_path in os.listdir(videos_folder):
        if '.ts' in video_path:
            if max_vids is not None:
                if vids >= max_vids:
                    break
            video_name = video_path.split('.')[0]
            annotation_name = ''
            for annotation_path in os.listdir(labels_folder):
                if video_name in annotation_path:
                    annotation_name = annotation_path
                    break

            if annotation_name == '':
                continue

            annotation_objects = load_annotation_objects(os.path.join(labels_folder, annotation_name))

            print("Loading video: {}".format(os.path.join(videos_folder, video_path)))
            video = cv2.VideoCapture(os.path.join(videos_folder, video_path))

            frame_num = 0

            while True:

                if max_frames is not None:
                    if frame_num >= max_frames * FRAME_STEP:
                        break

                return_value, frame = video.read()

                if not return_value:
                    break

                if frame_num % FRAME_STEP == 0:

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_objects = get_objects_in_frame(annotation_objects, frame_num)
                    frame_bboxes = get_frame_bboxes(frame_objects)

                    for frame_bbox in frame_bboxes:
                        coords = frame_bbox[0]
                        label = frame_bbox[1]
                        id_ = frame_bbox[2]
                        occlusion = frame_bbox[3]

                        if label in VEHICLES_NO_LP:
                            cropped_frame = frame[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0], :]
                            for frame_bbox2 in frame_bboxes:
                                if frame_bbox2[1] == 'license':
                                    if frame_bbox2[2] == id_:
                                        if id_ != '-1':
                                            y_len = len(cropped_frame)
                                            x_len = len(cropped_frame[0])
                                            new_ytl = frame_bbox2[0][0][1] - coords[0][1]
                                            new_xtl = frame_bbox2[0][0][0] - coords[0][0]
                                            new_xbr = frame_bbox2[0][1][0] - coords[1][0] + x_len
                                            new_ybr = frame_bbox2[0][1][1] - coords[1][1] + y_len
                                            new_coords = ((new_xtl, new_ytl), (new_xbr, new_ybr))
                                        else:
                                            new_coords = 'None'
                                        vehicle_image = Image.fromarray(cropped_frame)
                                        vehicle_image_name = str(vehicle_num) + '.jpg'
                                        vehicle_image.save(os.path.join('./vehicles/', vehicle_image_name))
                                        lp_writer.writerow([vehicle_image_name, new_coords, occlusion])
                                        vehicle_num += 1


                        elif label == 'person':
                            cropped_frame = frame[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0], :]
                            for frame_bbox2 in frame_bboxes:
                                if frame_bbox2[1] == 'face':
                                    if frame_bbox2[2] == id_:
                                        if id_ != '-1':
                                            y_len = len(cropped_frame)
                                            x_len = len(cropped_frame[0])
                                            new_ytl = frame_bbox2[0][0][1] - coords[0][1]
                                            new_xtl = frame_bbox2[0][0][0] - coords[0][0]
                                            new_xbr = frame_bbox2[0][1][0] - coords[1][0] + x_len
                                            new_ybr = frame_bbox2[0][1][1] - coords[1][1] + y_len
                                            new_coords = ((new_xtl, new_ytl), (new_xbr, new_ybr))
                                        else:
                                            new_coords = 'None'
                                        person_image = Image.fromarray(cropped_frame)
                                        person_image_name = str(person_num) + '.jpg'
                                        person_image.save(os.path.join('./people/', person_image_name))
                                        face_writer.writerow([person_image_name, new_coords, occlusion])
                                        person_num += 1

                frame_num += 1

            vids += 1

    license_plate_lables.close()
    face_labels.close()
