"""
This file contains the functions for generating Darknet datasets. 

E6792 Spring 2026
"""
import os
import cv2
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

from .utils import del_folder_contents, parse_config, load_test_config, read_validation_video_names
from .load_annotations import load_annotation_objects, get_objects_in_frame, get_frame_bboxes

FRAME_STEP = 6

def make_darknet_dataset(cfg_path, class_groups=None):
    """
    This function generates a dataset in the format specified by the
    obj.data file

    cfg_path: path to dataset configuration file
    class_groups: dictionary of classes to be grouped together
    
        Example: if we want to group all vehicle classes into the class 'vehicle'
        
        class_groups = {'vehicle' : ['car', 'bus', 'truck']}
    """
    
    print("Decoding evaluation parameters. (Video validation)")
    # decode config file
    options = parse_config(cfg_path, cfg_type='darknet_dataset')
    
    obj_data_filename = str(options["obj_data_filename"])
    train_path = str(options["train_path"])
    val_path = str(options["val_path"])
    
    videos_path = str(options["videos_path"])
    labels_path = str(options["labels_path"])
    
    val_video_names_path = str(options["val_video_names"])
    
    if options["max_vids"] != 'None':
        max_vids = int(options["max_vids"])
    else:
        max_vids = None
        
    if options["max_frames"] != 'None':
        max_frames = int(options["max_frames"])
    else:
        max_frames = None
        
    with open(obj_data_filename, 'r') as obj_data: # get configurations from .data file
        lines = obj_data.readlines()
        lines = [line for line in lines if line[0] not in ["#", "\n"]]
        num_classes = int(lines[0].split('=')[1].strip())
        train_txt_paths = lines[1].split('=')[1].strip() 
        val_txt_paths = lines[2].split('=')[1].strip()
        class_names_path = lines[3].split('=')[1].strip()
    
    # if os.getcwd().split('/')[-1] != 'darknet':
    #     class_names_path = class_names_path[1:]
    #     train_txt_paths = train_txt_paths[1:]
    #     val_txt_paths = val_txt_paths[1:]
    
    with open(class_names_path, 'r') as classes: # get class dictionary
        class_dict = {}
        for index, line in enumerate(classes.readlines()):
            class_dict.update({ line.strip() : str(index) })
            
    if class_groups is not None:
        # get index of 'superclass' in class_dict (line number in .names file) for each entry to class_groups
        for superclass in list(class_groups.keys()):
            original_classes = class_groups[superclass]
            for original_class in original_classes:
                class_dict.update({ original_class : class_dict[superclass] })
                            
            
    if not os.path.exists(train_path): # create train and val paths if they don't exist. Empty if they do exist
        os.makedirs(train_path)       
    else: 
        del_folder_contents(train_path)
    
    if not os.path.exists(val_path): 
        os.makedirs(val_path)
    else: 
        del_folder_contents(val_path)
        
    vids = 0
    frames = 0 # global frame count (accross multiple videos)
    
    train_image_list_file = open(train_txt_paths, 'w')
    val_image_list_file = open(val_txt_paths, 'w')
    
    validation_video_names = read_validation_video_names(val_video_names_path)

    for video_path in os.listdir(videos_path):
        if '.ts' in video_path:
            if max_vids is not None:
                if vids >= max_vids:
                    break
            video_name = video_path.split('.')[0]
            annotation_name = ''
            for annotation_path in os.listdir(labels_path):
                if video_name in annotation_path:
                    annotation_name = annotation_path
                    break
            print("Total frames: {}".format(frames))
            if annotation_name == '':
                print("Annotations not found for {}".format(video_path))
                continue

            annotation_objects = load_annotation_objects(os.path.join(labels_path, annotation_name))

            print("Loading video: {}".format(os.path.join(videos_path, video_path)))
            video = cv2.VideoCapture(os.path.join(videos_path, video_path))

            frame_num = 0 # local frame count (for one video)

            while True:

                if max_frames is not None:
                    if frame_num >= max_frames * FRAME_STEP:
                        break
                
                
                return_value, frame = video.read()

                if not return_value:
                    break

                if frame_num % FRAME_STEP == 0:

                    frame_name = str(frames) + '.jpg '

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]

                    frame_objects = get_objects_in_frame(annotation_objects, frame_num)
                    frame_bboxes = get_frame_bboxes(frame_objects)

                    if video_path in validation_video_names:
                        label_file = open(os.path.join(val_path, str(frames) + '.txt'), 'w')
                    else:
                        label_file = open(os.path.join(train_path, str(frames) + '.txt'), 'w')

                    for frame_bbox in frame_bboxes:

                        line = ''

                        label = frame_bbox[1]
                        
                        if label in list(class_dict.keys()):
                            line += class_dict[label] + ' '
                        else:
                            continue

                        coords = frame_bbox[0]
                        
                        ###################################################
                        # ---------- YOUR IMPLEMENTATION HERE ----------- #
                        ###################################################

                        # raise Exception('darknet_utils.make_dataset.make_darknet_dataset() not implemented!') # delete me
                        
                        # 1) Read the coordinates from the Mudd dataset format (absolute coordinates)
                        # print(coords)
                        x_top_left = coords[0][0]
                        y_top_left = coords[0][1]
                        
                        x_bottom_right = coords[1][0]
                        y_bottom_right = coords[1][1]
                        
                        # 2) Find height and width of the object (bounding box)
                        bounding_box_width = x_bottom_right - x_top_left
                        bounding_box_height = y_bottom_right - y_top_left
                        
                        # 3) Find the center of the object  (bounding box)
                        x_center = x_top_left + (bounding_box_width / 2)
                        y_center = y_top_left + (bounding_box_height / 2)
                        x_center = x_center / frame_width
                        y_center = y_center / frame_height
                        
                        bounding_box_width = bounding_box_width / frame_width
                        bounding_box_height = bounding_box_height / frame_height 
                        
                        # 4) Write the object's information into a string coords_string in the Darknet format
                        coords_string = f"{x_center} {y_center} {bounding_box_width} {bounding_box_height}\n"
       
                        
                        ###################################################
                        # ----------- END YOUR IMPLEMENTATION ----------- #
                        ###################################################
                
                        line += coords_string

                        label_file.write(line)

                    label_file.close()

                    frame_image = Image.fromarray(frame) # ensure newline character is removed. OpenCV (used by darknet) cannot handle newline character in filename
                    frame_image_name = frame_name[:-1]

                    if video_path in validation_video_names:
                        frame_image.save(os.path.join(val_path, frame_image_name))
                        val_image_list_file.write(os.path.join(val_path, frame_image_name) + '\n')
                    else:
                        frame_image.save(os.path.join(train_path, frame_image_name))
                        train_image_list_file.write(os.path.join(train_path, frame_image_name) + '\n')

                    frames += 1
                    
                    if frames % 100 == 0:
                        print('Frames read: {}'.format(frames))

                frame_num += 1

            vids += 1


    train_image_list_file.close()
    val_image_list_file.close()
        
    print("Darknet dataset (video validation) generated from {} videos with {} frames.".format(str(vids), str(frames)))    

    
def inspect_darknet_dataset(dataset_path, tests=3):
    """
    Visualize random darknet training/validation images and their corresponding labels.
   
    params:
        dataset_path (string): path to the directory containing .jpg images and 
                               corresponding .txt label files
        tests (int): the number of randomly selected examples to visualize
    """
    ###################################################
    # ---------- YOUR IMPLEMENTATION HERE ----------- #
    ###################################################

    # raise Exception('darknet_utils.make_dataset.inspect_darknet_dataset() not implemented!') # delete me
    images = []
    for i in os.listdir(dataset_path):
        if i.endswith('.jpg'):
            images.append(i)
    

    if len(images) == 0:
        print(f"No .jpg found in {(dataset_path)}")
        return

    num_tests = min(tests, len(images))
    selected_images = random.sample(images, num_tests)

    names = {'0': 'people', '1': 'vehicle'}
    colors = {'0': (0, 255, 0),'1': (255, 0, 0)}

    for file in selected_images:
        image_path = os.path.join(dataset_path, file)
        label_path = os.path.join(dataset_path, file.replace('.jpg', '.txt'))

        # print("image path:", image_path)
        # print("label_path:", label_path)

        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image {image_path}")
            print("Pleas check!!!!!!!!!!!!")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[0], img.shape[1]

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
                # for i in range(0,5):
                #     print(labes[i])
        else:
            print("No label path read! Pleases check .data and .cfg")
            labels = []

        for label in labels:
            parts = label.strip().split()
            # print(parts)
            
            if len(parts) != 5:
                continue


            # for i in parts:
            #     print(type(i))
                
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            bounding_box_w = float(parts[3])
            bounding_box_h = float(parts[4])

            x_center = x_center * img_w
            y_center = y_center * img_h
            bounding_box_w = bounding_box_w * img_w
            bounding_box_h = bounding_box_h * img_h

            x_h = int(x_center - bounding_box_w / 2)
            y_h = int(y_center - bounding_box_h / 2)
            x_w = int(x_center + bounding_box_w / 2)
            y_w = int(y_center + bounding_box_h / 2)

            x_h = max(0, x_h)
            y_h = max(0, y_h)
            x_w = min(img_w - 1, x_w)
            y_w = min(img_h - 1, y_w)

            display_name = names.get(class_id, class_id)
            color = colors.get(class_id, (255, 255, 0))

            cv2.rectangle(img, (x_h, y_h), (x_w, y_w), color, 2)
            cv2.putText(img,display_name,(x_h, max(y_h - 5, 0)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(file)
        plt.axis('off')
        plt.show()
    ###################################################
    # ----------- END YOUR IMPLEMENTATION ----------- #
    ###################################################