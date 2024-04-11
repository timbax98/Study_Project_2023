# region Setup

import os
import sys
import zipfile
import cv2
import numpy as np
import yolov5
import tensorflow as tf
import matplotlib.pyplot as plt

# Configure Hyperparameters
MAX_BATCH_SIZE = 32 # maximum batch size of transformer training dataset
EPSILON = 100 # pixel threshold for tokenizing the successful overlap of hand and target (EOS); which range?
#EPSILON = vid_res[(batch+1)*(seq+1)][1] / 12  # alternative: use video resolution dependent epsilon
MAX_MISSING = 10 # maximum length of missing detection sequence in a video, otherwise removed
MIN_DETECTS = 1/4 # minimum number of frames in which object is detected, otherwise removed
MIN_SCORE = 0.55 # minimum score threshold for evaluating true hand candidates in a video with multiple hand detections
TRACKING_AREA = 100 # maximum distance for detecting hands after a true hand is lost; which range?

# Create path variables
folderpath = './data/trials/'
labelpath = './data/labels/'
weights_objs = './weights/yolo/objects.pt'
weights_hands = './weights/yolo/hands.pt'

# Read all video files in the data folder
folder = [f for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f)) and (f.endswith('.mp4') or f.endswith('.mov'))]

# endregion

# region Helpers

def GetBoundingBoxes(videopath, weightpath):
    """
    Extract bounding boxes for objects from video.

    Args:
    videopath -- path to video to extract bounding boxes from.
    weightpath -- path to object detector weights.
    """

    # Load model
    model = yolov5.load(weightpath)

    bbs = np.zeros((0, 7))
    remove_vid = False

    # Open the video file
    cap = cv2.VideoCapture(videopath)

    # Get the frames per second (fps) and frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    true_hand_found = False

    # Loop through each frame and detection
    for frame_num in range(frame_count):
        ret, frame = cap.read()

        if not ret:
            remove_vid = True
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(frame)
        #result.show()

        # get object classes
        if frame_num == 0:
            classes = result.names

        # (frame, class, x_center, y_center, bbwidth, bbheight, trackerID)
        xywh = result.xywh[0].cpu().numpy()
        n = len(xywh)
        cls = xywh[:, 5].reshape((n,1))
        xywh = xywh[:, 0:4]

        frame_count_n = np.repeat(frame_num, n).reshape((n,1))
        data = np.concatenate((frame_count_n, cls, xywh), axis=1)
        data = np.hstack((data, np.zeros((n, 1))))

        # Filtering: Multiple hand detections per frame
        # do this only for hands, as only one hand should be detected (not using class filtering)
        if weightpath == weights_hands:
            
            # Get all hand detections in this frame
            detections = result.xywh[0].cpu().numpy()

            # if there are any detections
            if len(detections) != 0:
                # if the "ground truth" hand has not been found yet
                if not true_hand_found:

                    max_score = 0 # within one frame
                    for _, d in enumerate(detections):
                        _, ycenter, _, bbheight, conf, _ = d
                        vidheight = frameHeight
                        # evaluation of true_hand candidate detections
                        score = conf * ((ycenter - (bbheight//2)) / vidheight)

                        # search for true hand using the eval score (confidence weighted y-pos)
                        if score >= MIN_SCORE and score > max_score:
                            detection = d[0:4].reshape((1,4))
                            data = np.concatenate((np.array([[frame_num, 333]]), detection, np.array([[0]])), axis=1)
                            last_detection = detection
                            max_score = score
                            true_hand_found = True
                else:
                    distances = np.linalg.norm(detections[:, :4] - last_detection, axis=1)

                    # tracking area avoids jumps to a wrong hand if the true hand is lost in a frame
                    if min(np.abs(distances)) <= TRACKING_AREA:
                        closest_index = np.argmin(np.abs(distances))
                        detection = detections[closest_index, :4].reshape((1, 4))
                        data = np.concatenate((np.array([[frame_num, 333]]), detection, np.array([[0]])), axis=1)
                        last_detection = detection
        
        bbs = np.concatenate((bbs, data), axis=0)

    # Release the video capture object
    cap.release()

    return bbs , classes, remove_vid


def Center(videopath, bounding_boxes):

    """
    Create a video where a target object is always centered.

    Args:
    videopath -- path to video to center
    bounding_boxes -- bbs of object to center on
    """

    # 1. Read video
    cap = cv2.VideoCapture(videopath)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    video = np.zeros((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    for frame_num in range(frameCount):
        ret, frame = cap.read()

        if ret:
            video[frame_num] = frame

    cap.release()

    # 2. Correct video
    bbs = bounding_boxes
    width = frameWidth
    height = frameHeight
    center_x = (width/2)
    center_y = (height/2)
    x_dists = []
    y_dists = []

    # Instead keep the size after centering constant across videos
    # For this use a padding based on maximally possible offset

    # ??? What is the correct centered size? (width + width) or (width + width*2) ? 
    # Double should be correct (offset in each direction is max half the videos size in that dimension)

    # 3. Center video
    video_centered = np.zeros((len(video),
                              int(height + height), # ehemals variabel: int(height + max(y_dists)*2),
                              int(width + width), # ehemals variabel: int(width + max(x_dists)*2),
                              3), dtype=int)

    # get coords for placement height and width
    start_row = (video_centered.shape[1] - height) // 2
    start_col = (video_centered.shape[2] - width) // 2

    offsets = np.zeros(shape=(len(bbs),2))

    for idx, frame in enumerate(video):

        # video length and length of bb df is not the same, because we have a cut-off at the end
        if idx == len(bbs):
            break

        # get matching bb
        box_curr = bbs[idx]

        # If there is no information on fruit, color the frame black (skip iter)
        if np.isnan(box_curr).any():
            continue
        else:
            # get center of bb
            center_x_bb = box_curr[0]
            center_y_bb = box_curr[1]

            # get offset of center
            # pos if bounding box is to right of center, else negative
            x_offset = int(center_x_bb - center_x)
            # pos if bounding box is below center, else negative
            y_offset = int(center_y_bb - center_y)

            #!coordinates until here are for old video

            #get fitting indices for new video
            fixed_start_row = start_row - y_offset
            fixed_start_col = start_col - x_offset

            fixed_end_row = fixed_start_row + height
            fixed_end_col = fixed_start_col + width

            # Save centered + corrected in new video
            video_centered[idx][fixed_start_row:fixed_end_row,
                              fixed_start_col:fixed_end_col] = frame

            #Save offsets
            offsets[idx,:] = x_offset, y_offset


    return offsets, (start_row, start_col), video_centered


def Export(video, targetBBs, handBBs, labelpath):

    video_centered = np.uint8(video)
    video_centered = video_centered[:len(targetBBs), :, :, :]

    height_new = int(video_centered.shape[1])
    width_new = int(video_centered.shape[2])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(labelpath[:-4]+'_centered.mp4', fourcc, 30, (width_new, height_new), True)

    for i, frame in enumerate(video_centered):

        # Draw Bounding Boxes for target and hand
        if targetBBs[i,1] == i:
            if not np.isnan(targetBBs[i]).any():
                xc = int(targetBBs[i, 2])
                yc = int(targetBBs[i, 3])
                width = int(targetBBs[i, 4])
                height = int(targetBBs[i, 5])
                cv2.rectangle(frame, (xc - width//2, yc - height//2), (xc + width//2, yc + height//2), (0, 255, 0), 2)

            if not np.isnan(handBBs[i]).any():
                xc = int(handBBs[i, 0])
                yc = int(handBBs[i, 1])
                width = int(handBBs[i, 2])
                height = int(handBBs[i, 3])
                cv2.rectangle(frame, (xc - width//2, yc - height//2), (xc + width//2, yc + height//2), (0, 0, 255), 2)

        # Calculate Angle
                
        # If not at the end of the video
        if i < len(handBBs) - 1:
                
            # If current and next frame have a detection
            if not np.isnan(handBBs[i]).any() and not np.isnan(handBBs[i + 1]).any():

                # Get hand BBox center of the current frame
                xc_current = handBBs[i, 0] 
                yc_current = handBBs[i, 1]

                # Get hand BBox center of the next frame
                xc_next = handBBs[i + 1, 0]
                yc_next = handBBs[i + 1, 1]
                
                # Calculate the angle between the line passing through the centers of the hand bounding boxes and the positive x-axis
                angle = np.arctan2(yc_next-yc_current,xc_next-xc_current)

                # Calculate vector components based on angle
                magnitude = 600  # Adjust the magnitude of the vectors as needed
                dx = int(magnitude * np.cos(angle))
                dy = int(magnitude * np.sin(angle))

                #print(yc_next, yc_current, xc_next, xc_current)
                #print(f"angle: {angle}")

                # Draw the vector on the frame
                cv2.arrowedLine(frame, (int(xc_current), int(yc_current)), (int(xc_current + dx), int(yc_current + dy)), (255, 0, 0), 2)
                
                # Draw the angle in radians and degrees on the frame
                cv2.putText(frame, f"{angle:.2f}", (int(xc_current + dx), int(yc_current + dy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"{np.degrees(angle):.2f}", (int(xc_current + dx), int(yc_current + dy + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            

        out.write(frame)
    out.release()


def check_overlap(bbox1, bbox2):

    x1_box1, x2_box1, y1_box1, y2_box1 = bbox1
    x1_box2, x2_box2, y1_box2, y2_box2 = bbox2

    # Check if the bounding boxes overlap in the x-axis
    if x1_box1 > x2_box2 or x2_box1 < x1_box2:
        return False

    # Check if the bounding boxes overlap in the y-axis
    if y1_box1 > y2_box2 or y2_box1 < y1_box2:
        return False

    return True


def delete_overlap_rows(hands, targets, frameCount, buffer=3):
    # in bb_arrays [0,1,2,3,4,5] = [video, frame, xc_target, yc_target, bbw_target, bbh_target]

    # Convert coordinates to frame, x1,x2,y1,y2

    # Calculate the coordinates for the corners of the hands bounding boxes
    hands_detec_frames = np.array([int(bbx[1]) for bbx in hands])
    hands_x1 = np.array(hands[:, 2] - hands[:, 4] / 2)
    hands_x2 = np.array(hands[:, 2] + hands[:, 4] / 2)
    hands_y1 = np.array(hands[:, 3] - hands[:, 5] / 2)
    hands_y2 = np.array(hands[:, 3] + hands[:, 5] / 2)

    hands_corners = np.stack((hands_detec_frames, hands_x1, hands_x2, hands_y1, hands_y2), axis=1)

    # Calculate the coordinates for the corners of the targets bounding boxes
    targets_detec_frames = np.array([int(bbx[1]) for bbx in targets])
    targets_x1 = np.array(targets[:, 2] - targets[:, 4] / 2)
    targets_x2 = np.array(targets[:, 2] + targets[:, 4] / 2)
    targets_y1 = np.array(targets[:, 3] - targets[:, 5] / 2)
    targets_y2 = np.array(targets[:, 3] + targets[:, 5] / 2)

    targets_corners = np.stack((targets_detec_frames, targets_x1, targets_x2, targets_y1, targets_y2), axis=1)

    # Check for overlap between hands and targets for all rows and safe overlap idx
    first_overlap_frame = None
    for frame_idx in range(frameCount):
        #check if detection for targets and for hands
        if (frame_idx in hands_detec_frames) and (frame_idx in targets_detec_frames):
            #get corner values for the frame. It is only a single row, although I use list comprehension
            hand_bb = [row for row in hands_corners if row[0]==frame_idx][0] #Take the first
            target_bb = [row for row in targets_corners if row[0]==frame_idx][0]

            # check if targets and hands overlap in this frame
            if check_overlap(hand_bb[1:], target_bb[1:]): # [1:] because frame ids are not needed for computation of the overlap
                #if so, save the frame idx and stop the loop, we only need the first one
                first_overlap_frame = frame_idx
                break

    frame_cutoff = frameCount
    # If we found an overlap (else we keep all bounding boxes)
    if first_overlap_frame is not None:
        # Calculate frame after which bbs should be deleted based on the buffer
        frame_cutoff = first_overlap_frame + buffer

        # Delete all frames after overlap with some buffer
        hands = [row for row in hands if row[1] < frame_cutoff]
        targets = [row for row in targets if row[1] < frame_cutoff]

    # Do not permit greater frame_cutoff than original frameCount
    if frame_cutoff > frameCount:
        frame_cutoff = frameCount

    return hands, targets, frame_cutoff


def calculate_batch_size(num_inputs, max_batch_size=128):
    for batch_size in range(min(num_inputs, max_batch_size), 0, -1):
        if num_inputs % batch_size == 0:
            return batch_size


def PlotTrajectory(path, name, input):

    if not os.path.exists(path):
        os.makedirs(path)

    video_data = input

    # Plotting centers of hand and object bounding boxes
    object_centers = video_data[:, [2, 3]]  # x and y centers of the object
    object_widths = video_data[:, 4]
    object_heights = video_data[:, 5]

    hand_centers = video_data[:, [6, 7]]  # x and y centers of the hand
    hand_widths = video_data[:, 8]
    hand_heights = video_data[:, 9]

    video_width = 1920
    video_height = 1080

    # Use the first object center as a reference for the plot's center
    ref_ox, ref_oy = object_centers[0]

    #print(f"\n Printing object centers for video with index {video_index} ({len(object_centers)} frames). Should all be the same. \n {object_centers} \n \n")

    plt.figure(figsize=(10, 6))
    # For every frame in the current video, plot the boundingboxes
    for frame in range(len(video_data)):
        ox, oy = object_centers[frame]
        ow = object_widths[frame]
        oh = object_heights[frame]

        hx, hy = hand_centers[frame]
        hw = hand_widths[frame]
        hh = hand_heights[frame]

        # Plot the bounding box for the object
        plt.plot([ox - ow / 2, ox + ow / 2, ox + ow / 2, ox - ow / 2, ox - ow / 2],
                 [oy - oh / 2, oy - oh / 2, oy + oh / 2, oy + oh / 2, oy - oh / 2],
                 'b-')  # Object bounding box in blue

        # Plot the bounding box for the hand
        plt.plot([hx - hw / 2, hx + hw / 2, hx + hw / 2, hx - hw / 2, hx - hw / 2],
                 [hy - hh / 2, hy - hh / 2, hy + hh / 2, hy + hh / 2, hy - hh / 2],
                 'r-')  # Hand bounding box in red

        # Plot the actual video size in the centered(padded) video
        plt.plot([ref_ox - video_width / 2, ref_ox + video_width / 2, ref_ox + video_width / 2, ref_ox - video_width / 2, ref_ox - video_width / 2],
                 [ref_oy - video_height / 2, ref_oy - video_height / 2, ref_oy + video_height / 2, ref_oy + video_height / 2, ref_oy - video_height / 2],
                 'g-')  # actual video in green

        plt.plot(ox, oy, 'bo')  # Object center in blue
        plt.plot(hx, hy, 'ro')  # Hand center in red


    plt.title(f"Object and Hand Centers for Video {name}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.legend(["Object Center", "Hand Center"])

    # Set the plot limits to keep the object center in the middle of the plot
    plt.xlim(0, ref_ox*2)
    plt.ylim(0, ref_oy*2)

    # Ensure the object center is visually marked at the center
    plt.scatter(ref_ox, ref_oy, c='blue', marker='x', s=100, label='Fixed Object Center')

    # Flip Y-axis and save the plot to a file
    plt.gca().invert_yaxis()
    plt.savefig(path+video[:-4]+".png")
    plt.close()  # Close the plot to free up memory


# endregion

# region Input
        
"""
## Create Input X

This step is done to filter out head movements to get the true hand movements and subsequently create Input X.

**Procedure:**
1. Get the bounding boxes of all objects and hands in each raw video.
2. Filter for hand and target object bounding boxes.
3. Target centering: fixate the target object (TARGET.mp4) using interpolation.
4. Input concatenation.
"""

# Initialize array for concatenation of hand and target bounding boxes
input = np.zeros((0,10)) # (video, frame, xc_target, yc_target, bbw_target, bbh_target, xc_hand, yc_hand, bbw_hand, bbh_target)
input[:] = np.nan # default: no detection
videos_removed = 0 # counter for videos that were removed due to not meeting our requirements (e.g. number of frames detected)

# Iterate through every video file in the folder
for i, video in enumerate(folder):

    ### 0. Setup ###    

    # Get length of input array (number of frames)
    cap = cv2.VideoCapture(folderpath+video)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\n Video {i+1}/{len(folder)}: {video} ({frameCount} frames) \n")


    ### 1. Get all bbs of this video ###

    # Get bbs from objects model
    bbs_objects, classes_objs, remove_vid = GetBoundingBoxes(folderpath+video, weights_objs)
    video_count = np.repeat(i, len(bbs_objects)).reshape((len(bbs_objects),1))
    bbs_objects = np.concatenate((video_count, bbs_objects), axis=1)

    # if a frame could not be read due to technical issues, remove the video
    if remove_vid:
        videos_removed += 1
        print("Video skipped because of problems reading the video (objects).")
        continue

    # Get bbs from hands model
    bbs_hands, classes_hands, remove_vid = GetBoundingBoxes(folderpath+video, weights_hands)
    video_count = np.repeat(i, len(bbs_hands)).reshape((len(bbs_hands),1))
    bbs_hands = np.concatenate((video_count, bbs_hands), axis=1)

    # if a frame could not be read due to technical issues, remove the video
    if remove_vid:
        videos_removed += 1
        print("Video skipped because of problems reading the video (hands).")
        continue


    ### 2. Filter for target by class name ###

    if ('.mov' in video) or ('.mp4' in video):
        # returns the string name without '000.mp4' ending
        target_class = video[:-7]

    # Get class number from lookup table
    target_class = list(classes_objs.keys())[list(classes_objs.values()).index(target_class)]
    # get median value of all classes in hand detections to get "true" label 
    # --> results in too few detections --> we filter in GetBoundingBoxes (only hands)
    #hand_class = int(np.median(bbs_hands[:, 2]))

    target_bbs = bbs_objects[(bbs_objects[:, 2] == target_class)]
    target_bbs = target_bbs[:, [0,1,3,4,5,6]]
    hand_bbs = bbs_hands[(bbs_hands[:, 2] == 333)] # true hand evaluator class ID
    hand_bbs = hand_bbs[:, [0,1,3,4,5,6]]

    # Cut frames after overlap of hand and target
    hand_bbs, target_bbs, frameCount = delete_overlap_rows(hand_bbs, target_bbs, frameCount)


    ### 3. Interpolation: take last known bounding box positions ###

    vid_input = np.zeros((frameCount, 10))
    vid_input[:] = np.nan

    ## Perform interpolation for targets ##

    # get x values for which detection of target was successful
    x_detects_target = [int(target_bb[1]) for target_bb in target_bbs]
    possible_detections = 0

    # if there are not enough frames detected (50%) or a too large missing frame sequence (10 frames), we do not keep the video
    if len(x_detects_target) > 1:
        possible_detections = x_detects_target[-1] - x_detects_target[0]
        differences = [x_detects_target[i + 1] - x_detects_target[i] for i in range(len(x_detects_target) - 1)]
    else:
        print("Video skipped because there are less than 2 detections (objects).")
        videos_removed += 1
        continue

    if len(x_detects_target) < possible_detections * MIN_DETECTS or any(diff > MAX_MISSING for diff in differences):
        print("Video skipped because there are not enough detections or missing frame sequences are too long (objects).")
        #print(f"x_detects_target: {len(x_detects_target)}, possible detections: {possible_detections}, threshold: {possible_detections * MIN_DETECTS}")
        videos_removed += 1
        continue

    # get x values for which interpolation has to be performed
    first_detection_frame = x_detects_target[0]
    x_to_interp = [xframe for xframe in range(len(vid_input)) if ((xframe not in x_detects_target) and (xframe >= first_detection_frame))]

    # only interpolate if there are any detections
    if len(x_detects_target) != 0:

        # get the values for successful detections
        x_centers_targets_detec = np.asarray([frame_value[2] for frame_value in target_bbs])
        y_centers_targets_detec = np.asarray([frame_value[3] for frame_value in target_bbs])
        bbwidths_targets_detec = np.asarray([frame_value[4] for frame_value in target_bbs])
        bbheights_targets_detec = np.asarray([frame_value[5] for frame_value in target_bbs])

        # Interpolate
        #numpy.interp(x, xp, fp, left=None, right=None, period=None)
        inter_x_centers_target = np.interp(x_to_interp, x_detects_target, x_centers_targets_detec)
        inter_y_centers_target = np.interp(x_to_interp, x_detects_target, y_centers_targets_detec)
        inter_bbwidths_target = np.interp(x_to_interp, x_detects_target, bbwidths_targets_detec)
        inter_bbheights_target = np.interp(x_to_interp, x_detects_target, bbheights_targets_detec)

        # Combine values into single array
        # Add frame and video indices for new array
        # frame indices
        all_frames = [frame for frame in range(len(vid_input))]
        vid_input[all_frames,1] = range(len(vid_input))
        # video indices
        vid_input[all_frames,0] = i

        # pseudo: vid_input[all_detec_xs][einzeln 2 bis 4 für die werte (in versch zeilen)] =  x_centers_targets_detec bis bbheights_targets_detec in den zeilen
        vid_input[x_detects_target,2] = x_centers_targets_detec
        vid_input[x_detects_target,3] = y_centers_targets_detec
        vid_input[x_detects_target,4] = bbwidths_targets_detec
        vid_input[x_detects_target,5] = bbheights_targets_detec

        # same for interpolated
        vid_input[x_to_interp,2] = inter_x_centers_target
        vid_input[x_to_interp,3] = inter_y_centers_target
        vid_input[x_to_interp,4] = inter_bbwidths_target
        vid_input[x_to_interp,5] = inter_bbheights_target

    ## Perform interpolation for hands ##

    # get x values for which detection of hand was successful
    x_detects_hand = [int(hand_bb[1]) for hand_bb in hand_bbs]
    possible_detections = 0

    # if there are not enough frames detected (50%) or a too large missing frame sequence (10 frames), we do not keep the video
    if len(x_detects_hand) > 1:
        possible_detections = x_detects_hand[-1] - x_detects_hand[0]
        differences = [x_detects_hand[i + 1] - x_detects_hand[i] for i in range(len(x_detects_hand) - 1)]
    else:
        print("Video skipped because there are less than 2 detections (hands).")
        videos_removed += 1
        continue

    if len(x_detects_hand) < possible_detections * MIN_DETECTS or any(diff > MAX_MISSING for diff in differences):
        print("Video skipped because there are not enough detections or missing frame sequences are too long (hands).")
        #print(f"x_detects_target: {len(x_detects_hand)}, possible detections: {possible_detections}, threshold: {possible_detections * MIN_DETECTS}")
        videos_removed += 1
        continue

    # get x values for which interpolation has to be performed
    first_detection_frame = x_detects_hand[0]
    x_to_interp = [xframe for xframe in range(len(vid_input)) if ((xframe not in x_detects_hand) and (xframe >= first_detection_frame))]

    # only save data if there are any detections
    if len(x_detects_hand) != 0:

        # get the values for successful detections
        x_centers_hand_detec = np.asarray([frame_value[2] for frame_value in hand_bbs])
        y_centers_hand_detec = np.asarray([frame_value[3] for frame_value in hand_bbs])
        bbwidths_hand_detec = np.asarray([frame_value[4] for frame_value in hand_bbs])
        bbheights_hand_detec = np.asarray([frame_value[5] for frame_value in hand_bbs])

        # Interpolate
        #numpy.interp(x, xp, fp, left=None, right=None, period=None)
        inter_x_centers_hand = np.interp(x_to_interp, x_detects_hand, x_centers_hand_detec)
        inter_y_centers_hand = np.interp(x_to_interp, x_detects_hand, y_centers_hand_detec)
        inter_bbwidths_hand = np.interp(x_to_interp, x_detects_hand, bbwidths_hand_detec)
        inter_bbheights_hand = np.interp(x_to_interp, x_detects_hand, bbheights_hand_detec)

        # Combine values into single array
        # Add frame and video indices for new array
        # frame indices
        all_frames = [frame for frame in range(len(vid_input))]
        vid_input[all_frames,1] = range(len(vid_input))
        # video indices
        vid_input[all_frames,0] = i

        # pseudo: vid_input[all_detec_xs][einzeln 2 bis 4 für die werte (in versch zeilen)] =  x_centers_targets_detec bis bbheights_targets_detec in den zeilen
        vid_input[x_detects_hand,6] = x_centers_hand_detec
        vid_input[x_detects_hand,7] = y_centers_hand_detec
        vid_input[x_detects_hand,8] = bbwidths_hand_detec
        vid_input[x_detects_hand,9] = bbheights_hand_detec

        # same for interpolated
        vid_input[x_to_interp,6] = inter_x_centers_hand
        vid_input[x_to_interp,7] = inter_y_centers_hand
        vid_input[x_to_interp,8] = inter_bbwidths_hand
        vid_input[x_to_interp,9] = inter_bbheights_hand


    ### 4. Center: Center on the correct bounding boxes of the target object ###
    offsets, start_cords, video_centered = Center(folderpath+video, vid_input[:, 2:6])

    new_target_bbs = vid_input[:,:6]
    new_hand_bbs = vid_input[:,6:]

    # Get offsets of target from center of OG video
    x_offsets = offsets[:,0]
    y_offsets = offsets[:,1]

    start_x = start_cords[1]
    start_y = start_cords[0]

    # Calculate Bounding boxes in bigger video before centering
    # Adjust x
    new_target_bbs[:,2] += start_x
    new_hand_bbs[:,0] += start_x

    # Adjust y
    new_target_bbs[:,3] += start_y
    new_hand_bbs[:,1] += start_y

    # Now adjust for after centering based on offsets
    # Adjust x
    new_target_bbs[:,2] -= x_offsets
    new_hand_bbs[:,0] -= x_offsets

    # Adjust y
    new_target_bbs[:,3] -= y_offsets
    new_hand_bbs[:,1] -= y_offsets

    ### Debug: Export centered videos with BBs
    Export(video_centered, new_target_bbs, new_hand_bbs, labelpath+video)


    ### 5. Input concatenation ###

    bbs = np.concatenate((new_target_bbs, new_hand_bbs), axis=1)
    bbs = bbs[~np.isnan(bbs).any(axis=1)] # remove frames without detections
    input = np.concatenate((input, bbs), axis=0)

    ### Debug: Plot grasping trajectory
    PlotTrajectory("./data/trajectories/", video, bbs)

print(f"\n INPUT CREATED \n Number of videos (total): {len(folder)}, Removed: {videos_removed}, Left: {len(folder)-videos_removed}")

# Debugging: check input
"""
with np.printoptions(threshold=sys.maxsize, suppress=True):
    print("\n \n Printing the first 200 input rows.")
    print(input[:200])
"""

# endregion

# region Tokenization

# define tokens; has to be unusual in sequences (0 to max dim of video res) as well as labels (radiands: -2pi to 2pi)
start_token = -333 # for seqs and labels
end_token = -1 # for labels only (sequences do not need an end token, as they are all separated by start tokens)
padding_token = -2 # for seqs only (labels are generated from padded sequences)

# Slice data into single videos
uniques = np.unique(input[:,0])
sliced_seqs = []
sliced_labels = []

video_width = 1920
video_height = 1080

context_SOS = np.array([video_width/2, video_height/2, video_width/10, video_height/5, video_width/2, -1, video_width/5, video_height/5])
context_EOS = np.array([video_width/2, video_height/2, video_width/8, video_height/3, video_width/2, video_height/2, video_width/6, video_height/6])
context_PAD = np.array([padding_token] * 8)

for vid in uniques:
    # subset input per video and append to list in which each array is one video (sequence)
    subset = input[input[:,0] == vid,:]

    #slice off video and frame count
    subset = subset[:,2:]

    # add fake embedded start token to beginning of seq
    subset = np.vstack((context_SOS, subset))

    # add fake embedded end token to end of seq
    subset = np.vstack((subset, context_EOS))

    sliced_seqs.append(subset)

    

# Debug
"""
sets_to_print = 1
print("\n \n Printing {} Tokenized subsets".format(sets_to_print))
for i in range(sets_to_print):
    print(sliced_seqs[i])
"""

# endregion

# region Batching

# calculate batch size
BATCH_SIZE = calculate_batch_size(len(folder)-videos_removed, MAX_BATCH_SIZE)

# endregion

# region Padding

# store all lengths to but end token at last point in sequence
real_sequence_lengths = [seq.shape[0] for seq in sliced_seqs]

print(f"real_sequence_lengths: {real_sequence_lengths}")

# find max length over all batches and sequences
max_length = max(seq.shape[0] for seq in sliced_seqs)

# Debugging
"""
for seq in sliced_seqs:
    print(f"sliced_seq.shape: {seq.shape}")
"""

# pad each sequence to max
padded_sequences = []
for seq in sliced_seqs:
    padded_seq = [tf.pad(seq, paddings=[[0, max_length - seq.shape[0]], [0, 0]], mode="CONSTANT", constant_values=padding_token)]
    padded_sequences.append(tf.stack(padded_seq))

# Slice of unnecessary axis 0 in (1,max_length,8): --> (max_length,8)
padded_sequences = [tf.stack(np.squeeze(seq, axis=0)) for seq in padded_sequences]

# Debug
"""
for seq in padded_sequences:
    print(f"padded seq.shape: {seq.shape}")

elements_to_print = 1
print("\n \n Printing {} Elements".format(elements_to_print))
for element_number in range(elements_to_print):
    print(padded_sequences[element_number])

print(f"max_length: {max_length}")
"""

# endregion

# region Labels

padded_lbl_sequences = []

seq_counter = 0
for seq in padded_sequences:
    seq_length = real_sequence_lengths[seq_counter]
    seq = seq.numpy()
    lbl = np.zeros((seq.shape[0], 1))

    #print(f"seq.shape: {seq.shape}")
    #print(f"lbl.shape: {lbl.shape}")

    for row in range(lbl.shape[0]):
        # sos tokens
        if np.all(np.array(seq[row])  == context_SOS):
            lbl[row] = start_token
            continue

        # padding tokens
        if np.all(seq[row] == padding_token):
            lbl[row] = padding_token
            continue

        # fixed tokens for last point in sequence
        if row == seq_length:
            lbl[row] = end_token
            continue

        # EOS token if hand overlaps with target (with a little wiggle room epsilon)
        object_loc = np.array([seq[row, 0], seq[row, 1]])
        hand_loc = np.array([seq[row, 4], seq[row, 5]])

        # this implementation entails that after an EOS token, there do not necessarily have to follow EOS tokens until the end
        # -> good or bad for training/ online application?
        if np.linalg.norm(hand_loc - object_loc) < EPSILON or row+1 == len(seq):
            lbl[row] = end_token
            continue

        # else neither start- nor end-token -> calculate angle
        # Calculate the angle between the line passing through the centers of two subsequent hand bounding boxes and the positive x-axis
        # calculate angle in degrees, range (0,360)
        lbl[row] = int((np.arctan2(seq[row+1,5] - seq[row,5], seq[row+1,4] - seq[row,4]) * 180 / np.pi) + 180)

        #print(seq[row+1,5], seq[row,5],seq[row+1,4], seq[row+1,4])
        #print(f"label_angle: {np.arctan2(seq[row+1,5] - seq[row,5], seq[row+1,4] - seq[row,4])}")

    padded_lbl_sequences.append(tf.stack(lbl))

    seq_counter += 1

# Convert outer lists from list to tensor
padded_lbl_sequences = tf.stack(padded_lbl_sequences)
padded_input_sequences = tf.stack(padded_sequences)

# Debugging: check padded sequences and corresponding labels
"""
with np.printoptions(threshold=sys.maxsize, suppress=True):
    print(padded_input_sequences)
    print()
    print(padded_lbl_sequences)
"""

# endregion

# region Dataset
    
# Convert outer lists from list to tensor
padded_lbl_sequences = tf.stack(padded_lbl_sequences)
padded_input_sequences = tf.stack(padded_sequences)

# 1. Create dataset
dataset = tf.data.Dataset.from_tensor_slices((padded_input_sequences, padded_lbl_sequences))

print(f"BATCH_SIZE: {BATCH_SIZE}")

# Create dataset of shape: (context, input), target
# Function to create correct labels for forced teacher learning

def create_context_input_target_structure(padded_input_sequence, padded_lbl_sequence):
    """
    print(f"padded_input_sequence: {padded_input_sequence}")
    print(f"padded_lbl_sequence: {padded_lbl_sequence}")
    print(f"padded_input_sequence shape: {tf.shape(padded_input_sequence)}")
    print(f"padded_lbl_sequence shape: {tf.shape(padded_lbl_sequence)}")
    """
    # we have: (x,y)
    # we want: ((x, y[:-1]), y[1:])
    # but we need to consider the padding!
    # --> instead of shortening to lose start and end token respectively, replace tokens with padding instead
    # --> but for target, we need to shift the sequence
    context = padded_input_sequence
    # Input: replace end token with padding token
    input = tf.where(padded_lbl_sequence==end_token, tf.constant(padding_token, dtype=tf.float64), padded_lbl_sequence)

    # Target: delete start token and add a padding token to the end to get constant sequence length
    target = padded_lbl_sequence[1:] # delete start token

    # get shape of single point in sequence that is to be replaced
    element_shape = tf.shape(padded_lbl_sequence[0])
    padding_to_add = tf.fill(element_shape, tf.constant(padding_token, dtype=tf.float64))

    target = tf.concat([target, [padding_to_add]], axis=0)  # add padding token to end of sequence to keep the same length

    #input_bb, label_angle = datapoint
    new_datapoint = context, input, target   # old without padding considered: new_datapoint = (input_bb, label_angle[:-1]), label_angle[1:]

    return new_datapoint

dataset = dataset.map(create_context_input_target_structure)

"""
samples_to_print = 1
print("\n \n Printing {} dataset".format(samples_to_print)) #.take(samples_to_print)
for i, (context_data, input_data, target_data) in enumerate(dataset):
    print(f"Training pair: {i}")
    print("Context Data:")
    print(context_data.shape)
    print(context_data.numpy())
    print()
    print("Input Data:")
    print(input_data.shape)
    print(input_data.numpy())
    print()
    print("Target Data:")
    print(target_data.shape)
    print(target_data.numpy())
    print()
"""

dataset = dataset.batch(BATCH_SIZE)


# Split dataset into train and test
num_batches = 0
for batch in dataset:
    num_batches += 1

print(f"num_batches in full dataset: {num_batches}")
train_size = int(0.8 * num_batches) # roughly 80% train split
test_size = num_batches - train_size

dataset = dataset.shuffle(100)
train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size).take(test_size)


# Debugging: check train dataset
""""
samples_to_print = 1
print("\n \n Printing {} train_ds".format(samples_to_print))
for i, (context_data, input_data, target_data) in enumerate(train_ds.take(samples_to_print)):
    print(f"Training pair: {i}")
    print("Context Data:")
    print(context_data.shape)
    print(context_data.numpy())
    print()
    print("Input Data:")
    print(input_data.shape)
    print(input_data.numpy())
    print()
    print("Target Data:")
    print(target_data.shape)
    print(target_data.numpy())
    print()


# Debugging: check test dataset
samples_to_print = 1
print("\n \n Printing {} test_ds".format(samples_to_print))
for i, (context_data, input_data, target_data) in enumerate(test_ds.take(samples_to_print)):
    print(f"Training pair: {i}")
    print("Context Data:")
    print(context_data.shape)
    print(context_data.numpy())
    print()
    print("Input Data:")
    print(input_data.shape)
    print(input_data.numpy())
    print()
    print("Target Data:")
    print(target_data.shape)
    print(target_data.numpy())
    print()
"""


# 2. Export dataset

# Define a file path for saving the zipped dataset
export_paths = ['./data/train_ds.zip', './data/test_ds.zip']

for export_path in export_paths:
    # Create a TFRecordWriter to save the zipped dataset
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    with tf.io.TFRecordWriter(export_path, options=options) as writer:
        ds = train_ds if export_path == export_paths[0] else test_ds
        for context, input, target in ds:
            tf_example = tf.train.Example(features=tf.train.Features(
                feature={
                    'context': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(context).numpy()])
                    ),
                    'input': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(input).numpy()])
                    ),
                    'target': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(target).numpy()])
                    )
                }
            ))
            writer.write(tf_example.SerializeToString())


# endregion
