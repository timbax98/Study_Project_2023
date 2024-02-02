# region Setup

import os
import sys
import zipfile
import cv2
import numpy as np
import ultralytics
import tensorflow as tf
import yolov5

# Configure Hyperparameters
MAX_BATCH_SIZE = 32
# alternative: use video resolution dependent epsilon; only care about width atm
#EPSILON = vid_res[(batch+1)*(seq+1)][1] / 12
EPSILON = 100 # what is a good range?
MAX_MISSING = 10 # 999999999 placeholder to remove no videos
MIN_DETECTS = 1/4

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

    # Loop through each frame and detection
    for frame_num in range(frame_count):  #range(frame_count)
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

            # handle first frame (no previous detection)
            if frame_num == 0:
                last_detection = np.full((1, 7), np.nan)
                if n > 1:
                    # assumption: hand moves into the frame from the bottom
                    data = data[np.argmax(data[:,3])].reshape((1, 7)) # not taken as last_detection!

            # if multiple detections per frame, take the one closest to last frame's BB
            if n > 1 and not np.isnan(last_detection).any():
                distances = np.linalg.norm(data[:, 2:4] - last_detection[:, 2:4], axis=1)
                closest_index = np.argmin(np.abs(distances))
                data = data[closest_index].reshape((1, 7))

            # update the last detection if there was any
            if len(data) > 0:
                # If initially multiple detections occur in a later frame, this has to be reduced to 1D
                if len(data) > 1: # we can take len() because it is 2D
                    mean = np.mean(data[:, 2:6], axis=0)
                    data = data[0].reshape((1, 7))
                    data[:, 2:6] = mean
                last_detection = data
        
        bbs = np.concatenate((bbs, data), axis=0)

    # Release the video capture object
    cap.release()

    return bbs , classes, remove_vid


def Center(videopath, labelpath, bounding_boxes):

    """
    Create a video where a target object is always centered.

    Args:
    videopath -- path to video to center
    labelpath -- location to export centered video to
    bounding_boxes -- bbs of object to center on
    width -- output video dim x, default=1920
    height -- output video dim y, default=1120
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

    for box_pos in bbs:
        if np.isnan(box_pos).any():
            # if all values were nan, then 0 would be the max, so the corrected video would be the original video
            x_dists.append(0)
            y_dists.append(0)
        else:
            # get center of bb
            #center_x_bb = (box_pos[0] + box_pos[2]) / 2
            #center_y_bb = (box_pos[1] + box_pos[3]) / 2
            center_x_bb = box_pos[0]
            center_y_bb = box_pos[1]
            # calculate distances
            x_dists.append(abs(center_x-center_x_bb))
            y_dists.append(abs(center_y-center_y_bb))

    video_corrected = np.zeros((len(video),
                              int(height + max(y_dists)*2),
                              int(width + max(x_dists)*2),
                              3), dtype=int)

    # 3. Center video
    # get coords for placement height and width. may be switched
    start_row = (video_corrected.shape[1] - height) // 2
    start_col = (video_corrected.shape[2] - width) // 2

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
            video_corrected[idx][fixed_start_row:fixed_end_row,
                              fixed_start_col:fixed_end_col] = frame

            #Save offsets
            offsets[idx,:] = x_offset, y_offset

    # 4. Save video (Debug)
    """
    video_corrected = np.uint8(video_corrected)

    height_new = int(video_corrected.shape[1])
    width_new = int(video_corrected.shape[2])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(labelpath[:-4]+'_centered.mp4', fourcc, FPS, (width_new, height_new), True)
    for idx in range(len(video)):
        out.write(video_corrected[idx])
    out.release()
    """

    return offsets, (start_row, start_col)


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


def delete_overlap_rows(hands, targets, frameCount, buffer=5):
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

    print(f"\n Video {i+1}/{len(folder)}: {video}")

    # Get length of input array (number of frames)
    cap = cv2.VideoCapture(folderpath+video)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Initial total frame count: {frameCount} \n")


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
    #print("Hand BBs")
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
        # returns the string name without '00_centered.mp4' ending
        target_class = video[:-7]

    # Get class number from lookup table
    target_class = list(classes_objs.keys())[list(classes_objs.values()).index(target_class)]
    # get median value of all classes in hand detections to get "true" label 
    # --> results in too few detections --> we filter in GetBoundingBoxes (only hands)
    #hand_class = int(np.median(bbs_hands[:, 2]))

    target_bbs = bbs_objects[(bbs_objects[:, 2] == target_class)]
    target_bbs = target_bbs[:, [0,1,3,4,5,6]]
    #hand_bbs = bbs_hands[(bbs_hands[:, 2] == hand_class)]
    hand_bbs = bbs_hands[:, [0,1,3,4,5,6]] # hand_bbs[:, [0,1,3,4,5,6]]

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
        print(f"x_detects_target: {len(x_detects_target)}, possible detections: {possible_detections}, threshold: {possible_detections * MIN_DETECTS}")
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
        print(f"x_detects_target: {len(x_detects_hand)}, possible detections: {possible_detections}, threshold: {possible_detections * MIN_DETECTS}")
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
    offsets, start_cords = Center(folderpath+video, labelpath+video, vid_input[:, 2:6])

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


    ### 5. Input concatenation ###

    bbs = np.concatenate((new_target_bbs, new_hand_bbs), axis=1)
    bbs = bbs[~np.isnan(bbs).any(axis=1)] # remove frames without detections
    input = np.concatenate((input, bbs), axis=0)


print("\n Input created.")
print(f"Number of videos (total): {len(folder)}, Removed: {videos_removed}, Left: {len(folder)-videos_removed}")

# Debugging: check input
with np.printoptions(threshold=sys.maxsize, suppress=True):
    print("\n \n Printing the first 200 input rows.")
    print(input[:200])

# endregion

# region Tokenization

# define tokens; has to be unusual in sequences (0 to max dim of video res) as well as labels (radiands: -2pi to 2pi)
start_token = -333 # for seqs and labels
end_token = -666 # for labels only (sequences do not need an end token, as they are all separated by start tokens)
padding_token = -999 # for seqs only (labels are generated from padded sequences)

# Slice data into single videos
uniques = np.unique(input[:,0])
sliced_seqs = []
sliced_labels = []
#vid_res = [] # save video center for resolution-dependent epsilon later

for vid in uniques:
    # subset input per video and append to list in which each array is one video (sequence)
    subset = input[input[:,0] == vid,:]
    # this is the target object detection center x and y, i.e. half of the video's resolution
    #vid_res.append((subset[0,2], subset[0,3]))
    # prepend start token to sequence
    subset = np.vstack([np.full((1, subset.shape[1]), start_token), subset])
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

# create list that contains lists of size batch_size, each one containing single videos (sequences) as arrays
BATCH_SIZE = calculate_batch_size(len(folder)-videos_removed, MAX_BATCH_SIZE)
seq_batches = [sliced_seqs[i:i+BATCH_SIZE] for i in range(0, len(sliced_seqs), BATCH_SIZE)]

# endregion

# region Padding

# Pad sequences to the maximum sequence length within each batch
padded_seq_batches = []
for batch in seq_batches:
    max_length = max(seq.shape[0] for seq in batch)
    # [:,2:] slices the sequence to remove columns for video count and frame count
    padded_batch = [tf.pad(seq[:,2:], paddings=[[max_length - seq.shape[0], 0], [0, 0]], mode="CONSTANT", constant_values=padding_token) for seq in batch]
    padded_seq_batches.append(tf.stack(padded_batch))

# Debug
"""
batches_to_print = 1
print("\n \n Printing {} Padded batches".format(batches_to_print))
for i in range(batches_to_print):
    print(padded_seq_batches[i])
"""

# endregion

# region Labels

padded_lbl_batches = []
for batch in padded_seq_batches:
    labels = []

    for seq in batch:
        seq = seq.numpy()
        lbl = np.zeros((seq.shape[0], 1))

        for row in range(lbl.shape[0]):

            # sos tokens
            if np.all(seq[row] == start_token):
                lbl[row] = start_token
                continue

            # padding tokens
            if np.all(seq[row] == padding_token):
                lbl[row] = padding_token
                continue

            # EOS token if hand overlaps with target (with a little wiggle room epsilon)
            object_loc = np.array(seq[row, 0], seq[row, 1])
            hand_loc = np.array(seq[row, 4], seq[row, 5])

            # this implementation entails that after an EOS token, there do not necessarily have to follow EOS tokens until the end
            # -> good or bad for training/ online application?
            if np.linalg.norm(hand_loc - object_loc) < EPSILON or row+1 == len(seq):
                lbl[row] = end_token
                continue

            # else neither start- nor end-token -> calculate angle
            lbl[row] = np.arctan2(seq[row,7] - seq[row+1,7], seq[row,6] - seq[row+1,6]) # angle between current and next frame

        labels.append(lbl)

    padded_lbl_batches.append(tf.stack(labels))

# Debugging: check padded sequences and corresponding labels
"""
with np.printoptions(threshold=sys.maxsize, suppress=True):
    print(padded_seq_batches)
    print()
    print(padded_lbl_batches)
"""

# endregion

# region Dataset

# 1. Create dataset

# Define a generator function to yield each sequence batch
def seq_generator():
    for batch in padded_seq_batches:
        yield batch

# Define a generator function to yield each label batch
def lbl_generator():
    for batch in padded_lbl_batches:
        yield batch

# Create a tf.data.Dataset from the generator
X = tf.data.Dataset.from_generator(seq_generator, output_signature=tf.TensorSpec(shape=(BATCH_SIZE, None, 8), dtype=tf.float32))
Y = tf.data.Dataset.from_generator(lbl_generator, output_signature=tf.TensorSpec(shape=(BATCH_SIZE, None, 1), dtype=tf.float32))
train_ds = tf.data.Dataset.zip((X, Y))

# Debugging: check train dataset
samples_to_print = 1
print("\n \n Printing {} train_ds".format(samples_to_print))
for i, (train_data, label_data) in enumerate(train_ds.take(samples_to_print)):
    if i < 5: # print at most the first 5 examples
        print(f"Training pair: {i}")
        print("Train Data:")
        print(train_data.shape)
        print(train_data.numpy())
        print()
        print("Label Data:")
        print(label_data.shape)
        print(label_data.numpy())
        print()


# 2. Export dataset

# Define a file path for saving the zipped dataset
export_path = './data/train_ds.zip'

# Create a TFRecordWriter to save the zipped dataset
options = tf.io.TFRecordOptions(compression_type='GZIP')
with tf.io.TFRecordWriter(export_path, options=options) as writer:
    for example_x, example_y in train_ds:
        tf_example = tf.train.Example(features=tf.train.Features(
            feature={
                'x': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example_x).numpy()])
                ),
                'y': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(example_y).numpy()])
                )
            }
        ))
        writer.write(tf_example.SerializeToString())

# endregion