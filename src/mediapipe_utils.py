import cv2
import os
import mediapipe as mp
import time
import json
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

def get_annotated_detection(detection: detection_pb2.Detection, width, height):
    """Extract detection keypoints and bounding box from results

    Args:
        detection: A detection proto message to be annotated.
        width: frame width
        height: frame height
    """
    if not detection.location_data:
        return [], None

    location = detection.location_data
    if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
        raise ValueError('LocationData must be relative for this function to work.')
  
    # Get keypoints
    keypoints = [_normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height) for keypoint in location.relative_keypoints]
    
    # Get bounding box if exists
    if not location.HasField('relative_bounding_box'):
        return keypoints, None

    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin, relative_bounding_box.ymin, width, height)
    rect_end_point = _normalized_to_pixel_coordinates(relative_bounding_box.xmin + relative_bounding_box.width, relative_bounding_box.ymin + +relative_bounding_box.height, width, height)
    rect_bbox = (rect_start_point, rect_end_point)

    return keypoints, rect_bbox

def get_annotated_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList, width, height, visibility_threshold: float = 0.5, presence_threshold: float = 0.5):
    """Extract landmarks from results

    Args:
        landmark_list: A normalized landmark list proto message to be annotated.
        width: frame width
        height: frame height
        visibility_threshold: landmark visibility threshold (default: 0.5)
        presence_threshold: landmark presence threshold (default: 0.5)
    """
    if not landmark_list:
        return []

    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < visibility_threshold) or
            (landmark.HasField('presence') and
            landmark.presence < presence_threshold)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    return [landmark_px for landmark_px in idx_to_coordinates.values()]


def get_video_metadata(file_name: Path):
    """
    Retrieve video metadata given an input file.
    :param file_name: input file name (video)
    :return: a tuple with fps and total frames
    """
    vidcap = cv2.VideoCapture(str(file_name))

    fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    total_frames = round(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = round(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vidcap.release()

    return (fps, total_frames, width, height)

def frame_stream_from_file(file_name: Path, start_frame=0, end_frame=None):
    """
    Produce a stream of video frames from a given input file.
    :param file_name: input file name (video)
    :return: pipe stream of video frames
    """
    vidcap = cv2.VideoCapture(str(file_name))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    curr_frame = start_frame
    success, image = vidcap.read()

    if not success:
        raise ValueError('Could not read the video file!')
    while success:
        curr_frame += 1
        # Convert the BGR image to RGB before processing.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        yield image
        if end_frame is not None and curr_frame == end_frame:
            break  # Chunk completed
        success, image = vidcap.read()
    vidcap.release()


def model_factory(model_type: str, min_detection_confidence: float, min_tracking_confidence: float):
    if model_type == 'mediapipe_holistic':
        # See: https://google.github.io/mediapipe/solutions/holistic.html
        return mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=2, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
    if model_type == 'mediapipe_pose':
        # See: https://google.github.io/mediapipe/solutions/pose.html
        return mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=min_detection_confidence)
    if model_type == 'mediapipe_objectron':
        # See: https://google.github.io/mediapipe/solutions/objectron.html
        return mp.solutions.objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=min_detection_confidence, model_name='Shoe')
    if model_type == 'mediapipe_face_detection':
        # See: https://google.github.io/mediapipe/solutions/face_detection
        return mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)
    if model_type == 'mediapipe_face_mesh':
        # See: https://google.github.io/mediapipe/solutions/face_mesh.html
        return mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    raise ValueError(f'Unknown model type specified ({model_type})')

def frame_drawing(model_type: str, source_frame: str, results: any):

    if results is None:
        return source_frame

    annotated_frame = source_frame.copy()
    annotated_frame.flags.writeable = True

    if model_type == 'mediapipe_holistic':
        if results.face_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.face_landmarks, mp.solutions.holistic.FACE_CONNECTIONS)
        if results.left_hand_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks is not None:            
            mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if results.pose_landmarks is not None:            
            mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
    
    elif model_type == 'mediapipe_pose':
        if results.pose_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(annotated_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    elif model_type == 'mediapipe_objectron':
        if results.detected_objects is not None:
            for detected_object in results.detected_objects:
                mp.solutions.drawing_utils.draw_landmarks(annotated_frame, detected_object.landmarks_2d, mp.solutions.objectron.BOX_CONNECTIONS)
                mp.solutions.drawing_utils.draw_axis(annotated_frame, detected_object.rotation, detected_object.translation)
    
    elif model_type == 'mediapipe_face_detection':
        if results.detections is not None:
            for detection in results.detections:
                mp.solutions.drawing_utils.draw_detection(annotated_frame, detection)
    
    elif model_type == 'mediapipe_face_mesh':
        drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(annotated_frame, face_landmarks, mp.solutions.face_mesh.FACE_CONNECTIONS, drawing_spec, drawing_spec)
    
    annotated_frame.flags.writeable = False
    return annotated_frame

def results_dump(model_type: str, output_file: any, results: any, frame_idx: int, width: int, height: int):
    
    if results is None:
        return

    frame_annotation_dump = { "frame": frame_idx, "keypoints": {} }

    if model_type == 'mediapipe_holistic':
        frame_annotation_dump["keypoints"]["face"] = get_annotated_landmarks(results.face_landmarks, width, height)
        frame_annotation_dump["keypoints"]["left_hand"] = get_annotated_landmarks(results.left_hand_landmarks, width, height)
        frame_annotation_dump["keypoints"]["right_hand"] = get_annotated_landmarks(results.right_hand_landmarks, width, height)
        frame_annotation_dump["keypoints"]["pose"] = get_annotated_landmarks(results.pose_landmarks, width, height)
        
    output_file.write(json.dumps(frame_annotation_dump))
    output_file.write('\n')

def processor(processor_index, model_type, input_file, output_folder, frame_start, frame_end, fps, width, height, min_detection_confidence, min_tracking_confidence):

    progress_bar = tqdm(range(frame_end-frame_start), desc=f'[{processor_index:02}] Processing frames ({frame_start:05}-{frame_end:05})', position=processor_index)

    model = model_factory(model_type, min_detection_confidence, min_tracking_confidence)

    idx = frame_start
       
    # Prepare partial chunk output files
    fourcc = cv2.cv2.VideoWriter_fourcc(*'mp4v')
    chunk_video_output_file_path = Path.joinpath(output_folder, f'{input_file.stem}_{frame_start}_{frame_end}.mp4')
    chunk_video_out = cv2.VideoWriter(str(chunk_video_output_file_path), fourcc, fps, (width, height))

    chunk_annotations_output_file_path = Path.joinpath(output_folder, f'{input_file.stem}_{frame_start}_{frame_end}.json')
    chunk_annotations_out = open(chunk_annotations_output_file_path, "w")
    
    chunk_start_time = time.time()
    frame_proc_time_total = 0.0

    frame_stream = frame_stream_from_file(input_file, frame_start, frame_end)

    for frame in frame_stream:

        frame_start_time = time.time()
        idx += 1

        results = model.process(frame)

        # Draw annotations on each frame, depending on the selected model
        annotated_frame = frame_drawing(model_type, frame, results)
        
        # Save frame in temp video file
        chunk_video_out.write(annotated_frame)

        # Store frame annotations in partial JSON file, depending on the selected model
        results_dump(model_type, chunk_annotations_out, results, idx, width, height)
        
        frame_proc_time_total += time.time() - frame_start_time

        progress_bar.update(1)
    
    model.close()
    chunk_video_out.release()
    chunk_annotations_out.close()
    progress_bar.close()

    return chunk_video_output_file_path, chunk_annotations_output_file_path, frame_proc_time_total / (frame_end - frame_start), time.time() - chunk_start_time


def mediapipe_multiproc(input_file: Path, output_folder: Path, processor_count: int, model_type: str, delete_temp_files: bool, min_detection_confidence: float, min_tracking_confidence: float):

    start_time = time.time()

    output_video_file_path = Path.joinpath(output_folder, f'{model_type}_{input_file.stem}.mp4')
    output_annotations_file_path = Path.joinpath(output_folder, f'{model_type}_{input_file.stem}.json')
    
    print(f'Processing {input_file}')

    fps, total_frames, width, height = get_video_metadata(input_file)

    chunk_size = total_frames // processor_count

    # Multi-proc
    tasks = []
    for processor_index in range(processor_count):
        start_frame = start_frame = processor_index * chunk_size
        end_frame = start_frame + chunk_size
        if processor_index == (processor_count - 1):
            # last processor may handle more frames, due to non-exact divisibility of total frames by number of processors
            if end_frame < total_frames:
                end_frame = total_frames
        tasks.append(delayed(processor)(processor_index, model_type, input_file, output_folder, start_frame, end_frame, fps, width, height, min_detection_confidence, min_tracking_confidence))
    
    processed_file_chunks = Parallel(n_jobs=processor_count)(tasks)

    # Prepare output video file (concatenate all video chunks)
    save_progress_bar = tqdm(range(total_frames), desc='Saving output video file', position=0)
    fourcc = cv2.cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_video_file_path), fourcc, fps, (width, height))
    for input_video_chunk_file, _, frame_processing_mean_time, chunk_processing_total_time in processed_file_chunks:
        for frame in frame_stream_from_file(input_video_chunk_file):
            out_video.write(frame)
            save_progress_bar.update(1)
    save_progress_bar.close()
    out_video.release()

    # Prepare annotations file (concatenate all annotation file chunks)
    save_progress_bar = tqdm(range(len(processed_file_chunks)), desc='Saving output annotations file', position=0)
    with open(output_annotations_file_path, "w") as output_json_file:
        for _, chunk_annotations_output_file_path, _, _ in processed_file_chunks:
            with open(chunk_annotations_output_file_path, "r") as input_json_file:
                for line in input_json_file:
                    output_json_file.write(line)
            save_progress_bar.update(1)
    save_progress_bar.close()

    if delete_temp_files:
        for input_video_chunk_file, chunk_annotations_output_file_path, _, _ in processed_file_chunks:
            os.remove(input_video_chunk_file)
            os.remove(chunk_annotations_output_file_path)

    print(f'File processed in {time.time() - start_time:.2f}s')
