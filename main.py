from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
import argparse

import numpy as np

def main(args):
    """
    Main function to demonstrate how to use the football analysis project.
    This script will walk you through loading models, assigning clubs, tracking objects and players, and processing the video.
    """

    # 1. Load the object detection model
    # Adjust the 'conf' value as per your requirements.
    obj_tracker = ObjectTracker(
        model_path='models/weights/object-detection.pt',    # Object Detection Model Weights Path
        conf=.5,                                            # Object Detection confidence threshold
        ball_conf=.05                                        # Ball Detection confidence threshold
    )

    # 2. Load the keypoints detection model
    # Adjust the 'conf' and 'kp_conf' values as per your requirements.
    kp_tracker = KeypointsTracker(
        model_path='models/weights/keypoints-detection.pt', # Keypoints Model Weights Path
        conf=.3,                                            # Field Detection confidence threshold
        kp_conf=.5,                                         # Keypoint confidence threshold
    )

    # Create a ClubAssigner Object to automatically assign players and goalkeepers 
    # to their respective clubs based on jersey colors (player) and positions (goalkeeper)
    club_assigner = ClubAssigner(obj_tracker, video_source=args.input)
    club1 = club_assigner.club1
    club2 = club_assigner.club2

    # 4. Initialize the BallToPlayerAssigner object
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    # 5. Define the keypoints for a top-down view of the football field (from left to right and top to bottom)
    # These are used to transform the perspective of the field.
    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],             # 0-5 (left goal line)
        [32, 122], [32, 229],                                                # 6-7 (left goal box corners)
        [64, 176],                                                           # 8 (left penalty dot)
        [96, 57], [96, 122], [96, 229], [96, 293],                           # 9-12 (left penalty box)
        [263, 0], [263, 122], [263, 229], [263, 351],                        # 13-16 (halfway line)
        [431, 57], [431, 122], [431, 229], [431, 293],                       # 17-20 (right penalty box)
        [463, 176],                                                          # 21 (right penalty dot)
        [495, 122], [495, 229],                                              # 22-23 (right goal box corners)
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351], # 24-29 (right goal line)
        [210, 176], [317, 176]                                               # 30-31 (center circle leftmost and rightmost points)
    ])

    # 6. Initialize the video processor
    # This processor will handle every task needed for analysis.
    processor = FootballVideoProcessor(obj_tracker,                                   # Created ObjectTracker object
                                       kp_tracker,                                    # Created KeypointsTracker object
                                       club_assigner,                                 # Created ClubAssigner object
                                       ball_player_assigner,                          # Created BallToPlayerAssigner object
                                       top_down_keypoints,                            # Created Top-Down keypoints numpy array
                                       field_img_path='input_videos/field_2d_v2.png', # Top-Down field image path
                                       save_tracks_dir='output_videos',               # Directory to save tracking information.
                                       draw_frame_num=True                            # Whether or not to draw current frame number on 
                                                                                      #the output video.
                                       )
    
    # 7. Process the video
    # Specify the input video path and the output video path. 
    # The batch_size determines how many frames are processed in one go.
    process_video(processor,                                # Created FootballVideoProcessor object
                  video_source=args.input, # Video source (in this case video file path)
                  output_video=args.output,    # Output video path (Optional)
                  batch_size=10                             # Number of frames to process at once
                  )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Footbal Analysis Tool')
    parser.add_argument('-i', '--input', type=str, required=True, help='path of football video')
    parser.add_argument('-o', '--output', type=str, default='result.mp4', help='path of output video')
    args = parser.parse_args()


    main(args)
