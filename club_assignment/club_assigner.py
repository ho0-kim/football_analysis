from .club import Club

import os
from sklearn.cluster import KMeans
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional

import supervision as sv
from tracking import ObjectTracker
import torch
from sports.common.team import TeamClassifier

from collections import Counter, deque

class ClubAssigner:
    def __init__(self, obj_tracker: ObjectTracker, video_source:str, 
                 images_to_save: int = 0, images_save_path: Optional[str] = None) -> None:
        """
        Initializes the ClubAssigner with club information and image saving parameters.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
            images_to_save (int): The number of images to save for analysis.
            images_save_path (Optional[str]): The directory path to save images.
        """
        # Saving images for analysis
        self.images_to_save = images_to_save
        self.output_dir = images_save_path

        if not images_save_path:
            images_to_save = 0
            self.saved_images = 0
        else:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        
            self.saved_images = len([name for name in os.listdir(self.output_dir) if name.startswith('player')])


        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)

        self.club1, self.club2 = self._create_clubs(obj_tracker, video_source)

        self.model = ClubAssignerModel(self.club1, self.club2)
        self.club_colors: Dict[str, Any] = {
            self.club1.name: self.club1.player_jersey_color,
            self.club2.name: self.club2.player_jersey_color
        }
        self.goalkeeper_colors: Dict[str, Any] = {
            self.club1.name: self.club1.goalkeeper_jersey_color,
            self.club2.name: self.club2.goalkeeper_jersey_color
        }

        self.previous_club:Dict[int, int] = {}
        self.pred_history:Dict[Any, deque] = {}


    def apply_mask(self, image: np.ndarray, green_threshold: float = 0.08) -> np.ndarray:
        """
        Apply a mask to an image based on green color in HSV space. 
        If the mask covers more than green_threshold of the image, apply the inverse of the mask.

        Args:
            image (np.ndarray): An image to apply the mask to.
            green_threshold (float): Threshold for green color coverage.

        Returns:
            np.ndarray: The masked image.
        """
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the green color range in HSV
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])

        # Create the mask
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Count the number of masked pixels
        total_pixels = image.shape[0] * image.shape[1]
        masked_pixels = cv2.countNonZero(cv2.bitwise_not(mask))
        mask_percentage = masked_pixels / total_pixels
        
        if mask_percentage > green_threshold:
            # Apply inverse mask
            return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        else:
            # Apply normal mask
            return image

    def clustering(self, img: np.ndarray) -> Tuple[int, int, int]:
        """
        Perform K-Means clustering on an image to identify the dominant jersey color.

        Args:
            img (np.ndarray): The input image.

        Returns:
            Tuple[int, int, int]: The jersey color in BGR format.
        """
        # Reshape image to 2D array
        img_reshape = img.reshape(-1, 3)
        
        # K-Means clustering
        self.kmeans.fit(img_reshape)
        
        # Get Cluster Labels
        labels = self.kmeans.labels_
        
        # Reshape the labels into the image shape
        cluster_img = labels.reshape(img.shape[0], img.shape[1])

        # Get Jersey Color
        corners = [cluster_img[0, 0], cluster_img[0, -1], 
                   cluster_img[-1, 0], cluster_img[-1, -1], 
                   cluster_img[img.shape[0] * 3 // 4, 0], 
                   cluster_img[img.shape[0] * 3 // 4, -1]]
        bg_cluster = max(set(corners), key=corners.count)

        # The other cluster is a player cluster
        player_cluster = 1 - bg_cluster

        jersey_color_bgr = self.kmeans.cluster_centers_[player_cluster]
        
        return (int(jersey_color_bgr[2]), int(jersey_color_bgr[1]), int(jersey_color_bgr[0]))

    def save_player_image(self, img: np.ndarray, player_id: int, is_goalkeeper: bool = False) -> None:
        """
        Save the player's image to the specified directory.

        Args:
            img (np.ndarray): The image of the player.
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.
        """
        # Use 'goalkeeper' or 'player' prefix based on is_goalkeeper flag
        prefix = 'goalkeeper' if is_goalkeeper else 'player'
        filename = os.path.join(self.output_dir, f"{prefix}_{player_id}.png")
        if os.path.exists(filename):
            return
        cv2.imwrite(filename, img)
        print(f"Saved {prefix} image: {filename}")
        # Increment the count of saved images
        self.saved_images += 1

    def get_jersey_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], player_id: int, is_goalkeeper: bool = False) -> Tuple[int, int, int]:
        """
        Extract the jersey color from a player's bounding box in the frame.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.

        Returns:
            Tuple[int, int, int]: The jersey color in BGR format.
        """
        # Save player images only if needed
        if self.saved_images < self.images_to_save:
            img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            img_top = img[0:img.shape[0] // 2, :] 
            if player_id > -1:
                self.save_player_image(img_top, player_id, is_goalkeeper)  # Pass is_goalkeeper here

        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        masked_img = self.apply_mask(img, green_threshold=0.08)
        jersey_color = self.clustering(masked_img)
        
        return jersey_color

    def get_player_club(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], player_id: int, player_tracks:Dict[int, Any], is_goalkeeper: bool = False) -> Tuple[str, int]:
        """
        Determine the club associated with a player based on their jersey color.

        Args:
            frame (np.ndarray): The current video frame.
            bbox (Tuple[int, int, int, int]): The bounding box coordinates (x1, y1, x2, y2).
            player_id (int): The unique identifier for the player.
            is_goalkeeper (bool): Flag to indicate if the player is a goalkeeper.

        Returns:
            Tuple[str, int]: The club name and the predicted class index.
        """
        color = self.get_jersey_color(frame, bbox, player_id, is_goalkeeper)
        pred = self.model.predict(color, bbox, player_tracks, is_goalkeeper)
        pred = self._average_filter(player_id, pred)
        
        return list(self.club_colors.keys())[pred], pred
    

    def assign_clubs(self, frame: np.ndarray, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
        """
        Assign clubs to players and goalkeepers based on their jersey colors.

        Args:
            frame (np.ndarray): The current video frame.
            tracks (Dict[str, Dict[int, Any]]): The tracking data for players and goalkeepers.

        Returns:
            Dict[str, Dict[int, Any]]: The updated tracking data with assigned clubs.
        """
        tracks = tracks.copy()

        for track_type in ['player', 'goalkeeper']:
            for player_id, track in tracks[track_type].items():
                bbox = track['bbox']
                is_goalkeeper = (track_type == 'goalkeeper')
                club, _ = self.get_player_club(frame, bbox, player_id, tracks['player'], is_goalkeeper)
                
                tracks[track_type][player_id]['club'] = club
                tracks[track_type][player_id]['club_color'] = self.club_colors[club]
        
        return tracks
    
    def _average_filter(self, player_id:int, pred:int) -> int:
        if player_id not in self.pred_history:
            self.pred_history[player_id] = deque(maxlen=75)
            self.previous_club[player_id] = pred
        
        self.pred_history[player_id].append(pred)
        average = sum(self.pred_history[player_id]) / len(self.pred_history[player_id])

        if average != 0.5: self.previous_club[player_id] = round(average)
        return self.previous_club[player_id]
    
    def _sample_frame(self, video_path:str, interval:int=30, maximum:int=30) -> List[np.ndarray]:
            video_info = sv.VideoInfo.from_video_path(video_path)
            end = min(interval * maximum, video_info.total_frames-1)

            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            frames = []

            while True:
                ret, frame = cap.read()

                if not ret or frame_count == end:
                    break

                if frame_count % interval == 0:
                    frames.append(frame)

                frame_count += 1

            cap.release()

            return frames
    
    def _create_clubs(self, obj_tracker: ObjectTracker, video_source:str) -> Tuple[Club, Club]:
        """
        Initializes Club objects by recognizing jersey colors of players and goalkeepers
        """
        frames = self._sample_frame(video_source)
        results = obj_tracker.detect(frames)

        jersey_colors = []
        for result, frame in zip(results, frames):
            tracks = obj_tracker.track(result)
            players = tracks['player']
            for player in players.values():
                scaled_bbox = player['bbox']
                jersey_colors.append(self.get_jersey_color(frame, scaled_bbox, player_id=-1))
        obj_tracker.reset_tracker()

        self.kmeans.fit(jersey_colors)

        club1 = Club(name='Club1',
                     player_jersey_color=self.kmeans.cluster_centers_[0],
                     goalkeeper_jersey_color=(0,0,0))
        club2 = Club(name='Club2',
                     player_jersey_color=self.kmeans.cluster_centers_[1],
                     goalkeeper_jersey_color=(0,0,0))
        
        return club1, club2

class ClubAssignerModel:
    """ Assign club based on jersey color """
    def __init__(self, club1: Club, club2: Club) -> None:
        """
        Initializes the ClubAssignerModel with jersey colors for the clubs.

        Args:
            club1 (Club): The first club object.
            club2 (Club): The second club object.
        """
        self.club1_name = club1.name
        self.club2_name = club2.name
        self.player_centroids = np.array([club1.player_jersey_color, club2.player_jersey_color])
        self.goalkeeper_centroids = np.array([club1.goalkeeper_jersey_color, club2.goalkeeper_jersey_color])

    def predict(self, extracted_color: Tuple[int, int, int], bbox: Tuple[int, int, int, int], player_tracks:Dict[int, Any], is_goalkeeper: bool = False) -> int:
        """
        Predict the club for a given jersey color based on the centroids (Player).
        Predict the club based on distance from field players (Goalkeeper).

        Args:
            extracted_color (Tuple[int, int, int]): The extracted jersey color in BGR format.
            is_goalkeeper (bool): Flag to indicate if the color is for a goalkeeper.

        Returns:
            int: The index of the predicted club (0 or 1).
        """
        if is_goalkeeper:
            goalkeeper_xy = [(bbox[0]+bbox[2])/2, bbox[3]]

            players_xy:Dict[str, Any] = {self.club1_name: [], self.club2_name: []}
            for player_track in player_tracks.values():
                club = player_track['club']
                pl_bbox = player_track['bbox']
                players_xy[club].append([(pl_bbox[0]+pl_bbox[2])/2, pl_bbox[3]])
                
            club1_centroid = np.array(players_xy[self.club1_name]).mean(axis=0)
            club2_centroid = np.array(players_xy[self.club2_name]).mean(axis=0)

            dist_1 = np.linalg.norm(goalkeeper_xy - club1_centroid)
            dist_2 = np.linalg.norm(goalkeeper_xy - club2_centroid)

            return 0 if dist_1 < dist_2 else 1
        else:
            # Calculate distances
            distances = np.linalg.norm(extracted_color - self.player_centroids, axis=1)
            return np.argmin(distances)

