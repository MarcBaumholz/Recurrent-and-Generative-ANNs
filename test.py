import logging
import math
import os
import random
from multiprocessing import Pool, cpu_count
from typing import Tuple, List, Dict
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, Sequence
from tqdm import tqdm
from config_loader import load_config, Config  # Import the dataclass-based config loader

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the configuration file using dataclasses
config: Config = load_config("config.yaml")

# Constants from the configuration
frame_step = config.video_processing.frame_step
video_sample_length = config.video_processing.video_sample_length
length_video = config.video_processing.length_video
image_size = tuple(config.video_processing.image_size)
drop_random_frames_count = config.video_processing.drop_random_frames_count

class VideoProcessor_val:
    """
    Handles video loading, frame extraction, and batch creation.
    """

    @staticmethod
    def load_frames_from_folder(folder_path: str, frame_step: int, image_size=image_size) -> List[np.ndarray]:
        """
        Load every `frame_step` frame from the specified folder and resize.
        """
        frames = []
        filenames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tiff')])
        
        for idx, filename in enumerate(filenames):
            if idx % frame_step == 0:
                filepath = os.path.join(folder_path, filename)
                frame = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                frame_resize = cv2.resize(frame, image_size)
                frame_normalized = DataPreprocessor_train.preprocess_frame(frame_resize)
                frames.append(frame_normalized)
        return frames

    @staticmethod
    def create_sliding_window_batches(frames: List[np.ndarray], window_size: int, step_size: int) -> List[np.ndarray]:
        """
        Create overlapping batches using a sliding window approach.
        """
        return [frames[start:start + window_size]
                for start in range(0, len(frames) - window_size + 1, step_size)]

    @staticmethod
    def binary_label_conversion(labels: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Convert labels to binary labels where 1 = 'stress' and 0 = 'exercise'.
        """
        binary_labels = np.where(labels == 'stress', 1, 0)
        num_exercises = len(np.unique(binary_labels))
        return binary_labels, num_exercises
    @staticmethod
    def process_directory(dir_path, frame_step=frame_step, window_size=video_sample_length, step_size=10, label_counts=None):
        """
        Load and process all frames from a directory, returning batches and corresponding labels.
        """
        label = os.path.basename(dir_path)  # The folder name is the label
        
        # Initialize label counts if not provided
        if label_counts is None:
            label_counts = {}
    
        batches = []
    
        for subdir, _, _ in os.walk(dir_path):
            frames = VideoProcessor_val.load_frames_from_folder(subdir, frame_step=frame_step)
            
            if len(frames) >= window_size:  # Ensure at least one sequence can be generated
                video_batches = VideoProcessor_val.create_sliding_window_batches(frames, window_size=window_size, step_size=step_size)
    
                for batch in video_batches:
                    batches.append((batch, label))
                    label_counts[label] = label_counts.get(label, 0) + 1
    
        
        return batches
    @staticmethod 
    def process_directory_wrapper(args):
        return VideoProcessor_val.process_directory(*args)
    @staticmethod
    def load_all_videos_and_labels(base_dir, frame_step=frame_step, window_size=video_sample_length, step_size=10, validation=False):
        """
        Load all videos and generate corresponding labels based on folder names.
    
        :param base_dir: The root directory containing class folders with videos.
        :param frame_step: Step size for loading frames from each video.
        :param window_size: Size of the sliding window to generate sequences.
        :param step_size: Step size for the sliding window.
        :param validation: Whether this is validation data. If True, skip balancing. Default is False.
        :return: A tuple of (videos, labels).
            - videos: List of arrays, where each array represents a video.
            - labels: List of labels corresponding to each video.
        """
        videos = []
        labels = []
        total_non_stress_count = 0
    
        # Get directory paths for all categories
        dir_paths = [os.path.join(base_dir, dir_name) for dir_name in os.listdir(base_dir)]
    
        # Separate stress and non-stress categories
        stress_dirs = [dir_path for dir_path in dir_paths if "stress" in dir_path.split("/validation/")[-1]]
        non_stress_dirs = [dir_path for dir_path in dir_paths if dir_path not in stress_dirs]
    
        # Load non-stress categories first
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(VideoProcessor_val.process_directory_wrapper, 
                                          [(dir_path, frame_step, window_size, step_size) for dir_path in non_stress_dirs]), 
                                total=len(non_stress_dirs), desc="Processing Non-Stress Directories"))
    
        for result in results:
            if result:
                video_batches, batch_labels = zip(*result)
                videos.extend(video_batches)
                labels.extend(batch_labels)
                total_non_stress_count += len(batch_labels)
    
        print(f"Total non-stress videos loaded: {total_non_stress_count}")
    
        # Load stress category only if it appears after "validation"
        stress_loaded_count = 0
        with Pool(processes=cpu_count()) as pool:
            stress_results = list(tqdm(pool.imap(VideoProcessor_val.process_directory_wrapper, 
                                                 [(dir_path, frame_step, window_size, step_size) for dir_path in stress_dirs]), 
                                       total=len(stress_dirs), desc="Processing Stress Directories"))
    
        for result in stress_results:
            if result:
                video_batches, batch_labels = zip(*result)
                
                for batch, label in zip(video_batches, batch_labels):
                    if stress_loaded_count >= total_non_stress_count:
                        break
                    videos.append(batch)
                    labels.append(label)
                    stress_loaded_count += 1
    
        print(f"Stress videos loaded: {stress_loaded_count}")
    
        videos = np.array(videos)
        labels = np.array(labels)
    
        return np.array(videos), np.array(labels)

class DirectoryProcessor:
    """
    Handles loading and processing of directories, including balancing binary labels.
    """

    @staticmethod
    def process_directory(dir_path: str) -> Tuple[str, str]:
        """
        Return the directory path and its binary label ('stress' or 'exercise').
        """
        label = os.path.basename(dir_path).lower()
        binary_label = 'stress' if label == "stress" else 'exercise'
        return dir_path, binary_label

    @staticmethod
    def balance_binary_pointers(dir_paths_by_label: Dict[str, List[str]], max_counts: int, debug: bool = False) -> \
            Tuple[List[str], List[str]]:
        """
        Ensure each binary category ('stress' or 'exercise') has exactly `max_counts` entries.
        """
        balanced_paths = []
        balanced_labels = []

        for label, paths in dir_paths_by_label.items():
            if debug:
                # Debug mode: Limit to 1 entry per category
                paths = paths[:1]
            elif len(paths) < max_counts:
                extras_needed = max_counts - len(paths)
                duplicates = random.choices(paths, k=extras_needed)
                paths.extend(duplicates)
            balanced_paths.extend(paths[:max_counts])
            balanced_labels.extend([label] * min(max_counts, len(paths)))

        return balanced_paths, balanced_labels

    @staticmethod
    def load_directory_pointers(base_dir: str, max_counts: int, debug: bool = False) -> Tuple[List[str], List[str]]:
        """
        Load all main directories and balance them to ensure equal distribution.
        """
        dir_paths = [os.path.join(base_dir, dir_name) for dir_name in os.listdir(base_dir)]
        dir_paths_by_label = {}

        # Process directories in parallel
        with Pool(processes=cpu_count()) as pool:
            results = list(
                tqdm(pool.imap(DirectoryProcessor.process_directory, dir_paths),
                     total=len(dir_paths), desc="Processing Directories")
            )

        # Organize paths by label
        for dir_path, binary_label in results:
            dir_paths_by_label.setdefault(binary_label, []).append(dir_path)

        # Balance the directory pointers
        balanced_paths, balanced_labels = DirectoryProcessor.balance_binary_pointers(dir_paths_by_label, max_counts,
                                                                                     debug=debug)

        # Shuffle combined paths and labels
        combined = list(zip(balanced_paths, balanced_labels))
        random.shuffle(combined)

        return zip(*combined)


class DataPreprocessor_train:
    """
    Handles preprocessing tasks, including one-hot encoding and video loading.
    """

    @staticmethod
    def one_hot_encoding(labels: List[str]) -> Tuple[np.ndarray, pd.DataFrame, LabelEncoder]:
        """
        One-hot encode the labels and create a mapping DataFrame.
        """
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        num_classes = len(np.unique(labels_encoded))

        label_mapping_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Integer': range(len(label_encoder.classes_))
        })

        # Create a dictionary for quick lookup
        label_to_int = pd.Series(label_mapping_df.Integer.values, index=label_mapping_df.Category).to_dict()

        # Replace the string labels in labels_val with their corresponding integer values
        labels_int = np.array([label_to_int[label] for label in labels])
        labels_one_hot = to_categorical(labels_int, num_classes=num_classes)

        return labels_one_hot, label_mapping_df, label_encoder, num_classes

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """
        Normalize a single frame to [0, 1].
        """
        frame = frame.astype(np.float32)
        normalized = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
        if len(normalized.shape) == 2:  # If grayscale, add channel dimension
            normalized = normalized[..., np.newaxis]
        return normalized

    @staticmethod
    def load_video_frames(file_paths: List[str]) -> np.ndarray:
        """
        Load and preprocess frames from a list of file paths.
        """
        video = []
        for filepath in file_paths:
            frame = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            frame_resized = cv2.resize(frame, image_size)
            video.append(DataPreprocessor_train.preprocess_frame(frame_resized))
        return np.array(video)

    @staticmethod
    def load_videos(base_dir: str, min_length: int, debug: bool = False) -> Dict[str, List[np.ndarray]]:
        """
        Load all videos from the base directory with a minimum length.
        """
        videos = {}
        video_tasks = []

        for category in os.listdir(base_dir):
            category_path = os.path.join(base_dir, category)
            if not os.path.isdir(category_path):
                continue

            for video_folder in os.listdir(category_path):
                video_path = os.path.join(category_path, video_folder)
                if os.path.isdir(video_path):
                    tiff_files = sorted(
                        [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.tiff')])
                    if tiff_files:
                        video_tasks.append((category, tiff_files, min_length))

        # Debug mode: Limit to 1 video per category
        if debug:
            logging.info("Debug mode enabled. Limiting to 1 video per category.")
            tasks_by_category = {task[0]: task for task in video_tasks}
            video_tasks = list(tasks_by_category.values())[:2]  # Only 2 categories for testing

        # Process videos in parallel
        # num_cpus = min(cpu_count(), 4)
        num_cpus = cpu_count()
        logging.info(f"Using {num_cpus} CPU cores for data loading.")
        results = []

        with Pool(processes=num_cpus) as pool:
            for result in tqdm(pool.imap_unordered(DataPreprocessor_train.process_video_task, video_tasks),
                               total=len(video_tasks), desc="Loading Videos"):
                if result is not None:
                    results.append(result)

        # Assemble the video dictionary
        for category, video in results:
            videos.setdefault(category, []).append(video)

        logging.info(f"Total videos loaded: {sum(len(v) for v in videos.values())}")
        return videos

    @staticmethod
    def process_video_task(task: Tuple[str, List[str], int]) -> Tuple[str, np.ndarray]:
        """
        Process a single video loading task.
        """
        category, file_paths, min_length = task
        video = DataPreprocessor_train.load_video_frames(file_paths)
        return (category, video) if len(video) >= min_length else None


# data generator to match pointer und videos of training 
class DataGenerator_stress(Sequence):
    def __init__(self, all_videos: Dict[str, List[np.ndarray]], batch_size: int,
                 folder_pointers: List[str], labels_one_hot: List[np.ndarray],
                 image_size=image_size, z_shift_range=25, x_shift_range=25, y_shift_range=25,
                 rotation_range_z=15, noise_sigma=0.02, validation=False, log_file="generatetext.txt", shuffle=True):
        """
        Initialization
        """
        self.batch_size = batch_size
        self.folders = folder_pointers  # List of folder paths (folder_pointers)
        self.labels = labels_one_hot  # One-hot encoded labels (labels_one_hot)
        self.image_size = image_size
        self.z_shift_range = z_shift_range
        self.x_shift_range = x_shift_range
        self.y_shift_range = y_shift_range
        self.rotation_range_z = rotation_range_z
        self.noise_sigma = noise_sigma
        self.validation = validation
        self.log_file = log_file
        self.shuffle = shuffle
        self.indices = np.arange(len(self.folders))  # Initialize indices

        # Store all videos
        self.videos = all_videos

        self.on_epoch_end()  # Shuffle data if needed

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch
        """
        # TODO: Limits here, should be more?
        return math.ceil(len(self.folders) / self.batch_size)

    def __getitem__(self, idx: int, return_labels=False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate one batch of data, optionally including label names.
        """
        # Generate indices for the batch
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Select data and labels based on indices
        batch_folders = [self.folders[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        # Generate data and cast to float32
        X, y, label_names = self._generate_training_data(batch_folders, batch_labels)

        # Cast data to float32
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if self.validation is False:
            # Add noise: TODO: Slow and should be in sub function! Also on uint16!
            random_image_batch = np.random.uniform(-0.25, 0.25, X.shape).astype(np.float32)
            noisy_batch = cv2.add(X, random_image_batch)
            X = np.clip(noisy_batch, 0, 1)

        if return_labels:
            return X, y, label_names
        return X, y  # For training, only return X and y

    def _generate_training_data(self, batch_folders, batch_labels):
        """
        Generate data containing batch_size samples.
        Returns video data, one-hot labels, and label names (strings) for each video in the batch.
        """
        batch_videos = []
        new_batch_labels = []
        label_names = []  # To store the label names (strings) for each video
        if self.validation is False:
            for category_folder, label in zip(batch_folders, batch_labels):
                # Extract the label name to use it as a key to access all_videos
                label_name = os.path.basename(category_folder)  # Get the label name
    
                # Select a random video associated with the label from all_videos
                video_list = self.videos[label_name]  # Get the list of videos for the label
                video = random.choice(video_list)  # Select a random preloaded video
    
                # Ensure there are at least enough frames by extending if necessary
                if len(video) < length_video:
                    while len(video) < length_video:
                        video = np.concatenate((video, video[:length_video - len(video)]), axis=0)
    
                # Pick a random start index for the sliding window
                max_start_frame = len(video) - length_video - (drop_random_frames_count * frame_step)
    
                # Duplicate frames when missing: TODO: Not well designed!
                if max_start_frame < 0:
                    num_duplicates = abs(max_start_frame)
    
                    # Randomly select indices to duplicate
                    duplicate_indices = np.random.choice(len(video), num_duplicates, replace=False)
    
                    # Sort the indices in descending order to avoid messing up the array when inserting
                    duplicate_indices = np.sort(duplicate_indices)[::-1]
    
                    # Duplicate frames and add right after
                    for idx in duplicate_indices:
                        # Insert the frame right after the original frame
                        video = np.insert(video, idx + 1, video[idx], axis=0)
    
                    max_start_frame = len(video) - length_video - (drop_random_frames_count * frame_step)
    
                start_frame = np.random.randint(0, max_start_frame + 1)
                end_frame = start_frame + length_video + (drop_random_frames_count * frame_step)
    
                # Select the sliding window of frames:
                selected_frames = video[start_frame:end_frame:frame_step]
    
                if random.random() > 0.5:
                    selected_frames = selected_frames[::-1]
    
                # TODO: Before downsample?
    
                indices_to_remove = np.random.choice(selected_frames.shape[0], size=drop_random_frames_count, replace=False)
                modified_array = np.delete(selected_frames, indices_to_remove, axis=0)
    
                selected_frames = self.augment_video(modified_array)
    
                batch_videos.append(selected_frames)
                new_batch_labels.append(label)  # Append the one-hot encoded label
                label_names.append(label_name)  # Append the label name (string)
        else:
                batch_videos = batch_folders
                new_batch_labels = batch_labels
        # Convert the results back into NumPy arrays
        return np.array(batch_videos), np.array(new_batch_labels), label_names

    def augment_video(self, video):
        """
        Apply augmentations to the entire video.
        """
        # Flip vertically with 50% probability (axis 2)
        if np.random.rand() < 0.5:
            video = np.flip(video, axis=2)

        # Random translation
        tx = np.random.uniform(-self.x_shift_range, self.x_shift_range)
        ty = np.random.uniform(-self.y_shift_range, self.y_shift_range)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        video = np.array([cv2.warpAffine(frame, translation_matrix, self.image_size) for frame in video])

        # Random rotation
        angle = np.random.uniform(-self.rotation_range_z, self.rotation_range_z)
        rotation_matrix = cv2.getRotationMatrix2D((self.image_size[0] // 2, self.image_size[1] // 2), angle, 1.0)
        video = np.array([cv2.warpAffine(frame, rotation_matrix, self.image_size) for frame in video])

        # Add random noise
        noise = np.random.normal(0, self.noise_sigma, video.shape)
        video = np.clip(video + noise, 0, 1)

        # Random brightness adjustment
        if np.random.rand() < 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            video = np.clip(video * brightness_factor, 0, 1)

        if video.ndim == 3:
            video = video[..., np.newaxis]

        return video

    def on_epoch_end(self):
        """
        Shuffle indices after each epoch if shuffle is True.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)



