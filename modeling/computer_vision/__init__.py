"""
Computer Vision Package

This package provides comprehensive computer vision capabilities including:
- Image preprocessing and augmentation
- Feature extraction and descriptors
- Object detection and recognition
- Convolutional Neural Networks (CNNs)
- Transfer learning with pre-trained models
- Image classification and segmentation
- Real-time video processing
- Advanced computer vision algorithms
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
warnings.filterwarnings('ignore')

# Image processing imports
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import skimage
    from skimage import filters, feature, segmentation, measure, morphology
    from skimage.feature import hog, local_binary_pattern, corner_harris
    from skimage.filters import gaussian, sobel
    from skimage.segmentation import slic, watershed
    from skimage.measure import regionprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Deep learning for computer vision
try:
    import tensorflow as tf
    import keras
    from keras import layers, models
    from keras.applications import imagenet_utils, VGG16, ResNet50, MobileNetV2
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import applications
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Provide stubs for keras/tensorflow symbols
    class models:
        @staticmethod
        def Sequential(*args, **kwargs):
            raise ImportError("Keras not available")
    class layers:
        @staticmethod
        def __getattr__(name):
            raise ImportError("Keras not available")
    class applications:
        @staticmethod
        def __getattr__(name):
            raise ImportError("TensorFlow applications not available")
    class imagenet_utils:
        @staticmethod
        def preprocess_input(x):
            return x

# Machine learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# Add base module to path
base_path = os.path.join(os.path.dirname(__file__), '..')
if base_path not in sys.path:
    sys.path.insert(0, base_path)

try:
    from base import ModelMetadata, ModelStatus, ProblemType, ModelType
except ImportError:
    # If base.py does not exist, define stubs for compatibility
    class ModelMetadata: pass
    class ModelStatus: pass
    class ProblemType: pass
    class ModelType: pass


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"


class ColorSpace(Enum):
    """Color space representations."""
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    LAB = "lab"
    YUV = "yuv"


class FeatureType(Enum):
    """Feature extraction types."""
    HOG = "hog"
    LBP = "lbp"
    SIFT = "sift"
    ORB = "orb"
    HARRIS_CORNERS = "harris_corners"
    EDGE_HISTOGRAM = "edge_histogram"


class AugmentationType(Enum):
    """Image augmentation types."""
    ROTATION = "rotation"
    FLIP = "flip"
    CROP = "crop"
    ZOOM = "zoom"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    NOISE = "noise"


@dataclass
class ImageInfo:
    """Information about an image."""
    width: int
    height: int
    channels: int
    format: str
    size_bytes: int
    color_space: str = "RGB"
    dtype: str = "uint8"


@dataclass
class DetectionResult:
    """Object detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    area: int


@dataclass
class SegmentationResult:
    """Image segmentation result."""
    segments: np.ndarray
    num_segments: int
    segment_properties: List[Dict]
    processing_time: float


class ImageProcessor:
    """Core image processing functionality."""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
    def load_image(self, image_path: str, color_space: ColorSpace = ColorSpace.RGB) -> np.ndarray:
        """Load an image from file."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with OpenCV (BGR by default)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert color space
        if color_space == ColorSpace.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == ColorSpace.GRAY:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif color_space == ColorSpace.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == ColorSpace.LAB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # BGR is default, no conversion needed
        
        return image
    
    def save_image(self, image: np.ndarray, output_path: str, 
                  color_space: ColorSpace = ColorSpace.RGB) -> bool:
        """Save an image to file."""
        try:
            # Convert to BGR for OpenCV saving
            if color_space == ColorSpace.RGB:
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif color_space == ColorSpace.GRAY:
                save_image = image
            else:
                save_image = image
            
            return cv2.imwrite(output_path, save_image)
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def get_image_info(self, image: np.ndarray) -> ImageInfo:
        """Get information about an image."""
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
        
        return ImageInfo(
            width=width,
            height=height,
            channels=channels,
            format="array",
            size_bytes=image.nbytes,
            dtype=str(image.dtype)
        )
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """Resize an image."""
        if maintain_aspect:
            # Calculate aspect ratio preserving resize
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            aspect_ratio = w / h
            if aspect_ratio > target_w / target_h:
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad to target size if needed
            if new_w < target_w or new_h < target_h:
                top = (target_h - new_h) // 2
                bottom = target_h - new_h - top
                left = (target_w - new_w) // 2
                right = target_w - new_w - left
                
                resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            resized = cv2.resize(image, target_size)
        
        return resized
    
    def normalize_image(self, image: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize image values."""
        image_float = image.astype(np.float32)
        
        if method == "minmax":
            return (image_float - image_float.min()) / (image_float.max() - image_float.min())
        elif method == "standard":
            return (image_float - image_float.mean()) / image_float.std()
        elif method == "unit":
            return image_float / 255.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class ImageAugmentor:
    """Image augmentation for data augmentation."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        return rotated
    
    def flip(self, image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
        """Flip image horizontally or vertically."""
        if direction == "horizontal":
            return cv2.flip(image, 1)
        elif direction == "vertical":
            return cv2.flip(image, 0)
        elif direction == "both":
            return cv2.flip(image, -1)
        else:
            raise ValueError("Direction must be 'horizontal', 'vertical', or 'both'")
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def add_noise(self, image: np.ndarray, noise_type: str = "gaussian", 
                 intensity: float = 0.1) -> np.ndarray:
        """Add noise to image."""
        if noise_type == "gaussian":
            noise = np.random.normal(0, intensity * 255, image.shape)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
            return noisy.astype(np.uint8)
        elif noise_type == "salt_pepper":
            noisy = image.copy()
            # Salt noise
            salt_coords = tuple([np.random.randint(0, i - 1, int(intensity * image.size * 0.5)) 
                               for i in image.shape])
            noisy[salt_coords] = 255
            
            # Pepper noise
            pepper_coords = tuple([np.random.randint(0, i - 1, int(intensity * image.size * 0.5)) 
                                 for i in image.shape])
            noisy[pepper_coords] = 0
            
            return noisy
        else:
            raise ValueError("Noise type must be 'gaussian' or 'salt_pepper'")
    
    def random_crop(self, image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """Randomly crop image."""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        if crop_h > h or crop_w > w:
            raise ValueError("Crop size larger than image")
        
        start_y = np.random.randint(0, h - crop_h + 1)
        start_x = np.random.randint(0, w - crop_w + 1)
        
        return image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    def augment_batch(self, images: List[np.ndarray], 
                     augmentations: List[AugmentationType]) -> List[np.ndarray]:
        """Apply random augmentations to a batch of images."""
        augmented = []
        
        for image in images:
            aug_image = image.copy()
            
            for aug_type in augmentations:
                if aug_type == AugmentationType.ROTATION:
                    angle = np.random.uniform(-30, 30)
                    aug_image = self.rotate(aug_image, angle)
                elif aug_type == AugmentationType.FLIP:
                    if np.random.random() > 0.5:
                        direction = np.random.choice(["horizontal", "vertical"])
                        aug_image = self.flip(aug_image, direction)
                elif aug_type == AugmentationType.BRIGHTNESS:
                    factor = np.random.uniform(0.7, 1.3)
                    aug_image = self.adjust_brightness(aug_image, factor)
                elif aug_type == AugmentationType.CONTRAST:
                    factor = np.random.uniform(0.7, 1.3)
                    aug_image = self.adjust_contrast(aug_image, factor)
                elif aug_type == AugmentationType.NOISE:
                    intensity = np.random.uniform(0.05, 0.15)
                    aug_image = self.add_noise(aug_image, "gaussian", intensity)
            
            augmented.append(aug_image)
        
        return augmented


class FeatureExtractor:
    """Extract various features from images."""
    
    def __init__(self):
        self.sift_detector = None
        self.orb_detector = None

        # Initialize OpenCV detectors if available
        # Provide stubs if not available
        if hasattr(cv2, 'SIFT_create'):
            try:
                self.sift_detector = cv2.SIFT_create()
            except Exception:
                self.sift_detector = None
        else:
            class SIFTStub:
                def detectAndCompute(self, *args, **kwargs):
                    raise ImportError("cv2.SIFT_create not available")
            self.sift_detector = SIFTStub()
        if hasattr(cv2, 'ORB_create'):
            try:
                self.orb_detector = cv2.ORB_create()
            except Exception:
                self.orb_detector = None
        else:
            class ORBStub:
                def detectAndCompute(self, *args, **kwargs):
                    raise ImportError("cv2.ORB_create not available")
            self.orb_detector = ORBStub()

    def extract_hog_features(self, image: np.ndarray, 
                           orientations: int = 9,
                           pixels_per_cell: Tuple[int, int] = (8, 8),
                           cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features."""
        if SKIMAGE_AVAILABLE:
            # Import hog from skimage.feature
            from skimage.feature import hog
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            features = hog(
                gray, 
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=False,
                feature_vector=True
            )

            return features
        else:
            raise ImportError("scikit-image required for HOG features")

    def extract_lbp_features(self, image: np.ndarray, 
                           radius: int = 1, 
                           n_points: int = 8) -> np.ndarray:
        """Extract LBP (Local Binary Pattern) features."""
        if SKIMAGE_AVAILABLE:
            # Import local_binary_pattern from skimage.feature
            from skimage.feature import local_binary_pattern
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

            # Create histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                 range=(0, n_points + 2), density=True)

            return hist
        else:
            raise ImportError("scikit-image required for LBP features")
    
    def extract_sift_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract SIFT keypoints and descriptors."""
        if self.sift_detector is None:
            raise RuntimeError("SIFT detector not available")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.sift_detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_orb_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ORB keypoints and descriptors."""
        if self.orb_detector is None:
            raise RuntimeError("ORB detector not available")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create edge histogram
        hist, _ = np.histogram(edges.ravel(), bins=256, range=(0, 256), density=True)
        
        return hist
    
    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Extract color histogram features."""
        if len(image.shape) == 3:
            # Multi-channel histogram
            hist = []
            for channel in range(image.shape[2]):
                channel_hist, _ = np.histogram(image[:, :, channel], 
                                             bins=bins, range=(0, 256), density=True)
                hist.extend(channel_hist)
            return np.array(hist)
        else:
            # Grayscale histogram
            hist, _ = np.histogram(image.ravel(), bins=bins, range=(0, 256), density=True)
            return hist


class ImageSegmentor:
    """Image segmentation algorithms."""
    
    def __init__(self):
        pass
    
    def kmeans_segmentation(self, image: np.ndarray, k: int = 3) -> SegmentationResult:
        """K-means based image segmentation."""
        start_time = time.time()
        
        # Reshape image for clustering
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            data = image.reshape(-1, image.shape[2])
        else:
            data = image.reshape(-1, 1)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Reshape back to image
        segments = labels.reshape(h, w)
        
        processing_time = time.time() - start_time
        
        # Calculate segment properties
        properties = []
        for i in range(k):
            mask = segments == i
            properties.append({
                'segment_id': i,
                'pixel_count': np.sum(mask),
                'percentage': np.sum(mask) / (h * w) * 100,
                'centroid': kmeans.cluster_centers_[i].tolist()
            })
        
        return SegmentationResult(
            segments=segments,
            num_segments=k,
            segment_properties=properties,
            processing_time=processing_time
        )
    
    def slic_segmentation(self, image: np.ndarray, 
                         n_segments: int = 100, 
                         compactness: float = 10.0) -> SegmentationResult:
        """SLIC superpixel segmentation."""
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for SLIC segmentation")
        
        start_time = time.time()
        
        from skimage.segmentation import slic
        from skimage.measure import regionprops

        segments = slic(image, n_segments=n_segments, compactness=compactness)
        
        processing_time = time.time() - start_time
        
        # Calculate segment properties
        props = regionprops(segments + 1)  # +1 to avoid 0 labels
        properties = []
        
        for prop in props:
            properties.append({
                'segment_id': prop.label - 1,  # Convert back to 0-based
                'area': prop.area,
                'centroid': prop.centroid,
                'bbox': prop.bbox
            })
        
        return SegmentationResult(
            segments=segments,
            num_segments=len(np.unique(segments)),
            segment_properties=properties,
            processing_time=processing_time
        )
    
    def threshold_segmentation(self, image: np.ndarray, 
                             threshold: Optional[int] = None) -> SegmentationResult:
        """Threshold-based segmentation."""
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Auto-threshold if not provided
        if threshold is None:
            otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = int(otsu_threshold)
        
        # Apply threshold
        _, segments = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        segments = segments // 255  # Convert to 0/1
        
        processing_time = time.time() - start_time
        
        # Calculate properties
        properties = [
            {
                'segment_id': 0,
                'pixel_count': np.sum(segments == 0),
                'percentage': np.sum(segments == 0) / segments.size * 100
            },
            {
                'segment_id': 1,
                'pixel_count': np.sum(segments == 1),
                'percentage': np.sum(segments == 1) / segments.size * 100
            }
        ]
        
        return SegmentationResult(
            segments=segments,
            num_segments=2,
            segment_properties=properties,
            processing_time=processing_time
        )


class CNNClassifier:
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                 num_classes: int = 10, 
                 architecture: str = "simple"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None
        self.history = None
        
        if TENSORFLOW_AVAILABLE:
            self._build_model()
        else:
            raise ImportError("TensorFlow required for CNN functionality")
    
    def _build_model(self):
        """Build the CNN architecture."""
        if self.architecture == "simple":
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        elif self.architecture == "deep":
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
           validation_split: float = 0.2,
           epochs: int = 10,
           batch_size: int = 32) -> 'CNNClassifier':
        """Train the CNN model."""
        if self.model is None:
            raise RuntimeError("Model not built")
        
        # Normalize data
        X_norm = X.astype('float32') / 255.0
        
        self.history = self.model.fit(
            X_norm, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        X_norm = X.astype('float32') / 255.0
        predictions = self.model.predict(X_norm)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        X_norm = X.astype('float32') / 255.0
        return self.model.predict(X_norm)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        import io
        summary_buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        return summary_buffer.getvalue()


class TransferLearningClassifier:
    """Transfer learning using pre-trained models."""
    
    def __init__(self, base_model: str = "VGG16", 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 10):
        self.base_model_name = base_model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        if TENSORFLOW_AVAILABLE:
            self._build_transfer_model()
        else:
            raise ImportError("TensorFlow required for transfer learning")
    
    def _build_transfer_model(self):
        """Build transfer learning model."""
        # Load pre-trained base model
        if self.base_model_name == "VGG16" and TENSORFLOW_AVAILABLE:
            base_model = applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == "ResNet50":
            base_model = applications.ResNet50(
                weights="imagenet",
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == "MobileNetV2":
            base_model = applications.mobilenet_v2.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom classification head
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        if TENSORFLOW_AVAILABLE:
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray,
           epochs: int = 10,
           batch_size: int = 32) -> 'TransferLearningClassifier':
        """Train the transfer learning model."""
        if self.model is None:
            raise RuntimeError("Model not built")
        
        # Normalize data for pre-trained models
        X_norm = applications.imagenet_utils.preprocess_input(X.astype('float32'))
        
        self.history = self.model.fit(
            X_norm, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        X_norm = imagenet_utils.preprocess_input(X.astype('float32'))
        predictions = self.model.predict(X_norm)
    

# Convenience functions
def create_cnn_classifier(input_shape: Tuple[int, int, int] = (224, 224, 3),
                         num_classes: int = 10,
                         architecture: str = "simple") -> CNNClassifier:
    """Create a CNN classifier."""
    return CNNClassifier(input_shape, num_classes, architecture)


def create_transfer_learning_classifier(base_model: str = "VGG16",
                                       input_shape: Tuple[int, int, int] = (224, 224, 3),
                                       num_classes: int = 10) -> TransferLearningClassifier:
    """Create a transfer learning classifier."""
    return TransferLearningClassifier(base_model, input_shape, num_classes)


def generate_sample_images(num_images: int = 100, 
                          image_size: Tuple[int, int] = (64, 64),
                          num_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sample images for demonstration."""
    images = []
    labels = []

    for i in range(num_images):
        img = np.zeros((*image_size, 3), dtype=np.uint8)
        label = 0
        # Create synthetic image
        if num_classes == 3:
            if i % 3 == 0:
                # Circle
                center = (image_size[0] // 2, image_size[1] // 2)
                radius = min(image_size) // 4
                cv2.circle(img, center, radius, (255, 100, 100), -1)
                label = 0
            elif i % 3 == 1:
                # Rectangle
                pt1 = (image_size[0] // 4, image_size[1] // 4)
                pt2 = (3 * image_size[0] // 4, 3 * image_size[1] // 4)
                cv2.rectangle(img, pt1, pt2, (100, 255, 100), -1)
                label = 1
            else:
                # Triangle
                pts = np.array([
                    [image_size[0] // 2, image_size[1] // 4],
                    [image_size[0] // 4, 3 * image_size[1] // 4],
                    [3 * image_size[0] // 4, 3 * image_size[1] // 4]
                ], np.int32)
                cv2.fillPoly(img, [pts], (100, 100, 255))
                label = 2

        # Add some noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


# Export main classes and functions
__all__ = [
    'ImageProcessor',
    'ImageAugmentor', 
    'FeatureExtractor',
    'ImageSegmentor',
    'CNNClassifier',
    'TransferLearningClassifier',
    'ImageInfo',
    'DetectionResult',
    'SegmentationResult',
    'ImageFormat',
    'ColorSpace',
    'FeatureType',
    'AugmentationType',
    'create_cnn_classifier',
    'create_transfer_learning_classifier',
    'generate_sample_images'
]
