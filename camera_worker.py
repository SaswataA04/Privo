
import os
import time

import cv2
import numpy as np
import mediapipe as mp

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from alert_engine import AlertEngine


class FaceTracker:
    """Lightweight centroid tracker assigning temporary IDs to detected faces."""

    def __init__(self, max_distance=110, max_missed=15):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}  # ID -> {center, bbox, missed}

    @staticmethod
    def center_of(box):
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def update(self, detections):
        """Return track IDs aligned with detections in the current frame."""
        detections = [tuple(map(int, box)) for box in detections]
        centers = [self.center_of(box) for box in detections]

        # Age existing tracks before matching.
        for track in self.tracks.values():
            track['missed'] += 1

        # Build candidate track/detection matches by center distance.
        candidates = []
        for track_id, track in self.tracks.items():
            old_x, old_y = track['center']
            for detection_index, (new_x, new_y) in enumerate(centers):
                distance = ((old_x - new_x) ** 2 + (old_y - new_y) ** 2) ** 0.5
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_index))

        candidates.sort(key=lambda item: item[0])
        used_tracks = set()
        used_detections = set()
        detection_ids = [None] * len(detections)

        # Greedy nearest-neighbor matching, with each ID/detection used once.
        for distance, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            detection_ids[detection_index] = track_id
            self.tracks[track_id].update({
                'center': centers[detection_index],
                'bbox': detections[detection_index],
                'missed': 0
            })

        # Start new tracks for unmatched detections.
        for detection_index, box in enumerate(detections):
            if detection_ids[detection_index] is not None:
                continue
            track_id = self.next_id
            self.next_id += 1
            detection_ids[detection_index] = track_id
            self.tracks[track_id] = {
                'center': centers[detection_index],
                'bbox': box,
                'missed': 0
            }

        # Remove tracks that have been missing for too many frames.
        expired_ids = [
            track_id for track_id, track in self.tracks.items()
            if track['missed'] > self.max_missed
        ]
        for track_id in expired_ids:
            del self.tracks[track_id]

        return detection_ids


class CameraWorker(QThread):

    # Sends the processed webcam frame to the PyQt dashboard.
    frame_ready = pyqtSignal(QImage)

    # Sends current status and face counts to the dashboard.
    status_changed = pyqtSignal(str, int, int)

    # Reports errors to the main application.
    error_occurred = pyqtSignal(str)

    def __init__(self, settings=None):
        super().__init__()

        self.base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.running = False
        self.paused = False
        self.camera = None

        # Temporary face IDs across frames. These IDs are tracking labels,
        # not verified real-world identities and are not stored persistently.
        self.face_tracker = FaceTracker(
            max_distance=110,
            max_missed=15
        )

        settings = settings or {}

        # -----------------------------------------------------
        # APPLICATION SETTINGS
        # -----------------------------------------------------

        self.confidence_threshold = int(
            settings.get("confidence_threshold", 65)
        )

        self.trigger_delay = float(
            settings.get("trigger_delay", 1.0)
        )

        self.clear_delay = float(
            settings.get("clear_delay", 1.5)
        )

        self.camera_index = int(
            settings.get("camera_index", 0)
        )

        # Existing visual alert engine.
        self.alert_engine = AlertEngine(
            trigger_delay=self.trigger_delay,
            clear_delay=self.clear_delay
        )

        # -----------------------------------------------------
        # HAAR CASCADE FACE DETECTOR
        # -----------------------------------------------------

        cascade_path = os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_detector.empty():
            raise RuntimeError(
                "Unable to load Haar Cascade face detector."
            )

        # -----------------------------------------------------
        # EXISTING LBPH FACE RECOGNIZER
        # -----------------------------------------------------

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.recognizer_ready = False

        self.dataset_folder = os.path.join(
            self.base_dir,
            "dataset",
            "saswata"
        )

        self.model_path = os.path.join(
            self.base_dir,
            "privacy_guard_model.yml"
        )

        # -----------------------------------------------------
        # MEDIAPIPE FACE LANDMARKER
        # -----------------------------------------------------

        self.landmarker = None

        self.landmarker_model_path = os.path.join(
            self.base_dir,
            "models",
            "face_landmarker.task"
        )

        # Video timestamps must increase monotonically.
        self.last_timestamp_ms = 0

        # -----------------------------------------------------
        # HEAD POSE ESTIMATION CONFIGURATION
        # -----------------------------------------------------

        # These are the MediaPipe landmark indices used to estimate
        # the orientation of a human face.
        #
        # The selected landmarks:
        # 1   -> Nose tip
        # 152 -> Chin
        # 33  -> Left eye outer corner
        # 263 -> Right eye outer corner
        # 61  -> Left mouth corner
        # 291 -> Right mouth corner

        self.pose_landmark_indices = [
            1,      # Nose tip
            152,    # Chin
            33,     # Left eye outer corner
            263,    # Right eye outer corner
            61,     # Left mouth corner
            291     # Right mouth corner
        ]

        # Approximate 3D facial coordinates.
        #
        # These points represent a simplified generic face model.
        # They are NOT measurements of your actual face.
        #
        # solvePnP compares this generic 3D model against the 2D
        # landmarks detected by MediaPipe to estimate face rotation.

        self.face_3d_model_points = np.array(
            [
                (0.0, 0.0, 0.0),             # Nose
                (0.0, -330.0, -65.0),         # Chin
                (-225.0, 170.0, -135.0),       # Left eye
                (225.0, 170.0, -135.0),        # Right eye
                (-150.0, -150.0, -125.0),      # Left mouth
                (150.0, -150.0, -125.0),       # Right mouth
            ],
            dtype=np.float64
        )

        # Camera matrix is generated dynamically using the current
        # webcam frame width and height.
        self.camera_matrix = None

        # Assuming zero lens distortion for this initial version.
        self.distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64
        )








    # =========================================================
            # SHOULDER-SURFING DETECTION CONFIGURATION
        # =========================================================

        # Time a sideways head orientation must persist before
        # it is classified as suspicious.
        self.shoulder_surf_threshold = 3.0

        # Tracks when a sideways orientation first began.
        self.sideways_start_time = None

        # Tracks the currently detected sideways direction.
        self.sideways_direction = None

        # Current suspicion state.
        self.shoulder_surfing_detected = False

        # Duration of the current sideways head orientation.
        self.sideways_duration = 0.0

    # =========================================================
    # RESET ALERT ENGINE
    # =========================================================

    def reset_alert_state(self):

        self.alert_engine = AlertEngine(
            trigger_delay=self.trigger_delay,
            clear_delay=self.clear_delay
        )

    # =========================================================
    # INITIALIZE MEDIAPIPE
    # =========================================================

    def initialize_landmarker(self):

        if not os.path.isfile(self.landmarker_model_path):

            raise FileNotFoundError(
                "MediaPipe Face Landmarker model was not found:\n"
                f"{self.landmarker_model_path}\n\n"
                "Download face_landmarker.task and place it "
                "inside the models folder."
            )

        # Configure MediaPipe to process consecutive video frames.
        options = mp.tasks.vision.FaceLandmarkerOptions(

            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.landmarker_model_path
            ),

            running_mode=mp.tasks.vision.RunningMode.VIDEO,

            # Maximum number of faces to process simultaneously.
            num_faces=5,

            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,

            output_face_blendshapes=False
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

        print(
            "[Privacy Guard] MediaPipe Face Landmarker initialized."
        )

    # =========================================================
    # DETECT FACIAL LANDMARKS
    # =========================================================

    def detect_landmarks(self, rgb_frame):

        if self.landmarker is None:
            raise RuntimeError(
                "MediaPipe Face Landmarker is not initialized."
            )

        # Convert RGB NumPy frame into a MediaPipe image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate a monotonically increasing video timestamp.
        timestamp_ms = int(
            time.monotonic_ns() // 1_000_000
        )

        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        # Perform landmark detection and tracking.
        results = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return results.face_landmarks

    # =========================================================
    # INITIALIZE CAMERA MATRIX FOR POSE ESTIMATION
    # =========================================================

    def initialize_camera_matrix(self, width, height):

        # Approximate focal length using frame width.
        #
        # Without webcam calibration data, this is a reasonable
        # initial approximation for pose estimation.
        focal_length = width

        center_x = width / 2.0
        center_y = height / 2.0

        self.camera_matrix = np.array(
            [
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1],
            ],
            dtype=np.float64
        )

        self.distortion_coefficients = np.zeros(
            (4, 1),
            dtype=np.float64
        )

    # =========================================================
    # ESTIMATE HEAD POSE USING SOLVEPnP
    # =========================================================

    def estimate_head_pose(self, face_landmarks, frame_width, frame_height):

        # Pose estimation requires the camera matrix.
        if self.camera_matrix is None:

            self.initialize_camera_matrix(
                frame_width,
                frame_height
            )

        # -----------------------------------------------------
        # STEP 1: EXTRACT 2D LANDMARK COORDINATES
        # -----------------------------------------------------

        image_points = []

        for index in self.pose_landmark_indices:

            landmark = face_landmarks[index]

            # MediaPipe landmarks are normalized between 0 and 1.
            # Convert them into actual webcam pixel coordinates.
            x = landmark.x * frame_width
            y = landmark.y * frame_height

            image_points.append(
                (x, y)
            )

        image_points = np.array(
            image_points,
            dtype=np.float64
        )

        # -----------------------------------------------------
        # STEP 2: MATCH 2D POINTS WITH 3D FACE MODEL
        # -----------------------------------------------------

        # solvePnP estimates the rotation and translation of the
        # 3D face model that best matches the observed 2D points.

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                self.face_3d_model_points,
                image_points,
                self.camera_matrix,
                self.distortion_coefficients,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        )

        if not success:
            return None

        # -----------------------------------------------------
        # STEP 3: CONVERT ROTATION VECTOR TO ROTATION MATRIX
        # -----------------------------------------------------

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        # -----------------------------------------------------
        # STEP 4: EXTRACT EULER ANGLES
        # -----------------------------------------------------

        # Convert the rotation matrix into Euler angles.
        #
        # OpenCV's RQDecomp3x3 provides angles in degrees.
        # These angles represent rotation around the three axes.

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
            rotation_matrix
        )

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Return the estimated pose.
        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "rotation_vector": rotation_vector,
            "translation_vector": translation_vector
        }

    # =========================================================
    # DETERMINE HEAD ORIENTATION LABEL
    # =========================================================

    def get_head_direction(self, yaw, pitch):

        # These thresholds are deliberately conservative.
        # We'll calibrate their signs and sensitivity after testing.

        yaw_threshold = 18.0
        pitch_threshold = 15.0

        # Head turned left or right.
        if yaw < -yaw_threshold:
            return "LEFT"

        if yaw > yaw_threshold:
            return "RIGHT"

        # Head tilted upward or downward.
        if pitch < -pitch_threshold:
            return "UP"

        if pitch > pitch_threshold:
            return "DOWN"

        return "CENTER"




    # =========================================================
    # SHOULDER-SURFING TEMPORAL DETECTION
    # =========================================================

    def detect_shoulder_surfing(self, direction):
        """Flag sustained LEFT/RIGHT head orientation as suspicious.


        This is an orientation-based heuristic, not proof that another
        person is looking at the screen.
        """
        current_time = time.monotonic()

        if direction in ("LEFT", "RIGHT"):
            if (
                self.sideways_start_time is None
                or self.sideways_direction != direction
            ):
                self.sideways_start_time = current_time
                self.sideways_direction = direction
                self.sideways_duration = 0.0


            self.sideways_duration = current_time - self.sideways_start_time
            self.shoulder_surfing_detected = (
                self.sideways_duration >= self.shoulder_surf_threshold
            )
        else:
            self.sideways_start_time = None
            self.sideways_direction = None
            self.sideways_duration = 0.0
            self.shoulder_surfing_detected = False



        return self.shoulder_surfing_detected, self.sideways_duration



    # =========================================================
    # DRAW SHOULDER-SURFING WARNING
    # =========================================================

    def draw_shoulder_surf_warning(self, frame, detected, duration):
        """Draw a visual-only sustained side-orientation indicator."""
        height, width, _ = frame.shape



        if not detected:
            text = (
                f"SIDE ORIENTATION: {duration:.1f}s"
                if duration > 0
                else "SIDE ORIENTATION: CLEAR"
            )
            cv2.putText(
                frame, text, (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 0), 2, cv2.LINE_AA
            )
            return frame



        # This banner indicates sustained side orientation only.
        cv2.rectangle(frame, (0, 90), (width, 140), (0, 0, 200), -1)
        warning_text = f"SIDE ORIENTATION SUSPECTED | {duration:.1f}s"
        cv2.putText(
            frame, warning_text, (15, 122),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        return frame



    # =========================================================
    # DRAW HEAD POSE INFORMATION
    # =========================================================

    def draw_head_pose(
        self,
        frame,
        face_landmarks,
        pose,
        face_box=None
    ):

        height, width, _ = frame.shape

        # -----------------------------------------------------
        # DRAW FACE CENTER REFERENCE
        # -----------------------------------------------------

        nose = face_landmarks[1]

        nose_x = int(nose.x * width)
        nose_y = int(nose.y * height)

        # Draw a reference point at the nose.
        cv2.circle(
            frame,
            (nose_x, nose_y),
            5,
            (255, 255, 255),
            -1
        )

        if pose is None:

            cv2.putText(
                frame,
                "POSE: UNAVAILABLE",
                (nose_x + 10, nose_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )

            return frame

        pitch = pose["pitch"]
        yaw = pose["yaw"]
        roll = pose["roll"]

        direction = self.get_head_direction(
            yaw,
            pitch
        )

        # -----------------------------------------------------
        # COLOR BASED ON ORIENTATION
        # -----------------------------------------------------

        if direction == "CENTER":

            pose_color = (0, 255, 0)

        else:

            pose_color = (0, 165, 255)

        # -----------------------------------------------------
        # DRAW HEAD ORIENTATION LABEL
        # -----------------------------------------------------

        pose_text = (
            f"{direction} | "
            f"Y:{yaw:+.1f} "
            f"P:{pitch:+.1f} "
            f"R:{roll:+.1f}"
        )

        if face_box is not None:

            x, y, w, h = face_box

            text_x = max(10, x)
            text_y = min(
                height - 10,
                y + h + 25
            )

        else:

            text_x = max(
                10,
                min(nose_x - 100, width - 300)
            )

            text_y = min(
                height - 10,
                nose_y + 35
            )

        cv2.putText(
            frame,
            pose_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            pose_color,
            2,
            cv2.LINE_AA
        )

        # -----------------------------------------------------
        # DRAW DIRECTION INDICATOR
        # -----------------------------------------------------

        direction_text = (
            f"HEAD: {direction}"
        )

        cv2.putText(
            frame,
            direction_text,
            (text_x, max(20, text_y - 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            pose_color,
            2,
            cv2.LINE_AA
        )

        return frame

    # =========================================================
    # DRAW FACIAL LANDMARKS
    # =========================================================

    def draw_landmarks(self, frame, all_faces):

        height, width, _ = frame.shape

        for face_landmarks in all_faces:

            # Draw a subset of points to keep the preview readable.
            for index, landmark in enumerate(face_landmarks):

                if index % 3 != 0:
                    continue

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Ignore points outside the image.
                if not (
                    0 <= x < width
                    and 0 <= y < height
                ):
                    continue

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 255),
                    -1
                )

            # Highlight selected central facial landmarks.
            key_indices = [
                1, 4, 10, 152, 33, 263, 61, 291
            ]

            for index in key_indices:

                if index >= len(face_landmarks):
                    continue

                landmark = face_landmarks[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:

                    cv2.circle(
                        frame,
                        (x, y),
                        3,
                        (255, 255, 0),
                        -1
                    )

        return frame

    # =========================================================
    # TRAIN LBPH RECOGNITION MODEL
    # =========================================================

    def train_recognizer(self):

        if not os.path.isdir(self.dataset_folder):

            raise FileNotFoundError(
                "Enrollment folder not found:\n"
                + self.dataset_folder
            )

        training_faces = []

        supported_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        )

        for filename in os.listdir(self.dataset_folder):

            if not filename.lower().endswith(
                supported_extensions
            ):
                continue

            image_path = os.path.join(
                self.dataset_folder,
                filename
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(image, (200, 200))
            image = cv2.equalizeHist(image)

            training_faces.append(image)

        if not training_faces:

            raise ValueError(
                "No valid enrolled face images were found "
                "inside dataset/saswata."
            )

        labels = np.zeros(
            len(training_faces),
            dtype=np.int32
        )

        self.recognizer.train(
            training_faces,
            labels
        )

        self.recognizer.save(
            self.model_path
        )

        self.recognizer_ready = True

        print(
            f"[Privacy Guard] Trained using "
            f"{len(training_faces)} images."
        )

    # =========================================================
    # RECOGNIZE FACE USING LBPH
    # =========================================================

    def recognize_face(self, face):

        if not self.recognizer_ready:

            raise RuntimeError(
                "Recognition model is not ready."
            )

        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)

        label, confidence = self.recognizer.predict(face)

        is_authorized = (
            label == 0
            and confidence < self.confidence_threshold
        )

        return is_authorized, confidence

    # =========================================================
    # CAMERA THREAD
    # =========================================================

    def run(self):

        self.running = True

        try:

            # Train the existing recognition model.
            self.train_recognizer()

            # Initialize MediaPipe before opening the camera.
            self.initialize_landmarker()

            # Open the configured webcam.
            self.camera = cv2.VideoCapture(
                self.camera_index
            )

            if not self.camera.isOpened():

                raise RuntimeError(
                    f"Unable to access camera index "
                    f"{self.camera_index}. "
                    "Check camera permissions or select another camera."
                )

            self.camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                800
            )

            self.camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                600
            )

            # Initialize camera matrix for solvePnP.
            self.initialize_camera_matrix(
                800,
                600
            )

            print(
                "[Privacy Guard] Monitoring started."
            )

            while self.running:

                if self.paused:

                    self.msleep(100)
                    continue

                success, frame = self.camera.read()

                if not success:

                    raise RuntimeError(
                        "Unable to read webcam frame."
                    )

                frame = cv2.resize(
                    frame,
                    (800, 600)
                )

                # -------------------------------------------------
                # STEP 1: HAAR CASCADE FACE DETECTION
                # -------------------------------------------------

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(60, 60)
                )

                authorized_count = 0
                unknown_count = 0

                # -------------------------------------------------
                # STEP 2: MULTI-PERSON FACE TRACKING
                # -------------------------------------------------

                # Assign a temporary track ID to each face detection.
                # IDs persist while detections remain spatially close.
                face_track_ids = self.face_tracker.update(faces)

                # -------------------------------------------------
                # STEP 3: EXISTING LBPH RECOGNITION
                # -------------------------------------------------

                for face_index, (x, y, w, h) in enumerate(faces):

                    track_id = face_track_ids[face_index]

                    face = gray[y:y + h, x:x + w]

                    is_authorized, confidence = (
                        self.recognize_face(face)
                    )

                    if is_authorized:

                        authorized_count += 1
                        color = (0, 200, 0)
                        name = f"ID {track_id} | AUTHORIZED"

                    else:

                        unknown_count += 1
                        color = (0, 0, 255)
                        name = f"ID {track_id} | UNKNOWN"

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        name,
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

                # -------------------------------------------------
                # STEP 4: MEDIAPIPE FACIAL LANDMARK DETECTION
                # -------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                landmark_faces = self.detect_landmarks(
                    rgb_frame
                )

                # -------------------------------------------------
                # STEP 5: DRAW FACIAL LANDMARKS
                # -------------------------------------------------

                frame = self.draw_landmarks(
                    frame,
                    landmark_faces
                )

                # -------------------------------------------------
                # STEP 6: ESTIMATE HEAD POSE FOR EACH FACE
                # -------------------------------------------------

                primary_direction = "CENTER"
                primary_pose_available = False

                for face_index, face_landmarks in enumerate(landmark_faces):

                    pose = self.estimate_head_pose(
                        face_landmarks,
                        frame.shape[1],
                        frame.shape[0]
                    )

                    frame = self.draw_head_pose(
                        frame,
                        face_landmarks,
                        pose
                    )

                    # Use only the first MediaPipe face as the primary
                    # subject for the shoulder-surfing orientation timer.
                    if face_index == 0 and pose is not None:
                        primary_direction = self.get_head_direction(
                            pose["yaw"], pose["pitch"]
                        )
                        primary_pose_available = True

                # No detected face or failed pose estimation resets the
                # temporal detector to avoid stale warnings.
                if not primary_pose_available:
                    primary_direction = "CENTER"

                shoulder_warning, sideways_duration = (
                    self.detect_shoulder_surfing(primary_direction)
                )

                frame = self.draw_shoulder_surf_warning(
                    frame,
                    shoulder_warning,
                    sideways_duration
                )

                # -------------------------------------------------
                # STEP 7: MEDIAPIPE STATUS
                # -------------------------------------------------

                landmark_status = (
                    f"LANDMARKS: ACTIVE | "
                    f"FACES: {len(landmark_faces)} | "
                    f"TRACKS: {len(self.face_tracker.tracks)}"
                )

                cv2.putText(
                    frame,
                    landmark_status,
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                # -------------------------------------------------
                # STEP 8: EXISTING VISUAL ALERT ENGINE
                # -------------------------------------------------

                alert_active = self.alert_engine.update(
                    unknown_count
                )

                if alert_active:

                    status = "PRIVACY WARNING"
                    banner_color = (0, 0, 220)

                else:

                    status = "PROTECTED"
                    banner_color = (0, 150, 0)

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (800, 45),
                    banner_color,
                    -1
                )

                cv2.putText(
                    frame,
                    status,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # -------------------------------------------------
                # STEP 9: SEND FRAME TO PYQT DASHBOARD
                # -------------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                height, width, channels = rgb_frame.shape

                qt_image = QImage(
                    rgb_frame.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888
                ).copy()

                self.frame_ready.emit(qt_image)

                self.status_changed.emit(
                    status,
                    authorized_count,
                    unknown_count
                )

        except Exception as error:

            self.error_occurred.emit(str(error))

        finally:

            # Release webcam resources.
            if self.camera is not None:

                self.camera.release()
                self.camera = None

            # Release MediaPipe resources.
            if self.landmarker is not None:

                self.landmarker.close()
                self.landmarker = None

            self.running = False

            print(
                "[Privacy Guard] Camera worker stopped."
            )

    # =========================================================
    # PAUSE MONITORING
    # =========================================================

    def pause_monitoring(self):

        self.paused = True
        self.reset_alert_state()

        # Do not count time spent paused as a sideways orientation.
        self.detect_shoulder_surfing("CENTER")

    # =========================================================
    # RESUME MONITORING
    # =========================================================

    def resume_monitoring(self):

        self.reset_alert_state()
        self.detect_shoulder_surfing("CENTER")
        self.paused = False

    # =========================================================
    # STOP MONITORING
    # =========================================================

    def stop_monitoring(self):

        self.running = False
        self.paused = False

        if self.isRunning():
            self.wait()
