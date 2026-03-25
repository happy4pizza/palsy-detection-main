from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp


# Face oval indices, still useful for left/right edge anchors
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

# MediaPipe Face Mesh landmark groups
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]

# Stable anchor landmarks
FOREHEAD_TOP_IDX = 10
CHIN_IDX = 152


def get_face_landmarks(image_bgr: np.ndarray, face_mesh) -> np.ndarray | None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    h, w = image_bgr.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark

    pts = []
    for lm in landmarks:
        x = lm.x * w
        y = lm.y * h
        pts.append([x, y])

    return np.array(pts, dtype=np.float32)


def get_eye_center(landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
    return landmarks[indices].mean(axis=0)


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    left_eye = get_eye_center(landmarks, LEFT_EYE_IDX)
    right_eye = get_eye_center(landmarks, RIGHT_EYE_IDX)

    eye_dx = right_eye[0] - left_eye[0]
    eye_dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(eye_dy, eye_dx))
    eyes_center = ((left_eye + right_eye) / 2.0).astype(np.float32)

    rot_mat = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)

    h, w = image_bgr.shape[:2]
    aligned_image = cv2.warpAffine(
        image_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    landmarks_h = np.hstack([landmarks, ones])
    aligned_landmarks = (rot_mat @ landmarks_h.T).T

    return aligned_image, aligned_landmarks


def crop_from_stable_landmarks(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    margin_x_ratio: float = 0.18,
    margin_top_ratio: float = 0.20,
    margin_bottom_ratio: float = 0.04
) -> np.ndarray:
    face_oval_pts = landmarks[FACE_OVAL_IDX]

    left_x = np.min(face_oval_pts[:, 0])
    right_x = np.max(face_oval_pts[:, 0])

    top_y = landmarks[FOREHEAD_TOP_IDX, 1]
    bottom_y = landmarks[CHIN_IDX, 1]

    face_width = right_x - left_x
    face_height = bottom_y - top_y

    margin_x = face_width * margin_x_ratio
    margin_top = face_height * margin_top_ratio
    margin_bottom = face_height * margin_bottom_ratio

    x1 = int(max(left_x - margin_x, 0))
    y1 = int(max(top_y - margin_top, 0))
    x2 = int(min(right_x + margin_x, image_bgr.shape[1]))
    y2 = int(min(bottom_y + margin_bottom, image_bgr.shape[0]))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop box after landmark-based cropping.")

    return image_bgr[y1:y2, x1:x2]


def resize_and_pad(
    image_bgr: np.ndarray,
    target_size: int = 224,
    pad_value: int = 0
) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Image has invalid dimensions after cropping.")

    scale = min(target_size / w, target_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value)
    )

    return padded

def isolate_face_to_224(
    image_path: str | Path,
    output_path: str | Path,
    target_size: int = 224
) -> bool:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Could not read image: {image_path}")
        return False

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False
    ) as face_mesh:
        landmarks = get_face_landmarks(image_bgr, face_mesh)

    if landmarks is None:
        print(f"No face found: {image_path}")
        return False

    aligned_image, aligned_landmarks = align_face(image_bgr, landmarks)

    cropped = crop_from_stable_landmarks(
        aligned_image,
        aligned_landmarks,
        margin_x_ratio=0.18,
        margin_top_ratio=0.20,
        margin_bottom_ratio=0.04
    )

    final_img = resize_and_pad(cropped, target_size=target_size, pad_value=0)

    ok = cv2.imwrite(str(output_path), final_img)
    return ok


#Darken face background. Is not used in final pipeline, but keeping code here for potential future use.
# def darken_background_outside_face(
#     image_bgr: np.ndarray,
#     landmarks: np.ndarray,
#     darken_factor: float = 0.45,
#     feather: int = 31
# ) -> np.ndarray:
#     """
#     Darken everything outside the face oval using a soft mask.

#     Parameters
#     ----------
#     image_bgr : np.ndarray
#         Input aligned image.
#     landmarks : np.ndarray
#         Aligned facial landmarks of shape (N, 2).
#     darken_factor : float
#         Multiplier for pixels outside the face.
#         1.0 = no darkening
#         0.45 = moderately dark
#         0.25 = aggressive darkening
#     feather : int
#         Gaussian blur kernel size for soft mask edges.
#         Must be odd.
#     """
#     if not (0.0 < darken_factor <= 1.0):
#         raise ValueError("darken_factor must be in (0, 1].")

#     if feather % 2 == 0:
#         feather += 1

#     h, w = image_bgr.shape[:2]
#     face_oval_pts = landmarks[FACE_OVAL_IDX].astype(np.int32)

#     # Binary face mask
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillConvexPoly(mask, face_oval_pts, 255)

#     # Feather edges so transition is natural
#     soft_mask = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
#     soft_mask = np.expand_dims(soft_mask, axis=2)

#     # Darkened version of the whole image
#     darkened = np.clip(image_bgr.astype(np.float32) * darken_factor, 0, 255).astype(np.uint8)

#     # Keep face original, outside darkened
#     blended = (
#         image_bgr.astype(np.float32) * soft_mask +
#         darkened.astype(np.float32) * (1.0 - soft_mask)
#     )

#     return blended.astype(np.uint8)


# def isolate_face_to_224(
#     image_path: str | Path,
#     output_path: str | Path,
#     target_size: int = 224
# ) -> bool:
#     image_path = Path(image_path)
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     image_bgr = cv2.imread(str(image_path))
#     if image_bgr is None:
#         print(f"Could not read image: {image_path}")
#         return False

#     with mp.solutions.face_mesh.FaceMesh(
#         static_image_mode=True,
#         max_num_faces=1,
#         refine_landmarks=False
#     ) as face_mesh:
#         landmarks = get_face_landmarks(image_bgr, face_mesh)

#     if landmarks is None:
#         print(f"No face found: {image_path}")
#         return False

#     aligned_image, aligned_landmarks = align_face(image_bgr, landmarks)

#     darkened_image = darken_background_outside_face(
#         aligned_image,
#         aligned_landmarks,
#         darken_factor=0.45,
#         feather=31
#     )

#     cropped = crop_from_stable_landmarks(
#         darkened_image,
#         aligned_landmarks,
#         margin_x_ratio=0.18,
#         margin_top_ratio=0.20,
#         margin_bottom_ratio=0.04
#     )

#     final_img = resize_and_pad(cropped, target_size=target_size, pad_value=0)

#     ok = cv2.imwrite(str(output_path), final_img)
#     return ok

