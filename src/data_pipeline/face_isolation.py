from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.data_pipeline.path_utils import relativize_to_project_root, resolve_manifest_filepath


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
LEFT_EYE_CORNER_IDX = [33, 133]
RIGHT_EYE_CORNER_IDX = [362, 263]
NOSE_BRIDGE_IDX = [168, 6, 197]

# Stable anchor landmarks
FOREHEAD_TOP_IDX = 10
CHIN_IDX = 152

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MAN_DIR = DATA_DIR / "manifests"

MANIFEST_KIND_ALIASES = {
    "images_only": "images_only",
    "single_image": "single_image",
    "single_images": "single_image",
}

DEFAULT_INPUT_MANIFESTS = {
    "images_only": MAN_DIR / "manifest_image_only.parquet",
    "single_image": MAN_DIR / "manifest_single_image.parquet",
}

REQUIRED_MANIFEST_COLUMNS = {"patient_id", "filepath"}

DEFAULT_MARGIN_X_RATIO = 0.18
DEFAULT_MARGIN_TOP_RATIO = 0.20
DEFAULT_MARGIN_BOTTOM_RATIO = 0.04
DEFAULT_PAD_VALUE = 0


@dataclass(frozen=True)
class FaceCropSpec:
    width: int
    height: int
    anchor_offset_x: float
    anchor_offset_y: float


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


def get_eye_midpoint(landmarks: np.ndarray) -> np.ndarray:
    left_eye = get_eye_center(landmarks, LEFT_EYE_CORNER_IDX)
    right_eye = get_eye_center(landmarks, RIGHT_EYE_CORNER_IDX)
    return ((left_eye + right_eye) / 2.0).astype(np.float32)


def get_nose_bridge_center(landmarks: np.ndarray) -> np.ndarray:
    return landmarks[NOSE_BRIDGE_IDX].mean(axis=0).astype(np.float32)


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Eye corners are more stable than eyelid landmarks when patients blink or wink.
    left_eye = get_eye_center(landmarks, LEFT_EYE_CORNER_IDX)
    right_eye = get_eye_center(landmarks, RIGHT_EYE_CORNER_IDX)

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


def get_stable_face_extents(landmarks: np.ndarray) -> tuple[float, float, float, float]:
    face_oval_pts = landmarks[FACE_OVAL_IDX]

    left_x = float(np.min(face_oval_pts[:, 0]))
    right_x = float(np.max(face_oval_pts[:, 0]))
    top_y = float(landmarks[FOREHEAD_TOP_IDX, 1])
    bottom_y = float(landmarks[CHIN_IDX, 1])
    return left_x, right_x, top_y, bottom_y


def get_stable_face_box(
    landmarks: np.ndarray,
    margin_x_ratio: float = DEFAULT_MARGIN_X_RATIO,
    margin_top_ratio: float = DEFAULT_MARGIN_TOP_RATIO,
    margin_bottom_ratio: float = DEFAULT_MARGIN_BOTTOM_RATIO,
) -> tuple[float, float, float, float]:
    left_x, right_x, top_y, bottom_y = get_stable_face_extents(landmarks)

    face_width = right_x - left_x
    face_height = bottom_y - top_y

    margin_x = face_width * margin_x_ratio
    margin_top = face_height * margin_top_ratio
    margin_bottom = face_height * margin_bottom_ratio

    return (
        left_x - margin_x,
        top_y - margin_top,
        right_x + margin_x,
        bottom_y + margin_bottom,
    )


def build_reference_crop_spec(
    landmarks: np.ndarray,
    margin_x_ratio: float = DEFAULT_MARGIN_X_RATIO,
    margin_top_ratio: float = DEFAULT_MARGIN_TOP_RATIO,
    margin_bottom_ratio: float = DEFAULT_MARGIN_BOTTOM_RATIO,
) -> FaceCropSpec:
    x1, y1, x2, y2 = get_stable_face_box(
        landmarks,
        margin_x_ratio=margin_x_ratio,
        margin_top_ratio=margin_top_ratio,
        margin_bottom_ratio=margin_bottom_ratio,
    )

    crop_left = int(np.floor(x1))
    crop_top = int(np.floor(y1))
    crop_right = int(np.ceil(x2))
    crop_bottom = int(np.ceil(y2))

    width = crop_right - crop_left
    height = crop_bottom - crop_top
    if width <= 0 or height <= 0:
        raise ValueError("Invalid reference crop box after landmark-based cropping.")

    box_center = np.array(
        [crop_left + (width / 2.0), crop_top + (height / 2.0)],
        dtype=np.float32,
    )
    # A nose-bridge anchor is less sensitive to blinks, winks, and fully closed eyes.
    anchor_point = get_nose_bridge_center(landmarks)
    anchor_offset = box_center - anchor_point

    return FaceCropSpec(
        width=width,
        height=height,
        anchor_offset_x=float(anchor_offset[0]),
        anchor_offset_y=float(anchor_offset[1]),
    )


def crop_with_padding(
    image_bgr: np.ndarray,
    x1: int,
    y1: int,
    width: int,
    height: int,
    pad_value: int = DEFAULT_PAD_VALUE,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Crop size must be positive.")

    x2 = x1 + width
    y2 = y1 + height

    src_x1 = max(x1, 0)
    src_y1 = max(y1, 0)
    src_x2 = min(x2, image_bgr.shape[1])
    src_y2 = min(y2, image_bgr.shape[0])

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        raise ValueError("Fixed crop box fell completely outside the image.")

    cropped = np.full((height, width, image_bgr.shape[2]), pad_value, dtype=image_bgr.dtype)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    cropped[dst_y1:dst_y2, dst_x1:dst_x2] = image_bgr[src_y1:src_y2, src_x1:src_x2]
    return cropped


def crop_from_reference_spec(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    crop_spec: FaceCropSpec,
    pad_value: int = DEFAULT_PAD_VALUE,
) -> np.ndarray:
    anchor_point = get_nose_bridge_center(landmarks)
    box_center = anchor_point + np.array(
        [crop_spec.anchor_offset_x, crop_spec.anchor_offset_y],
        dtype=np.float32,
    )

    x1 = int(np.round(box_center[0] - (crop_spec.width / 2.0)))
    y1 = int(np.round(box_center[1] - (crop_spec.height / 2.0)))

    return crop_with_padding(
        image_bgr,
        x1=x1,
        y1=y1,
        width=crop_spec.width,
        height=crop_spec.height,
        pad_value=pad_value,
    )


def crop_from_stable_landmarks(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    margin_x_ratio: float = DEFAULT_MARGIN_X_RATIO,
    margin_top_ratio: float = DEFAULT_MARGIN_TOP_RATIO,
    margin_bottom_ratio: float = DEFAULT_MARGIN_BOTTOM_RATIO
) -> np.ndarray:
    crop_spec = build_reference_crop_spec(
        landmarks,
        margin_x_ratio=margin_x_ratio,
        margin_top_ratio=margin_top_ratio,
        margin_bottom_ratio=margin_bottom_ratio,
    )
    return crop_from_reference_spec(image_bgr, landmarks, crop_spec, pad_value=DEFAULT_PAD_VALUE)


def resize_and_pad(
    image_bgr: np.ndarray,
    target_size: int = 224,
    pad_value: int = DEFAULT_PAD_VALUE
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


def load_and_align_face(
    image_path: str | Path,
    face_mesh,
) -> tuple[np.ndarray, np.ndarray] | None:
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logging.warning("Could not read image: %s", image_path)
        return None

    landmarks = get_face_landmarks(image_bgr, face_mesh)
    if landmarks is None:
        logging.warning("No face found: %s", image_path)
        return None

    return align_face(image_bgr, landmarks)


def isolate_face_image(
    image_path: str | Path,
    output_path: str | Path,
    face_mesh,
    target_size: int = 224,
    reference_crop_spec: FaceCropSpec | None = None,
) -> tuple[bool, FaceCropSpec | None]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aligned_face = load_and_align_face(image_path, face_mesh)
    if aligned_face is None:
        return False, reference_crop_spec

    aligned_image, aligned_landmarks = aligned_face

    crop_spec = reference_crop_spec or build_reference_crop_spec(aligned_landmarks)

    try:
        cropped = crop_from_reference_spec(
            aligned_image,
            aligned_landmarks,
            crop_spec,
            pad_value=DEFAULT_PAD_VALUE,
        )
    except ValueError as exc:
        logging.warning("Cropping failed for %s: %s", image_path, exc)
        return False, reference_crop_spec

    final_img = resize_and_pad(cropped, target_size=target_size, pad_value=DEFAULT_PAD_VALUE)

    ok = cv2.imwrite(str(output_path), final_img)
    if not ok:
        logging.warning("Could not write image: %s", output_path)

    return ok, crop_spec


def isolate_face_to_224(
    image_path: str | Path,
    output_path: str | Path,
    target_size: int = 224
) -> bool:
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False
    ) as face_mesh:
        ok, _ = isolate_face_image(
            image_path=image_path,
            output_path=output_path,
            face_mesh=face_mesh,
            target_size=target_size,
        )
    return ok


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face isolation for an image manifest.")
    parser.add_argument("--manifest-kind", default="images_only", choices=sorted(MANIFEST_KIND_ALIASES))
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--faces-dir", type=Path)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def normalize_manifest_kind(kind: str) -> str:
    try:
        return MANIFEST_KIND_ALIASES[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported manifest kind '{kind}'. Expected one of {sorted(MANIFEST_KIND_ALIASES)}"
        ) from exc


def resolve_manifest_paths(
    manifest_kind: str,
    input_manifest: Path | None,
    output_manifest: Path | None,
    target_size: int,
) -> tuple[Path, Path]:
    resolved_input_manifest = input_manifest or DEFAULT_INPUT_MANIFESTS[manifest_kind]

    if output_manifest is not None:
        resolved_output_manifest = output_manifest
    elif manifest_kind == "images_only":
        resolved_output_manifest = MAN_DIR / f"manifest_face{target_size}.parquet"
    else:
        resolved_output_manifest = MAN_DIR / f"manifest_single_image_face{target_size}.parquet"

    return resolved_input_manifest, resolved_output_manifest


def resolve_faces_dir(faces_dir: Path | None, target_size: int) -> Path:
    return faces_dir or (DATA_DIR / f"faces_{target_size}")


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Input manifest is empty: {path}")

    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input manifest missing required columns: {sorted(missing)}")

    return df


def build_face_output_path(image_path: Path, patient_id: str, faces_dir: Path, target_size: int) -> Path:
    rel_name = f"{image_path.stem}_face{target_size}.jpg"
    return faces_dir / patient_id / rel_name


def run_manifest_face_isolation(
    manifest_kind: str,
    input_manifest: Path | None = None,
    output_manifest: Path | None = None,
    faces_dir: Path | None = None,
    target_size: int = 224,
) -> Path:
    normalized_manifest_kind = normalize_manifest_kind(manifest_kind)
    resolved_input_manifest, resolved_output_manifest = resolve_manifest_paths(
        manifest_kind=normalized_manifest_kind,
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        target_size=target_size,
    )
    resolved_faces_dir = resolve_faces_dir(faces_dir, target_size).resolve()

    df = load_manifest(resolved_input_manifest.resolve())

    new_paths = pd.Series(index=df.index, dtype=object)
    success_flags = pd.Series(False, index=df.index, dtype=bool)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False
    ) as face_mesh:
        for patient_id, patient_rows in df.groupby("patient_id", sort=False):
            reference_crop_spec: FaceCropSpec | None = None

            # Keep a single crop box per patient so every output image shares the same framing.
            for idx, row in patient_rows.iterrows():
                old_path = resolve_manifest_filepath(row["filepath"], project_root=PROJECT_ROOT)
                patient_id_str = str(patient_id)
                new_path = build_face_output_path(
                    image_path=old_path,
                    patient_id=patient_id_str,
                    faces_dir=resolved_faces_dir,
                    target_size=target_size,
                )

                success, resolved_crop_spec = isolate_face_image(
                    image_path=old_path,
                    output_path=new_path,
                    face_mesh=face_mesh,
                    target_size=target_size,
                    reference_crop_spec=reference_crop_spec,
                )

                if reference_crop_spec is None and resolved_crop_spec is not None:
                    reference_crop_spec = resolved_crop_spec

                new_paths.at[idx] = (
                    relativize_to_project_root(new_path, project_root=PROJECT_ROOT)
                    if success
                    else None
                )
                success_flags.at[idx] = success

            if reference_crop_spec is None:
                logging.warning("Face isolation failed for every image in patient %s", patient_id)

    df["face_filepath"] = new_paths
    df["face_success"] = success_flags

    df = df[df["face_success"]].copy()
    if df.empty:
        raise ValueError(f"Face isolation failed for every row in {resolved_input_manifest}")

    df["filepath"] = df["face_filepath"]
    df = df.drop(columns=["face_filepath", "face_success"])

    resolved_output_manifest = resolved_output_manifest.resolve()
    resolved_output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(resolved_output_manifest, index=False)

    logging.info("Manifest kind: %s", normalized_manifest_kind)
    logging.info("Input rows: %s", len(new_paths))
    logging.info("Rows kept: %s", len(df))
    logging.info("Rows failed: %s", len(new_paths) - len(df))
    logging.info("Saved: %s", resolved_output_manifest)
    return resolved_output_manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    run_manifest_face_isolation(
        manifest_kind=args.manifest_kind,
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        faces_dir=args.faces_dir,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()


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
