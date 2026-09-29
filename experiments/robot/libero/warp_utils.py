# warping_utils.py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


# 기초 유틸리티
def get_sim_handle(env):
    for attr in ("sim", "env", "unwrapped"):
        if not hasattr(env, attr):
            continue
        candidate = getattr(env, attr)
        if hasattr(candidate, "sim"):
            return candidate.sim
        if attr == "sim":
            return candidate
    return None


def get_cam_T_w_c(sim, cam_id):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    transform[:3, 3] = sim.data.cam_xpos[cam_id].copy()
    return transform


def invert_T(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def relative_T(T_w_c_pre, T_w_c_post):
    return invert_T(T_w_c_pre) @ T_w_c_post


def K_from_fovy(width, height, fovy_rad):
    fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
    return np.array(
        [
            [fy * (width / height), 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def plane_in_cam_from_world(T_w_c, n_w, p0_w):
    T_c_w = invert_T(T_w_c)
    n_c = T_c_w[:3, :3] @ n_w
    b_c = -float(n_w @ p0_w) - float(
        n_w @ (T_c_w[:3, :3].T @ T_c_w[:3, 3])
    )
    norm = np.linalg.norm(n_c) + 1e-12
    return n_c / norm, float(-b_c / norm)


def homography_from_Rt_plane(K, rotation, translation, n_c, distance):
    return K @ (
        rotation
        - (translation.reshape(3, 1) @ n_c.reshape(1, 3)) / distance
    ) @ np.linalg.inv(K)


def adjust_K_for_resize_crop(K_raw, raw_wh, proc_wh):
    K = K_raw.copy()
    scale_x = proc_wh[0] / raw_wh[0]
    scale_y = proc_wh[1] / raw_wh[1]
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def compute_pixel_similarity(warped_pooled, current_pooled):
    return 1.0 - (warped_pooled - current_pooled).abs().mean(dim=1).reshape(-1)


def compute_static_patch_indices(
    img_pre: np.ndarray,
    img_post: np.ndarray,
    threshold: float = 0.9,
    feat_shape=(16, 16),
    use_cosine_similarity: bool = True,
):
    """
    warping 없이 img_pre와 img_post를 직접 비교해서 정적 패치 인덱스를 반환.
    fixed cam(고정 카메라)용 - 카메라가 움직이지 않으므로 homography 불필요.

    Args:
        img_pre: 이전 프레임 이미지 (H, W, 3), uint8
        img_post: 현재 프레임 이미지 (H, W, 3), uint8
        threshold: cosine similarity 임계값 (이 값 이상이면 static으로 판단)
        feat_shape: 패치 그리드 크기 (H_f, W_f), 기본 (16, 16) = 256 patches
        use_cosine_similarity: False면 모든 패치를 static으로 간주

    Returns:
        static_indices: 정적 패치 인덱스 리스트 (0-based, 길이 <= H_f * W_f)
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    height, width = feat_shape

    img_pre_tensor = torch.from_numpy(np.ascontiguousarray(img_pre)).permute(2, 0, 1).unsqueeze(0)
    img_post_tensor = torch.from_numpy(np.ascontiguousarray(img_post)).permute(2, 0, 1).unsqueeze(0)
    img_pre_tensor = img_pre_tensor.to(device=device, dtype=torch.float32).div_(255.0)
    img_post_tensor = img_post_tensor.to(device=device, dtype=torch.float32).div_(255.0)

    img_pre_pooled = F.adaptive_avg_pool2d(img_pre_tensor, (height, width))
    img_post_pooled = F.adaptive_avg_pool2d(img_post_tensor, (height, width))

    if use_cosine_similarity:
        static_mask = compute_pixel_similarity(img_pre_pooled, img_post_pooled) > threshold
    else:
        static_mask = torch.ones(height * width, dtype=torch.bool, device=device)

    return torch.where(static_mask)[0].cpu().tolist()


def safe_direction_cosine(curr_delta, prev_delta, eps: float = 1e-8):
    if curr_delta is None or prev_delta is None:
        return np.nan

    curr = np.asarray(curr_delta, dtype=np.float32)
    prev = np.asarray(prev_delta, dtype=np.float32)
    curr_norm = float(np.linalg.norm(curr))
    prev_norm = float(np.linalg.norm(prev))

    if curr_norm < eps or prev_norm < eps:
        return np.nan
    return float(np.dot(curr, prev) / (curr_norm * prev_norm + eps))


def rotation_delta_angle_rad(T_prev, T_curr):
    if T_prev is None or T_curr is None:
        return np.nan

    rotation_prev = np.asarray(T_prev[:3, :3], dtype=np.float32)
    rotation_curr = np.asarray(T_curr[:3, :3], dtype=np.float32)
    rotation_delta = rotation_curr @ rotation_prev.T
    cos_theta = float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cos_theta))

def compute_vla_cache_static_patch_indices(
    img_pre: np.ndarray,
    img_post: np.ndarray,
    *,
    patch_size: int = 14,
    resize_hw=(224, 224),
    top_k: int = 150,
    sim_threshold: float = 0.996,
):
    """VLA-Cache-style same-index static patch selection for A1.

    The supplied VLA-Cache utility uses 224x224 images, 14x14 RGB patches,
    cosine similarity, threshold 0.996, and a top-150 cap. Resizing here makes
    the behavior explicit even when the evaluator observation is still 256x256.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def _patch_vectors(image):
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        x = x.to(device=device, dtype=torch.float32).div_(255.0)
        if tuple(x.shape[-2:]) != tuple(resize_hw):
            x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
        h, w = resize_hw
        if h % patch_size or w % patch_size:
            raise ValueError("resize_hw must be divisible by patch_size")
        p = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        return p.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3 * patch_size * patch_size)

    pre = _patch_vectors(img_pre)
    post = _patch_vectors(img_post)
    similarity = F.cosine_similarity(pre, post, dim=1, eps=1e-8)
    valid = torch.where(similarity >= float(sim_threshold))[0]
    if valid.numel() == 0:
        return []
    order = torch.argsort(similarity[valid], descending=True, stable=True)
    return valid[order[: min(int(top_k), int(order.numel()))]].detach().cpu().tolist()


class WarpTracker:
    def __init__(self, keyframe_interval=None):
        self.plane_cached, self.cam_cached = False, False
        self.last_H_img = None
        self.last_img_pre = None
        self.last_T_w_c_pre = None
        self.last_within_bounds = None
        self.last_pixel_sim = None
        self._coord_cache = {}
        

    def init_env_info(self, sim, img_shape):
        if not self.cam_cached:
            camera_names = [sim.model.camera(i).name for i in range(sim.model.ncam)]
            camera_name = (
                "robot0_eye_in_hand"
                if "robot0_eye_in_hand" in camera_names
                else camera_names[0]
            )
            self.cam_id = camera_names.index(camera_name)
            self.cam_cached = True

        if not hasattr(self, "K_proc"):
            fovy = float(sim.model.cam_fovy[self.cam_id]) * np.pi / 180.0
            self.K_proc = adjust_K_for_resize_crop(
                K_from_fovy(256, 256, fovy),
                (256, 256),
                (img_shape[1], img_shape[0]),
            )

        if not self.plane_cached:
            geom_names = [sim.model.geom(i).name for i in range(sim.model.ngeom)]
            plane_name = next(
                name
                for name in geom_names
                if name and any(key in name.lower() for key in ("table", "desk", "floor"))
            )
            geom_id = geom_names.index(plane_name)
            z_axis = sim.data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
            self.table_p0_w = (
                sim.data.geom_xpos[geom_id]
                + z_axis * float(sim.model.geom_size[geom_id][2])
            )
            self.table_n_w = z_axis / (np.linalg.norm(z_axis) + 1e-12)
            self.plane_cached = True


    def _feature_coordinates(self, feat_shape, device):
        cache_key = (int(feat_shape[0]), int(feat_shape[1]), str(device))
        cached = self._coord_cache.get(cache_key)
        if cached is not None:
            return cached

        height, width = feat_shape
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        coordinates = torch.stack(
            [x, y, torch.ones_like(x)],
            dim=0,
        ).reshape(3, -1).float()
        self._coord_cache[cache_key] = coordinates
        return coordinates

    def step_warp(
        self,
        T_w_c_pre,
        img_pre,
        T_w_c_post,
        img_post,
        img_shape,
        threshold=0.9,
        feat_shape=(16, 16),
        use_cosine_similarity=True,
        include_warp_holes=False,
    ):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        height, width = feat_shape

        # --------------------------------------------------------------
        # MuJoCo camera frame -> CV optical frame.
        #
        # MuJoCo:
        #   +X right, +Y up, viewing direction -Z
        #
        # CV optical:
        #   +X right, +Y down, viewing direction +Z
        # --------------------------------------------------------------
        C4 = np.eye(4, dtype=np.float64)
        C4[:3, :3] = np.diag([1.0, -1.0, -1.0])

        T_w_cv_pre = np.asarray(T_w_c_pre, dtype=np.float64) @ C4
        T_w_cv_post = np.asarray(T_w_c_post, dtype=np.float64) @ C4

        # Transform a point from previous CV camera -> current CV camera.
        T_post_from_pre = invert_T(T_w_cv_post) @ T_w_cv_pre
        R_post_pre = T_post_from_pre[:3, :3]
        t_post_pre = T_post_from_pre[:3, 3]

        # Table plane in the previous CV-camera coordinates.
        T_cv_pre_w = invert_T(T_w_cv_pre)

        n_pre = (
            T_cv_pre_w[:3, :3]
            @ np.asarray(self.table_n_w, dtype=np.float64)
        )
        n_pre /= np.linalg.norm(n_pre) + 1e-12

        p0_pre = (
            T_cv_pre_w[:3, :3]
            @ np.asarray(self.table_p0_w, dtype=np.float64)
            + T_cv_pre_w[:3, 3]
        )

        d_pre = float(n_pre @ p0_pre)

        if abs(d_pre) < 1e-10:
            raise RuntimeError(
                "Degenerate table-plane distance in previous CV camera."
            )

        K = np.asarray(self.K_proc, dtype=np.float64)

        # Previous image -> current image.
        H_img = K @ (
            R_post_pre
            + (
                t_post_pre.reshape(3, 1)
                @ n_pre.reshape(1, 3)
            ) / d_pre
        ) @ np.linalg.inv(K)

        H_img /= H_img[2, 2]

        self.last_H_img = H_img
        self.last_img_pre = img_pre
        self.last_T_w_c_pre = T_w_c_pre

        # --------------------------------------------------------------
        # Current visual-token centers -> previous visual-token positions.
        #
        # Example for 256x256 and 16x16:
        # token centers are 7.5, 23.5, 39.5, ...
        # --------------------------------------------------------------
        current_coordinates = self._feature_coordinates(
            feat_shape,
            device,
        )

        img_h = int(img_shape[0])
        img_w = int(img_shape[1])

        current_u = (
            (current_coordinates[0] + 0.5)
            * (float(img_w) / float(width))
            - 0.5
        )
        current_v = (
            (current_coordinates[1] + 0.5)
            * (float(img_h) / float(height))
            - 0.5
        )

        current_pixels = torch.stack(
            [
                current_u,
                current_v,
                torch.ones_like(current_u),
            ],
            dim=0,
        )

        H_inverse_img = torch.as_tensor(
            np.linalg.inv(H_img),
            dtype=torch.float32,
            device=device,
        )

        # H_img is previous -> current, therefore H^-1 gives:
        # current target pixel -> previous source pixel.
        source_pixels = H_inverse_img @ current_pixels
        source_pixels[:2] /= source_pixels[2:3] + 1e-8

        source_u = source_pixels[0]
        source_v = source_pixels[1]

        # Previous pixel -> continuous previous visual-token coordinate.
        source_x = (
            (source_u + 0.5)
            * (float(width) / float(img_w))
            - 0.5
        )
        source_y = (
            (source_v + 0.5)
            * (float(height) / float(img_h))
            - 0.5
        )

        source_coordinates = torch.stack(
            [
                source_x,
                source_y,
                torch.ones_like(source_x),
            ],
            dim=0,
        )

        source_x_round = torch.round(source_x).long()
        source_y_round = torch.round(source_y).long()

        within_bounds = (
            (source_x_round >= 0)
            & (source_x_round < width)
            & (source_y_round >= 0)
            & (source_y_round < height)
        )

        # target current token i -> previous source token warp_mapping[i]
        warp_mapping = torch.where(
            within_bounds,
            source_y_round * width + source_x_round,
            torch.zeros_like(source_x_round),
        )


        source_grid = torch.stack(
            [
                (source_u / max(img_w - 1, 1)) * 2.0 - 1.0,
                (source_v / max(img_h - 1, 1)) * 2.0 - 1.0,
            ],
            dim=-1,
        ).reshape(1, height, width, 2)

        img_pre_tensor = torch.from_numpy(np.ascontiguousarray(img_pre)).permute(2, 0, 1).unsqueeze(0)
        img_post_tensor = torch.from_numpy(np.ascontiguousarray(img_post)).permute(2, 0, 1).unsqueeze(0)
        img_pre_tensor = img_pre_tensor.to(device=device, dtype=torch.float32).div_(255.0)
        img_post_tensor = img_post_tensor.to(device=device, dtype=torch.float32).div_(255.0)
        
        # 4. 워핑 수행
        warped_pooled = F.grid_sample(
            img_pre_tensor,
            source_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        current_pooled = F.adaptive_avg_pool2d(img_post_tensor, (height, width))
        # 함수 분리 적용
        pixel_similarity = compute_pixel_similarity(warped_pooled, current_pooled)

        static_mask = within_bounds
        # cosine similarity 쓸 때만 threshold로 static patch 고르는 부분임
        if use_cosine_similarity:
            static_mask = static_mask & (pixel_similarity > threshold)

        # warp hole도 재사용 후보에 넣을지 정하는 부분임
        if include_warp_holes:
            static_mask = static_mask | ~within_bounds

        self.last_within_bounds = within_bounds.detach().cpu()
        self.last_pixel_sim = pixel_similarity.detach().cpu()
        static_indices = torch.where(static_mask)[0].cpu().tolist()
        return warp_mapping, static_indices
