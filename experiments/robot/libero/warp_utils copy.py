import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

def _as_np(x):
    return np.asarray(x, dtype=np.float64)


# 기초 유틸리티
def get_sim_handle(env):
    for attr in ["sim", "env", "unwrapped"]:
        if hasattr(env, attr):
            res = getattr(env, attr)
            if hasattr(res, "sim"): return res.sim
            if attr == "sim": return res
    return None

def get_cam_T_w_c(sim, cam_id):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    T[:3, 3] = sim.data.cam_xpos[cam_id].copy()
    return T

def invert_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3], Ti[:3, 3] = R.T, -R.T @ t
    return Ti

def relative_T(T_w_c_t, T_w_c_t1): return invert_T(T_w_c_t) @ T_w_c_t1

def K_from_fovy(width, height, fovy_rad):
    fy = (height / 2.0) / np.tan(fovy_rad / 2.0)
    return np.array([[fy * (width / height), 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]], dtype=np.float64)

def plane_in_cam_from_world(T_w_c, n_w, p0_w):
    T_c_w = invert_T(T_w_c)
    n_c = T_c_w[:3, :3] @ n_w
    b_c = -float(n_w @ p0_w) - float(n_w @ (T_c_w[:3, :3].T @ T_c_w[:3, 3]))
    norm = np.linalg.norm(n_c) + 1e-12
    return n_c / norm, float(-b_c / norm)

def homography_from_Rt_plane(K, R, t, n_c, d):
    return K @ (R - (t.reshape(3,1) @ n_c.reshape(1,3)) / d) @ np.linalg.inv(K)

def adjust_K_for_resize_crop(K_raw, raw_wh, proc_wh):
    K = K_raw.copy()
    sx, sy = proc_wh[0] / raw_wh[0], proc_wh[1] / raw_wh[1]
    K[0,0]*=sx; K[1,1]*=sy; K[0,2]*=sx; K[1,2]*=sy
    return K

def compute_pixel_similarity(img_w_pooled, img_p_pooled, threshold=0.9):
    pixel_sim = 1.0 - (img_w_pooled - img_p_pooled).abs().mean(dim=1).reshape(-1)
    return pixel_sim

def compute_static_patch_indices(
    img_pre: np.ndarray,
    img_post: np.ndarray,
    threshold: float = 0.9,
    feat_shape: tuple = (16, 16),
    use_cosine_similarity: bool = True,
) -> list:
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
    dtype = torch.float32
    H_f, W_f = feat_shape

    # 이미지 → GPU tensor, (1, 3, H_f, W_f)로 풀링
    img_pre_t  = torch.from_numpy(np.ascontiguousarray(img_pre)).permute(2, 0, 1).unsqueeze(0).to(device, dtype) / 255.0
    img_post_t = torch.from_numpy(np.ascontiguousarray(img_post)).permute(2, 0, 1).unsqueeze(0).to(device, dtype) / 255.0

    img_pre_pooled  = F.adaptive_avg_pool2d(img_pre_t,  (H_f, W_f))
    img_post_pooled = F.adaptive_avg_pool2d(img_post_t, (H_f, W_f))

    if use_cosine_similarity:
        # compute_pixel_similarity와 동일한 방식
        pixel_sim = compute_pixel_similarity(img_pre_pooled, img_post_pooled)
        static_mask = pixel_sim > threshold
    else:
        # cosine similarity 미사용 시 전부 static
        static_mask = torch.ones(H_f * W_f, dtype=torch.bool, device=device)

    static_indices = torch.where(static_mask)[0].cpu().tolist()
    return static_indices

class WarpTracker:
    def __init__(self, keyframe_interval=None):
        # keyframe_interval은 외부(run_libero_eval)에서 관리하므로 사용하지 않음
        # 시그니처 호환을 위해 인자만 유지
        self.plane_cached, self.cam_cached = False, False
        self.last_H_img = None
        self.last_img_pre = None
        self.last_T_w_c_pre = None

        self.last_within_bounds = None
        self.last_pixel_sim = None
        

    def init_env_info(self, sim, img_shape):
        if not self.cam_cached:
            cam_names = [sim.model.camera(i).name for i in range(sim.model.ncam)]
            self.cam_id = cam_names.index("robot0_eye_in_hand" if "robot0_eye_in_hand" in cam_names else cam_names[0])
            self.cam_cached = True
        if not hasattr(self, "K_proc"):
            fovy = float(sim.model.cam_fovy[self.cam_id]) * np.pi / 180.0
            self.K_proc = adjust_K_for_resize_crop(K_from_fovy(256, 256, fovy), (256, 256), (img_shape[1], img_shape[0]))
        if not self.plane_cached:
            geom_names = [sim.model.geom(i).name for i in range(sim.model.ngeom)]
            gid = geom_names.index(next(g for g in geom_names if g and ("table" in g.lower() or "desk" in g.lower() or "floor" in g.lower())))
            z_axis = sim.data.geom_xmat[gid].reshape(3, 3)[:, 2]
            self.table_p0_w = sim.data.geom_xpos[gid] + z_axis * float(sim.model.geom_size[gid][2])
            self.table_n_w = z_axis / (np.linalg.norm(z_axis) + 1e-12)
            self.plane_cached = True

    def step_warp(
        self,
        T_w_c_pre,
        img_pre,
        T_w_c_post,
        img_post,
        img_shape,
        threshold=0.9,
        feat_shape=(16,16),
        use_cosine_similarity=True,
        include_warp_holes=False,
    ):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        dtype = torch.float32

        # 1. 매 호출마다 인자로 받은 (T_w_c_pre, img_pre)를 keyframe으로 사용
        # keyframe 결정은 호출자(run_libero_eval) 책임
        n_c_key, d_key = plane_in_cam_from_world(T_w_c_pre, self.table_n_w, self.table_p0_w)
        if d_key < 0:
            n_c_key, d_key = -n_c_key, -d_key
        img_key_gpu = torch.from_numpy(img_pre.copy()).permute(2, 0, 1).unsqueeze(0).to(device, dtype) / 255.0

        # 2. 호모그래피 및 피처용 변환 행렬 계산
        T_rel = relative_T(T_w_c_pre, T_w_c_post)
        H_img = homography_from_Rt_plane(self.K_proc, T_rel[:3, :3].T, T_rel[:3, 3], n_c_key, d_key)
        H_img /= H_img[2, 2]
        self.last_H_img = H_img
        self.last_img_pre = img_pre
        self.last_T_w_c_pre = T_w_c_pre

        H_f, W_f = feat_shape # f_post 대신 인자로 받은 shape 사용

        scale_x, scale_y = W_f / img_shape[1], H_f / img_shape[0]
        S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        H_feat = S @ H_img @ np.linalg.inv(S)
        H_inv = torch.tensor(np.linalg.inv(H_feat), dtype=torch.float32, device=device)

        # 3. 좌표 추적 및 Grid 생성
        y, x = torch.meshgrid(torch.arange(H_f, device=device), torch.arange(W_f, device=device), indexing='ij')
        curr_coords = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1).float()
        src_coords = torch.matmul(H_inv, curr_coords)
        src_coords[:2, :] /= (src_coords[2:3, :] + 1e-8)

        # F.grid_sample용 그리드 생성
        grid = torch.stack([
                (src_coords[0, :] / (W_f - 1)) * 2 - 1,
                (src_coords[1, :] / (H_f - 1)) * 2 - 1
            ], dim=-1).reshape(1, H_f, W_f, 2).to(dtype)

        # 4. 워핑 수행
        img_w_pooled = F.grid_sample(img_key_gpu, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        img_post_t = torch.from_numpy(img_post).permute(2,0,1).unsqueeze(0).to(device, dtype) / 255.0
        img_p_pooled = F.adaptive_avg_pool2d(img_post_t, (H_f, W_f))
        
        # 함수 분리 적용
        pixel_sim = compute_pixel_similarity(img_w_pooled, img_p_pooled)

        src_x, src_y = torch.round(src_coords[0, :]).long(), torch.round(src_coords[1, :]).long()
        within_bounds = (src_x >= 0) & (src_x < W_f) & (src_y >= 0) & (src_y < H_f)
        self.last_within_bounds = within_bounds.detach().cpu()
        self.last_pixel_sim = pixel_sim.detach().cpu()
        warp_mapping = torch.where(within_bounds, src_y * W_f + src_x, torch.zeros_like(src_x))

        # cosine similarity 쓸 때만 threshold로 static patch 고르는 부분임
        if use_cosine_similarity:
            static_mask = (pixel_sim > threshold) & within_bounds
        else:
            static_mask = within_bounds

        # warp hole도 재사용 후보에 넣을지 정하는 부분임
        if include_warp_holes:
            static_mask = static_mask | (~within_bounds)

        static_indices = torch.where(static_mask)[0].cpu().tolist()

        return warp_mapping, static_indices


    def step_warp_geom_only(
        self,
        T_w_c_pre,
        img_pre,
        T_w_c_post,
        img_shape,
        feat_shape=(16, 16),
    ):
        """
        step_warp의 경량 버전: pixel_similarity/static_indices 계산 생략.
        attn_warp용 — warp_mapping과 within_bounds만 필요한 경우.

        수치적으로 step_warp와 동일한 H, src_coords, within_bounds, warp_mapping 생성.
        (같은 K_proc, 같은 공식, 같은 round, 같은 bounds 체크)

        생략되는 작업:
        - img_pre / img_post GPU 업로드
        - grid_sample, adaptive_avg_pool2d
        - compute_pixel_similarity
        - static_indices의 .cpu().tolist() (GPU→CPU sync)
        """
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # 1. Plane 파라미터 (step_warp와 동일)
        n_c_key, d_key = plane_in_cam_from_world(T_w_c_pre, self.table_n_w, self.table_p0_w)
        if d_key < 0:
            n_c_key, d_key = -n_c_key, -d_key

        # 2. Homography (step_warp와 동일)
        T_rel = relative_T(T_w_c_pre, T_w_c_post)
        H_img = homography_from_Rt_plane(self.K_proc, T_rel[:3, :3].T, T_rel[:3, 3], n_c_key, d_key)
        H_img /= H_img[2, 2]
        self.last_H_img = H_img
        self.last_img_pre = img_pre
        self.last_T_w_c_pre = T_w_c_pre

        H_f, W_f = feat_shape
        scale_x, scale_y = W_f / img_shape[1], H_f / img_shape[0]
        S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
        H_feat = S @ H_img @ np.linalg.inv(S)
        H_inv = torch.tensor(np.linalg.inv(H_feat), dtype=torch.float32, device=device)

        # 3. 좌표 매핑 (step_warp와 동일)
        y, x = torch.meshgrid(torch.arange(H_f, device=device), torch.arange(W_f, device=device), indexing='ij')
        curr_coords = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1).float()
        src_coords = torch.matmul(H_inv, curr_coords)
        src_coords[:2, :] /= (src_coords[2:3, :] + 1e-8)

        # 4. within_bounds & warp_mapping (step_warp와 동일)
        src_x, src_y = torch.round(src_coords[0, :]).long(), torch.round(src_coords[1, :]).long()
        within_bounds = (src_x >= 0) & (src_x < W_f) & (src_y >= 0) & (src_y < H_f)
        self.last_within_bounds = within_bounds.detach().cpu()
        warp_mapping = torch.where(within_bounds, src_y * W_f + src_x, torch.zeros_like(src_x))

        return warp_mapping, within_bounds



    #################################################################
    ################### Warp 관련 시각화 도구 #########################
    #################################################################

    def _make_patch_overlay(self, base_img, indices, color_in, color_out, alpha=0.4):
        overlay = base_img.copy()
        p_sz = 224 // 16
        indices_set = set(indices) if indices else set()
        for i in range(16):
            for j in range(16):
                patch_idx = i * 16 + j
                color = color_in if patch_idx in indices_set else color_out
                cv2.rectangle(overlay, (j * p_sz, i * p_sz), ((j + 1) * p_sz, (i + 1) * p_sz), color, -1)
        return cv2.addWeighted(overlay, alpha, base_img, 1.0 - alpha, 0)

    def _add_label(self, img, label):
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(canvas, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def visualize_2x3(self, img_post, save_path, static_indices,
                       critical_indices=None, reusable_indices=None,
                       img_key=None, H_img=None):
        # img_key/H_img가 주어지지 않으면 마지막 step_warp 호출의 값을 사용함
        if img_key is None:
            img_key = self.last_img_pre
        if H_img is None:
            H_img = self.last_H_img
        if img_key is None or H_img is None:
            return

        h, w = img_post.shape[:2]
        img_warped = cv2.warpPerspective(img_key, H_img, (w, h), flags=cv2.INTER_LINEAR)

        rgb_key = cv2.cvtColor(cv2.resize(img_key, (224, 224)), cv2.COLOR_RGB2BGR)
        rgb_warp_base = cv2.cvtColor(cv2.resize(img_warped, (224, 224)), cv2.COLOR_RGB2BGR)
        rgb_post = cv2.cvtColor(cv2.resize(img_post, (224, 224)), cv2.COLOR_RGB2BGR)

        # Static: 파랑=static, 빨강=non-static
        static_panel = self._make_patch_overlay(
            rgb_warp_base, static_indices,
            color_in=[255, 0, 0], color_out=[0, 0, 255],
        )

        # Critical: 빨강=critical(재계산 필요), 파랑=non-critical(중요하지 않음)
        critical_panel = self._make_patch_overlay(
            rgb_post, critical_indices or [],
            color_in=[0, 0, 255], color_out=[255, 0, 0],
        )

        # Reusable: 파랑=reuse, 빨강=recompute (최종 결과)
        reusable_panel = self._make_patch_overlay(
            rgb_post, reusable_indices or static_indices,
            color_in=[255, 0, 0], color_out=[0, 0, 255],
        )

        row = np.concatenate([
            self._add_label(rgb_key, "Keyframe"),
            self._add_label(static_panel, f"Static ({len(static_indices) if static_indices else 0})"),
            self._add_label(critical_panel, f"Critical ({len(critical_indices) if critical_indices else 0})"),
            self._add_label(reusable_panel, f"Reusable ({len(reusable_indices) if reusable_indices else 0})"),
            self._add_label(rgb_post, "Current"),
        ], axis=1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, row)


    