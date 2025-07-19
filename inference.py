import pycolmap
from models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
import moviepy.editor as mp
from models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import glob
import json
from rich import print
import argparse
import decord
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri

def natural_sort_key(s):
    """自然排序，处理纯数字文件名如00001.png"""
    # 提取文件名（去掉路径）
    filename = os.path.basename(s)
    # 分离文件名和扩展名
    name_without_ext = os.path.splitext(filename)[0]
    # 将文件名转换为整数（去掉前导零）
    return int(name_without_ext)

def generate_coords_4d(track2d_pred, depth_tensor, vis_pred, intrs, mask_list):
    """
    生成类似track_3dtraj_dynamic.py中的coords_4d格式
    
    Args:
        track2d_pred: 2D追踪预测 (T, N, 2)
        depth_tensor: 深度图 (T, H, W) 
        vis_pred: 可见性预测 (T, N)
        intrs: 相机内参 (T, 3, 3)
        mask: 二值掩码 (H, W)
    
    Returns:
        coords_4d: numpy array of shape (1, T, 5, H, W) where 5 represents (x, y, z, visibility, in_mask)
    """
    T, N, _ = track2d_pred.shape
    _, H, W = depth_tensor.shape if len(depth_tensor.shape) == 3 else (depth_tensor.shape[0], depth_tensor.shape[1], depth_tensor.shape[2])
    
    # 初始化coords_4d
    coords_4d = np.zeros((1, T, 5, H, W), dtype=np.float32)
    
    # 为每个像素创建网格坐标
    v_grid, u_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    for t in range(T):
        # 获取当前帧的相机内参
        fx, fy = intrs[t, 0, 0], intrs[t, 1, 1]
        cx, cy = intrs[t, 0, 2], intrs[t, 1, 2]
        
        # 获取当前帧的深度图
        if isinstance(depth_tensor, torch.Tensor):
            depth_map_t = depth_tensor[t].cpu().numpy()
        else:
            depth_map_t = depth_tensor[t]
        
        # 计算每个像素的3D坐标
        coords_4d[0, t, 0] = (u_grid - cx) * depth_map_t / fx  # x坐标
        coords_4d[0, t, 1] = (v_grid - cy) * depth_map_t / fy  # y坐标
        coords_4d[0, t, 2] = depth_map_t  # z坐标（深度）
        
        # 可见性：创建全图的可见性图
        visibility_map = np.zeros((H, W), dtype=np.float32)
        if N > 0 and len(track2d_pred[t]) > 0:
            # 将追踪点的可见性映射到对应像素位置
            for n in range(N):
                if t < len(vis_pred) and n < len(vis_pred[t]):
                    u, v = int(track2d_pred[t, n, 0]), int(track2d_pred[t, n, 1])
                    if 0 <= u < W and 0 <= v < H:
                        vis_value = vis_pred[t, n]
                        if isinstance(vis_value, np.ndarray):
                            vis_value = float(vis_value.item())
                        elif hasattr(vis_value, 'cpu'):
                            vis_value = float(vis_value.cpu().numpy().item())
                        else:
                            vis_value = float(vis_value)
                        visibility_map[v, u] = vis_value
        
        coords_4d[0, t, 3] = visibility_map
        
        # 掩码：获取当前帧的mask
        mask_t = mask_list[t]
        
        # 确保mask尺寸匹配
        if mask_t.shape != (H, W):
            mask_resized = cv2.resize(mask_t.astype(np.uint8), (W, H)).astype(bool)
            coords_4d[0, t, 4] = mask_resized.astype(np.float32)
        else:
            coords_4d[0, t, 4] = mask_t.astype(np.float32)
    
    return coords_4d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGBD")
    parser.add_argument("--data_dir", type=str, default="assets/example0")
    parser.add_argument("--video_name", type=str, default="snowboard")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--use_npz", action="store_true", 
                       help="使用NPZ文件格式而不是原始数据集格式")
    parser.add_argument("--use_video", action="store_true",
                       help="RGB模式下使用MP4视频文件")
    parser.add_argument("--camera_coords", action="store_true",
                       help="输出相机坐标系下的3D轨迹而不是世界坐标系")
    parser.add_argument("--output_coords_4d", action="store_true",
                       help="输出coords_4d格式: (x,y,z,visibility,in_mask) 为每个像素")
    parser.add_argument("--no_resize", action="store_true",
                       help="不缩放输出结果，保持原始图像尺寸")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_dir = args.data_dir + "/results"
    # fps
    fps = int(args.fps)
    mask_dir = args.data_dir + f"/{args.video_name}.png"
    
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    if args.data_type == "RGBD":    

        if args.use_npz:
                # 使用NPZ文件格式（原来的方式）
                npz_dir = args.data_dir + f"/{args.video_name}.npz"
                data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
                video_tensor = data_npz_load["video"] * 255
                video_tensor = torch.from_numpy(video_tensor)
                video_tensor = video_tensor[::fps]
                depth_tensor = data_npz_load["depths"]
                depth_tensor = depth_tensor[::fps]
                intrs = data_npz_load["intrinsics"]
                intrs = intrs[::fps]
                extrs = np.linalg.inv(data_npz_load["extrinsics"])
                extrs = extrs[::fps]
                unc_metric = None
        else:
                # 直接从数据集目录读取文件
                rgb_dir = os.path.join(args.data_dir, "rgb")
                depth_dir = os.path.join(args.data_dir, "depth")
                intrinsics_path = os.path.join(args.data_dir, "cam_K.txt")
                extrinsics_path = os.path.join(args.data_dir, "camera_params.json")
                
                # 加载RGB图像 - 支持00001.png格式
                rgb_files = glob.glob(os.path.join(rgb_dir, "*.png"))
                if not rgb_files:
                    rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
                if not rgb_files:
                    rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpeg"))
                rgb_files.sort(key=natural_sort_key)
                
                print(f"找到 {len(rgb_files)} 个RGB图像文件")
                if rgb_files:
                    print(f"第一个文件: {os.path.basename(rgb_files[0])}")
                    print(f"最后一个文件: {os.path.basename(rgb_files[-1])}")
                
                # 读取第一张图像获取尺寸
                first_img = cv2.imread(rgb_files[0])
                height, width = first_img.shape[:2]
                
                # 加载所有RGB图像
                video_list = []
                for img_path in rgb_files[::fps]:
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video_list.append(img_rgb)
                
                video_tensor = np.array(video_list).transpose(0, 3, 1, 2).astype(np.float32)
                video_tensor = torch.from_numpy(video_tensor)
                
                # 加载深度数据 - 支持00001.npz格式
                depth_files = glob.glob(os.path.join(depth_dir, "*.npz"))
                depth_files.sort(key=natural_sort_key)
                print(f"找到 {len(depth_files)} 个深度文件")
                if depth_files:
                    print(f"第一个深度文件: {os.path.basename(depth_files[0])}")
                    print(f"最后一个深度文件: {os.path.basename(depth_files[-1])}")
                
                depth_list = []
                for i, depth_path in enumerate(depth_files[::fps]):
                    depth_data = np.load(depth_path)
                    # 假设深度数据的键名为 'depth' 或第一个键
                    depth_key = 'depth' if 'depth' in depth_data.keys() else list(depth_data.keys())[0]
                    depth_list.append(depth_data[depth_key].astype(np.float32))
                
                depth_tensor = np.array(depth_list)
                
                # 加载相机内参
                intrinsics_3x3 = np.loadtxt(intrinsics_path)
                num_frames = len(video_list)
                intrs = np.tile(intrinsics_3x3[None, :, :], (num_frames, 1, 1))
                
                # 加载相机外参
                with open(extrinsics_path, 'r') as f:
                    extrinsics_data = json.load(f)
                
                extrinsics_list = []
                if isinstance(extrinsics_data, list):
                    # 格式: [{"model_matrix": [[...]], ...}, ...]
                    for frame_data in extrinsics_data[::fps]:
                        if "model_matrix" in frame_data:
                            extrinsics_list.append(np.array(frame_data["model_matrix"]))
                
                extrinsics_array = np.array(extrinsics_list)
                
                # 确保外参数量匹配帧数
                if len(extrinsics_array) != num_frames:
                    if len(extrinsics_array) < num_frames:
                        # 重复最后一个外参
                        last_extrinsic = extrinsics_array[-1]
                        padding = np.tile(last_extrinsic[None, :, :], 
                                        (num_frames - len(extrinsics_array), 1, 1))
                        extrinsics_array = np.concatenate([extrinsics_array, padding], axis=0)
                    else:
                        # 截取前面的部分
                        extrinsics_array = extrinsics_array[:num_frames]
                
                extrs = np.linalg.inv(extrinsics_array)
                unc_metric = None
            
            # 创建data_npz_load字典以保持兼容性
        data_npz_load = {}

    elif args.data_type == "RGB":
        if args.use_video:
            # 使用MP4视频文件
            vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
            video_reader = decord.VideoReader(vid_dir)
            video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)  # Convert to tensor and permute to (N, C, H, W)
            video_tensor = video_tensor[::fps].float()
        else:
            # 直接从RGB图片文件夹读取
            rgb_dir = os.path.join(args.data_dir, "rgb")
            
            # 加载RGB图像 - 支持00001.png格式
            rgb_files = glob.glob(os.path.join(rgb_dir, "*.png"))
            if not rgb_files:
                rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
            if not rgb_files:
                rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpeg"))
            rgb_files.sort(key=natural_sort_key)
            
            print(f"RGB模式: 找到 {len(rgb_files)} 个RGB图像文件")
            if rgb_files:
                print(f"第一个文件: {os.path.basename(rgb_files[0])}")
                print(f"最后一个文件: {os.path.basename(rgb_files[-1])}")
            
            # 加载所有RGB图像
            video_list = []
            for img_path in rgb_files[::fps]:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video_list.append(img_rgb)
            
            # 转换为张量格式 (T, C, H, W)
            video_tensor = np.array(video_list).transpose(0, 3, 1, 2).astype(np.float32)
            video_tensor = torch.from_numpy(video_tensor)
        
           
        # process the image tensor
        # video_tensor = preprocess_image(video_tensor)[None]
        video_tensor = video_tensor.unsqueeze(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = vggt4track_model(video_tensor.cuda()/255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
        
        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        #NOTE: 20% of the depth is not reliable
        # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

        data_npz_load = {}
    
    # 加载mask - 支持每帧对应的mask
    masks_dir = os.path.join(args.data_dir, "masks")
    single_mask_path = args.data_dir + f"/{args.video_name}.png"
    
    if  os.path.exists(masks_dir):
        # 每帧对应的mask文件夹
        mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
        
        
        if mask_files:
            mask_files.sort(key=natural_sort_key)
            print(f"找到 {len(mask_files)} 个mask文件")
            if mask_files:
                print(f"第一个mask文件: {os.path.basename(mask_files[0])}")
                print(f"最后一个mask文件: {os.path.basename(mask_files[-1])}")
            
            # 加载每帧的mask
            mask_list = []
            for mask_path in mask_files[::fps]:
                mask_img = cv2.imread(mask_path)
                if mask_img is not None:
                    # 调整mask尺寸并转换为二值
                    mask_resized = cv2.resize(mask_img, (video_tensor.shape[3], video_tensor.shape[2]))
                    mask_binary = mask_resized.sum(axis=-1) > 0
                    mask_list.append(mask_binary)
                else:
                    # 如果无法加载，使用全1mask
                    mask_list.append(np.ones((video_tensor.shape[2], video_tensor.shape[3]), dtype=bool))
            
            # 确保mask数量与帧数匹配
            num_frames = len(video_tensor)
            if len(mask_list) != num_frames:
                if len(mask_list) < num_frames:
                    # 重复最后一个mask
                    last_mask = mask_list[-1] if mask_list else np.ones((video_tensor.shape[2], video_tensor.shape[3]), dtype=bool)
                    mask_list.extend([last_mask] * (num_frames - len(mask_list)))
                else:
                    # 截取前面的部分
                    mask_list = mask_list[:num_frames]
            
            mask = mask_list  # 现在mask是一个列表
            print(f"使用每帧对应的mask，共 {len(mask_list)} 个mask")
        else:
            print("masks文件夹存在但没有找到mask文件，使用全1mask")
            mask = np.ones_like(video_tensor[0,0].numpy()) > 0
    
    else:
        # 没有找到mask，使用全1mask
        mask = np.ones_like(video_tensor[0,0].numpy()) > 0
        print("未找到mask文件，使用全1mask")


        
    # get all data pieces
    viz = True
    os.makedirs(out_dir, exist_ok=True)
        
    # with open(cfg_dir, "r") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = easydict.EasyDict(cfg)
    # cfg.out_dir = out_dir
    # cfg.model.track_num = args.vo_points
    # print(f"Downloading model from HuggingFace: {cfg.ckpts}")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = args.vo_points
    
    model.eval()
    model.to("cuda")
    viser = Visualizer(save_dir=out_dir, grayscale=True, 
                     fps=10, pad_value=0, tracks_leave_trace=5)
    
    grid_size = args.grid_size

    # get frame H W
    if video_tensor is  None:
        cap = cv2.VideoCapture(video_path)
        frame_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    # Sample mask values at grid points and filter out points where mask=0
    if isinstance(mask, list):
        # 如果是每帧对应的mask列表，使用第一帧的mask进行过滤
        first_mask = mask[0]
        if isinstance(first_mask, np.ndarray) and first_mask.size > 0:
            grid_pts_int = grid_pts[0].long()
            mask_values = first_mask[grid_pts_int[...,1], grid_pts_int[...,0]]
            grid_pts = grid_pts[:, mask_values]
    elif isinstance(mask, np.ndarray) and mask.size > 0:
        # 单个mask的情况
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
  

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
     
        
        # resize the results to avoid too large I/O Burden
        # depth and image, the maximum side is 336
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1 and not args.no_resize:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))

        

        if viz:
            viser.visualize(video=video[None],
                                tracks=track2d_pred[None][...,:2],
                                visibility=vis_pred[None],filename="test")
            
        if args.camera_coords:
            # 输出相机坐标系下的3D轨迹
            data_npz_load["coords"] = track3d_pred[:,:,:3].cpu().numpy()
        else:
            # 输出世界坐标系下的3D轨迹（原来的方式）
            data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        
        # 如果请求输出coords_4d格式
        if args.output_coords_4d:
            coords_4d = generate_coords_4d(
                track2d_pred.cpu().numpy() if isinstance(track2d_pred, torch.Tensor) else track2d_pred,
                depth_tensor,
                vis_pred.cpu().numpy() if isinstance(vis_pred, torch.Tensor) else vis_pred,
                intrs.cpu().numpy() if isinstance(intrs, torch.Tensor) else intrs,
                mask
            )
            import ipdb
            ipdb.set_trace()
            data_npz_load["coords_4d"] = coords_4d
            print(f"Generated coords_4d with shape: {coords_4d.shape}")

        # save as the tapip3d format   
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        # 保存 point_indices
        point_indices = grid_pts[0].long().cpu().numpy()  # shape (N, 2)
        x_indices = point_indices[:, 0]
        y_indices = point_indices[:, 1]
        data_npz_load["point_indices"] = np.stack([y_indices, x_indices], axis=0)  # shape (2, N)
        np.savez(os.path.join(out_dir, f'result.npz'), **data_npz_load)

        print(f"Results saved to {out_dir}.\nTo visualize them with tapip3d, run: [bold yellow]python tapip3d_viz.py {out_dir}/result.npz[/bold yellow]")
