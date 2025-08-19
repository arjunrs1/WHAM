import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

from pathlib import Path
import pandas as pd

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False,
        detector=None,
        extractor=None):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    frame_skip = 5
    # Preprocess
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):
            
            if detector is None or extractor is None:
                detector = DetectionModel(cfg.DEVICE.lower())
                extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            
            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            print(f"###########Processing video {video}###########")
            frame_idx = 0
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                if frame_idx % frame_skip == 0:
                    # 2D detection and tracking
                    detector.track(img, fps/frame_skip, length//frame_skip)
                    
                    # SLAM
                    if slam is not None: 
                        slam.track()
                    
                    bar.next()
                frame_idx += 1
            tracking_results = detector.process(fps/frame_skip)
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion

            for track in tracking_results.keys():
                tracking_results[track]['frame_id'] = tracking_results[track]['frame_id'] * frame_skip
            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data (commented out to save space)
            # joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            # joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
            # logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        # if False:
        if args.run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        
        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id
    
    # Handle tracklet continuity and select main actor
    logger.info(f"Processing {len(results)} tracklets for main actor identification")
    if len(results) > 1:
        # First, analyze all tracklets to understand their temporal distribution
        track_info = {}
        for track_id in results.keys():
            # Skip tracks without necessary data
            if 'frame_ids' not in results[track_id] or len(results[track_id]['frame_ids']) == 0:
                continue
                
            frame_ids = results[track_id]['frame_ids']
            start_frame = frame_ids.min() if isinstance(frame_ids, np.ndarray) else min(frame_ids)
            end_frame = frame_ids.max() if isinstance(frame_ids, np.ndarray) else max(frame_ids)
            
            # Get bounding box information
            if track_id in tracking_results and 'bbox' in tracking_results[track_id]:
                bbox_data = tracking_results[track_id]['bbox']
                if len(bbox_data) > 0:
                    # Calculate average size
                    if bbox_data.shape[1] == 3:  # [cx, cy, scale]
                        avg_size = np.mean(bbox_data[:, 2])
                    elif bbox_data.shape[1] == 4:  # [x1, y1, x2, y2]
                        areas = (bbox_data[:, 2] - bbox_data[:, 0]) * (bbox_data[:, 3] - bbox_data[:, 1])
                        avg_size = np.mean(areas)
                    
                    # Calculate motion
                    total_motion = 0
                    if bbox_data.shape[0] > 1:
                        # Calculate center points
                        if bbox_data.shape[1] == 3:  # [cx, cy, scale]
                            centers = bbox_data[:, :2]
                        else:  # [x1, y1, x2, y2]
                            centers = np.column_stack([
                                (bbox_data[:, 0] + bbox_data[:, 2]) / 2,
                                (bbox_data[:, 1] + bbox_data[:, 3]) / 2
                            ])
                        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
                        total_motion = np.sum(displacements)
                    
                    # Additional motion from translation data if available
                    if 'trans' in results[track_id] and results[track_id]['trans'].shape[0] > 1:
                        trans_motion = np.linalg.norm(np.diff(results[track_id]['trans'], axis=0), axis=1)
                        trans_motion_total = np.sum(trans_motion)
                        total_motion = max(total_motion, trans_motion_total)
                    
                    # Calculate motion density (motion per frame)
                    num_frames = end_frame - start_frame + 1
                    motion_density = total_motion / num_frames if num_frames > 0 else 0
                    
                    # Calculate starting position (center of first bbox)
                    if bbox_data.shape[1] == 3:  # [cx, cy, scale]
                        start_pos = bbox_data[0, :2]
                    else:  # [x1, y1, x2, y2]
                        start_pos = [(bbox_data[0, 0] + bbox_data[0, 2]) / 2, 
                                    (bbox_data[0, 1] + bbox_data[0, 3]) / 2]
                    
                    # Calculate ending position (center of last bbox)
                    if bbox_data.shape[1] == 3:  # [cx, cy, scale]
                        end_pos = bbox_data[-1, :2]
                    else:  # [x1, y1, x2, y2]
                        end_pos = [(bbox_data[-1, 0] + bbox_data[-1, 2]) / 2, 
                                  (bbox_data[-1, 1] + bbox_data[-1, 3]) / 2]
                    
                    track_info[track_id] = {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'avg_size': avg_size,
                        'total_motion': total_motion,
                        'motion_density': motion_density,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'num_frames': len(frame_ids)
                    }
                    
                    logger.info(f"Track {track_id}: Frames={start_frame}-{end_frame} ({len(frame_ids)}), "
                               f"Size={avg_size:.4f}, Motion={total_motion:.4f}, Density={motion_density:.4f}")
        
        # Step 1: Find potential main actor tracklets using a combination of size and motion metrics
        main_tracklets = []
        size_threshold = 0.8  # Consider tracklets with size at least 80% of the maximum
        max_size = max([info['avg_size'] for info in track_info.values()]) if track_info else 0
        size_min = max_size * size_threshold
        
        # If we have multiple large tracklets of similar size, consider motion as well
        similar_size_tracklets = []
        for track_id, info in track_info.items():
            # Check if tracklet is large enough to be considered
            if info['avg_size'] >= size_min:
                similar_size_tracklets.append((track_id, info))
        
        # If we have multiple similar-sized tracklets, use motion to select
        if len(similar_size_tracklets) > 1:
            logger.info("Multiple similar-sized tracklets found, using motion heuristic for selection")
            
            # Find the tracklet with highest motion that's at least 70% of max size
            high_motion_threshold = 0.7  # Consider tracklets with at least 70% of max motion
            motion_scores = [(tid, info['total_motion']) for tid, info in similar_size_tracklets]
            max_motion = max([score for _, score in motion_scores])
            motion_min = max_motion * high_motion_threshold
            
            # Compute a combined score of size and motion
            # This formula gives higher weight to motion when tracklets are similarly sized
            combined_scores = []
            for track_id, info in similar_size_tracklets:
                # Normalize size and motion to 0-1 scale
                norm_size = info['avg_size'] / max_size
                norm_motion = info['total_motion'] / max_motion if max_motion > 0 else 0
                
                # When tracklets are similar in size (all above size_threshold),
                # give more weight to motion
                combined_score = norm_size * 0.4 + norm_motion * 0.6
                combined_scores.append((track_id, combined_score, norm_size, norm_motion))
                
                logger.info(f"Track {track_id}: Size Score={norm_size:.4f}, Motion Score={norm_motion:.4f}, "
                           f"Combined Score={combined_score:.4f}")
            
            # Choose tracklets with the highest combined scores
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Use the top tracklet(s)
            top_score = combined_scores[0][1]
            score_threshold = top_score * 0.9  # Consider tracklets with at least 90% of top score
            main_tracklets = [tid for tid, score, _, _ in combined_scores if score >= score_threshold]
            
            logger.info(f"Selected top tracklet(s) by combined size+motion score: {main_tracklets}")
        else:
            # If there's only one large tracklet, or no similar-sized tracklets,
            # just use the size heuristic
            for track_id, info in track_info.items():
                if info['avg_size'] >= size_min:
                    main_tracklets.append(track_id)
        
        if not main_tracklets:
            # If no tracklets meet the criteria, just take the largest one
            if track_info:
                largest_id = max(track_info.items(), key=lambda x: x[1]['avg_size'])[0]
                main_tracklets = [largest_id]
                logger.info(f"No tracklets met criteria, defaulting to largest: {largest_id}")
        
        logger.info(f"Identified {len(main_tracklets)} potential main actor tracklets: {main_tracklets}")
        
        # Step 2: Check if these tracklets could be the same actor reappearing
        if len(main_tracklets) > 1:
            # Sort by start frame
            main_tracklets.sort(key=lambda x: track_info[x]['start_frame'])
            
            # Check for temporal continuity and spatial proximity
            continuous_groups = []
            current_group = [main_tracklets[0]]
            
            for i in range(1, len(main_tracklets)):
                prev_id = current_group[-1]
                curr_id = main_tracklets[i]
                prev_info = track_info[prev_id]
                curr_info = track_info[curr_id]
                
                # Check temporal gap
                frame_gap = curr_info['start_frame'] - prev_info['end_frame']
                
                # Check spatial proximity between end of previous and start of current
                distance = np.linalg.norm(np.array(prev_info['end_pos']) - np.array(curr_info['start_pos']))
                
                # Time threshold: allow gaps of up to 2 seconds (assuming 30fps)
                time_threshold = 120  # frames
                # Distance threshold: 30% of image width/height
                dist_threshold = max(width, height) * 0.3
                
                # If tracks are close in time and space, they might be the same person
                if frame_gap < time_threshold and distance < dist_threshold:
                    current_group.append(curr_id)
                else:
                    continuous_groups.append(current_group)
                    current_group = [curr_id]
            
            # Don't forget the last group
            if current_group:
                continuous_groups.append(current_group)
            
            logger.info(f"Found {len(continuous_groups)} continuous actor groups")
            
            # Step 3: Select the best group based on duration, size and motion
            if continuous_groups:
                # Calculate scores for each group incorporating both duration and motion
                group_scores = []
                for group in continuous_groups:
                    total_frames = sum(track_info[tid]['num_frames'] for tid in group)
                    total_motion = sum(track_info[tid]['total_motion'] for tid in group)
                    avg_size = np.mean([track_info[tid]['avg_size'] for tid in group])
                    
                    # Combined score prioritizes both duration and motion
                    # Scale factors ensure neither metric dominates unfairly
                    combined_score = total_frames * avg_size * (1 + 0.5 * total_motion)
                    group_scores.append((group, combined_score, total_frames, total_motion))
                    
                    logger.info(f"Group {group}: Frames={total_frames}, Motion={total_motion:.4f}, "
                               f"Avg Size={avg_size:.4f}, Score={combined_score:.4f}")
                
                # Select group with highest score
                best_group, best_score, frames, motion = max(group_scores, key=lambda x: x[1])
                logger.info(f"Selected best group with score {best_score:.4f}, {frames} frames, "
                           f"motion {motion:.4f}")
                
                # Merge all tracklets in the best group
                merged_results = defaultdict(dict)
                for track_id in best_group:
                    for key, value in results[track_id].items():
                        if key not in merged_results:
                            merged_results[key] = value
                        else:
                            if isinstance(value, np.ndarray) and isinstance(merged_results[key], np.ndarray):
                                merged_results[key] = np.concatenate([merged_results[key], value], axis=0)
                
                # Replace results with the merged tracklet
                merged_id = min(best_group)  # Use the first track ID as the merged ID
                results = {merged_id: merged_results}
                logger.info(f"Merged {len(best_group)} tracklets into a single continuous track with ID {merged_id}")
            else:
                # If there are no good groups, just take the largest tracklet
                main_actor_id = max(track_info.items(), key=lambda x: x[1]['avg_size'] * x[1]['num_frames'])[0]
                results = {main_actor_id: results[main_actor_id]}
                logger.info(f"Selected track {main_actor_id} as main actor (no continuous groups found)")
        else:
            # Only one main tracklet found
            if main_tracklets:
                main_actor_id = main_tracklets[0]
                results = {main_actor_id: results[main_actor_id]}
                logger.info(f"Selected track {main_actor_id} as the only main actor candidate")
            else:
                logger.warning("Could not determine main actor, keeping all tracklets")
    
    # Concatenate all 'pose_world' tracklets and save it
    keys_in_order = list(results.keys())
    arrays = []
    for k in keys_in_order:
        arr = results[k].get('pose_world', None)
        if arr is None:
            # no entry at all
            continue
        
        # If we have frame_ids, we should sort the pose data by frame
        if 'frame_ids' in results[k]:
            frame_ids = results[k]['frame_ids']
            # Check if we need to sort (in case of merged tracklets)
            if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) >= 0):
                logger.info(f"Tracklet {k} has non-sequential frames, sorting by frame ID")
                # Get sorting indices
                sort_idx = np.argsort(frame_ids)
                # Apply sorting to pose data
                arr = arr[sort_idx]
        
        # handle numpy arrays or torch tensors
        length = arr.shape[0] if hasattr(arr, 'shape') else (arr.size(0) if hasattr(arr, 'size') else 0)
        if length == 0:
            # empty array
            continue
        arrays.append(arr)
        logger.info(f"Adding tracklet {k} with {length} frames to output")

    if arrays:
        all_pose_world = np.concatenate(arrays, axis=0)
        logger.info(f"Total frames in final pose_world: {all_pose_world.shape[0]}")
    else:
        # fallback if *all* tracklets were empty
        all_pose_world = np.empty((0,))
        logger.warning("No valid pose data found")
    all_pose_world_tensor = torch.from_numpy(all_pose_world)
    torch.save(all_pose_world_tensor, osp.join(output_pth, "pose_world.pt"))

    # Save camera-relative pose and translation
    all_pose_cam = []
    all_trans_cam = []

    for k in keys_in_order:
        if 'pose' in results[k] and results[k]['pose'].shape[0] > 0:
            pose_data = results[k]['pose']
            # Sort by frame_ids if available and needed (for merged tracklets)
            if 'frame_ids' in results[k]:
                frame_ids = results[k]['frame_ids']
                if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) >= 0):
                    logger.info(f"Sorting pose_cam for tracklet {k} by frame ID")
                    sort_idx = np.argsort(frame_ids)
                    pose_data = pose_data[sort_idx]
            all_pose_cam.append(pose_data)
            
        if 'trans' in results[k] and results[k]['trans'].shape[0] > 0:
            trans_data = results[k]['trans']
            # Sort by frame_ids if available and needed (for merged tracklets)
            if 'frame_ids' in results[k]:
                frame_ids = results[k]['frame_ids']
                if len(frame_ids) > 1 and not np.all(np.diff(frame_ids) >= 0):
                    logger.info(f"Sorting trans_cam for tracklet {k} by frame ID")
                    sort_idx = np.argsort(frame_ids)
                    trans_data = trans_data[sort_idx]
            all_trans_cam.append(trans_data)

    if all_pose_cam:
        all_pose_cam_tensor = torch.from_numpy(np.concatenate(all_pose_cam, axis=0))
        torch.save(all_pose_cam_tensor, osp.join(output_pth, "pose_cam.pt"))
        logger.info(f"Saved pose_cam with {all_pose_cam_tensor.shape[0]} frames")

    if all_trans_cam:
        all_trans_cam_tensor = torch.from_numpy(np.concatenate(all_trans_cam, axis=0))
        torch.save(all_trans_cam_tensor, osp.join(output_pth, "trans_cam.pt"))
        logger.info(f"Saved trans_cam with {all_trans_cam_tensor.shape[0]} frames")

    # Also save camera parameters if available
    if calib is not None:
        camera_data = np.load(calib, allow_pickle=True).item()
        torch.save(camera_data, osp.join(output_pth, "camera_params.pt"))
    
    if save_pkl:
        # pose_world_tensor = torch.from_numpy(pred_pose_world)
        # torch.save(pose_world_tensor, osp.join(output_pth, "pose_world.pt"))
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))
     
    # Visualize
    if visualize:
        print("Visualizing")
        print(f"Output path: {output_pth}")
        from lib.vis.run_vis import run_vis_on_demo
        with torch.no_grad():
            run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_false',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')
    parser.add_argument("--csv_path", default="/vision/asomaya1/share/ego_exo_demonstrator_proficiency/ego_exo_splits/448pFull_v2/exo1/train.csv",
                        type=Path,
                        help="CSV whose first column holds absolute/relative "
                             "video paths ending in .mp4")
    parser.add_argument("--output_root", default="egoexo_exo_out", type=Path,
                        help="Root directory where outputs will be written")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    # sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    # output_pth = osp.join(args.output_pth, sequence)
    # os.makedirs(output_pth, exist_ok=True)
    
    df            = pd.read_csv(args.csv_path, header=None, usecols=[0])
    #Take only rows where 'basketball' or 'soccer' is in the lowercase video name
    df          = df[df[0].str.lower().str.contains('unc_basketball_03-31-23_02_37')]
    print(f"[info] Found {len(df)} video(s) in {args.csv_path}")
    root_dir = "/vision/asomaya1/share/ego_exo_demonstrator_proficiency/data"
    shuffled_idx  = np.random.permutation(len(df))
    detector      = DetectionModel(cfg.DEVICE.lower())
    extractor     = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
    failed_log    = "failed.txt"

    video_bar = Bar('Processing videos', fill='#', max=len(shuffled_idx))
    for idx in shuffled_idx:
        video      = df.iloc[idx, 0]
        video_path = os.path.join(root_dir, video.split(' ')[0])
        take_name  = Path(video_path).parts[7]
        sequence   = Path(video_path).stem
        out_root   = args.output_root / take_name / sequence

        # skip already‐done
        if out_root.exists():
            video_bar.next()
            continue
        
        out_root.mkdir(parents=True, exist_ok=False)
        detector.initialize_tracking()

        try:
            run(
                cfg,
                video_path,
                str(out_root),
                network,
                args.calib,
                run_global=not args.estimate_local_only,
                save_pkl=args.save_pkl,
                visualize=args.visualize,
                detector=detector,
                extractor=extractor
            )
        except Exception as e:
            print(f"[error] Failed on {video_path}: {e}")
            entry = (video_path + "\n").encode("utf-8")
            fd = os.open(failed_log,
                         os.O_CREAT | os.O_WRONLY | os.O_APPEND,
                         mode=0o644)
            os.write(fd, entry)
            os.close(fd)
            # (optionally) clean up partial output:
            # shutil.rmtree(out_root, ignore_errors=True)
        finally:
            video_bar.next()

    video_bar.finish()
    print()
    logger.info("Done!")