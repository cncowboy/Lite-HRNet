# -*- coding: utf-8 -*-
import os
import sys
import cv2
import json
import time

sys.path.append("..")
from utils.data_form import Form

from mmpose.apis.inference import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)
from mmdet.apis.inference import (inference_detector, init_detector)


class Play():
    def det__init__(self, config, checkpoint, device="cuda:0"):
        """
        Input:
            Detction -> config, checkpoint, device
        """
        det_config = config
        det_checkpoint = checkpoint
        det_device = device
        self.det_model = init_detector(det_config, det_checkpoint, device=det_device)

    def pose__init__(self, config, checkpoint, device="cuda:0"):
        """
        Input:
            Pose -> config, checkpoint, device
        """
        pose_config = config
        pose_checkpoint = checkpoint
        pose_device = device
        self.pose_model = init_pose_model(pose_config, pose_checkpoint, device=pose_device)


    def det_Pose_Video(self, user_video,  outpath="result", option=True):

        DET_CAT_ID = 1      # Category id for bounding box detection model
        RESULT_BOX = {}
        SHOW = False

        # 3. 영상 파일을 불러오기
        cap = cv2.VideoCapture(user_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(outpath, f'DancerFlow_{os.path.basename(user_video)}'), fourcc, fps, size)
        
        idx = 0
        mmdet_time = 0
        posedet_time = 0
        while cap.isOpened():
            idx += 1
            flag, img = cap.read()
            if not flag:
                break
            
            start_time = time.time()
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(self.det_model, img)
            end_time = time.time()
            mmdet_time = mmdet_time + (end_time - start_time)
            start_time = end_time

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, DET_CAT_ID)
            # test a single image, with a list of bboxes.
            start_time = time.time()
            pose_results, returned_outputs = inference_top_down_pose_model(
                self.pose_model,
                img,
                person_results,
                bbox_thr=0.5,
                format='xyxy',
                return_heatmap=False,
                outputs=None)

            end_time = time.time()
            posedet_time = posedet_time + (end_time - start_time)
            start_time = end_time


            bounding_Box = pose_results[0]["bbox"]
            result_dict={}
            for i, p_point in enumerate(pose_results[0]["keypoints"]):
                result_dict_1 = Form.data_form(dict=result_dict, i = i, keypoint = p_point)
            data_1 = Form.make_dic(idx, bounding_Box, result_dict_1)
            RESULT_BOX[idx] = (data_1)

            # show the results
            vis_img = vis_pose_result(
                self.pose_model,
                img,
                pose_results,
                kpt_score_thr=0.5,
                radius=4,               # 원 크기
                thickness=2,            # 관절 두께
                show=False)

            if SHOW == True:
                cv2.imshow('Image', vis_img)
            videoWriter.write(vis_img)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            print("finish write frame:", idx)
            
        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()
        print("person detect:", mmdet_time / idx)
        print("pose detect:", posedet_time / idx)

        ## 결과값을 json파일로 저장
        with open("./{}.json".format(os.path.basename(user_video)), "w") as outfile:
            json.dump(RESULT_BOX, outfile, indent=4)
        

if __name__ == '__main__':
    
    # Detction 설정
    DET_CONFIG_FASTER_R_CNN_R50_FPN_COCO = "./configs/detection/faster_rcnn_r50_fpn_coco.py"                                                                                                        # Detection config 파일
    DET_CHECKPOINT_FASTER_R_CNN_R50_FPN_COCO = "./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"        # Detection 훈련 모델 파일
    
    # Pose 설정
    POSE_CONFIG_HRNET_W48_COCO_256X192 = "./configs/top_down/lite_hrnet/coco/litehrnet_30_coco_384x288.py"                                                               # Pose config 파일
    POSE_CHECKPOINT_HRNET_W48_COCO_256X192 = "./checkpoints/litehrnet_30_coco_384x288.pth"                                             # Pose 훈련 모델 파일
    
    # 영상 경로
    VIDEO_1 = "./sample_data/demo.mov" # 유저 업로드 영상

    play= Play()
    play.det__init__(DET_CONFIG_FASTER_R_CNN_R50_FPN_COCO, DET_CHECKPOINT_FASTER_R_CNN_R50_FPN_COCO)
    play.pose__init__(POSE_CONFIG_HRNET_W48_COCO_256X192, POSE_CHECKPOINT_HRNET_W48_COCO_256X192)
    play.det_Pose_Video(VIDEO_1, outpath="result")
