import os
import time
import pybullet as p
import pybullet_data
from urdf_models import models_data
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycococreatortools import pycococreatortools
from config import CATEGORIES
# initialize the GUI and others


class ObjectEnv:
    def __init__(self):
        # load urdf data
        self.clientID = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.camera_parameters = {
            'width': 640,
            'height': 640,
            'fov': 60,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector': [1, 0, 0],
            'light_direction': [0.5, 0, 1],  # the direction is from the light source position to the origin of the world frame.
        }
        # load model list
        self.models = models_data.model_lib()
        self.namelist = self.models.model_name_list
        print("Look at what we have {}".format(self.namelist))
        self.objects = []
        self.model_configs = \
            {"cup": [
                {"class": "white_cup", "color": np.array([0.9, 0.9, 0.9, 1.0])},
                {"class": "cyan_cup", "color": np.array([187, 255, 255, 255]) / 255},                      
                {"class": "green_cup", "color": np.array([127, 255, 0, 255]) / 255},
                ],
            #  "dish": [
            #     {"class": "red_cup", "color": np.array([0.8, 0.8, 0.8, 1.0])},
            #     {"class": "cyan_cup", "color": np.array([187, 255, 255, 255]) / 255},                      
            #     {"class": "green_cup", "color": np.array([127, 255, 0, 255]) / 255},
            #  ],
             "apple": [
                 {"class": "apple", "color": None}
             ],
             "banana": [
                 {"class": "banana", "color": None}
             ]
            }
        self._init_env()

    def _init_env(self):
        p.resetSimulation()
        p.resetDebugVisualizerCamera(3, 90, -30, [0.0, -0.0, -0.0])
        p.setTimeStep(1. / 240.)
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

    def _get_random_object(self, pos):
        # load the randomly picked model
        flags = p.URDF_USE_INERTIA_FROM_FILE
        # randomly get a model
        # for i in range(4):
        #     xp = float(random.randint(-15, 15)) / 100
        #     yp = float(random.randint(-15, 15)) / 100
        #     random_model = namelist[random.randint(0, len(namelist))]
        #     p.loadURDF(models[random_model], [xp, yp, 0.7 + 0.15*i], flags=flags)
        model_superclass = random.choice(list(self.model_configs.keys()))
        model_subclass = random.choice(self.model_configs[model_superclass])
        rotation = random.randint(-10, 10) / 10
        uid = p.loadURDF(self.models[model_superclass], pos, [0,0, rotation,1], flags=flags)
        if model_subclass["color"] is not None:
            p.changeVisualShape(uid, -1, rgbaColor=model_subclass["color"])
        return uid

    def getImage(self):
        cam_position = np.array([ 1, 0,  1]) * 0.3 
        # cam_posture = np.array([1.00000000e+00,  6.31104058e-09, -1.88537852e-09, -2.30836413e-05])
        quad = p.getQuaternionFromEuler([0, -3*np.pi/4, 0])
        cam_posture = np.array(quad)
        # position, orientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=server_id)
        r_mat = p.getMatrixFromQuaternion(cam_posture)
        tx_vec = np.array([r_mat[0], r_mat[3], r_mat[6]])
        ty_vec = np.array([r_mat[1], r_mat[4], r_mat[7]])
        tz_vec = np.array([r_mat[2], r_mat[5], r_mat[8]])
        camera_position = np.array(cam_position)
        target_position = camera_position + 1 * tz_vec ## -z轴向下

        view_mat = p.computeViewMatrix(cameraEyePosition=camera_position,
                                        cameraTargetPosition=target_position,
                                        cameraUpVector= tx_vec) #-y轴向上

        proj_mat = p.computeProjectionMatrixFOV(fov=self.camera_parameters["fov"],  # 摄像头的视线夹角
                                                aspect=1.0,
                                                nearVal=0.01,  # 摄像头视距min
                                                farVal=10  # 摄像头视距max
                                                )

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=self.camera_parameters["width"],
                                                        height=self.camera_parameters["height"],
                                                        viewMatrix=view_mat,
                                                        projectionMatrix=proj_mat,
                                                        physicsClientId=self.clientID,
                                                        shadow=True,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # cv2.imwrite("./end.png", cv2.cvtColor(rgbImg[:, :, :3], cv2.COLOR_BGR2RGB))
        # print("image_update")
        # print(np.max(segImg))
        # segImg2 = segImg.astype(np.uint8)
        # segImg2[segImg2 != 0] = 255
        # cv2.imshow("111", segImg2)
        return width, height, rgbImg[:, :, :3], depthImg, segImg
    
    def generate(self, n):
        pos = []
        for i in range(n):
            while True:
                x = random.randint(-10, 7)
                y = random.randint(-6, 6)
                if (x, y) not in pos:
                    pos.append((x,y))
                    break
            xp = float(x) / 100 * 2
            yp = float(y) / 100 * 2
            zp = 0.05 * (i+1)
            self._get_random_object([xp, yp, zp])
            p.stepSimulation()
    
    def test(self, sample_num=10):
        self._init_env()
        object_count = np.random.randint(2, 5)
        self.generate(object_count)
        for i in range(100):
            print(i)
            p.stepSimulation()
            _,_,img,_,seg = self.getImage()
            time.sleep(1./240.)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("t.png", img)
        plt.imshow(seg)
        plt.show()
    
    def mask2coco(self):
        return
    
    def createAnno(self, ):
        return


if __name__ == "__main__":
    env = ObjectEnv()
    env.test(2)