import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import pickle
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from train_pose import MLP

filename = "pickled_body_pos"

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    count = 0 # for pickle. This is the number of samples (of 18 values)
    dataset = []

    net = MLP()
    net.load_state_dict(torch.load("trained_params.pth"))
    net.eval()

    #start_time = time.time()
    #elapsed_time = 0
    # while True:
    while True:
        #elapsed_time = time.time() - start_time
        
        ret_val, image = cam.read()

        #logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        #logger.debug('image process+')
        humans = e.inference(image)
        
        '''
		My code here
		Need to make data invariant before feeding to neural net. Need to divide.
        '''
        all_x_pos = []
        all_y_pos = []
        label = 0    # 1 for active, 0 for non active
        for human in humans:
            print(len(human.body_parts))
            if len(human.body_parts) == 18:
                for i in range(len(human.body_parts)):	# this will run 18 times
                    #print(human.body_parts[i].x, human.body_parts[i].y)
                    # save_body_pos.append([count, human.body_parts[i].x, human.body_parts[i].y])
                    all_x_pos.append(human.body_parts[i].x)
                    all_y_pos.append(human.body_parts[i].y)
                    count += 1
                    # Now it has collected the x,y of all 18 body parts
                hip_avg_x = (all_x_pos[8] + all_x_pos[11]) / 2.0
                hip_avg_y = (all_y_pos[8] + all_y_pos[11]) / 2.0
                #print("hip avg x :" + str(hip_avg_x))
                
                for i in range(len(human.body_parts)):
                    all_x_pos[i] -= hip_avg_x
                    all_y_pos[i] -= hip_avg_y
                
                # Feed it to the neural network:
                
                nn_input = all_x_pos + all_y_pos
                nn_input = torch.FloatTensor(nn_input)
                nn_input = Variable(nn_input)
                #print(nn_input)
                nn_input = nn_input.view(1,36)
                #print(nn_input)

                out = net(nn_input)
                print("out = " + str(out))
                classification = out[0].float()
		print("Classification = " + str(classification.data[0]))
                if (classification.data[0]> 0.5):
                    print("Human is ready")
                else:
                    print("Human is not ready")		
                #dataset.append([all_x_pos, all_y_pos, label])


        ##################################################################################33
        

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()

    # SAVING THE BODY POSITIONS
    # open the file for writing
    fileObject = open(filename,'wb')

    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(dataset,fileObject)
    print("Pickle success!") 
    # here we close the fileObject
    fileObject.close()
