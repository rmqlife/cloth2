from expert2 import *
import wrinkle2
import util
import cv2
import matplotlib.pyplot as plt

expert = arcsim_expert()
remesh_fag = False
render_image = True
render_depth = False
MOVEMENT_THRESH = 1e-5
BETA = 0.95

expert.create_sheet(1,5,2,10,"./python_sheet",[0,0,-9.81],[0,0,0],0,0,remesh_fag)
expert.setup("./python_sheet/sheet.json")

import time
path = "./"+time.strftime("%m%d-%H%M")

x = 1.0 #width of the cloth 
y = 2.0 #length of the cloth
delta = 0.01 # controller gain, multiplier
nr_frame = 100 # number of frames each setup consists
nr_pass = 30 # number of different passes (setups)
expert_controller = expert.expert_flat #expert type

if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

# initialize the positions
handles=[[0,0,0],[x,0,0],[0,y,0],[x,y,0]]
# frame_id
frame_id=0
tt_expert_pos = np.array([])
tt_pred_pos = np.array([])
tt_handles = np.array([])
tt_feat = np.array([])
# add at each run
tt_log = np.array([])

for p in range(nr_pass):
    print("TURN p:",p)
    hands=np.zeros((2,3))
    # pick two random positions for hands, which have at most x (won't break the cloth)
    while True:
        # random from [0,0,0] [x,0,0], which is the first two handles
        hands[0] = [random.uniform(-x/4,x/4),random.uniform(-x/4,x/4),random.uniform(-x/4,x/4)]
        hands[1] = [random.uniform(-x/4,x/4)+x,random.uniform(-x/4,x/4),random.uniform(-x/4,x/4)]
        if np.linalg.norm(np.subtract(hands[0],hands[1])) < x:
            break
    expert.advance()
    handles = expert.apply_hand(handles, hands, delta)
    expert.set_handle(handles)
    # save images
    filename = path+"/%04i"%frame_id;
    expert.save_frame(filename, render_image, render_depth)
    
    frame_id=frame_id+1
    print(filename)
    
    
    BETA_p = BETA**p
    # start each run
    for i in range(nr_frame):
        # get the visual feature
        im = cv2.imread(filename+'.png')
        feat = wrinkle2.xhist(im)
        print(handles)
        expert_pos = expert_controller(handles,x,y)
        # update the positions
        expert_pos = np.array(expert_pos).ravel()
        tt_expert_pos = np.vstack((tt_expert_pos, expert_pos)) if tt_expert_pos.size else expert_pos

        if p>0:
            pred_pos = rt.feat_to_act(feat.reshape(1,-1))
            tt_pred_pos = np.vstack((tt_pred_pos,pred_pos)) if tt_pred_pos.size else pred_pos
        
        # update the features
        tt_feat = np.vstack((tt_feat,feat)) if tt_feat.size else feat
        handles_ravel = np.array(handles).ravel();
        tt_handles = np.vstack((tt_handles,handles_ravel)) if tt_handles.size else handles_ravel
        
        expert.advance()
        handles_prev = handles
        # apply hands
        handles = expert.apply_hand(handles, hands, delta)
        # apply the robot controllers positions
                    

        if random.random()>BETA_p:
            pos = pred_pos
        else:
            pos = expert_pos
            
        pos = pos.reshape((2,3))    
        handles = expert.apply_expert(handles, pos, delta)    
        expert.set_handle(handles)
        # save images
        filename = path+"/%04i"%frame_id;
        expert.save_frame(filename, render_image, render_depth)
        frame_id=frame_id+1
        
        # determine whether to terminate the run
        d = np.array(handles_prev).ravel() - np.array(handles).ravel()
        d = sum(d*d)
        print("run:",p,"frame:",i,"movementv delta:",d)
        this_log = np.array([p,i,d]);
        tt_log = np.vstack((tt_log, this_log)) if tt_log.size else this_log
        np.savez(path+'/data',tt_expert_pos = tt_expert_pos, tt_pred_pos=tt_pred_pos, tt_feat = tt_feat, tt_handles = tt_handles, tt_log = tt_log)

        if d<MOVEMENT_THRESH:
            break;
        
    # begin training
    from rfdic import random_trees
    num_class = len(tt_handles)/30
    if num_class<2:
        num_class = 2
    rt = random_trees(tt_handles=tt_handles, tt_pos=tt_expert_pos, tt_feat=tt_feat, num_class=num_class, num_trees=10)
    
    
    
    
