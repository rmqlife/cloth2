from expert2 import *
import wrinkle2
import util
import cv2
import matplotlib.pyplot as plt
from dagger import *

remesh_fag = False
render_image = True
render_depth = False
MOVEMENT_THRESH = 1e-5
BETA = 0.95
cloth_x = 1.0 #width of the cloth 
cloth_y = 2.0 #length of the cloth
delta = 0.01 # controller gain, multiplier
nr_frame = 100 # number of frames each setup consists
nr_pass = 30 # number of different passes (setups)

expert = arcsim_expert()
expert.create_sheet(1,5,2,10,"./python_sheet",[0,0,-9.81],[0,0,0],0,0,remesh_fag)
expert.setup("./python_sheet/sheet.json")
expert_controller = expert.expert_flat #expert type

# initialize the positions
handles=[[0,0,0],[cloth_x,0,0],[0,cloth_y,0],[cloth_x,cloth_y,0]]
# frame_id
dagger = Dagger()

for p in range(nr_pass):
    hands = random_hands(x=cloth_x)
    pos = np.zeros((2,3))    
    BETA_p = BETA**p
    # start each run
    for i in range(nr_frame):
        # handles
        handles_prev = handles

        handles = expert.apply_hand(handles, hands, delta)
        handles = expert.apply_expert(handles, pos, delta)
        expert.set_handle(handles)
        expert.advance()
        # save images
        expert.save_frame(dagger.frame_path(), render_image, render_depth)
        print dagger.saved_frame_path()
        # get the visual feature
        im = cv2.imread(dagger.saved_frame_path())
        feat = wrinkle2.xhist(im)
        expert_pos = np.array(expert_controller(handles,cloth_x,cloth_y))
        if p>0:
            pred_pos = rt.feat_to_act(feat=feat.reshape(1,-1)).reshape(2,3)
        else:
            pred_pos = np.zeros(expert_pos.shape).reshape(2,3)

        
        # apply the robot controllers positions
        if random.random()>BETA_p:
            use_pred = 1
            pos = pred_pos
        else:
            use_pred = 0
            pos = expert_pos        

        this_log = np.array([p,i,use_pred]);
        dagger.update(log=this_log, handles=handles_prev, pred_pos=pred_pos, expert_pos=expert_pos, feat=feat)
        
        # determine whether to terminate the run
        d = np.array(handles_prev).ravel() - np.array(handles).ravel()
        d = sum(d*d)
        if d<MOVEMENT_THRESH:
            break;
        
    # begin training
    from rfdic import random_trees
    num_class = 10
    rt = random_trees(tt_handles=dagger.tt_handles, 
                      tt_pos=dagger.tt_expert_pos, 
                      tt_feat=dagger.tt_feat, 
                      num_class=num_class, 
                      num_trees=10)
