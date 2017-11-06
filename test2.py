from arcsim_expert_python import *
import wrinkle2
import util
import cv2
import matplotlib.pyplot as plt

expert = arcsim_expert()
remesh_fag = False
render_image = True
MOVEMENT_THRESH = 1e-5
BETA = 0.98

expert.create_sheet(1,5,2,10,"./python_sheet",[0,0,-9.81],[0,0,0],0,0,remesh_fag)
expert.setup("./python_sheet/sheet.json")

import time
path = "./"+time.strftime("%m%d-%H%M")

x = 1.0 #width of the cloth 
y = 2.0 #length of the cloth
delta = 0.01 # controller gain, multiplier
nr_frame = 100 # number of frames each setup consists
nr_pass = 100 # number of different passes (setups)
expert_controller = expert.expert_twist #expert type expert_flat

if os.path.exists(path):
    shutil.rmtree(path)
os.mkdir(path)

# initialize the positions
handles=[[0,0,0],[x,0,0],[0,y,0],[x,y,0]]
# frame_id
frame_id=0
tt_pos = np.array([])
tt_feat = np.array([])
# add log at each run
tt_log = np.array([])
tt_score = np.array([])
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
    expert.save_frame(filename, render_image)
    
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
        pos = np.array(expert_pos).ravel()
        tt_pos = np.vstack((tt_pos,pos)) if tt_pos.size else pos
        # update the features
        tt_feat = np.vstack((tt_feat,feat)) if tt_feat.size else feat
        
        expert.advance()
        handles_prev = handles
        # apply hands
        handles = expert.apply_hand(handles, hands, delta)
        # apply the robot controllers positions
        
        use_expert = 0
        if p>0 and random.random()>BETA_p:
            pred_pos = model.predict(feat.reshape(1,-1))
            pred_pos = pred_pos.reshape((2,3))
            handles = expert.apply_expert(handles, pred_pos, delta)
            use_expert = 0
        else:
            handles = expert.apply_expert(handles, expert_pos, delta)
            use_expert = 1
      
        expert.set_handle(handles)
        # save images
        filename = path+"/%04i"%frame_id;
        expert.save_frame(filename, render_image)
        frame_id=frame_id+1
        
        # determine whether to terminate the run
        d = np.array(handles_prev).ravel() - np.array(handles).ravel()
        d = sum(d*d)
        print("run:",p,"frame:",i,"movement delta:",d)
        
        
        log = np.array([frame_id,p,i,d,use_expert])
        tt_log = np.vstack((tt_log,log)) if tt_log.size else log
        np.savez(path+'/data',tt_pos = tt_pos, tt_feat = tt_feat, tt_score=tt_score)    
        if d<MOVEMENT_THRESH:
            break;
        
    # begin training
    from sklearn import linear_model
    # begin training
    model = linear_model.Lasso(1e-6)
    model.fit(tt_feat, tt_pos[:,-6:])
    score = model.score(tt_feat, tt_pos[:,-6:])
    #tt_score = tt_score.vstack((tt_score,score)) if tt_score.size else score
    
    
