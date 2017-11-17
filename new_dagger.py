from expert2 import *
import wrinkle2
import util
import cv2
import matplotlib.pyplot as plt

class Dagger:
    frame_id = 0
    def __init__(self):
        import time
        path = "./"+time.strftime("%m%d-%H%M")
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        self.tt_expert_pos = np.array([])
        self.tt_pred_pos = np.array([])
        self.tt_handles = np.array([])
        self.tt_feat = np.array([])
        self.tt_log = np.array([])
        self.path = path
        pass

    def update(self,handles, pred_pos, expert_pos, feat, log):
        handles = np.array(handles).ravel();
        expert_pos = np.array(expert_pos).ravel()
        pred_pos = np.array(pred_pos).ravel()
        

        self.tt_pred_pos = np.vstack((self.tt_pred_pos,pred_pos)) if self.tt_pred_pos.size else pred_pos
        self.tt_feat = np.vstack((self.tt_feat,feat)) if self.tt_feat.size else feat
        self.tt_handles = np.vstack((self.tt_handles, handles)) if self.tt_handles.size else handles
        self.tt_expert_pos = np.vstack((self.tt_expert_pos, expert_pos)) if self.tt_expert_pos.size else expert_pos
        self.tt_log = np.vstack((self.tt_log, this_log)) if self.tt_log.size else this_log
        
        np.savez(self.path+'/data',tt_expert_pos = self.tt_expert_pos,
                 tt_pred_pos = self.tt_pred_pos, tt_feat = self.tt_feat, 
                 tt_handles = self.tt_handles, tt_log = self.tt_log)
        pass
    
    def frame_path(self):
        self.frame_id = self.frame_id + 1
        return self.path+"/%04i"%(self.frame_id-1)
    
    def saved_frame_path(self):
        return self.path+"/%04i"%(self.frame_id-1)+'.png'
    
    
    
def random_hands(x):
    hands=np.zeros((2,3))
    # pick two random positions for hands, which have at most x (won't break the cloth)
    while True:
        # random from [0,0,0] [x,0,0], which is the first two handles
        hands[0] = [random.uniform(-x/4,x/4),random.uniform(-x/4,x/4),random.uniform(-x/4,x/4)]
        hands[1] = [random.uniform(-x/4,x/4)+x,random.uniform(-x/4,x/4),random.uniform(-x/4,x/4)]
        if np.linalg.norm(np.subtract(hands[0],hands[1])) < x:
            break
    
    return hands
    
    
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
