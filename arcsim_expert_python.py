import os,shutil,random
import ctypes as ct
import numpy as np
from vapory import Scene,POVRayElement,Camera,Background,   \
                   LightSource,Texture,Pigment,Finish,format_if_necessary

class ClothMesh(POVRayElement):
    def __init__(self,vss,iss,nss,*args):
        self.vss=vss
        self.iss=iss
        self.nss=nss
        self.args=list(args)
    def __str__(self):
        nrV = len(self.vss)
        nrI = len(self.iss)
        ret="mesh2{\n"
        ret=ret+"vertex_vectors{%d,\n"%nrV
        for i in xrange(nrV):
            ret=ret+("<%f,%f,%f>%s")%(self.vss[i][0],self.vss[i][1],self.vss[i][2],("," if i<nrV-1 else ""))
        ret=ret+"}\n"
        if len(self.nss) == nrV:
            ret=ret+"normal_vectors{%d,\n"%nrV
            for i in xrange(nrV):
                ret=ret+("<%f,%f,%f>%s")%(self.nss[i][0],self.nss[i][1],self.nss[i][2],("," if i<nrV-1 else ""))
            ret=ret+"}\n"
        ret=ret+"face_indices{%d,\n"%nrI
        for i in xrange(nrI):
            ret=ret+("<%d,%d,%d>%s")%(self.iss[i][0],self.iss[i][1],self.iss[i][2],("," if i<nrI-1 else ""))
        ret=ret+"}\n"
        ret_additional="".join([str(format_if_necessary(e)) for e in self.args])
        ret=ret+ret_additional
        ret=ret+"}"
        return ret

class arcsim_expert:
    def __init__(self):
        #load lib
        self.expert = ct.cdll.LoadLibrary("./libarcsim_expert.so")
        #register simulation functions
        self.expert.create_sheet_example_c.argtypes =   \
            [ct.c_double,ct.c_int,ct.c_double,ct.c_int,ct.c_char_p,
             ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),
             ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),
             ct.c_bool,ct.c_bool]
        self.expert.setup_c.argtypes = [ct.c_char_p]
        self.expert.set_handle_c.argtypes = [ct.POINTER(ct.c_double),ct.c_int]
        self.expert.save_frame_vtk_c.argtypes = [ct.c_char_p]
        self.expert.get_meshv_c.argtypes = [ct.POINTER(ct.c_int)]
        self.expert.get_meshv_c.restype = ct.POINTER(ct.c_double)
        self.expert.get_meshn_c.argtypes = [ct.POINTER(ct.c_int)]
        self.expert.get_meshn_c.restype = ct.POINTER(ct.c_double)
        self.expert.get_meshi_c.argtypes = [ct.POINTER(ct.c_int)]
        self.expert.get_meshi_c.restype = ct.POINTER(ct.c_int)
        self.expert.advance_c.argtypes = []
        self.expert.illustrate_c.argtypes = [ct.c_char_p,ct.c_int]
        #register expert functions
        self.expert.expert_flat_c.argtypes = [ct.POINTER(ct.c_double),ct.c_double,ct.c_double]
        self.expert.expert_flat_c.restype = ct.POINTER(ct.c_double)
        self.expert.expert_arc_c.argtypes = [ct.POINTER(ct.c_double),ct.c_double,ct.c_double]
        self.expert.expert_arc_c.restype = ct.POINTER(ct.c_double)
        self.expert.expert_twist_c.argtypes = [ct.POINTER(ct.c_double),ct.c_double,ct.c_double]
        self.expert.expert_twist_c.restype = ct.POINTER(ct.c_double)
        self.expert.apply_expert_c.argtypes = [ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_double]
        self.expert.apply_hand_c.argtypes = [ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.c_double]
        self.expert.free_vec_c.argtypes = [ct.POINTER(ct.c_double)]
        self.expert.free_veci_c.argtypes = [ct.POINTER(ct.c_int)]
        #image output size
        self.w=1024
        self.h=768
        #background color
        self.bk_r=1
        self.bk_g=1
        self.bk_b=1
        #cloth color
        self.cc_r=0.7
        self.cc_g=0.7
        self.cc_b=0.7
        #camera pos
        self.cx=2
        self.cy=2
        self.cz=2
        #focal point
        self.fx=0.5
        self.fy=1
        self.fz=0
        #view up
        self.ux=0
        self.uy=0
        self.uz=1
        #lights
        self.lights=[]
        self.lights=self.lights+[LightSource([0,0,2],'color',[1,1,1])]
        #material
        self.specular=0.4
    #simulation functionality
    def create_sheet(self, x, resX, y, resY,    \
                     path, g, w, wden, wdrag, remesh):
        g_c =(ct.c_double * 3)(*g)
        w_c =(ct.c_double * 3)(*w)
        wDen_c =(ct.c_double)(*[wden])
        wDrag_c =(ct.c_double)(*[wdrag])
        self.expert.create_sheet_example_c(x,resX,y,resY,path.encode(),  \
                                           g_c,w_c,wDen_c,wDrag_c,False,remesh)
    def setup(self, path):
        self.expert.setup_c(path.encode())
    def set_handle(self, handles):
        ptr = []
        for i in xrange(len(handles)):
            ptr = ptr+[handles[i][0],handles[i][1],handles[i][2]]
        ptr_c =(ct.c_double*len(ptr))(*ptr)
        self.expert.set_handle_c(ptr_c,len(handles))
    def save_frame_vtk(self, path):
        self.expert.save_frame_vtk_c(path.encode())
    def save_frame_image(self, path):
        #read
        nrv_c=(ct.c_int)()
        nrn_c=(ct.c_int)()
        nri_c=(ct.c_int)()
        vss_c = self.expert.get_meshv_c(nrv_c)
        nss_c = self.expert.get_meshn_c(nrn_c)
        iss_c = self.expert.get_meshi_c(nri_c)
        vss = [[0,0,0]]*nrv_c.value
        nss = [[0,0,0]]*nrn_c.value
        iss = [[0,0,0]]*nri_c.value
        assert nrv_c.value == nrn_c.value
        for i in xrange(nrv_c.value):
            vss[i]=[vss_c[i*3+0],vss_c[i*3+1],vss_c[i*3+2]]
            nss[i]=[nss_c[i*3+0],nss_c[i*3+1],nss_c[i*3+2]]
        for i in xrange(nri_c.value):
            iss[i]=[iss_c[i*3+0],iss_c[i*3+1],iss_c[i*3+2]]
        self.expert.free_vec_c(vss_c)
        self.expert.free_vec_c(nss_c)
        self.expert.free_veci_c(iss_c)
        #render
        tex = Texture(Pigment('color',[self.cc_r,self.cc_g,self.cc_b]),
                      Finish('specular',self.specular))
        scene = Scene(Camera('location',[self.cx,self.cy,self.cz],
                             'sky',[self.ux,self.uy,self.uz],
                             'look_at',[self.fx,self.fy,self.fz]),
                      objects=self.lights+
                      [Background("color",[self.bk_r,self.bk_g,self.bk_b]),
                       ClothMesh(vss,iss,nss,tex)])
        scene.render(path,width=self.w,height=self.h,
                     antialiasing=self.aa if hasattr(self,"aa") else 0.0)
    def save_frame(self, path, image):
        if image:
            self.save_frame_image(path+".png")
            self.save_frame_vtk(path+".vtk")
        else:
            self.save_frame_vtk(path+".vtk")
    def advance(self):
        self.expert.advance_c()
    def illustrate(self, path, nr_frame):
        self.expert.illustrate_c(path.encode(),nr_frame)
    def illustrate_python(self, path, nr_frame, image):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        for i in xrange(nr_frame):
            self.advance()
            self.save_frame(path+"/frm"+str(i),image)
    #expert functionality
    def expert_tpl(self, handles, x, y, func_ptr):
        assert len(handles) == 4
        ptr = []
        for i in xrange(len(handles)):
            ptr = ptr+[handles[i][0],handles[i][1],handles[i][2]]
        ptr_c = (ct.c_double*len(ptr))(*ptr)
        dss_c=func_ptr(ptr_c,x,y)
        dss_ret=[[dss_c[0],dss_c[1],dss_c[2]],
                 [dss_c[3],dss_c[4],dss_c[5]]]
        self.expert.free_vec_c(dss_c)
        return dss_ret
    def expert_flat(self, handles, x, y):
        return self.expert_tpl(handles,x,y,self.expert.expert_flat_c)
    def expert_arc(self, handles, x, y):
        return self.expert_tpl(handles,x,y,self.expert.expert_arc_c)
    def expert_twist(self, handles, x, y):
        return self.expert_tpl(handles,x,y,self.expert.expert_twist_c)
    def apply_expert(self, handles, robot, delta):
        assert len(handles) == 4 and len(robot) == 2
        xss = []
        for i in xrange(len(handles)):
            xss = xss+[handles[i][0],handles[i][1],handles[i][2]]
        xss_c = (ct.c_double*len(xss))(*xss)

        dss = []
        for i in xrange(len(robot)):
            dss = dss+[robot[i][0],robot[i][1],robot[i][2]]
        dss_c = (ct.c_double * len(dss))(*dss)

        self.expert.apply_expert_c(xss_c,dss_c,delta)
        return [[xss_c[0],xss_c[1],xss_c[2]],
                [xss_c[3],xss_c[4],xss_c[5]],
                [xss_c[6],xss_c[7],xss_c[8]],
                [xss_c[9],xss_c[10],xss_c[11]]]
    def apply_hand(self, handles, hand, delta):
        assert len(handles) == 4 and len(hand) == 2
        xss = []
        for i in xrange(len(handles)):
            xss = xss+[handles[i][0],handles[i][1],handles[i][2]]
        xss_c = (ct.c_double*len(xss))(*xss)
        dss0_c = (ct.c_double * 3)(*(hand[0]))
        dss1_c = (ct.c_double * 3)(*(hand[1]))

        self.expert.apply_hand_c(xss_c,dss0_c,dss1_c,delta)
        return [[xss_c[0],xss_c[1],xss_c[2]],
                [xss_c[3],xss_c[4],xss_c[5]],
                [xss_c[6],xss_c[7],xss_c[8]],
                [xss_c[9],xss_c[10],xss_c[11]]]
    def illustrate_expert_python(self, x, y, delta, \
                                 path, nr_frame, nr_pass, expert, image):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        handles=[[0,0,0],[x,0,0],[0,y,0],[x,y,0]]
        hands=[[0,0,0],[0,0,0]]
        j=0
        for p in xrange(nr_pass):
            #pick hand location
            while True:
                hands[0] = [random.uniform(-x/4,x/4),random.uniform(-x/4,x/4),random.uniform(-x/4,x/4)]
                hands[1] = [random.uniform(-x/4,x/4)+x,random.uniform(-x/4,x/4),random.uniform(-x/4,x/4)]
                if np.linalg.norm(np.subtract(hands[0],hands[1])) < x:
                    break
            #test
            for i in xrange(nr_frame):
                self.advance()
                handles=self.apply_hand(handles,hands,delta)
                handles =self.apply_expert(handles,expert(handles,x,y),delta)
                self.set_handle(handles)
                self.save_frame(path+"/frm"+str(j),image)
                j=j+1

if __name__== "__main__":
    expert = arcsim_expert()
    expert.create_sheet(1,5,2,10,"./python_sheet",[0,0,-9.81],[0,0,0],0,0,True)
    expert.setup("./python_sheet/sheet.json")
    #test simulation functionality
    #expert.illustrate_python("./python_sheet_output",100,True)
    #test expert functionality
    expert.illustrate_expert_python(1,2,0.01,"./python_sheet_expert_output",100,5,expert.expert_twist,True)
