import cv2
import wrinkle2
import numpy as np
import matplotlib.pyplot as plt

class random_trees():
    def __init__(self,tt_handles,tt_pos,tt_feat,num_class=100, num_trees=1):
        self.tt_handles = tt_handles
        self.tt_pos = tt_pos
        self.tt_feat = tt_feat
        # build the initial classifier
        # cluster the pos position:
        # give labels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_class, random_state=0).fit(tt_handles)
        self.tt_hidden = kmeans.labels_

        from sklearn.ensemble import RandomForestClassifier
        # using labels to construct the random trees
        self.clf = RandomForestClassifier(n_estimators=num_trees)
        # n_estimators : integer, optional (default=10) The number of trees in the forest.
        self.clf.fit(X=tt_feat, y=tt_hidden)


    

    def predict(self,feat,alpha):
        hidden = self.clf.predict(X=feat.reshape(1,-1))
        pos_sum = np.zeros(6)
        pos_num = 0
        
        inds = self.tt_hidden==hidden
        X = self.tt_feat[inds]
        y = self.tt_pos[inds]
        
        # regression
        # begin training
        from sklearn import linear_model
        model = linear_model.Lasso(alpha = alpha)
        model.fit(X, y)
        model.predict(feat)

