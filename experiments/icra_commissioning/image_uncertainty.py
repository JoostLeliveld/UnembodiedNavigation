#!/usr/bin/env python3
"""Small current-image uncertainty ablation; frozen NN mean and existing split.

Five inexpensive crop statistics are a probe, not a new perception architecture.
No reference-point label is clipped. Only the pixel extraction rectangle is bounded
by the available image. All geometry and bbox features remain stored in source rows.
"""
import json
import cv2
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from study import REPO,OUT,CAPTURE,load,readcsv,score,writejson
from model import covariance


def main():
    data,_=load(OUT);data=[r for r in data if r['role']!='mean_train']
    sources={(r['camera_id'],f"{r['pose_id']}:{r['repetition_id']}"):r for r in readcsv(REPO/CAPTURE/'bias_update_interpretations.csv')}
    cache=OUT/'crop_statistics.npz'
    if cache.exists():X=np.load(cache)['features']
    else:
        X=[]
        for r in data:
            s=sources[r['camera'],r['frame']];image=cv2.imread(str(REPO/CAPTURE/s['image']))
            if image is None:raise RuntimeError(s['image'])
            x0,y0,x1,y1=[float(s[k]) for k in ['x0','y0','x1','y1']]
            crop=image[max(0,int(np.floor(y0))):min(image.shape[0],int(np.ceil(y1))),
                       max(0,int(np.floor(x0))):min(image.shape[1],int(np.ceil(x1)))]
            if crop.size==0:raise ValueError('empty detected crop')
            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(float)/255
            # Color threshold is an image feature, not a semantic visibility annotation.
            B,G,R=cv2.split(crop.astype(float)/255)
            X.append([gray.mean(),gray.std(),np.log1p(cv2.Laplacian(gray,cv2.CV_64F).var()*1000),
                      np.mean((R>G*1.3)&(R>B*1.3)),np.mean(gray<.15)])
        X=np.array(X);np.savez_compressed(cache,features=X)
    assert len(X)==len(data)
    base=joblib.load(OUT/'models.joblib');models={}; predictions={}
    for c in sorted({r['camera'] for r in data}):
        idx=np.array([r['camera']==c and r['role']=='covariance_fit' for r in data])
        scaler=StandardScaler().fit(X[idx]);km=KMeans(n_clusters=3,n_init=10,random_state=509).fit(scaler.transform(X[idx]))
        labels=km.predict(scaler.transform(X));bias=base[c,'constant'].bias
        e=np.array([r['z']-r['truth']-bias for r in data]);overall=covariance(e[idx])
        covs={};means={};counts={}
        for k in range(3):
            residual=e[idx&(labels==k)];n=len(residual)
            covs[k]=(n*covariance(residual)+20*overall)/(n+20);means[k]=residual.mean(axis=0);counts[k]=n
        models[c]=dict(scaler=scaler,cluster=km,covs=covs,bias=bias,cell_means=means,counts=counts)
        for j,r in enumerate(data):
            if r['camera']==c:predictions[j]=(e[j],covs[labels[j]])
    idx=[j for j,r in enumerate(data) if r['role']=='selection'];groups=np.array([data[j]['group'] for j in idx])
    e=np.array([predictions[j][0] for j in idx]);R=np.array([predictions[j][1] for j in idx]);candidates=[]
    for tau in [0.,.25,.5,.75,1.]:
        C=(1-tau)*R+tau*np.trace(R,axis1=1,axis2=2)[:,None,None]*np.eye(2)/2
        nees=np.einsum('ni,ni->n',e,np.linalg.solve(C,e[...,None])[...,0])
        scale=max(.05,float(np.mean([nees[groups==g].mean()/2 for g in set(groups)])))
        candidates.append((score(e,C*scale,groups)['group_mean_nll'],tau,scale))
    loss,tau,scale=min(candidates)
    idx=[j for j,r in enumerate(data) if r['role']=='evaluation'];e=np.array([predictions[j][0] for j in idx]);R=np.array([predictions[j][1] for j in idx])
    R=scale*((1-tau)*R+tau*np.trace(R,axis1=1,axis2=2)[:,None,None]*np.eye(2)/2)
    result=score(e,R,[data[j]['group'] for j in idx])
    writejson(OUT/'image_results.json',dict(status='development_current_image_probe',score=result,
        selection_nll=loss,scale=scale,isotropic_shrink=tau,
        features=['gray_mean','gray_std','log_laplacian_variance','red_fraction','dark_fraction'],
        limitations=['Handcrafted crop statistics; no claim about a trained image encoder.',
        'Driving RGB crops were not saved in selected legacy logs; no fusion result for this branch.',
        'Same 1172 evaluation observations, shared frozen NN and per-camera bias.']))
    joblib.dump(dict(models=models,scale=scale,isotropic_shrink=tau),OUT/'image_models.joblib')
    print(result)

if __name__=='__main__':main()
