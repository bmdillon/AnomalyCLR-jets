import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def translate_jets( batch, width=1.0 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi translated jets, same shape as input
    '''
    mask = (batch[:,0] > 0) # 1 for constituents with non-zero pT, 0 otherwise
    ptp_eta  = np.ptp(batch[:,1,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    ptp_phi  = np.ptp(batch[:,2,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    low_eta  = -width*ptp_eta
    high_eta = +width*ptp_eta
    low_phi  = np.maximum(-width*ptp_phi, -np.pi-np.amin(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    high_phi = np.minimum(+width*ptp_phi, +np.pi-np.amax(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    shift_eta = mask*np.random.uniform(low=low_eta, high=high_eta, size=(batch.shape[0], 1))
    shift_phi = mask*np.random.uniform(low=low_phi, high=high_phi, size=(batch.shape[0], 1))
    shift = np.stack([np.zeros((batch.shape[0], batch.shape[2])), shift_eta, shift_phi], 1)
    shifted_batch = batch+shift
    return shifted_batch


def rotate_jets( batch ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets rotated independently in eta-phi, same shape as input
    '''
    rot_angle = np.random.rand(batch.shape[0])*2*np.pi
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)
    o = np.ones_like(rot_angle)
    z = np.zeros_like(rot_angle)
    rot_matrix = np.array([[o, z, z], [z, c, -s], [z, s, c]]) # (3, 3, batchsize)
    return np.einsum('ijk,lji->ilk', batch, rot_matrix)

def normalise_pts( batch ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    '''
    batch_norm = batch.copy()
    batch_norm[:,0,:] = np.nan_to_num(batch_norm[:,0,:]/np.sum(batch_norm[:,0,:], axis=1)[:, np.newaxis], posinf = 0.0, neginf = 0.0 )
    return batch_norm

def rescale_pts( batch ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by 600, same shape as input
    '''
    batch_rscl = batch.copy()
    batch_rscl[:,0,:] = np.nan_to_num(batch_rscl[:,0,:]/600, posinf = 0.0, neginf = 0.0 )
    return batch_rscl

def crop_jets( batch, nc ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, 3, nc)
    '''
    batch_crop = batch.copy()
    return batch_crop[:,:,0:nc]

def distort_jets( batch, strength=0.1, pT_clip_min=0.1 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with each constituents position shifted independently, shifts drawn from normal with mean 0, std strength/pT, same shape as input
    '''
    pT = batch[:,0]   # (batchsize, n_constit)
    shift_eta = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / pT.clip(min=pT_clip_min), posinf = 0.0, neginf = 0.0 )# * mask
    shift_phi = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / pT.clip(min=pT_clip_min), posinf = 0.0, neginf = 0.0 )# * mask
    shift = np.stack( [ np.zeros( (batch.shape[0], batch.shape[2]) ), shift_eta, shift_phi ], 1)
    return batch + shift

def rescale_pt(dataset):
    for i in range(0, dataset.shape[0]):
        dataset[i,0,:] = dataset[i,0,:]/600
    return dataset

def recentre_jet(batch):
    batchc = batch.copy()
    nj = batch.shape[0]
    for i in range( nj ):
        pts = batch[i,0,:]
        etas = batch[i,1,:]
        phis = batch[i,2,:]
        nc = len( pts )
        eta_shift = np.sum( [ pts[j]*etas[j] for j in range( nc ) ] ) / np.sum( pts )
        phi_shift = np.sum( [ pts[j]*phis[j] for j in range( nc ) ] ) / np.sum( pts )
        batchc[i,1,:] = batch[i,1,:] - eta_shift
        batchc[i,2,:] = batch[i,2,:] - phi_shift
    return batchc

def drop_constits_jet( batch, prob=0.3 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where each jet has some fraction of missing constituents
    Note: rescale pts so that the augmented jet pt matches the original
    '''
    batchc = batch.copy()
    nj = batchc.shape[0]
    nc = batchc.shape[2]
    nzs = np.array( [ np.where( batchc[:,0,:]>0.0 )[0].shape[0] for i in range(len(batch)) ] )
    mask = np.array( np.random.rand( nj, nc ) > prob, dtype='int' )
    for i in range( nj ):
        for j in range( nc ):
            if mask[i][j]==0:
                batchc[i,:,j] = np.array([0.0,0.0,0.0])
    pts = np.sum( batch[:,0,:], axis=1 )
    pts_aug = np.sum( batchc[:,0,:], axis=1 )
    pt_rescale = [ pts[i]/pts_aug[i] for i in range(nj) ]
    for i in range(nj):
        batchc[i,0,:] = batchc[i,0,:]*pt_rescale[i]
    return recentre_jet( batchc )

def subjet_shift_jet( batch, nsubs=1, probs=[0.5], R=0.8 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where a subset of constituents have been shifted
    Note: probs should be a vector of length nsubs, drop anything that falls outside the jet radius 
        and re-scale pts so that augmented jets have the same pTs as the originals,
        so choose sensible parameters in the function.
    '''
    batchc = batch.copy()
    nj = batchc.shape[0]
    nc = batchc.shape[2]
    eta_shifts = np.array( [ ( np.random.rand( nj ) - 0.5 )/(1*R) for i in range(nsubs) ] )
    phi_shifts = np.array( [ ( np.random.rand( nj ) - 0.5 )/(1*R) for i in range(nsubs) ] )
    masks = np.array( [ np.array( np.random.rand( nj, nc ) > probs[i], dtype='int' ) for i in range(nsubs) ] )
    for i in range( nj ):
        for j in range( nc ):
            for k in range( nsubs ):
                if np.all( np.array( [ masks[l][i][j] for l in range(k+1) ] ) == 0 ):
                    batchc[i,1,j] = batchc[i,1,j] + eta_shifts[k][i]
                    batchc[i,2,j] = batchc[i,2,j] + phi_shifts[k][i]
    batchc = recentre_jet( batchc )
    for i in range( nj ):
        for j in range( nc ):
            if np.sqrt( batchc[i,1,j]**2 + batchc[i,2,j]**2 ) > R:
                batchc[i,:,j] = np.array( [0.0,0.0,0.0] )
    pts = np.sum( batch[:,0,:], axis=1 )
    pts_aug = np.sum( batchc[:,0,:], axis=1 )
    pt_rescale = [ pts[i]/pts_aug[i] for i in range(nj) ]
    for i in range(nj):
        batchc[i,0,:] = batchc[i,0,:]*pt_rescale[i]
    batchc = recentre_jet( batchc )
    return batchc

def pt_reweight_jet( batch, beta=1.5 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    Dim 1 ordering: (pT, eta, phi)
    Output: batch of jets where the pt of the constituents in each jet has has been re-weighted by some power
    Note: rescale pts so that the augmented jet pt matches the original
    '''
    batchc = batch.copy()
    nj = batchc.shape[0]
    nc = batchc.shape[2]
    for i in range( nj ):
        for j in range( nc ):
            batchc[i,0,j] = batch[i,0,j]**beta
    pts = np.sum( batch[:,0,:], axis=1 )
    pts_aug = np.sum( batchc[:,0,:], axis=1 )
    pt_rescale = [ pts[i]/pts_aug[i] for i in range(nj) ]
    for i in range(nj):
        batchc[i,0,:] = batchc[i,0,:]*pt_rescale[i]
    return recentre_jet( batchc )


