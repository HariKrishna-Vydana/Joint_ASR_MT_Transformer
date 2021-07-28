#! /usr/bin/python

import sys
import os
import subprocess

from os.path import join, isdir, isfile
import torch
import json

import numpy as np
from torch import autograd, nn, optim
import torch.nn.functional as F

from argparse import Namespace

model_dir = str(sys.argv[1])
SWA_random_tag = int(sys.argv[2])

est_cpts = str(sys.argv[3])
ignore_cpts = int(sys.argv[4])

if est_cpts.strip()=='None':
        est_cpts=None
else:
        est_cpts=int(est_cpts)


from Load_Trained_ASR_MT_model import Load_Transformer_ASR_MT_model
Load_Transformer_ASR_MT_model(model_dir, SWA_random_tag,est_cpts,ignore_cpts)
print("SWA averagining is done:")









