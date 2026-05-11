# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:39:57 2026

@author: senev
"""
import deepchem as dc
import os


import	torch

pytorch_model	=	torch.nn.Sequential(
torch.nn.Linear(1024,	1000),
torch.nn.ReLU(),
				torch.nn.Dropout(0.5),
				torch.nn.Linear(1000,	1)
)

tasks,	datasets,	transformers	=	dc.molnet.load_delaney(featurizer='ECFP',	splitter='random')
train_dataset,	valid_dataset,	test_dataset	=	datasets

metric	=	dc.metrics.Metric(dc.metrics.pearson_r2_score)

model	=	dc.models.TorchModel(pytorch_model,	dc.models.losses.L2Loss())

model.fit(train_dataset, nb_epoch=20)

print('training	set	score:',	model.evaluate(train_dataset,[metric]))
print('test	set	score:',	model.evaluate(test_dataset,[metric]))