# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
from deepchem.models.torch_models.layers import GraphConv, GraphGather, GraphPool

import deepchem as dc

tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

n_tasks = len(tasks)


from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
import numpy as np
batch_size=100

def data_generator(dataset, epochs=1):
  for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size, epochs,
                                                                   deterministic=False, pad_batches=True)):
    multiConvMol = ConvMol.agglomerate_mols(X_b)
    inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, np.array(multiConvMol.membership)]
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
      inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
    labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
    weights = [w_b]
    yield (inputs, labels, weights)
    
    
sample_batch = next(data_generator(train_dataset))
node_features = sample_batch[0][0]
num_input_features = node_features.shape[1]
print(f"Number of input features: {num_input_features}")


class GraphConvModelTorch(nn.Module):
    def __init__(self):
        super(GraphConvModelTorch, self).__init__()

        self.gc1 = GraphConv(out_channel=128, number_input_features=num_input_features, activation_fn=nn.Tanh())
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.gp1 = GraphPool()

        self.gc2 = GraphConv(out_channel=128, number_input_features=128, activation_fn=nn.Tanh())
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.gp2 = GraphPool()

        self.dense1 = nn.Linear(128, 256)
        self.act3 = nn.Tanh()
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.readout = GraphGather(batch_size=batch_size, activation_fn=nn.Tanh())

        self.dense2 = nn.Linear(512, n_tasks * 2)

        self.logits = lambda data: data.view(-1, n_tasks, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        gc1_output = self.gc1(inputs)
        batch_norm1_output = self.batch_norm1(gc1_output)
        gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

        gc2_output = self.gc2([gp1_output] + inputs[1:])
        batch_norm2_output = self.batch_norm2(gc2_output)
        gp2_output = self.gp2([batch_norm2_output] + inputs[1:])

        dense1_output = self.act3(self.dense1(gp2_output))
        batch_norm3_output = self.batch_norm3(dense1_output)
        readout_output = self.readout([batch_norm3_output] + inputs[1:])

        dense2_output = self.dense2(readout_output)
        logits_output = self.logits(dense2_output)
        softmax_output = self.softmax(logits_output)
        return softmax_output
    
    
    
model = dc.models.TorchModel(GraphConvModelTorch(), loss=dc.models.losses.CategoricalCrossEntropy())
model.fit_generator(data_generator(train_dataset, epochs=5))

metric	=	dc.metrics.Metric(dc.metrics.roc_auc_score)

print('Training set score:', model.evaluate_generator(data_generator(train_dataset), [metric], transformers))
print('Test set score:', model.evaluate_generator(data_generator(test_dataset), [metric], transformers))



