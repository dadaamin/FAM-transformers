import torch
from torch.autograd import grad
import numpy as np
from numpy import linalg as LA
import random
import functorch
from collections import Mapping

from util import torch_conv_layer_to_affine

# example call: FAMreg(inputs, targets, F.cross_entropy, autogradHessian, j_layer)

def FAMreg(inputs, targets, hessian_function):
    regularizer = RelativeFlatness(hessian_function, hessian_function.f_layer)

    return regularizer(inputs, targets)

class RelativeFlatness():
    def __init__(self, hessian_function, feature_layer, norm_function='neuronwise'):
        self.hessian_function = hessian_function
        self.feature_layer = feature_layer
        self.norm_function = norm_function

    def __call__(self, inputs, targets):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        shape = self.feature_layer.shape

        H = self.hessian_function(inputs, targets, device)
        if self.norm_function == 'neuronwise':
            regularization = torch.tensor(torch.zeros(1), requires_grad=True).to(device)
            for neuron_i in range(shape[0]):
                neuron_i_weights = self.feature_layer[neuron_i, :]
                for neuron_j in range(shape[0]):
                    neuron_j_weights = self.feature_layer[neuron_j, :]
                    hessian = torch.tensor(H[:, neuron_j, neuron_i, :], requires_grad=True).to(device)
                    norm_trace = torch.trace(hessian) / (1.0 * shape[1])
                    regularization = regularization + torch.dot(neuron_i_weights, neuron_j_weights) * norm_trace
        return regularization

class LayerHessian():
    def __init__(self, model, layer_id, loss_function, method='functorch', padding=None, stride=None, input_shape=None):
        # here we can call torch_conv_layer_to_affine(self.parameters[self.layer_id], padding, stride, input_size)
        # to transform convolutional layer into fully connected
        self.model = model
        for i, p in enumerate(self.model.named_parameters()):
            if i == layer_id:
                self.f_name = p[0]
                self.f_layer = p[1]
                break
        self.loss_function = loss_function
        self.method = method

    def __call__(self, inputs, targets, device):
        if self.method == 'functorch':
            if isinstance(inputs, Mapping):
                func = lambda params: self.loss_function(
                    torch.nn.utils.stateless.functional_call(self.model, {self.f_name: params}, (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])).logits.to(device),
                    targets)
            else:
                func = lambda params: self.loss_function(
                    torch.nn.utils.stateless.functional_call(self.model, {self.f_name: params}, inputs).to(device),
                    targets)

            hessian = functorch.jacfwd(functorch.jacrev(func), randomness='same')(self.f_layer)
            return hessian
        elif self.method == 'autograd':
            loss = self.loss_function(self.model(inputs), targets)
            shape = self.f_layer.shape
            # need to retain_graph for further hessian calculation
            # need allow_unused for the layer that is transformed from conv2d
            layer_jacobian = grad(loss, self.f_layer, create_graph=True, retain_graph=True, allow_unused=True)
            drv2 = torch.tensor(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).to(device)
            for ind, n_grd in enumerate(layer_jacobian[0].T):
                for neuron_j in range(shape[0]):
                    drv2[ind][neuron_j] = grad(n_grd[neuron_j].to(device), self.f_layer, retain_graph=True)[0].to(device)
            return drv2
        else:
            print("No such method of hessian computation")
        return None

'''
def FAMloss(output, target, loss_function, feature_layer, lmb=0.01):
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_part = loss_function(output, target)

    regularization = Variable(torch.zeros(1), requires_grad=True).to(device)
    shape = feature_layer.shape
    layer_jacobian = grad(loss_part, feature_layer, create_graph=True, retain_graph=True)
    drv2 = Variable(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).to(device)
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = grad(n_grd[neuron_j].to(device), feature_layer, retain_graph=True)[0].to(device)

    for neuron_i in range(shape[0]):
        neuron_i_weights = feature_layer[neuron_i, :]
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :]
            hessian = Variable(torch.tensor(drv2[:, neuron_j, neuron_i, :]), requires_grad=True).to(device)
            norm_trace = torch.trace(hessian) / (1.0 * shape[1])
            regularization = regularization + torch.dot(neuron_i_weights, neuron_j_weights) * norm_trace

    end = time.time()
    print("Elapsed time", end - start)

    print(regularization)
    return loss_part + lmb*regularization
'''

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_loss_on_data(model, loss, x, y):
    output = model(x)
    loss_value = loss(output, y)
    return output, np.sum(loss_value.data.cpu().numpy())

def softmax_accuracy(output, y):
    labels_np = y.data.cpu().numpy()
    output_np = output.data.cpu().numpy()
    acc = 0
    for i in range(len(labels_np)):
        if labels_np[i] == output_np[i].argmax():
            acc += 1
    return acc

def calculateNeuronwiseHessians_fc_layer(feature_layer, train_loss, alpha, normalize = False):
    shape = feature_layer.shape

    layer_jacobian = grad(train_loss, feature_layer, create_graph=True, retain_graph=True)
    layer_jacobian_out = layer_jacobian[0]
    drv2 = Variable(torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True).cuda()
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = grad(n_grd[neuron_j].cuda(), feature_layer, retain_graph=True)[0].cuda()
    print("got hessian")

    trace_neuron_measure = 0.0
    maxeigen_neuron_measure = 0.0
    for neuron_i in range(shape[0]):
        neuron_i_weights = feature_layer[neuron_i, :].data.cpu().numpy()
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :].data.cpu().numpy()
            hessian = drv2[:,neuron_j,neuron_i,:]
            trace = np.trace(hessian.data.cpu().numpy())
            if normalize:
                trace /= 1.0*hessian.shape[0]
            trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * trace
            if neuron_j == neuron_i:
                eigenvalues = LA.eigvalsh(hessian.data.cpu().numpy())
                maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * eigenvalues[-1]
                # adding regularization term
                if alpha:
                    trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha
                    maxeigen_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha

    return trace_neuron_measure, maxeigen_neuron_measure


