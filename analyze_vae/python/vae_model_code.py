import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics.pairwise import euclidean_distances
import vae_utils

EPS = 1e-6

def locmap():
    '''
    :return: location of each neuron
    '''
    x = np.arange(0, vae_utils.params.params['latent_shape'][0], dtype=np.float32)
    y = np.arange(0, vae_utils.params.params['latent_shape'][1], dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv, (xv.size, 1))
    yv = np.reshape(yv, (yv.size, 1))
    return np.hstack((xv, yv))


def lateral_effect():
    '''
    :return: functions of lateral effect
    '''
    locations = locmap()
    weighted_distance_matrix = euclidean_distances(locations, locations)/vae_utils.params.params['sigma']

    if vae_utils.params.params['lateral'] == 'mexican':
        S = (1.0-0.5*np.square(weighted_distance_matrix))*np.exp(-0.5*np.square(weighted_distance_matrix))
        return S-np.eye(len(locations))

    if vae_utils.params.params['lateral'] == 'rbf':
        S = np.exp(-0.5*np.square(weighted_distance_matrix))
        return S-np.eye(len(locations))
    
    print('no lateral effect is chosen')
    
    return np.zeros(weighted_distance_matrix.shape, dtype=np.float32)


class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, vae_utils.params.params['layers'][0], bias=False),
            #nn.Tanh()
            
        )

        self.layer2 = nn.Sequential(
            nn.Linear(vae_utils.params.params['layers'][0], vae_utils.params.params['layers'][1], bias=False),
            #nn.Tanh()
            
        )

        self.layer3 = nn.Sequential(
            nn.Linear(vae_utils.params.params['layers'][1], vae_utils.params.params['latent_shape'][0]*vae_utils.params.params['latent_shape'][1], bias=False),
            # nn.Softplus()
            nn.ReLU()
        )

    def forward(self, x):
        if vae_utils.params.params['cuda']:
            self.cuda()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(vae_utils.params.params['latent_shape'][0]*vae_utils.params.params['latent_shape'][1], input_size, bias=False),
        )

    def forward(self, x):
        output = self.layer1(x)
        return output


class VAE(nn.Module):
    def __init__(self, encoder, decoder, lateral):
        super(VAE, self).__init__()
        if vae_utils.params.params['cuda']:
            self.cuda()
        self.encoder = encoder
        self.decoder = decoder
        self.lateral = torch.from_numpy(lateral).type(torch.FloatTensor) # not positive definite
        self.dropout = nn.Dropout(vae_utils.params.params['dropout']/100) # convert from percentage
        
    def forward(self, inputs):
        if vae_utils.params.params['cuda']:
            self.cuda()
            inputs = inputs.cuda()
        #inputs = inputs/40.0
        rates = self.encoder(inputs)

        # dropout layer
        rates = self.dropout(rates)+0.0001
        
        if vae_utils.params.params['sampling'] == 'bernoulli':
            self.posterior = torch.distributions.Bernoulli(probs=rates)
            samples = self.posterior.sample([vae_utils.params.params['n_samples']])
            samples = torch.transpose(samples, 0, 1)
            samples.clamp(max = vae_utils.params.params['n_samples'])
            return torch.mean(self.decoder(samples), 1), rates

        if vae_utils.params.params['sampling'] == 'poisson':
            self.posterior = torch.distributions.Poisson(rates*vae_utils.params.params['n_samples'])
            samples = self.posterior.sample()
            return self.decoder(samples/vae_utils.params.params['n_samples']), rates

        if vae_utils.params.params['sampling'] == 'none':
            self.posterior = rates
            return self.decoder(rates), rates


    def kl_divergence(self):
        if vae_utils.params.params['sampling'] == 'bernoulli':
            prior = torch.distributions.Bernoulli(probs = torch.ones_like(self.posterior.probs)*vae_utils.params.params['default_rate'])
            kl = torch.distributions.kl_divergence(self.posterior, prior)
            return torch.mean(kl)

        if vae_utils.params.params['sampling'] == 'poisson':
            prior = torch.distributions.Poisson(torch.ones_like(self.posterior.mean) * \
                                                vae_utils.params.params['default_rate'] * vae_utils.params.params['n_samples'])
            kl = torch.distributions.kl_divergence(self.posterior, prior)
            return torch.mean(kl)

        if vae_utils.params.params['sampling'] == 'none':
            return 0.0

    def lateral_loss(self):
        if vae_utils.params.params['sampling'] == 'bernoulli':
            rates = torch.squeeze(self.posterior.probs)
        if vae_utils.params.params['sampling'] == 'poisson':
            rates = torch.squeeze(self.posterior.mean)
        if vae_utils.params.params['sampling'] == 'none':
            rates = torch.squeeze(self.posterior)

        latent_size = vae_utils.params.params['latent_shape'][0]*vae_utils.params.params['latent_shape'][1]
        
        n = rates.norm(2, 1).view(-1, 1).repeat(1, latent_size)
        rates = rates/n
        if vae_utils.params.params['cuda']:
            A = rates.mm(self.lateral.cuda()).mm(rates.t())/latent_size
        else:
            A = rates.mm(self.lateral).mm(rates.t())/latent_size # self.lateral is a lower triangular matrix
        loss = torch.diag(A)
        return -torch.mean(loss)

    def normalise_weight(self):
        weight = self.decoder.layer[0].weight.data
        tmp = torch.norm(weight, dim=0)
        self.decoder.layer[0].weight.data = weight/tmp.repeat([input_size, 1])

    def save(self):
        torch.save(self.state_dict(), vae_utils.params.params['save_path'])

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self,*datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def vaf(x,xhat):
    x = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    return (1-(np.sum(np.square(x-xhat))/np.sum(np.square(x))))*100

#%%