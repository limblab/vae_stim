import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import euclidean_distances
from torchsummary import summary
from tensorboardX import SummaryWriter
import argparse
import torchvision.utils as vutils
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument('--batch_size', type=int, default=1024)
argparser.add_argument('--latent_shape', type=int, default=[40, 40], help='dimensions of S1 map')
argparser.add_argument('--n_epochs', type=int, default=600)
argparser.add_argument('--sampling', type=str, default='poisson')
argparser.add_argument('--n_samples', type=int, default=20)
argparser.add_argument('--layers', type=int, default=[20, 40])
argparser.add_argument('--cuda', action='store_true')
argparser.add_argument('--sigma', type=float, default=2.0, help='neibourhood factor')
argparser.add_argument('--eta', type=float, default=0.00001, help='learning rate')
argparser.add_argument('--lateral', type=str, default='mexican', help='type of lateral influence')
argparser.add_argument('--lambda_l', type=float, default=1, help='strength of lateral influence')
argparser.add_argument('--default_rate', type=float, default=0.1, help='expectation of rate')
argparser.add_argument('--save_path', type=str, default='D:\\Lab\\Data\\StimModel', help='path of saved model')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

latent_size = args.latent_shape[0]*args.latent_shape[1]
writer = SummaryWriter()

EPS = 1e-6


def locmap():
    '''
    :return: location of each neuron
    '''
    x = np.arange(0, args.latent_shape[0], dtype=np.float32)
    y = np.arange(0, args.latent_shape[1], dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv, (xv.size, 1))
    yv = np.reshape(yv, (yv.size, 1))
    return np.hstack((xv, yv))


def lateral_effect():
    '''
    :return: functions of lateral effect
    '''
    locations = locmap()
    weighted_distance_matrix = euclidean_distances(locations, locations)/args.sigma

    if args.lateral is 'mexican':
        S = (1.0-0.5*np.square(weighted_distance_matrix))*np.exp(-0.5*np.square(weighted_distance_matrix))
        return S-np.eye(len(locations))

    if args.lateral is 'rbf':
        S = np.exp(-0.5*np.square(weighted_distance_matrix))
        return S-np.eye(len(locations))
    print('no lateral effect is chosen')
    return np.zeros(weighted_distance_matrix.shape, dtype=np.float32)


class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, args.layers[0], bias=False),
            #nn.Tanh()
            
        )

        self.layer2 = nn.Sequential(
            nn.Linear(args.layers[0], args.layers[1], bias=False),
            #nn.Tanh()
            
        )

        self.layer3 = nn.Sequential(
            nn.Linear(args.layers[1], latent_size, bias=False),
            # nn.Softplus()
            nn.ReLU()
        )

    def forward(self, x):
        if args.cuda:
            self.cuda()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_size, input_size, bias=False),
        )

    def forward(self, x):
        output = self.layer1(x)
        return output


class VAE(nn.Module):
    def __init__(self, encoder, decoder, lateral):
        super(VAE, self).__init__()
        if args.cuda:
            self.cuda()
        self.encoder = encoder
        self.decoder = decoder
        self.lateral = torch.from_numpy(lateral).type(torch.FloatTensor) # not positive definite
        self.dropout = nn.Dropout(0.75)
        
    def forward(self, inputs):
        if args.cuda:
            self.cuda()
            inputs = inputs.cuda()
        #inputs = inputs/40.0
        rates = self.encoder(inputs)

        # dropout layer
        rates = self.dropout(rates)+0.01
        
        if args.sampling is 'bernoulli':
            self.posterior = torch.distributions.Bernoulli(probs=rates)
            samples = self.posterior.sample([args.n_samples])
            samples = torch.transpose(samples, 0, 1)
            samples.clamp(max = args.n_samples)
            return torch.mean(self.decoder(samples), 1)

        if args.sampling is 'poisson':
            self.posterior = torch.distributions.Poisson(rates*args.n_samples)
            samples = self.posterior.sample()
            return self.decoder(samples/args.n_samples)

        if args.sampling is 'none':
            self.posterior = rates
            return self.decoder(rates)


    def kl_divergence(self):
        if args.sampling is 'bernoulli':
            prior = torch.distributions.Bernoulli(probs = torch.ones_like(self.posterior.probs)*args.default_rate)
            kl = torch.distributions.kl_divergence(self.posterior, prior)
            return torch.mean(kl)

        if args.sampling is 'poisson':
            prior = torch.distributions.Poisson(torch.ones_like(self.posterior.mean) * \
                                                args.default_rate * args.n_samples)
            kl = torch.distributions.kl_divergence(self.posterior, prior)
            return torch.mean(kl)

        if args.sampling is 'none':
            return 0.0

    def lateral_loss(self):
        if args.sampling is 'bernoulli':
            rates = torch.squeeze(self.posterior.probs)
        if args.sampling is 'poisson':
            rates = torch.squeeze(self.posterior.mean)
        if args.sampling is 'none':
            rates = torch.squeeze(self.posterior)

        n = rates.norm(2, 1).view(-1, 1).repeat(1, latent_size)
        rates = rates/n
        if args.cuda:
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
        torch.save(self.state_dict(), args.save_path)

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

my_data = np.genfromtxt(r'D:\Lab\Data\StimModel\Han_20201204_RT3D_SmoothNormalizedJointCart_50ms.txt', delimiter=',')[:,:]
train = my_data[:10000]
test = my_data[10000:]

x_tr = torch.from_numpy(train[:,:]).type(torch.FloatTensor)
y_tr = torch.from_numpy(train[:,:]).type(torch.FloatTensor)

x_te = torch.from_numpy(test[:,:]).type(torch.FloatTensor)
y_te = test[:,:]

dataloader = DataLoader(ConcatDataset(x_tr,y_tr), batch_size=args.batch_size,
                                              shuffle=True)

test_data = (x_te,y_te)

input_size = len(x_tr[0])
output_size = len(y_tr[0])
encoder = Encoder(input_size=input_size)
decoder = Decoder(input_size=output_size)
lateral = lateral_effect()


#%%
vae = VAE(encoder, decoder, lateral)

if args.cuda:
    vae.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=args.eta)

for epoch in range(args.n_epochs):
    print(epoch)
    for i_batch, (x_batch, y_batch) in enumerate(dataloader):
        if args.cuda:
           x_batch = x_batch.cuda()
           y_batch = y_batch.cuda()
           vae.cuda()
        
        yhat = vae(x_batch)
        recon_error = criterion(yhat,y_batch)
        kl = vae.kl_divergence()
        lateral_loss = vae.lateral_loss()
        loss = 10.0*recon_error +args.lambda_l*lateral_loss + kl*0.005 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    test_result = vae(x_te)
    y_te_hat = test_result.cpu().detach().numpy()

    weight = vae.decoder.layer1[0].weight.data
    w = weight.reshape([-1, 1, args.latent_shape[0], args.latent_shape[1]])
    imgs = vutils.make_grid(w, normalize=True, scale_each=False)
    writer.add_image('Model/Weight', imgs, epoch)

    writer.add_scalar('loss/total_loss', loss, epoch)
    writer.add_scalar('loss/kl', kl, epoch)
    writer.add_scalar('loss/lateral', lateral_loss, epoch)
    writer.add_scalar('loss/recon', recon_error, epoch)
    writer.add_scalar('loss/VAF', vaf(y_te,y_te_hat), epoch)


    if epoch % 2 == 0:
        if args.cuda:
            vae.cuda()

        # test_result = torch.mean(test_result, 1)
        fig, ax = plt.subplots(10,1)
        for i in range(len(my_data[1,:])):
            if args.cuda:
                ax[i].plot(y_te[:,i])
                ax[i].plot(y_te_hat[:,i])
            else:
                ax[i].plot(y_te[:,i])
                ax[i].plot(y_te_hat[:,i])

        if args.sampling is 'bernoulli':
            rates = torch.squeeze(vae.posterior.probs)
        if args.sampling is 'poisson':
            rates = torch.squeeze(vae.posterior.mean)
        if args.sampling is 'none':
            rates = torch.squeeze(vae.posterior)
        rates = rates.reshape([-1, 1, args.latent_shape[0], args.latent_shape[1]])
       # print(rates.shape)
        response = vutils.make_grid(rates[0:1000:50], normalize=True, scale_each=False)
        writer.add_image('Model/Response', response, epoch)
        writer.add_figure('Model/test', fig, epoch)

    writer.flush()

#writer.export_scalars_to_json('./all_scalers.json')
writer.close()

#%% run test data set through model and save firing rates

my_data = np.genfromtxt(r'D:\Lab\Data\StimModel\Han_20201204_RT3D_SmoothNormalizedJointCart_50ms.txt', delimiter=',')[:,:]
my_data = torch.from_numpy(my_data[:,:]).type(torch.FloatTensor)
rates = vae.encoder(my_data)
rates = rates.detach().numpy()
np.savetxt("D:\\Lab\\Data\\StimModel\\vae_rates_Han_20201204_RT3D_jointCart_50ms.csv", rates,delimiter=",")

my_data_pred = vae(my_data)
test_vaf = vaf(my_data.detach().numpy(),my_data_pred.detach().numpy())
    
