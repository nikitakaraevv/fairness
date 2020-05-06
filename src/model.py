import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
### Define the CNN model ###

def get_classifier(n_outputs=1, n_filters=12):
    model = nn.Sequential(
        nn.Conv2d(3, 1*n_filters, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(1*n_filters),
        nn.Conv2d(1*n_filters, 2*n_filters, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(2*n_filters),
        nn.Conv2d(2*n_filters, 4*n_filters, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(4*n_filters),
        nn.Conv2d(4*n_filters, 6*n_filters, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.BatchNorm2d(6*n_filters),
        nn.Flatten(),
        nn.Linear(288, 512),
        nn.ReLU(),
        nn.Linear(512, n_outputs)
    )
    return model

class Decoder(nn.Module):
  # Functionally define the different layer types we will use
  # Build the decoder network using the Sequential API
    def __init__(self, n_filters=12, latent_dim=100):
        super(Decoder, self).__init__()
        self.n_filters = n_filters
        # Transform to pre-convolutional generation
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=4*4*6*n_filters)  # 4x4 feature maps (with 6N occurances)
        # Upscaling convolutions (inverse of encoder)
        self.convT1 = nn.ConvTranspose2d(in_channels=6*n_filters, out_channels=4*n_filters, kernel_size=3, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(in_channels=4*n_filters, out_channels=2*n_filters, kernel_size=3, stride=2, padding=0)
        self.convT3 = nn.ConvTranspose2d(in_channels=2*n_filters, out_channels=1*n_filters, kernel_size=4, stride=2, padding=0)
        self.convT4 = nn.ConvTranspose2d(in_channels=1*n_filters, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, input):
        bs = input.size()[0]
        x = F.relu(self.fc1(input))
        x = x.view(bs,6*self.n_filters, 4, 4)
        x = self.convT1(x)
        x = self.convT2(x)
        x = self.convT3(x)
        x = self.convT4(x)
        return x


class DB_VAE(nn.Module):
    def __init__(self, latent_dim):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim

        # Define the number of outputs for the encoder. Recall that we have 
        # `latent_dim` latent variables, as well as a supervised output for the 
        # classification.
        num_encoder_dims = 2*self.latent_dim + 1

        self.encoder = get_classifier(num_encoder_dims)
        self.decoder = Decoder(latent_dim=self.latent_dim)

    # function to feed images into encoder, encode the latent space, and output
    # classification probability 
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)

        # classification prediction
        y_logit = encoder_output[:,0].unsqueeze(-1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim+1] 
        z_logsigma = encoder_output[:, self.latent_dim+1:]

        return y_logit, z_mean, z_logsigma

  # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        z = sampling(z_mean, z_logsigma)
        return z

  # Decode the latent space and output reconstruction
    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction

    # The call function will be used to pass inputs x through the core VAE
    def forward(self, x): 
        # Encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)

        z = self.reparameterize(z_mean, z_logsigma)

        recon = self.decode(z)
        return y_logit, z_mean, z_logsigma, recon

    # Predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit


def vae_loss_function(input, output, mu, logsigma, kl_weight=0.0005):
    latent_loss = 0.5 * torch.mean(logsigma.exp() + mu.pow(2) - 1 - logsigma, dim=1)
    reconstruction_loss = torch.mean(torch.abs(input-output),  dim=(1,2,3))
    vae_loss = reconstruction_loss + kl_weight * latent_loss
    return vae_loss

def sampling(z_mean, z_logsigma):
    # By default, random.normal is "standard" (ie. mean=0 and std=1.0)
    std = z_logsigma.mul(0.5).exp_()
    eps = torch.empty_like(std).normal_()
    return eps.mul(std).add_(z_mean)


def debiasing_loss_function(input, output, y, y_logit, mu, logsigma):
    vae_loss = vae_loss_function(input, output, mu, logsigma) 
    classification_loss = F.binary_cross_entropy_with_logits(y_logit, y, reduction='none')
    total_loss = classification_loss.squeeze() + y.squeeze() * vae_loss
    return total_loss.sum(), classification_loss.sum()

def get_latent_mu(images, dbvae, batch_size=1024):
    with torch.no_grad():
        N = images.size()[0]
        mu = np.zeros((N, dbvae.latent_dim))
        for start_ind in range(0, N, batch_size):
            end_ind = min(start_ind+batch_size, N+1)
            batch = (images[start_ind:end_ind])
            _, batch_mu, _ = dbvae.encode(batch)
            mu[start_ind:end_ind] = batch_mu
    return mu

### Resampling algorithm for DB-VAE ###

'''Function that recomputes the sampling probabilities for images within a batch
      based on how they distribute across the training data'''
def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=0.001): 
    print("Recomputing the sampling probabilities")
    mu = get_latent_mu(images, dbvae) 
    # sampling probabilities for the images
    training_sample_p = np.zeros(mu.shape[0])
    
    # consider the distribution for each latent variable 
    for i in range(dbvae.latent_dim):
      
        latent_distribution = mu[:,i]
        # generate a histogram of the latent distribution
        hist_density, bin_edges =  np.histogram(latent_distribution, density=True, bins=bins)

        # find which latent bin every data sample falls in 
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')
        
        # TODO: call the digitize function to find which bins in the latent distribution 
        #    every data sample falls in to
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
        bin_idx = np.digitize(latent_distribution, bin_edges) # TODO

        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        # invert the density function 
        p = 1.0/(hist_smoothed_density[bin_idx-1])
        
        # TODO: normalize all probabilities
        p = p / np.sum(p)
        
        # TODO: update sampling probabilities by considering whether the newly
        #     computed p is greater than the existing sampling probabilities.
        training_sample_p = np.maximum(p, training_sample_p)
        
    # final normalization
    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p