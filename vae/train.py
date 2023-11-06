import torch
from .types_ import *
from tqdm import tqdm

def vae_train(autoencoder, data_loader: Tensor, epochs: int, learning_rate):
    """
    Given a dataset data, returns a trained VAE object
    :param data_loader: (Tensor) [N x B x 1536]
    """
    optimizer = torch.optim.Adam([
        dict(params=autoencoder.parameters(), lr=learning_rate),
    ])
    for i in range(epochs):
        td = tqdm(data_loader, desc=f"Train VAE epoch {i+1}/{epochs}")
        for (_, tokens, _) in td:
            tokens = torch.stack([x.to(autoencoder.device) for x in tokens], dim=0) # GPU
            optimizer.zero_grad()
            tokens_hat, mu, sigma = autoencoder(tokens)
            print(tokens_hat.requires_grad)
            print(mu.requires_grad)
            print(sigma.requires_grad)
            sigma = torch.exp(sigma)
            print(sigma.requires_grad)
            loss = ((tokens - tokens_hat)**2).sum() + (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            loss.backward()
            optimizer.step()
            td.set_postfix(loss=loss)
    return autoencoder