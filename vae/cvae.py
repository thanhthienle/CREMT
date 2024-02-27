import torch
from vae import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from tqdm import tqdm


class ConditionalVAE(BaseVAE):

    def __init__(self, args, glob2task_relid,
        num_classes: int = 4,
        in_channels: int = 1536,
        latent_dim: int = 64,
        hidden_dims: List = [],
        img_size:int = 1,
        **kwargs
    ) -> None:
        super(ConditionalVAE, self).__init__()

        self.glob2task_relid = glob2task_relid
        self.num_classes = num_classes

        self.latent_dim = latent_dim
        self.img_size = img_size

        self.embed_class = lambda x: x
        self.embed_data = nn.Linear(in_channels, in_channels)

        if not hidden_dims:
            hidden_dims = [768, 512, 256, 128]

        changing_in_channels = in_channels + 1 # To account for the extra label channel

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.extend([
                nn.Linear(changing_in_channels, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
            ])
            changing_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + 1, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ])

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_dims[-1], in_channels),
                            nn.Tanh())

        self.device = args.device

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        y = kwargs['labels']
        embedded_class = self.embed_class(y.float())
        embedded_class = embedded_class.unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        # z = torch.cat([z, F.one_hot(y, num_classes=self.num_classes).float()], dim = 1)
        z = torch.cat([z, embedded_class], dim = 1)
        return [self.decode(z), mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input, reduction="mean")

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device=None,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        if current_device is None:
            current_device = self.device

        local_label = self.glob2task_relid[kwargs["label"]]
        # y = F.one_hot(torch.tensor(local_label, device=current_device), num_classes=self.num_classes).float().expand(num_samples, -1)
        y = torch.tensor(local_label, device=current_device, dtype=torch.float32).expand(num_samples, 1)
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return [samples.detach().cpu().numpy()]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]

    def fit(self, data_loader: Tensor, num_total_train: int, epochs: int, learning_rate):
        """
        Given a dataset data, returns a trained VAE object
        :param data_loader: (Tensor) [N x B x 1536]
        """
        optimizer = torch.optim.Adam([
            dict(params=self.parameters(), lr=learning_rate),
        ])
        for i in range(epochs):
            td = tqdm(data_loader, desc=f"Train VAE epoch {i+1}/{epochs}")
            for (labels, tokens, _) in td:
                labels = labels.type(torch.LongTensor).to(self.device)
                tokens = torch.stack([x.to(self.device) for x in tokens], dim=0) # GPU
                optimizer.zero_grad()
                tokens_hat, mu, sigma = self.forward(tokens, labels=labels)

                # Loss
                M_N = tokens.shape[0] / num_total_train
                loss_dict = self.loss_function(tokens, tokens_hat, mu, sigma, M_N=M_N)
                loss = loss_dict["loss"]

                # Backward
                loss.backward()
                optimizer.step()
                td.set_postfix(
                    loss=loss.item(),
                    recon=loss_dict["Reconstruction_Loss"].item(),
                    kl_div=loss_dict["KLD"].item()
                )
