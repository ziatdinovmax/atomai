from typing import Tuple, Type, Optional, Union
import torch
import numpy as np


class viBaseTrainer:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_dim = None
        self.out_dim = None
        self.z_dim = 1
        self.encoder_net = None
        self.decoder_net = None
        self.train_iterator = None
        self.test_iterator = None
        self.optim = None
        self.metadict = {}
        self.loss_history = {"train_loss": [], "test_loss": []}

    def set_model(self, encoder_net, decoder_net):
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.encoder_net.to(self.device)
        self.decoder_net.to(self.device)

    def set_data(self,
                 X_train: Union[torch.Tensor, np.ndarray],
                 y_train: Union[torch.Tensor, np.ndarray] = None,
                 X_test: Union[torch.Tensor, np.ndarray] = None,
                 y_test: Union[torch.Tensor, np.ndarray] = None,
                 ) -> None:

        self.train_iterator = self._set_data(X_train, y_train)
        if X_test is not None:
            self.test_iterator = self._set_data(X_test, y_test)

    def _set_data(self, X, y=None):

        tor = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        X, y = tor(X), tor(y)

        if X is None:
            raise AssertionError(
                "You must provide input train/test data")

        if y is not None:  # VED or cVAE
            data_train = torch.utils.data.TensorDataset(X, y)
        else:  # VAE
            data_train = torch.utils.data.TensorDataset(X,)

        data_loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.batch_size,
            shuffle=True, drop_last=True)

        return data_loader

    def step(self) -> None:
        raise NotImplementedError

    def compile_trainer(self,
                        train_data: Tuple[Union[torch.Tensor, np.ndarray]],
                        test_data: Tuple[Union[torch.Tensor, np.ndarray]] = None,
                        loss: str = "mse",
                        optimizer: Optional[Type[torch.optim.Optimizer]] = None,
                        training_cycles: int = 1000,
                        batch_size: int = 32,
                        **kwargs
                        ) -> None:

        self.training_cycles = training_cycles
        self.batch_size = batch_size
        self.loss = loss

        if test_data is not None:
            self.set_data(*train_data, *test_data)
        else:
            self.set_data(*train_data)

        params = list(self.decoder_net.parameters()) +\
            list(self.encoder_net.parameters())
        if optimizer is None:
            self.optim = torch.optim.Adam(params, lr=1e-4)
        else:
            self.optim = optimizer(params)

        self.filename = kwargs.get("filename", "./model")

    @classmethod
    def reparameterize(cls,
                       z_mean: torch.Tensor,
                       z_sd: torch.Tensor
                       ) -> torch.Tensor:
        """
        Reparameterization trick
        """
        batch_dim = z_mean.size(0)
        z_dim = z_mean.size(1)
        eps = z_mean.new(batch_dim, z_dim).normal_()
        return z_mean + z_sd * eps

    def train_epoch(self):
        """
        Trains a single epoch
        """
        self.decoder_net.train()
        self.encoder_net.train()
        c = 0
        elbo_epoch = 0
        for x in self.train_iterator:
            if len(x) == 1:  # VAE mode
                x = x[0]
                y = None
            else:  # VED or cVAE mode
                x, y = x
            b = x.size(0)
            elbo = self.step(x) if y is None else self.step(x, y)
            loss = -elbo
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch)
            elbo_epoch += delta / c
        return elbo_epoch

    def evaluate_model(self):
        """
        Evaluates model on test data
        """
        self.decoder_net.eval()
        self.encoder_net.eval()
        c = 0
        elbo_epoch_test = 0
        for x in self.test_iterator:
            if len(x) == 1:
                x = x[0]
                y = None
            else:
                x, y = x
            b = x.size(0)
            if y is None:  # VAE mode
                elbo = self.step(x, mode="eval")
            else:  # VED or cVAE mode
                elbo = self.step(x, y, mode="eval")
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch_test)
            elbo_epoch_test += delta / c
        return elbo_epoch_test

    def save_model(self, *args: str) -> None:
        """
        Saves trained weights and the key model parameters
        """
        try:
            savepath = args[0]
        except IndexError:
            savepath = self.filename
        self.metadict["encoder"] = self.encoder_net.state_dict()
        self.metadict["decoder"] = self.decoder_net.state_dict()
        self.metadict["optimizer"] = self.optim
        torch.save(self.metadict, savepath + ".tar")

    def save_weights(self, *args: str) -> None:
        """
        Saves trained weights
        """
        try:
            savepath = args[0]
        except IndexError:
            savepath = self.filename + "weights"
        torch.save({"encoder": self.encoder_net.state_dict(),
                   "decoder": self.decoder_net.state_dict()},
                   savepath + ".tar")

    def load_weights(self, filepath: str) -> None:
        """
        Loads saved weights
        """
        weights = torch.load(filepath, map_location=self.device)
        encoder_weights = weights["encoder"]
        decoder_weights = weights["decoder"]
        self.encoder_net.load_state_dict(encoder_weights)
        self.encoder_net.eval()
        self.decoder_net.load_state_dict(decoder_weights)
        self.decoder_net.eval()
