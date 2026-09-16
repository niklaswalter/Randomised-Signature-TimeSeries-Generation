"""
Out-of-sample evaluation for the conditional generators
"""

import torch

from rsig_wgan.discriminator_models.sigcw1 import compute_sig, fit_lr_sig, predict_lr_sig
from rsig_wgan.utils import compute_rsig, fit_lr_rsig, predict_lr_rsig

from .metrics import acf_diff, cov_diff


class ConditionalEvaluator:
    """
    Evaluates a conditional generator on held-out pasts.

    The conditional expectation is fitted on the training pasts only and then applied to the
    test pasts, so the reported metric is out-of-sample rather than the best training loss.
    """

    def __init__(
        self,
        generator,
        x_train,
        x_test,
        p,
        q,
        mc_num,
        discriminator_id,
        device,
        A1=None,
        A2=None,
        xi1=None,
        xi2=None,
        dim_res=None,
        activation=None,
        trunc=None,
        augmented=True
    ):
        self.generator = generator
        self.p = p
        self.q = q
        self.mc_num = mc_num
        self.discriminator_id = discriminator_id
        self.device = device
        self.A1, self.A2, self.xi1, self.xi2 = A1, A2, xi1, xi2
        self.dim_res = dim_res
        self.activation = activation
        self.trunc = trunc
        self.augmented = augmented

        self.x_train_past = x_train[:, :self.p].to(self.device)
        self.x_train_future = x_train[:, self.p:].to(self.device)
        self.x_test_past = x_test[:, :self.p].to(self.device)
        self.x_test_future = x_test[:, self.p:].to(self.device)

        self.estimator = self.fit_estimator()

        with torch.no_grad():
            self.x_fake_train = self.generate_conditional(self.x_train_past)
            self.x_fake_test = self.generate_conditional(self.x_test_past)

            self.train_error = self.conditional_error(self.x_train_past)
            self.test_error = self.conditional_error(self.x_test_past)

            self.acf_train_error = acf_diff(self.x_train_future, self.x_fake_train, lag=self.q // 2)
            self.acf_test_error = acf_diff(self.x_test_future, self.x_fake_test, lag=self.q // 2)

            self.cov_train_error = cov_diff(self.x_train_future, self.x_fake_train)
            self.cov_test_error = cov_diff(self.x_test_future, self.x_fake_test)

    def fit_estimator(self):
        if self.discriminator_id == "RSigCW1":
            return fit_lr_rsig(self.x_train_future, self.x_train_past, self.A1, self.A2, self.xi1, self.xi2,
                               self.dim_res, self.activation, self.device)
        elif self.discriminator_id == "SigCW1":
            return fit_lr_sig(self.x_train_future, self.x_train_past, self.trunc, self.device, self.augmented)
        raise ValueError(f"Unknown conditional discriminator id: {self.discriminator_id}")

    def predict(self, x_past):
        if self.discriminator_id == "RSigCW1":
            return predict_lr_rsig(self.estimator, x_past, self.A1, self.A2, self.xi1, self.xi2,
                                   self.dim_res, self.activation, self.device)
        return predict_lr_sig(self.estimator, x_past, self.trunc, self.device, self.augmented)

    def features(self, x):
        if self.discriminator_id == "RSigCW1":
            return compute_rsig(x, self.A1, self.A2, self.xi1, self.xi2, self.dim_res,
                                self.activation, self.device).reshape([x.shape[0], self.dim_res])
        return compute_sig(x, self.trunc, self.device, self.augmented)

    def generate_conditional(self, x_past, samples_per_past=None):
        """
        One future per past, for the distributional metrics
        """
        n = samples_per_past or 1
        paths = [self.generator(n, self.q, past.reshape(1, self.p, 1)).to(self.device) for past in x_past]
        return torch.cat(paths, dim=0)

    def monte_carlo_features(self, x_past):
        """
        Monte-Carlo estimate of the conditional expected (randomised) signature per past.

        Rows are grouped as [past_0 x mc_num, past_1 x mc_num, ...], so the average must be
        taken over the second axis of a (n_past, mc_num, -1) view.
        """
        fakes = [self.generator(self.mc_num, self.q, past.reshape(1, self.p, 1)).to(self.device)
                 for past in x_past]
        features = self.features(torch.cat(fakes, dim=0))
        return features.reshape(x_past.shape[0], self.mc_num, -1).mean(1)

    def conditional_error(self, x_past):
        predicted = self.predict(x_past).to(self.device)
        realised = self.monte_carlo_features(x_past)
        return torch.norm(predicted - realised, p=2, dim=1).mean()
