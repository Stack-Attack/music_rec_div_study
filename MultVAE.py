import numpy as np
from scipy import sparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numba as nb
from numba import jit
from torch.utils.tensorboard import SummaryWriter


@jit((nb.int32[:], nb.int32[:], nb.int32[:], nb.float32, nb.int64), nopython=True)
def hold_out_items_fast(data, cols, indptr, holdout, seed):
    # Set random seed
    np.random.seed(seed)

    # Get number of non-zero values for train and test based on split ratio
    total_non_zero_count = len(data)
    num_rows = len(indptr) - 1
    tr_non_zero = round(total_non_zero_count - (total_non_zero_count * holdout))
    te_non_zero = total_non_zero_count - tr_non_zero

    # Initialize arrays to store non-zero data for csr matrix. Add shape to account for worst-case rounding
    tr_data = np.zeros(tr_non_zero + num_rows, dtype=np.int32)
    tr_indptr = np.zeros(num_rows + 1, dtype=np.int32)
    tr_col = np.zeros(tr_non_zero + num_rows, dtype=np.int32)

    te_data = np.zeros(te_non_zero + num_rows, dtype=np.int32)
    te_indptr = np.zeros(num_rows + 1, dtype=np.int32)
    te_col = np.zeros(te_non_zero + num_rows, dtype=np.int32)

    tr_idx = 0
    te_idx = 0
    for row in range(num_rows):
        # Get the start and end indices of the row
        start_idx = indptr[row]
        end_idx = indptr[row + 1]

        # Get the indices of played songs (non-zero)
        non_zero_count = end_idx - start_idx

        # Use permutation to randomize the indices of the interactions
        non_zero_perm = np.random.permutation(non_zero_count)

        # This is the index to split the user interactions on
        split_idx = int(round(non_zero_count - (non_zero_count * holdout)))

        # Iteratively construct
        tr_data[tr_idx : tr_idx + split_idx] = data[start_idx:end_idx][
            non_zero_perm[:split_idx]
        ]
        tr_col[tr_idx : tr_idx + split_idx] = cols[start_idx:end_idx][
            non_zero_perm[:split_idx]
        ]
        tr_indptr[row] = tr_idx
        te_data[te_idx : te_idx + (non_zero_count - split_idx)] = data[
            start_idx:end_idx
        ][non_zero_perm[split_idx:]]
        te_col[te_idx : te_idx + (non_zero_count - split_idx)] = cols[
            start_idx:end_idx
        ][non_zero_perm[split_idx:]]
        te_indptr[row] = te_idx

        # Increment the indices tracking different rows
        tr_idx += split_idx
        te_idx += non_zero_count - split_idx

    tr_indptr[-1] = tr_idx
    te_indptr[-1] = te_idx

    return (tr_data[: tr_indptr[-1]], tr_col[: tr_indptr[-1]], tr_indptr), (
        te_data[: te_indptr[-1]],
        te_col[: te_indptr[-1]],
        te_indptr,
    )


class CSRDataset:
    def __init__(
        self,
        filename,
        user_holdout=0.10,
        int_holdout=0.15,
        r_seed=98765,
        binarize=False,
        toy=False,
    ):
        self.random_state = np.random.RandomState(seed=r_seed)

        if toy:
            self.data = self.data = sparse.load_npz("{}".format(filename)).astype(
                np.int32
            )[:, :10000]
        elif isinstance(filename, int):
            self.data = sparse.csr_matrix((1, filename), dtype=np.int32)
        else:
            self.data = sparse.load_npz("{}".format(filename)).astype(np.int32)
        self.users = self.data.shape[0]
        self.items = self.data.shape[1]
        self.user_holdout = user_holdout
        self.int_holdout = int_holdout

        # Binarize
        if binarize:
            self.data[self.data > 0] = 1

        # Get a list of random user indices for split
        rand_idx = self.random_state.permutation(self.users)

        # Indices to split data into train/val/test for strong generalization
        train_idx = int(self.users - (self.users * self.user_holdout))

        # Split data into train/val/test for strong generalization
        self.strong_train = self.data[rand_idx[:train_idx]]
        self.strong_val_tr, self.strong_val_te = self.hold_out_items(
            self.data[rand_idx[train_idx:]], self.int_holdout, r_seed
        )

        # Indices to split data into train/val/test for weak generalization
        self.weak, self.weak_test = self.hold_out_items(
            self.data, self.int_holdout, r_seed
        )
        self.weak_train, self.weak_val = self.hold_out_items(
            self.weak, self.int_holdout, r_seed
        )

    def hold_out_items(self, uim, holdout, r_seed):
        """
        Splits the uim into train and test sets with the same number of user but some items held out
        :param holdout: float representing ratio of items to put in test split
        :param uim: csr with m=len(user), n=len(items)
        :return:
        """
        tr_csr, te_csr = hold_out_items_fast(
            uim.data, uim.indices, uim.indptr, holdout, r_seed
        )

        return sparse.csr_matrix(tr_csr, shape=uim.shape), sparse.csr_matrix(
            te_csr, shape=uim.shape
        )


def ndcg_k(pred, holdout, already_sorted=False, k=100):

    if not torch.is_tensor(pred):
        pred = torch.FloatTensor(pred)
    if pred.device.type == "cuda":
        pred = pred.to("cpu")

    # Binarize the holdout data
    holdout[holdout > 0] = 1

    # Number of users in the batch
    batch_users = pred.shape[0]

    # Get the indices of topk predictions
    if already_sorted is False:
        _, topk_idx = pred.topk(k)
    else:
        topk_idx = pred

    # Build the discount template
    tp = 1.0 / np.log2(np.arange(2, k + 2))

    # Calculate the non-nomalized dcg for each user
    dcg = (holdout[np.arange(batch_users)[:, np.newaxis], topk_idx].toarray() * tp).sum(
        axis=1
    )

    # Calculate the ideal dcg for each user taking into consideration the number of items played
    idcg = np.array([(tp[: min(n, k)]).sum() for n in holdout.getnnz(axis=1)])

    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg != 0)


class MultVAE(nn.Module):
    def __init__(self, model_conf, dataset, device, lr=1e-3):
        """
        :param model_conf: A dictionary containing enc_dims, total_anneal_steps, anneal_cap, and dropout
        :param dataset: DataSet object containing sparse matrix data
        :param device: Torch.device to run the model on
        """
        super().__init__()
        self.dataset = dataset  # Set the dataset
        self.num_items = self.dataset.items  # Number of columns in the data matrix

        # Set the learning rate
        self.lr = lr

        if isinstance(model_conf["enc_dims"], str):
            model_conf.enc_dims = eval(
                model_conf["enc_dims"]
            )  # Some kind of function to evaluate

        self.enc_dims = [self.num_items] + model_conf[
            "enc_dims"
        ]  # Dimensions of encoding [in,
        self.dec_dims = self.enc_dims[::-1]  # Dimensions of decoding
        self.dims = self.enc_dims + self.dec_dims[1:]  # Total dimensions

        self.total_anneal_steps = model_conf["total_anneal_steps"]
        self.anneal_cap = model_conf["anneal_cap"]

        self.dropout = model_conf["dropout"]
        # self.reg = model_conf['reg'] # This wasn't in the config.

        self.eps = 1e-6  # Epsilon value (small near-0 value)

        self.anneal = 0.0  # Current annealing value
        self.update_count = 0  # Current update count

        self.device = device

        self.best_score = None

        self.logdir = "/scratch/kyle709/study_2/vae_checkpoints/{}_{}_{}_{}".format(
            self.enc_dims, self.dropout, self.anneal_cap, self.total_anneal_steps
        )

        self.writer = SummaryWriter(self.logdir)

        # This block builds the encoder by appending modules to a list iteratively
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            if (
                i == len(self.enc_dims[:-1]) - 1
            ):  # If encoded layer double output for mean and dev.
                d_out *= 2
            self.encoder.append(
                nn.Linear(d_in, d_out)
            )  # Add a linear layer with appropriate in/out
            if (
                i != len(self.enc_dims[:-1]) - 1
            ):  # If not encoding layer add act. function
                self.encoder.append(nn.Tanh())

        # This block builds the decoder by appending modules to a list iteratively
        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(
                nn.Linear(d_in, d_out)
            )  # Add a linear layer with appropriate in/out
            if (
                i != len(self.dec_dims[:-1]) - 1
            ):  # If not encoding layer add act. function
                self.decoder.append(nn.Tanh())

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

        self.to(self.device)

    def forward(self, uim, novel=True):
        """
        Completes one forward pass of the input data on the network.
        :param uim: sparse matrix where m=len(users), n=len(items)
        :return:
        """
        # Check if data is a tensor and if not convert it to one
        if not torch.is_tensor(uim):
            uim = torch.FloatTensor(uim.toarray()).to(self.device)

        # Encoder
        h = F.dropout(
            F.normalize(uim), p=self.dropout, training=self.training
        )  # Apply dropout (0 out half values)
        for layer in self.encoder:  # Feed data through encoder
            h = layer(h)

        # Sample
        mu_q = h[:, : self.enc_dims[-1]]  # First half of encoding layer for means
        logvar_q = h[:, self.enc_dims[-1] :]  # Second half of encoding layer for devs.
        std_q = torch.exp(0.5 * logvar_q)  # Somehow get standard deviations here

        epsilon = torch.zeros_like(std_q).normal_(
            mean=0, std=0.01
        )  # Some more fancy stats

        samples = (
            mu_q + self.training * epsilon * std_q
        )  # Compute samples from the distribution

        # Decoder
        output = samples  # Encoder output is data sampled from the distribution
        for layer in self.decoder:  # Feed data through the decoder
            output = layer(output)

        # Return the output OR compute loss if training
        if self.training:
            kl_loss = (
                (0.5 * (-logvar_q + torch.exp(logvar_q) + torch.pow(mu_q, 2) - 1)).sum(
                    1
                )
            ).mean()
            return output, kl_loss
        else:
            if novel:
                output[uim.nonzero(as_tuple=True)] = float("-inf")

            return output

    def train_one_epoch(self, batch_size, val=True, verbose=False):
        """
        Completes one epoch worth of training on the network.
        :param data: 'train' or 'data' for training or production
        :param batch_size:
        :param verbose:
        :return:
        """
        self.train()

        # Determines whether to train on only the test data or on the entirety of the dataset
        if val is True:
            train_matrix = self.dataset.weak_train
        else:
            train_matrix = self.dataset.data

        optimizer = self.optimizer

        num_training = train_matrix.shape[0]  # Count of users to train on
        num_batches = int(
            np.ceil(num_training / batch_size)
        )  # Count of batches to complete epoch

        perm = np.random.permutation(num_training)  # Get random user indices

        loss = 0.0
        kl = 0.0
        ce = 0.0
        # for b in tqdm(range(num_batches)):
        for b in range(num_batches):
            optimizer.zero_grad()

            # If final batch size, take remaining user ids, otherwise get correct splice
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size :]
            else:
                batch_idx = perm[b * batch_size : (b + 1) * batch_size]

            # Create matrix using user id's for batch and transfer it to device
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(
                self.device
            )

            # Anneal
            if self.total_anneal_steps > 0:
                self.anneal = min(
                    self.anneal_cap, 1.0 * self.update_count / self.total_anneal_steps
                )
            else:
                self.anneal = self.anneal_cap
            # Complete one forward pass on the network with the training batch
            pred_matrix, kl_loss = self.forward(batch_matrix)

            # Compute cross-entropy loss and total batch loss
            ce_loss = -(F.log_softmax(pred_matrix, 1) * batch_matrix).sum(1).mean()
            batch_loss = ce_loss + kl_loss * self.anneal

            batch_loss.backward()
            optimizer.step()

            self.update_count += 1

            kl += kl_loss
            ce += ce_loss
            loss += batch_loss
            if verbose and b % 50 == 0:
                print("(%3d / %3d) loss = %.4f" % (b, num_batches, batch_loss))

        return loss, ce, kl

    def predict_eval(self, pred, holdout, batch_size):
        """
        Predicts values for users in uim using batch size and returns each user ndcg@k
        :param uim:
        :param batch_size:
        :return:
        """
        num_training = pred.shape[0]  # Count of users to train on
        num_batches = int(
            np.ceil(num_training / batch_size)
        )  # Count of batches to complete epoch

        self.eval()
        ndcg_dist = []
        with torch.no_grad():
            # for b in tqdm(range(num_batches)):
            for b in range(num_batches):
                # If final batch size, take remaining user ids, otherwise get correct splice
                if (b + 1) * batch_size >= num_training:
                    batch_matrix = torch.FloatTensor(
                        pred[b * batch_size :].toarray()
                    ).to(self.device)

                    batch_holdout = holdout[b * batch_size :]
                else:
                    batch_matrix = torch.FloatTensor(
                        pred[b * batch_size : (b + 1) * batch_size].toarray()
                    ).to(self.device)

                    batch_holdout = holdout[b * batch_size : (b + 1) * batch_size]

                # Complete one forward pass on the network with the training batch
                batch_pred = self.forward(batch_matrix)

                # Get the ndcg for each user in batch and append it to others
                ndcg_dist.append(ndcg_k(batch_pred, batch_holdout))

        return np.concatenate(ndcg_dist)

    def train_val(
        self, epochs, batch_size, tr_batch_size, patience=float("inf"), start=1
    ):
        ndcg_list = []
        early_stop = 0
        anneal_patience = 5
        for i in tqdm(range(start, epochs + 1)):

            # Train one epoch of the model
            start_time = time.time()
            loss, ce, kl = self.train_one_epoch(batch_size, val=True, verbose=False)
            elapsed_time = time.time() - start_time

            # Evaluate ndcg@k on validation set
            self.eval()
            ndcg_dist = self.predict_eval(
                self.dataset.weak_train, self.dataset.weak_val, tr_batch_size
            )
            ndcg_ = ndcg_dist.mean()

            self.writer.add_scalar("Loss/ce", ce, i)
            self.writer.add_scalar("Loss/kl", kl, i)
            self.writer.add_scalar("Loss/NDCG@100", ndcg_, i)

            ndcg_list.append(ndcg_)
            #             fig = go.Figure(data=go.Scatter(y=ndcg_list))
            #             fig.update_xaxes(range=[0, epochs ])
            #             display.clear_output(wait=True)
            #             display.display(fig.show())
            #             display.display(self.anneal)

            if self.best_score is None or ndcg_ > self.best_score:
                self.best_score = ndcg_
                early_stop = False
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.logdir + ".pt",
                )

            else:
                early_stop += 1
                if early_stop >= patience:
                    print("Stopping early. . .")
                    break
                if early_stop >= anneal_patience and self.anneal_cap != self.anneal:
                    print("Stopping annealing. . .")
                    self.anneal_cap = self.anneal

    def train_final(self, epochs, batch_size, patience=float("inf"), start=1):
        self.logdir = self.logdir + "_final"
        self.writer = SummaryWriter(self.logdir)

        early_stop = 0
        for i in tqdm(range(start, epochs + 1)):

            # Train one epoch of the model
            loss, ce, kl = self.train_one_epoch(batch_size, val=False, verbose=False)

            self.writer.add_scalar("Loss/ce", ce, i)
            self.writer.add_scalar("Loss/kl", kl, i)
            self.writer.add_scalar("Loss/total", loss, i)

            if self.best_score is None or loss < self.best_score:
                self.best_score = loss
                early_stop = False
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.logdir + ".pt",
                )

            else:
                early_stop += 1
                if early_stop >= patience:
                    print("Stopping early. . .")
                    break
