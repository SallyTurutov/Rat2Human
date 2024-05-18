import torch
import torch.nn.functional as F


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, loss_function, opt, lambda_param=0.3):
        self.loss_function = loss_function
        self.opt = opt
        self.lambda_param = lambda_param

    def __call__(self, x, y, norm):
        loss = self.loss_function(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data * norm
