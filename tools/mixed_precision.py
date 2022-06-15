import torch


class MixedPrecision(object):
    def __init__(self, amp):
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def get_precision(self):
        return self if self.amp is False else torch.cuda.amp.autocast(enabled=True)

    def get_scale(self):
        return 1. if self.amp is False else self.scaler.get_scale()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        if self.amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
