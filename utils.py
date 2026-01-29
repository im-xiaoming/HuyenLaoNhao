import torch
import torch.nn as nn
import os, shutil


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis,True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms):

    assert stacked_embeddings.ndim == 3
    assert stacked_norms.ndim == 3

    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm



class EarlyStopping:
    def __init__(self, root, backup, model, head, optimizer, patience=3, eps=1e-6):
        self.path = os.path.join(root, 'checkpoints')
        self.backup = backup
        
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.backup, exist_ok=True)

        self.model = model
        self.head = head
        self.optimizer = optimizer
        self.eps = eps
        self.best_acc = float('-inf')
        self.count = 0
        self.stop = False
        self.patience = patience


    def copy(self, filename):        
        shutil.copy(filename, self.backup)


    def save(self, **kwargs):
        epoch = kwargs.get('epoch', 0)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'head_state_dict': self.head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': self.best_acc,
            'patience_count': self.count,
            
            'loss': kwargs.get('loss', None),
            'fpr_1e-4': kwargs.get('fpr_1e-4', None),
            'fpr_1e-5': kwargs.get('fpr_1e-5', None)
        }
        file = os.path.join(kwargs.get('root') if kwargs.get('root') else self.path,
                            f'checkpoint_{epoch}.pth')
        torch.save(checkpoint, file)
        print(f'Save checkpoint at epoch {epoch}\n')
        self.copy(file)
        return file

    def __call__(self, **kwargs):
        if kwargs.get('acc', 0) > self.best_acc + self.eps:
            print("Improved from previous best: {:.6f} to {:.6f}\n".format(
                self.best_acc, kwargs.get('acc', 0)))
            self.best_acc = kwargs.get('acc', 0)
            self.save(**kwargs)
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True
            print(f'No Improvement. Count: {self.count}/{self.patience}\n')
            
            
def load_checkpoint(path, model, head=None, optimizer=None,
                    scheduler=None, early_stopping=None, device='gpu'):
    
    statedict = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(statedict['model_state_dict'])

    head.load_state_dict(statedict['head_state_dict'])
    optimizer.load_state_dict(statedict['optimizer_state_dict'])

    epoch = statedict.get('epoch')
    acc = statedict.get('acc')
    count = statedict.get('count')

    scheduler.last_epoch = epoch - 1

    early_stopping.best_acc = acc
    early_stopping.count = count

    print(f"Successfully load model statedict with epoch {epoch}.\n")

    return epoch


def load_weights(path, model, device='gpu'):
    statedict = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(statedict['model_state_dict'])
    print(f"Successfully load model weights from {path}.\n")
    return model


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay