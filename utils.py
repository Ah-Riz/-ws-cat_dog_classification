import torch
import torch.nn as nn

def convert_params(params):
    for key, value in params.items():
        try:
            params[key] = int(value)
        except ValueError:
            pass
        except TypeError:
            pass
    return params

def get_loss_fn(selected):
    loss_fn_list={
        "L1Loss":nn.L1Loss(),
        "MSELoss": nn.MSELoss(),
        "CrossEntropyLoss": nn.CrossEntropyLoss(),
        "CTCLoss":nn.CTCLoss(),
        "NLLLoss":nn.NLLLoss(),
        "PoissonNLLLoss":nn.PoissonNLLLoss(),
        "GaussianNLLLoss":nn.GaussianNLLLoss(),
        "KLDivLoss":nn.KLDivLoss(),
    }
    return loss_fn_list[selected]
    
def get_optimizer(selected):
    optimizer_list={
        "Adam":torch.optim.Adam,
        "Adamax":torch.optim.Adamax,
        "RMSprop":torch.optim.RMSprop,
        "SGD":torch.optim.SGD
    }
    return optimizer_list[selected]