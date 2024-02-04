import torch
import torch.nn as nn

def l2_norm(param):
    size_reg = param.numel()
    return torch.sqrt(torch.sum(torch.pow(param, 2)) / size_reg)

def hg_loss(output, y, model, loss_func, reg_coeff=1, n_incoming=3):
    spaces = model.spaces[model.number_of_input_spaces:]
    spaces.extend(model.output_spaces)
    reg = 0
    for i, space in enumerate(spaces):
        if not space.pruned:
            if i <= model.number_of_spaces:
                coeff = len(space.incoming) - n_incoming
            else:
                coeff = 1
            coeff = max(coeff, 0)

            for param in space.parameters():
                reg += coeff * l2_norm(param)


    l1 = loss_func(output, y.to(output.device))
    l2 = reg_coeff * reg
    return l1 + l2, {"log_likelihood": l1.item(), 'regularization':l2.item()}


def hg_classification_loss(output, y, model, reg_coeff=1, n_incoming=3):
    return hg_loss(output, y, model, nn.functional.nll_loss,
                   reg_coeff=reg_coeff, n_incoming=n_incoming)