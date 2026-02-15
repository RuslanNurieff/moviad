from typing import List

import torch
from torch import Tensor

def fastflow_loss(hidden_variables: List[Tensor], jacobians: List[Tensor]) -> Tensor:
    """Calculate the Fastflow loss.

    Args:
        hidden_variables (List[Tensor]): Hidden variables from the fastflow model. f: X -> Z
        jacobians (List[Tensor]): Log of the jacobian determinants from the fastflow model.

    Returns:
        Tensor: Fastflow loss computed based on the hidden variables and the log of the Jacobians.
    """
    loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
    for (hidden_variable, jacobian) in zip(hidden_variables, jacobians):
        loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
    return loss