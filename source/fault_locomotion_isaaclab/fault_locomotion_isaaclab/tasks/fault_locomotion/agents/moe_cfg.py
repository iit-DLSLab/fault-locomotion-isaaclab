from isaaclab.utils.configclass import configclass

from pathlib import Path
from dataclasses import MISSING


@configclass
class MoECfg:
    """Configuration for using mixture of expert
    """

    class_name: str = "MixtureOfExpert"
    """The class name."""

    who = "actor+critic"
    """Which network to use the mixture of expert, actor, critic or both"""

    num_experts = None
    """The number of expert to use, from 1 to N"""

    gate_hidden_dims = None
    """The dimension of the gate, e.g. [128, 64] or None"""

    top_k = None
    """If not used, put None, otherwise we can do the softmax between 1 to K expert"""

    use_gate_loss = False
    """Regularization loss of the gate to not overspecialize expert"""

    use_load_balance_loss = False
    """Regularization loss of the gate to balance the load between experts"""

    use_explicit_expert = False
    """Whether to use an explicit expert"""

    use_shared_layers = False
    """Whether to use a shared backbone between experts. False, "backbone", "backbone+head" are supported."""

    expert_output_dims = None
    """The output dimensions of each expert, which will be concatenated to form the final output"""


# Mixture of Expert Stuff
moe_cfg = MoECfg(
    who = "actor+critic",
    num_experts = 4,
    gate_hidden_dims = [128],
    top_k = None,
    use_gate_loss = False,
    use_load_balance_loss = False,
    use_explicit_expert = True,  
    use_shared_layers = None, 
    expert_output_dims = [12, 12, 12, 12]
)

"""moe_cfg = MoECfg(
    who = "actor+critic",
    num_experts = 4,
    gate_hidden_dims = [128],
    top_k = None,
    use_gate_loss = False,
    use_load_balance_loss = False,
    use_explicit_expert = True,  
    use_shared_layers = "backbone", 
    expert_output_dims = [12, 9, 9, 6]
)"""
