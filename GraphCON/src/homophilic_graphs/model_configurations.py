from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc
from function_gcn import GCNFunc
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock


class BlockNotDefined(Exception):
    pass


class FunctionNotDefined(Exception):
    pass


def set_block(opt):
    ode_str = opt['block']
    if ode_str == 'attention':
        block = AttODEblock
    elif ode_str == 'constant':
        block = ConstantODEblock
    else:
        raise BlockNotDefined
    return block


def set_function(opt):
    ode_str = opt['function']
    if ode_str == 'laplacian':
        f = LaplacianODEFunc
    elif ode_str == 'GAT':
        f = ODEFuncAtt
    elif ode_str == 'transformer':
        f = ODEFuncTransformerAtt
    elif ode_str == 'gcn':
        f = GCNFunc
    else:
        raise FunctionNotDefined
    return f
