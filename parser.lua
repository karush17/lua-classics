require 'torch'

function build_parser()
    local cmd = torch.CmdLine()
    -- Hopfield Network args
    cmd:option('-alg', 'HopfieldNetwork', 'algorithm')
    cmd:option('-data', 'mnist', 'dataset (only mnist for now)')
    cmd:option('-samples', 20, 'number of training samples')
    cmd:option('-iter', 50, 'number of iterations')
    cmd:option('-async', true, 'asynchronous updates')
    cmd:option('-pred', 5, 'test samples')
    cmd:option('-log_dir', 'results/', 'logging directory')
    -- Boltzmann Machine args
    cmd:option('-n_visible', 1024, 'visible units')
    cmd:option('-n_hidden', 64, 'hidden units')
    cmd:option('-v_activation', 'relu', 'visible activation')
    cmd:option('-h_activation', 'relu', 'hidden activation')
    cmd:option('-lr', 0.01, 'learning rate')
    cmd:option('-batch_size', 10, 'mini batch size')
    cmd:option('-momentum', {0.5,0.9}, 'momentum')
    cmd:option('-momentum_after', {5}, 'final value of momentum')
    cmd:option('-cd_steps', 1, 'CD iterations')
    cmd:option('-use_states', true, 'use posterior as states')
    local params = cmd:parse(arg)
    return params
end
