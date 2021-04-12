require 'torch'
require 'nn'
require 'image'
require 'gnuplot'

local BoltzmannMachine = torch.class('BoltzmannMachine')

function BoltzmannMachine:__init(arg)
    self.args = arg
    self.n_visible = arg.n_visible
    self.n_hidden = arg.n_hidden
    -- unit type
    self.v_activation = arg.v_activation
    self.h_activation = arg.v_activation
    self.useStates = arg.use_states
    self.epochs = arg.iter
    -- learning
    self.learningRate = arg.lr
    self.minibatchSize = arg.batch_size
    self.momentum = arg.momemtum
    self.momentumAfter = arg.momentum_after
    self.CDsteps = arg.cd_steps
    -- initialize weights
    nv, nh = self.n_visible, self.n_hidden
    self.W = torch.Tensor(nv, nh):randn(nv, nh):mul(0.1)
    self.h_bias = torch.Tensor(1,nh):zeros(1,nh)
    self.v_bias = torch.Tensor(1,nv):zeros(1,nv)
    self.W_inc = torch.Tensor(nv,nh):zeros(nv, nh)
    -- setup samplers
    self.binarySampler = function(input)
        local a = 1 / (1 + torch.exp(-input))
        -- local a = nn.Sigmoid()(input)
        local s = torch.gt(a, torch.Tensor(a:size()):rand(a:size())):type(torch.getdefaulttensortype())
        return a, s
    end
    self.reluSampler = function(input)
        -- local n = torch.Tensor(input:size()):randn(input:size())
        local a = input
        a[a < 0] = 0
        -- local a = nn.ReLU()(input)
        return a, a
    end
    self.gaussSampler = function(input)
        return input, input+torch.Tensor(input:size()):randn(input:size())
    end
    -- setup hidden and visible samplers for chains
    if self.h_activation=='binary' then
        self.h_sampler = self.binarySampler
    elseif self.h_activation=='relu' then
        self.h_sampler = self.reluSampler
    else
        self.h_sampler = self.gaussSampler
    end
    if self.v_activation=='binary' then
        self.v_sampler = self.binarySampler
    elseif self.v_activation=='relu' then
        self.v_sampler = self.reluSampler
    else
        self.v_sampler = self.gaussSampler
    end
    -- get encoder decoder
    self.encoder, self.decoder = self:getNN()
end

function BoltzmannMachine:HgivenV(v_sample)
    local pre, post, states
    pre = torch.mm(v_sample, self.W):add(self.h_bias:repeatTensor(v_sample:size(1),1))
    post, states = self:h_sampler(pre)
    if self.useStates == false then
        states = post
    end
    return pre, post, states
end

function BoltzmannMachine:VgivenH(h_sample)
    local pre, post, states
    pre = torch.mm(h_sample, self.W:t()):add(self.v_bias:repeatTensor(h_sample:size(1),1))
    post, states = self:v_sampler(pre)
    if self.useStates == false then
        states = post
    end
    return pre, post, states
end

function BoltzmannMachine:sampleChain(h)
    start = h
    local v_mean, v_sample, h_mean, h_sample
    for i=1, self.CDsteps+1 do
        _, v_mean, v_sample = self:VgivenH(start)
        _, h_mean, h_sample = self:HgivenV(v_sample)
        start = h_sample
    end
    return v_mean, v_sample, h_mean, h_sample
end

function BoltzmannMachine:freeEnergy(sample)
    local wx_b = torch.mm(sample, self.W):add(self.h_bias:repeatTensor(sample:size(1),1))
    local vbias_term = torch.mm(sample, self.vbias:t())
    local hidden_term = torch.log(torch.add(wx_b:exp(),1)):sum(2)
    local e = -hidden_term-vbias_term
    return e
end

function BoltzmannMachine:getNN()
    local encoder, decoder
    -- construct encoder
    encoder = nn.Sequential()
    encoder:add(nn.Linear(self.n_visible, self.n_hidden))
    if self.h_activation=='binary' then
        encoder:add(nn.Sigmoid())
    elseif self.h_activation=='relu' then
        encoder:add(nn.ReLU())
    end
    encoder:get(1).weight = self.W:t()
    encoder:get(1).bias = self.h_bias[1]
    -- construct decoder
    decoder = nn.Sequential()
    decoder:add(nn.Linear(self.n_hidden, self.n_visible))
    if self.v_activation=='binary' then
        decoder:add(nn.Sigmoid())
    elseif self.v_activation=='relu' then
        decoder:add(nn.ReLU())
    end
    decoder:get(1).weight = self.W:t()
    decoder:get(1).bias = self.v_bias[1]
    return encoder, decoder
end

function BoltzmannMachine:fromNN(encoder, decoder)
    self.W = encoder.get(1).weight:t()
    self.h_bias[1] = encoder.get(1).bias
    self.h_activation = encoder.get(2)
    self.v_bias[1] = decoder.get(1).bias
    self.v_activation = decoder.get(2)
end

function BoltzmannMachine:update_params(v0)
    -- use momentum
    local momentum
    if self.momentum then
        momentum = self.momentum[1]
    end
    if self.momentum and self.epochs > self.momentumAfter then
        momentum = self.momentum[2]
    end
    -- sample first hidden layer
    local _, h0_mean, h0_sample = self:HgivenV(v0)
    local v_model_mean, v_model_sample, h_model_mean, h_model_sample = self:sampleChain(h0_sample)

    if momentum then
        local ww = self.Winc:clone()
        local vb = self.vbias:clone()
        local hb = self.hbias:clone()
    end

    -- calculate derivatives
    self.Winc = torch.mm(v0:t(), h0_mean)
    self.Winc:add(torch.mm(v_model_mean:t(), h_model_mean):mul(-1))
    self.Winc:div(v0:size(1))
    self.Winc:mul(self.learningRate)

    self.vbiasinc = v0:sum(1)
    self.vbiasinc:add(-v_model_mean:sum(1))
    self.vbiasinc:mul(self.learningRate)
    self.vbiasinc:div(v0:size(1))

    self.hbiasinc = h0_mean:sum(1)
    self.hbiasinc:add(-h_model_mean:sum(1))
    self.hbiasinc:mul(self.learningRate)
    self.hbiasinc:div(v0:size(1))

    if self.momentum and self.epochs > 1 then
        self.Winc:add(torch.mul(ww, momentum))
        self.vbiasinc:add(torch.mul(vb, momentum))
        self.hbiasinc:add(torch.mul(hb, momentum))
    end

    self.W:add(self.Winc)
    self.vbias:add(self.vbiasinc)
    self.hbias:add(self.hbiasinc)

    local v_w = torch.mean(v0:mul(self.W))
    local b_v = torch.mean(v0:mul(self.vbias))
    local v_w_h = torch.mean(h0_sample:mul(v_w))
    local c_h = torch.mean(h0_sample:mul(self.hbias))
    local free_energy = - b_v - c_h - v_w_h
    return free_energy
end

function BoltzmannMachine:train_weights(train_data)
    self.num_pixels = torch.sqrt(train_data:size(2))
    print('Starting Updates...')
    local e
    for e=1,self.epochs do
        local cd = 0
        for i=1,data:size(1), self.minibatchSize do
            local div = self:update_params(data[{{i,i+self.minibatchSize-1}, {}}])
            cd  = cd + div
        end
        collectgrabage()
        self.log(e, cd)
    end
    print('Updates Finished')
end

function BoltzmannMachine:predict(test_data, num_iter, threshold, asyn)
    mlp = nn.Sequential()
    mlp:add(self.encoder)
    mlp:add(self.decoder)
    v = mlp:forward(test_data)
    return v
end

function BoltzmannMachine:plot_weights()
    gnuplot.pngfigure('results/bm_weights.png')
    gnuplot.title('Network Weights')
    gnuplot.imagesc(self.W, 'color')
    gnuplot.plotflush()
    print('Weights Saved!')
end

function HopfieldNetwork:save_predictions(preds,num_saves)
    for j=1,num_saves do
        gnuplot.pngfigure('results/pred-'..tostring(j)..'.png')
        gnuplot.title('prediction-'..tostring(j))
        gnuplot.imagesc(preds[j]:view(self.num_pixels, self.num_pixels))
        gnuplot.plotflush()
    end
    print('Predictions Saved!')
end

function BoltzmannMachine:log(count, en)
    print('Iteration:'..tostring(count)..'/'..tostring(self.epochs)..'|'..'Energy:'..tostring(en))
end


