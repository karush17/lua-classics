require 'torch'
require 'nn'
require 'image'
require 'gnuplot'

local HopfieldNetwork = torch.class('HopfieldNetwork')

function HopfieldNetwork:__init() end

function HopfieldNetwork:train_weights(train_data)
    self.num_data = train_data:size(1)
    self.num_neuron = train_data:size(2)
    self.num_pixels = torch.sqrt(self.num_neuron)

    -- initialize weights
    self.W = torch.Tensor(self.num_neuron, self.num_neuron):zero()
    self.rho =  torch.sum(train_data)*(1/(self.num_data*self.num_neuron))

    -- Hebb rule
    for i=1,self.num_data do
        local t = train_data[i] - self.rho
        t = self:unsqueeze(t)
        self.W = self.W:add(torch.mm(t:t(),t))
    end

    -- make diagonal element of W as 0
    local diagW = torch.diag(torch.diag(self.W))
    self.W = (self.W-diagW)/self.num_data
end

function HopfieldNetwork:predict(data, num_iter, threshold, asyn)
    self.num_iter = num_iter
    self.threshold = threshold
    self.asyn = asyn

    copy_data = data:clone()

    preds = torch.Tensor(self.num_data,self.num_neuron)
    print('Starting Updates...')
    for i=1,self.num_data do
        preds[i] = self:_run(copy_data[i])
    end
    print('Updates Finished')
    return preds
end

function HopfieldNetwork:_run(init_s)
    -- synchronous update
    if self.asyn==false then
        local s = init_s
        s = self:unsqueeze(s)
        local e = self:energy(s)
        for i=1,self.num_iter do
            s = torch.sign(torch.mm(s,self.W)-self.threshold)
            local e_new = self:energy(s)
            self:log(i,e_new)
            if e==e_new then
                return s:t()
            end
            e = e_new
        end
        return s:t()
    -- asynchronous update
    else
        local s = init_s
        s = self:unsqueeze(s)
        local e = self:energy(s)
        for i=1,self.num_iter do
            for j=1,self.num_neuron do
                local idx = torch.random(1, self.num_neuron)
                s[1][idx] = torch.sign(torch.mm(self:unsqueeze(self.W[idx]),s:t())-self.threshold)
            end
            local e_new = self:energy(s)
            self:log(i,e_new)
            if e==e_new then
                return s
            end
            e = e_new
        end
        return s
    end
end

function HopfieldNetwork:energy(s)
    return torch.mm(s,torch.mm(self.W, s:t()))*(-0.5) + torch.sum(s*(-self.threshold))
end

function HopfieldNetwork:plot_weights()
    gnuplot.pngfigure('results/hn_weights.png')
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

function HopfieldNetwork:unsqueeze(tens)
    m = nn.Unsqueeze(1)
    return m:forward(tens)
end

function HopfieldNetwork:log(count, en)
    print('Iteration:'..tostring(count)..'/'..tostring(self.num_iter)..'|'..'Energy:'..tostring(en))
end
