require 'torch'
require 'nn'
require 'image'
require 'gnuplot'
require 'hopfield_net'
require 'boltzmann_machine'
require 'parser'
require 'mnist'
require 'paths'
require 'lfs'

function make_dir(args)
    if paths.dirp(args.log_dir) then
        print('Log Directory already exists')
    else
        lfs.mkdir(args.log_dir)
        print('Log Directory created')
    end
end

function reshape(data)
    local dim = math.floor(math.sqrt(#data))
    local data = torch.Tensor(data)
    local data = data:view(dim,dim)
    return data
end

function preprocessing(img, len)
    local flatten = img / 255
    mean = torch.mean(flatten)
    flatten[flatten:lt(mean)] = 0
    flatten = 2*flatten - 1
    flatten = flatten:view(len*len)
    return flatten
end

function main()
    local args = build_parser()
    make_dir(args)
    local samples = args.samples
    local train_d, test_d, len = load_mnist()
    -- make train dataset
    data = torch.Tensor(samples,len*len)
    for i=1,samples do
        flat = preprocessing(train_d.data[torch.random(1,train_d.data:size(1))][1], len)
        data[i] = flat
    end
    -- train weights
    if args.alg=='HopfieldNetwork' then
        model = HopfieldNetwork()
    else
        model = BoltzmannMachine(args)
    end
    model:train_weights(data)
    -- make test dataset
    test_data = torch.Tensor(samples,len*len)
    for i=1,samples do
        flat = preprocessing(test_d.data[torch.random(1,test_d.data:size(1))][1], len) 
        test_data[i] = flat
    end
    -- predict
    predicted = model:predict(test_data,args.iter,50,args.async)
    -- plot weights
    model:plot_weights()
    model:save_predictions(predicted,args.pred)
end

main()

