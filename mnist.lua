----------------------------------------------------------------------
-- This script downloads and loads the MNIST dataset
-- http://yann.lecun.com/exdb/mnist/
----------------------------------------------------------------------
require 'paths'
require 'torch'

function load_mnist()
   print '==> downloading dataset'

   -- Here we download dataset files. 

   -- Note: files were converted from their original LUSH format
   -- to Torch's internal format.

   -- The SVHN dataset contains 3 files:
   --    + train: training data
   --    + test:  test data

   tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

   if not paths.dirp('mnist.t7') then
      os.execute('wget ' .. tar)
      os.execute('tar xvf ' .. paths.basename(tar))
   end

   train_file = 'mnist.t7/train_32x32.t7'
   test_file = 'mnist.t7/test_32x32.t7'

   ----------------------------------------------------------------------
   print '==> loading dataset'

   -- We load the dataset from disk, it's straightforward

   trainData = torch.load(train_file,'ascii')
   testData = torch.load(test_file,'ascii')

   trainData.data = trainData.data:type(torch.getdefaulttensortype())
   testData.data = testData.data:type(torch.getdefaulttensortype())

   print('Training Data:')
   print(#trainData.data)
   print()

   print('Test Data:')
   print(#testData.data)
   print()

   img_len = trainData.data[1][1]:size(1)
   return trainData, testData, img_len
end

-- load_mnist()