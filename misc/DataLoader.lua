require 'hdf5'
require 'string'

local utils = require 'misc.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
  -- store settings locally
  self.batch_size = opt.batch_size
  self.gpuid = opt.gpuid
  self.train_size = opt.train_size
  self.val_size = opt.val_size
  self.test_size = opt.test_size
  self.feat_size_text = opt.feat_size_text
  self.h5_file_text_test = opt.h5_file_text_test
  self.feat_size_factors = opt.feat_size_factors

  -- text
  -- convert dataset in a tensor, where each row includes the indices 
  -- of the words of the timeline of each user
  print('DataLoader loading h5 file (timelines): ', opt.input_h5_text)
  self.d_txt  = "/timelines"
  self.h5_file_text = hdf5.open(opt.input_h5_text, 'r')
  print(self.h5_file_text:read(self.d_txt):dataspaceSize())

  print('DataLoader loading h5 file (factors): ', opt.input_h5_factors)
  self.d_im  = "/factors"
  self.h5_file_factors = hdf5.open(opt.input_h5_factors, 'r')
  print(self.h5_file_factors:read(self.d_im):dataspaceSize())

  --test set
  print('DataLoader loading h5 file (timelines) -TEST: ', opt.input_h5_text_test)
  self.d_txt  = "/timelines"
  self.h5_file_text_test = hdf5.open(opt.input_h5_text_test, 'r')
  print(self.h5_file_text_test:read(self.d_txt):dataspaceSize())

  -- TODO some sanity checks

  --Initialize indexes 
  self.split_ix = {}
  self.iterators = {}

  self:resetIndices('train')
  self:resetIndices('val')
  self:resetIndices('test')

  print('Assigned to train: ', (#self.split_ix['train'])[1])
  print('Assigned to val: ', (#self.split_ix['val'])[1])
  print('Assigned to test: ', (#self.split_ix['test'])[1])

  --pretrained lookup table
  if opt.use_pretrained_lu > 0 then
    local preFile = hdf5.open(opt.lookup_file, 'r')
    self.lookup = preFile:read('/lookup'):all()
    print('Loaded lookup: ')
    print(self.lookup:size())

    if opt.tfidf_file ~= nil then --NOTSURRRREEEEEEE TODO, maybe move to train3.lua so we dont need new dataloader
      preFile = hdf5.open(opt.tfidf_file, 'r')
      self.tfidf = preFile:read('/lookup'):all()
      print('Loaded tfidf: ')
      print(self.tfidf:size())
    end
  end

end

function DataLoader:resetIndices(split)
  -- all the indices referes to the original tables (hd5f)
  local gap

  if split == 'train' then
    self.split_ix[split] = torch.randperm(self.train_size)
    elseif split == 'val' then
      gap = torch.Tensor(self.val_size):fill(self.train_size)
      self.split_ix[split] = torch.randperm(self.val_size):add(1, gap)
      elseif split == 'test' then
        --gap = torch.Tensor(self.test_size):fill(self.train_size+self.val_size)
        --self.split_ix[split] = torch(self.test_size):add(1, gap)
        self.split_ix[split] = torch.Tensor():range(1,self.test_size)
      else
        error('error: unknown split - ' .. split)
  end

    self.iterators[split] = 1
end

function DataLoader:getFeatSizeText()
  return self.feat_size_text
end

function DataLoader:getFeatSizeFactors()
  return self.feat_size_factors
end

function DataLoader:getLookUp()
  return self.lookup
end

function DataLoader:getTfidf()
  return self.tfidf
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - ...
  - ...
--]]
function DataLoader:getBatch(split,neg_samples)
  
  local split_ix = self.split_ix[split]
  local batch_size = self.batch_size

  -- initialize one table per modality
  local text_batch = torch.FloatTensor(batch_size, self.feat_size_text):fill(0)
  local factors_batch = torch.FloatTensor(batch_size, self.feat_size_factors):fill(0)
  local labels_batch = torch.FloatTensor(batch_size):fill(1)

  local max_index = (#split_ix)[1]

  --if you are going to overflow, reset the indices 
  --(i know we are not processing some examples at the end, but they are random every time so it shouldn't harm)
  future_index = self.iterators[split] + batch_size
  if future_index >= max_index then
    self:resetIndices(split)
  end

  local si = self.iterators[split] -- use si for semplicity but update the self.iterator later
  local i = 1
  local loop_size = batch_size
  if neg_samples then loop_size = batch_size / 2 end --as we add 1 negative sample each good example

  for _ = 1,loop_size do
    ix = split_ix[si]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. si)

    -- read text and factors vectors (read line at position ix)
    text_batch[i] = self.h5_file_text:read(self.d_txt):partial({ix,ix},{1,self.feat_size_text})
    factors_batch[i] = self.h5_file_factors:read(self.d_im):partial({ix,ix},{1,self.feat_size_factors})
    labels_batch[i] = 1
    i = i+1

    -- negative examples
    if neg_samples then
      --get random example (verify it's not the same as the positive)
      local rand_ix = ix
      while rand_ix == ix do
        rand_ix = math.random(self.train_size) 
      end

      -- add negative sample
      text_batch[i] = self.h5_file_text:read(self.d_txt):partial({ix,ix},{1,self.feat_size_text})
      factors_batch[i] = self.h5_file_factors:read(self.d_im):partial({rand_ix,rand_ix},{1,self.feat_size_factors})
      labels_batch[i] = 1
      i = i+1
    end

    si = si + 1
  end
  self.iterators[split] = si

  local data = {}
  if self.gpuid<0 then
    data.text = text_batch
    data.factors = factors_batch
    data.labels = labels_batch
  else
    data.text = text_batch:cuda()
    data.factors = factors_batch:cuda()
    data.labels = labels_batch:cuda()
  end

  return data
end

  --------------------- different, we dont have factors here (clearer this way now)
  function DataLoader:getBatch_test()
  
  split = 'test'
  local split_ix = self.split_ix[split]

  local batch_size = 8 --5528 then 8 max possible batch

  -- initialize one table per modality
  local text_batch = torch.FloatTensor(batch_size, self.feat_size_text):fill(0)
  local factors_batch = torch.FloatTensor(batch_size, self.feat_size_factors):fill(0)
  local labels_batch = torch.FloatTensor(batch_size):fill(1)

  local max_index = (#split_ix)[1]

  --if you are going to overflow, reset the indices 
  --(i know we are not processing some examples at the end, but they are random every time so it shouldn't harm)
  --future_index = self.iterators[split] + batch_size
  --if future_index >= max_index then
  --  self:resetIndices(split)
  --end

  local si = self.iterators[split] -- use si for semplicity but update the self.iterator later
  for i = 1,batch_size do
    ix = split_ix[si]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. si)
    -- positive examples

    -- read text and factors vectors (read line at position ix)
    text_batch[i] = self.h5_file_text_test:read(self.d_txt):partial({ix,ix},{1,self.feat_size_text})
    si = si + 1
  end
  self.iterators[split] = si

  local data = {}
  if self.gpuid<0 then
    data.text = text_batch
    data.factors = factors_batch
    data.labels = labels_batch
  else
    data.text = text_batch:cuda()
    data.factors = factors_batch:cuda()
    data.labels = labels_batch:cuda()
  end
  return data
end
