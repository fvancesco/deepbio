require 'nn' 
--require 'nngraph'
require 'misc.Peek'
--require 'models.MyLookupTable'
require 'hdf5'
--require 'misc.LinearNB
local model2 = {}
function model2.model(opt,embeddings1,embeddings2)

  -- mean
  local net1 = nn.Sequential()
  local pretrainedLU1 = nn.LookupTable(opt.vocab_size, opt.embedding_dim, 0)
  if embeddings ~= nil then 
    pretrainedLU1.weight = embeddings1 
    print('Loaded pretrained embeddings mean')
  end
  net1:add(pretrainedLU1)
  net1:add(nn.Mean(2))
  --net1:add(nn.Linear(opt.embedding_dim,opt.output_size))
  net1:add(nn.Tanh())
  if opt.dropout > 0 then net1:add(nn.Dropout(opt.dropout)) end

  -- max
  local net2 = nn.Sequential()
  local pretrainedLU2 = nn.LookupTable(opt.vocab_size, opt.embedding_dim, 0)
  if embeddings ~= nil then 
    pretrainedLU2.weight = embeddings2
    print('Loaded pretrained embeddings max')
  end
  net2:add(pretrainedLU2)
  net2:add(nn.Max(2))
  --net2:add(nn.Linear(opt.embedding_dim,opt.output_size))
  net2:add(nn.Tanh())
  if opt.dropout > 0 then net2:add(nn.Dropout(opt.dropout)) end

  -------- Put mean and max together ---------
  -- 2 input 2 output
  local net = nn.Concat(2);
  net:add(net1)
  net:add(net2)

  local netout = nn.Sequential()
  netout:add(net)
  netout:add(nn.Linear(opt.embedding_dim*2,opt.output_size))
  return netout

end 

return model2


--[[
  -- language part
  local net1 = nn.Sequential()
  local pretrainedLU = nn.LookupTable(opt.vocab_size, opt.embedding_dim, 0)
  if embeddings ~= nil then 
    pretrainedLU.weight = embeddings 
    print('Loaded pretrained embeddings')
  end
  net1:add(pretrainedLU)
  net1:add(nn.Mean(2))
  --net1:add(nn.Max(2))
  net1:add(nn.Linear(opt.embedding_dim,opt.output_size))
  net1:add(nn.ReLU(true))
  if opt.dropout > 0 then net1:add(nn.Dropout(opt.dropout)) end

  return net1
end 
return model2
]]