require 'nn' 
--require 'nngraph'
require 'misc.Peek'
--require 'models.MyLookupTable'
require 'hdf5'
--require 'misc.LinearNB
local model1 = {}
function model1.model(opt,embeddings)

  -- language part
  local language_part = nn.Sequential()
  local pretrainedLU = nn.LookupTable(opt.vocab_size, opt.embedding_dim, 0)
  if embeddings ~= nil then 
    pretrainedLU.weight = embeddings 
    print('Loaded pretrained embeddings')
  end
  language_part:add(pretrainedLU)
  language_part:add(nn.Mean(2))
  --language_part:add(nn.Max(2))
  --language_part:add(nn.Linear(opt.embedding_dim,1000))
  --language_part:add(nn.Tanh())
  --language_part:add(nn.Linear(2000,2000))
  --language_part:add(nn.Tanh())
  --language_part:add(nn.Linear(2000,opt.output_size))
  language_part:add(nn.Linear(opt.embedding_dim,opt.output_size))
  language_part:add(nn.Tanh())
  if opt.dropout > 0 then language_part:add(nn.Dropout(opt.dropout)) end

  return language_part
end 
return model1

--[[
  -- language part
  local language_part = nn.Sequential()
  local pretrainedLU = nn.LookupTable(opt.vocab_size, opt.embedding_dim, 0)
  if embeddings ~= nil then 
    pretrainedLU.weight = embeddings 
    print('Loaded pretrained embeddings')
  end
  language_part:add(pretrainedLU)
  language_part:add(nn.Mean(2))
  --language_part:add(nn.Max(2))
  language_part:add(nn.Linear(opt.embedding_dim,opt.output_size))
  language_part:add(nn.ReLU(true))
  if opt.dropout > 0 then language_part:add(nn.Dropout(opt.dropout)) end

  return language_part
end 
return model1
]]