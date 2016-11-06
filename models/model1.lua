require 'nn' 
--require 'nngraph'
require 'misc.Peek'
--require 'models.MyLookupTable'
require 'hdf5'
--require 'misc.LinearNB
local model1 = {}
function model1.model(opt)

  -- nn with 2 inputs 2 outputs

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




--[[

  local h0t
  local h1t = text
  for _ =1,num_layers do
    h0t = nn.Linear(feat_size_text, hidden_size)(h1t)
    h1t = nn.Tanh()(h0t)
  end
  local resultText = nn.Linear(hidden_size, output_size)(h1t)

  
  local h0t = nn.Linear(feat_size_text, hidden_size)(text)
  local h1t = nn.Tanh()(h0t)
  local resultText = nn.Linear(hidden_size, output_size)(h1t)
  
  local h0v = nn.Linear(feat_size_visual, hidden_size)(image)
  local h1v = nn.Tanh()(h0v)
  local resultVisual = nn.Linear(hidden_size, output_size)(h1v)
--]]