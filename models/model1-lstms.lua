require 'rnn' 
--require 'nngraph'
require 'misc.Peek'
--require 'models.MyLookupTable'
require 'hdf5'
--require 'misc.LinearNB
local model1 = {}
function model1.model(opt, vocab_size, embedding_dim, feat_size_visual, num_layers, hidden_size, output_size, dropout, embeddings)

  -- Language part
  lookupDim = embedding_dim
  lookupDropout = tonumber(opt.lookupDropout)
  hiddenSizes = loadstring(" return " .. opt.hiddenSizes)()
  dropouts = loadstring(" return " .. opt.dropouts)()

  language_part = nn.Sequential()

  -- Transpose, such that input is seqLen x batchSize
  language_part:add(nn.Transpose({1,2}))

  -- LookupTable
  local pretrainedLU = nn.LookupTable(vocab_size, embedding_dim, 0)
  if embeddings ~= nil then 
    pretrainedLU.weight = embeddings 
    print('Loaded pretrained embeddings')
  end
  language_part:add(pretrainedLU)

  --local lookup = nn.LookupTableMaskZero(vocab_size, lookupDim)
  --language_part:add(lookup)
  if lookupDropout ~= 0 then language_part:add(nn.Dropout(lookupDropout)) end
  
  -- Recurrent layers
  local inputSize = lookupDim
  for i, hiddenSize in ipairs(hiddenSizes) do
    local rnn = nn.SeqLSTM(inputSize, hiddenSize) --SeqBRNN
    rnn.maskzero = true
    language_part:add(rnn)
    if dropouts[i] ~= 0 and dropouts[i] ~= nil then
      language_part:add(nn.Dropout(dropouts[i]))
    end
    inputSize = hiddenSize 
  end
  language_part:add(nn.Select(1, -1))
  language_part:add(nn.Linear(inputSize,output_size))
  language_part:add(nn.ReLU(true))

  -- visual part
  local visual_part = nn.Sequential()
  visual_part:add(nn.Linear(feat_size_visual,output_size))
  visual_part:add(nn.ReLU(true))

  -------- MM NET ---------
  -- 2 input 2 output
  local net = nn.ParallelTable()
  net:add(language_part)
  net:add(visual_part)
  --net:add(nn.Peek()) --PEEEEEEEK

  return net
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