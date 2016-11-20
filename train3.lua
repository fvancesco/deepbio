require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
local model3 = require 'models.model3'
--local model3 = require 'models.model3-lstms'

-------------------------------------------------------------------------------
-- input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('MM Twitter')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5_text','../../deepbio/dataset/timelines_train_tfidf.h5','path to the user tensors')
cmd:option('-input_h5_text_test','../../deepbio/dataset/timelines_test_tfidf.h5','path to the user tensors - test set')
cmd:option('-lookup_file','../../deepbio/dataset/lookup_tfidf.h5','path to gold factors')
cmd:option('-tfidf_file','../../deepbio/dataset/tfidf.h5','path to gold factors')

cmd:option('-input_h5_factors','../../deepbio/dataset/factors_train.h5','path to gold factors we want to learn')

cmd:option('-embedding_dim',300,'Pretrained Embedding size (dim of the lookup table)')
cmd:option('-vocab_size',200000,'Size of the vcabulary (defined in the script create_tensor_timeline.py)')
cmd:option('-feat_size_text',800,'Max number of words per timeline (defined in the script create_tensor_timeline.py)')
cmd:option('-feat_size_factors',200,'Max number of words per timeline (defined in the script create_tensor_timeline.py)')

--cmd:option('-feat_size_visual',4096,'The number of visual features')

-- Select model
cmd:option('-model','simple','What model to use')
cmd:option('-use_pretrained_lu',1,'Use pretrained embedding or not')
cmd:option('-crit','cosine','What criterion to use (only cosine so far)')
cmd:option('-margin',0.5,'Negative samples margin: L = max(0, cos(x1, x2) - margin)')
cmd:option('-neg_samples',true,'number of negative examples for each good example')
cmd:option('-num_layers', 1, 'number of hidden layers')
cmd:option('-hidden_size',300,'The size of the hidden layer')
cmd:option('-output_size',200,'The  dimension of the output vector')

--LSTM
--cmd:option('-lookupDim', 300, 'Lookup feature dimensionality.') same as embedding_dim
cmd:option('-lookupDropout', 0, 'Lookup feature dimensionality.')
cmd:option('-hiddenSizes', '{64, 64}', 'Hidden size for LSTM.')
cmd:option('-dropouts', '{0.1, 0.1}', 'Dropout on hidden representations.')

--cmd:option('-k',1,'The slope of sigmoid')
--cmd:option('-scale_output',0,'Whether to add a sigmoid at the output of the model')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',30,'what is the batch size in number of images per batch')
cmd:option('-batch_size_real',-1,'TODO REMOVE FROM HERE!!! real value of the batch with the negative examples')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
-- Optimization: for the model
cmd:option('-dropout', -1, 'strength of dropout in the model')
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')

cmd:option('-learning_rate_decay_every', -1, 'every how many iterations LR decay')
cmd:option('-learning_rate',0.0001,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')

cmd:option('-optim_alpha',0.1,'alpha for adagrad|rmsprop|momentum|adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-weight_decay',0,'Weight decay for L2 norm')

-- Evaluation/Checkpointing
cmd:option('-train_size', 21113, 'how many users to use for training set')
cmd:option('-val_size', 1000, 'how many users to use for validation set')
cmd:option('-test_size', 5528, 'how many users to use for the testing (the whole dataset to do some train eval. too)')
cmd:option('-save_checkpoint_every', 10000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'cp/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-output_path', '/home/fbarbieri/deepbio/out/', 'folder to save output vectors')
cmd:option('-save_output', 0.30, 'save if > save_output and > bestCosine')
cmd:option('-beta',1,'beta for f_x')
-- misc
cmd:option('-id', 'idcp', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-print_every',500,'Print some statistics')
cmd:option('-revert_params',-1,'Reverst parameters if you are doing worse on the validation (in the last print_every*batch_size) elements')
cmd:text()

------------------------------------------------------------------------------
-- Basic Torch initializations
------------------------------------------------------------------------------
local opt = cmd:parse(arg)
--opt.id = '_'..opt.model..'_h@'..opt.hidden_size..'_k@'..'_scOut@'..opt.scale_output..'_w@'..opt.weight_decay..'_lr@'..opt.learning_rate..'_dlr@'..opt.learning_rate_decay_every
print(opt)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
  print('Using GPU')
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader
--loader = DataLoader{train_size = opt.train_size, val_size = opt.val_size, json_file = opt.input_json, h5_file_text = opt.input_h5_text, h5_file_visual = opt.input_h5_visual, label_format = opt.crit, feat_size_text = opt.feat_size_text, feat_size_visual = opt.feat_size_visual, gpu = opt.gpuid, output_size = opt.output_size}

loader = DataLoader(opt)
local feat_size_text = loader:getFeatSizeText()
local feat_size_factors = loader:getFeatSizeFactors()
local lu = nil
local tfidf = nil
if opt.use_pretrained_lu > 0  then 
  lu = loader:getLookUp()
  tfidf = loader:getTfidf() 
end
best_sim = 0 --using as global (also needed in save_output)
-------------------------------------------------------------------------------
-- Initialize the network
-------------------------------------------------------------------------------
local protos = {}

--print(string.format('Parameters are model=%s feat_size=%d, output_size=%d\n',opt.model, feat_size,output_size))
-- create protos from scratch
if opt.model == 'simple' then 
  protos.model = model3.model(opt,lu,tfidf)
  --TfidfTable.weight = tfidf
  --LU.weight = lu
  else
    print(string.format('Wrong model:%s',opt.model))
  end

--add criterion
protos.criterion = nn.CosineEmbeddingCriterion(opt.margin)
protos.criterion.sizeAverage = false

-- ship protos to GPU
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
local params, grad_params = protos.model:getParameters()
print('total number of parameters in model: ', params:nElement())
assert(params:nElement() == grad_params:nElement())

collectgarbage()

-------------------------------------------------------------------------------
-- forward test set and save to file
-------------------------------------------------------------------------------
local function save_output_vectors()
  protos.model:evaluate()
  loader:resetIndices('test')

  local maxIter = torch.floor(opt.test_size / 8)
  local test_size_real = maxIter * 8
  -- initialize one table per modality
  text_out = torch.FloatTensor(test_size_real, opt.output_size)

  i = 1
  for _ = 1,maxIter do
    -- forward
    local data = loader:getBatch_test()
    local text_out_batch = protos.model:forward({data.text,data.text})
    for j = 1,8 do
      text_out[i] = text_out_batch[j]:float()
      i = i + 1
    end
  end

  --save h5
  local myFile = hdf5.open(opt.output_path .. 'factors_' .. best_sim .. '.h5', 'w')
  myFile:write('/factors', text_out)
  myFile:close()

end

------------------------------------
-- Evaluate on validation set
------------------------------------
local function eval()
  protos.model:evaluate()
  loader:resetIndices('val')

  local maxIter = torch.floor(opt.val_size / opt.batch_size)
  local val_size_real = maxIter * opt.batch_size
  local val_loss = 0

  -- initialize one table per modality
  if opt.gpuid<0 then
    text_out = torch.FloatTensor(val_size_real, opt.output_size)
    factors_out = torch.FloatTensor(val_size_real, opt.output_size)
  else
    text_out = torch.CudaTensor(val_size_real, opt.output_size)
    factors_out = torch.CudaTensor(val_size_real, opt.output_size)
  end

  i = 1
  for _ = 1,maxIter do
    -- forward
    local data = loader:getBatch('val',neg_samples)
    local text_out_batch = protos.model:forward({data.text,data.text})

    for j = 1,opt.batch_size do
      --print(i .. " - " .. j)
      text_out[i] = text_out_batch[j]
      factors_out[i] = data.factors[j]
      i = i + 1
    end

    local output = {}
    table.insert(output,text_out_batch)
    table.insert(output,data.factors)
    local labels = data.labels
    local loss = protos.criterion:forward(output, labels)
    local batch_loss = protos.criterion:forward(output, labels)
    val_loss = val_loss + batch_loss
  end

  -- normalize
  --text_out = torch.rand(val_size_real, opt.output_size):cuda() --do this if you want to see the random baseline
  local r_norm_text = text_out:norm(2,2)
  text_out:cdiv(r_norm_text:expandAs(text_out))

  local r_norm_factors = factors_out:norm(2,2)
  factors_out:cdiv(r_norm_factors:expandAs(factors_out))

  -- cosine
  local cosine = text_out * factors_out:transpose(1,2)

  -- trace
  local sim = cosine:float():trace() / cosine:size(1) 

  return sim, val_loss/val_size_real --we do not average
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
  protos.model:training() -- some flag, didnt undestand
  grad_params:zero()      -- very important to set deltaW to zero!

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  local data = loader:getBatch('train',neg_samples)

  -- get predicted ({text,visual})
  local output_lm = protos.model:forward({data.text,data.text})

  local output = {}
  table.insert(output,output_lm)
  table.insert(output,data.factors)
  local labels = data.labels
  local loss = protos.criterion:forward(output, labels)
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dpredicted = protos.criterion:backward(output, labels)

  -- backprop to  model
  local dummy = protos.model:backward({data.text,data.text}, dpredicted[1])

  return loss / opt.batch_size
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local iter = 0
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_acc_history = {}
local val_prop_acc_history = {}
local old_params
local checkpoint_path = opt.checkpoint_path .. 'cp_id' .. opt.id ..'.cp'
local learning_rate = opt.learning_rate

local timerTot = torch.Timer()
while true do  

    local timer = torch.Timer()

    local losses = lossFun()

    local time = timer:time().real
    local timeTot = timerTot:time().real  

    -- decay the learning rate
    if iter%opt.learning_rate_decay_every == 0 and opt.learning_rate_decay_every >= 0 then
      --local frac = (iter - opt.learning_rate_decay_every) / opt.learning_rate_decay_every
      --local decay_factor = math.pow(0.5, frac)
      local decay_factor = opt.learning_rate_decay
      learning_rate = learning_rate * decay_factor
      --print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. learning_rate)
    end

    -- perform a parameter update
    if opt.optim == 'rmsprop' then
      rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'adagrad' then
      adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'sgd' then
      sgd(params, grad_params, opt.learning_rate)
    elseif opt.optim == 'sgdm' then
      sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'sgdmom' then
      sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'adam' then
      adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
    else
      error('bad option opt.optim')
    end
    
   
    -- TODO apply normalization after param update
    --if opt.weight_decay >0 then
    --  for _,w in ipairs(reg) do
    --    w:add(-(opt.weight_decay*learning_rate), w)
    --  end
    --end

    -- stopping criterions
    iter = iter + 1
    if iter % 100 == 0 then collectgarbage() end
    if loss0 == nil then loss0 = losses end
    --if losses > loss0 * 20 then
    --  print('loss seems to be exploding, quitting.')
    --  break
    --end
    if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

    -- evaluate validation set
    if iter % opt.print_every == 0 then 
      --local rtext, rvisual, top5t, top5v, sim = eval_ranking()
      --local combined_rank = rtext + rvisual
      local sim, val_loss = eval('val')
      
      if sim > best_sim then
        best_sim = sim

        -- save output vectors (forward test set)
        if opt.save_output > 0 and sim > opt.save_output then --iter > 200 then 
          print('Found better model! Saving h5...')
          save_output_vectors()
        end
      end

      local epoch = iter*opt.batch_size / opt.train_size
      print(string.format('e:%.2f (i:%d) train/val loss: %f/%f sim: %f  bestSim: %f batch/total time: %.4f / %.0f', epoch, iter, losses, val_loss, sim, best_sim, time, timeTot/60)) --eval_split('val')
      print(string.format("lr= %6.4e grad norm = %6.4e, param norm = %6.4e, grad/param norm = %6.4e", learning_rate, grad_params:norm(), params:norm(), grad_params:norm() / params:norm()))
    end

  end
