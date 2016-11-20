require 'nn' 

--require 'nngraph'
require 'misc.Peek'
--require 'models.MyLookupTable'
require 'hdf5'
--require 'misc.LinearNB
local model3 = {}
function model3.model(opt,embeddings,tfidf)

  vocab_size = opt.vocab_size
  embedding_dim = opt.embedding_dim
  dropout = opt.dropout
  output_size = opt.output_size

  -- tfidf vector
  TfidfTable = nn.LookupTable(vocab_size,1, 0)
  if tfidf ~= nil then 
    TfidfTable.weight = tfidf 
    print('Loaded pretrained embeddings')
  end

  --lookup table
  LU = nn.LookupTable(vocab_size, embedding_dim, 0)
  if embeddings ~= nil then 
    LU.weight = embeddings 
    print('Loaded pretrained embeddings')
  end

  n0 = nn.Sequential()
  n0:add(TfidfTable)
  n0:add(nn.Replicate(embedding_dim,3))
  n0:add(nn.Select(4,-1))
  --n0:add(nn.Peek())

  n00 = nn.Sequential()
  n00:add(LU)
  --n00:add(nn.Peek())

  n1 = nn.ParallelTable()
  n1:add(n0)
  n1:add(n00)
  n2 = nn.Sequential()
  n2:add(n1)
  --n2:add(nn.Peek())
  n2:add(nn.CMulTable())
  if dropout > 0 then n2:add(nn.Dropout(dropout)) end --drop words before averaging
  n2:add(nn.Mean(2))
  --n2:add(nn.Peek())
  --n2:add(nn.Linear(embedding_dim,10000))
  --n2:add(nn.Linear(10000,10000))
  --n2:add(nn.Linear(10000,output_size))
  n2:add(nn.Linear(embedding_dim,output_size))
  n2:add(nn.Tanh())

  return n2
end 
return model3


--[[

x = torch.linspace(1, 3, 3)
M = torch.rand(3,5)

n0 = nn.Sequential()
n0:add(nn.Replicate(5))
n0:add(nn.Transpose({1,2}))
n1 = nn.ParallelTable()
n1:add(n0)
n1:add(nn.Identity()) --lookup
n2 = nn.Sequential()
n2:add(n1)
n2:add(nn.CMulTable())

n2:forward({x,M})

]]