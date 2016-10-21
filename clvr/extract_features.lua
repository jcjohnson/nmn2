require 'torch'
require 'nn'
require 'image'
local hdf5 = require 'hdf5'


local cmd = torch.CmdLine()
cmd:option('-image_list', '')
cmd:option('-model', '')
cmd:option('-layer', 30)  -- Last ReLU for VGG-16
cmd:option('-output_h5', '')
cmd:option('-output_h5_dset', '/feats')

cmd:option('image_height', 224)
cmd:option('image_width', 224)
cmd:option('-preprocessing', 'vgg')
cmd:option('-batch_size', 32)
cmd:option('-max_batches', 0)

cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
local opt = cmd:parse(arg)


--[[
ResNet-style image preprocessing.

Input:
- imgs: Tensor of RGB images of shape (N, C, H, W) with pixels in the range [0, 1]

Output:
- out: Tensor of preprocessed RGB images of shape (N, C, H, W)
--]]
local function resnet_preprocess(imgs)
  local mean = imgs.new({0.485, 0.456, 0.406})
  local std = imgs.new({0.229, 0.224, 0.225})

  local N, C = imgs:size(1), imgs:size(2)
  assert(C == 3)
  local H, W = imgs:size(3), imgs:size(4)
  mean = mean:view(1, 3, 1, 1):expand(N, C, H, W)
  std = std:view(1, 3, 1, 1):expand(N, C, H, W)

  return (imgs - mean):cdiv(std)
end


--[[
VGG-style image preprocessing. We need to rescale from [0, 1] to [0, 255],
convert from RGB to BGR, and subtract the mean.

Input:
- imgs: Tensor of RGB images of shape (N, C, H, W) with pixels in the range [0, 1]

Output:
- out: Tensor of preprocessed BGR images of shape (N, C, H, W).
--]]
local function vgg_preprocess(imgs)
  local mean = imgs.new{103.939, 116.779, 123.68}
  local N, C = imgs:size(1), imgs:size(2)
  assert(C == 3)
  local H, W = imgs:size(3), imgs:size(4)
  mean = mean:view(1, 3, 1, 1):expand(N, C, H, W)
  local perm = torch.LongTensor{3, 2, 1}
  return imgs:index(2, perm):mul(255):add(-1, mean)
end


local function run_batch(image_batch_list, model, opt)
  local batch = torch.cat(image_batch_list, 1):type(opt.dtype)
  if opt.preprocessing == 'vgg' then
    batch = vgg_preprocess(batch)
  elseif opt.preprocessing == 'resnet' then
    batch = resnet_preprocess(batch)
  end
  model:forward(batch)
  return model:get(opt.layer).output:float():clone()
end


local function main() 
  -- Set up GPU stuff
  local use_cudnn = false
  local dtype = 'torch.FloatTensor'
  if opt.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpu + 1)
    dtype = 'torch.CudaTensor'
    if opt.use_cudnn == 1 then
      require 'cudnn'
      use_cudnn = true
      cudnn.benchmark = true
    end
  end
  opt.dtype = dtype

  -- Load model
  if opt.model == '' then
    error('Must provide -model')
    return
  end
  local model = torch.load(opt.model)
  model:type(dtype)
  if use_cudnn then
    cudnn.convert(model, cudnn)
  end
  model:evaluate()
  print(model)

  -- Make sure output file was provided
  if opt.output_h5 == '' then
    error('Must provide -output_h5')
  end

  -- Load image list
  if opt.image_list == '' then
    error('Most provide -image_list')
  end
  local num_processed = 0
  local image_batch_list = {}
  local feats_list = {}
  local H, W = opt.image_height, opt.image_width
  for image_path in io.lines(opt.image_list) do
    local img = image.load(image_path, 3)
    img = image.scale(img, W, H, 'bicubic')
    img = img:view(1, 3, H, W)
    table.insert(image_batch_list, img)
    if #image_batch_list == opt.batch_size then
      local feats = run_batch(image_batch_list, model, opt)
      table.insert(feats_list, feats)
      image_batch_list = {}
      num_processed = num_processed + feats:size(1)
      print(string.format('Processed %d images', num_processed))
      if opt.max_batches > 0 and #feats_list == opt.max_batches then
        break
      end
    end
  end
  if #image_batch_list > 0 then
    local feats = run_batch(image_batch_list, model, opt)
    table.insert(feats_list, feats)
  end
  local all_feats = torch.cat(feats_list, 1)
  print(all_feats:type())
  print(#all_feats)

  -- Write features to HDF5 file
  local fout = hdf5.open(opt.output_h5, 'w')
  fout:write(opt.output_h5_dset, all_feats)
end


main()

