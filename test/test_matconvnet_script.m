net = load('retrievalSfM120k-gem-resnet101.mat');
net = dagnn.DagNN.loadobj(net.net) ;
net.mode = 'test' ;
im = imread('oxford_img_1.jpg') ;
im_ = single(im) ; % note: 255 range
im_ = single(im_) - mean(net.meta.normalization.averageImage(:));

index = net.getVarIndex('l2descriptor');
net.vars(index).precious = true;
net.eval({'input', im_})
scores = net.vars(net.getVarIndex('l2descriptor')).value ;
save('l2descriptor.mat', 'scores')