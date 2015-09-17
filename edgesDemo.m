% Demo for Structured Edge Detector (please see readme.txt first).

%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
%opts.modelDir='models/';          % model will be in models/forest
%opts.modelFnm='modelBsds';        % model name
opts.modelDir='/media/data1/work/results/SF_edges/';          % model will be in models/forest
opts.modelFnm='model';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results
%I = imread('peppers.png');
I = imread('/media/data1/work/datasets/CamVid/extracted_testing/Seq05VD_f01860_colors.png');
tic, [E,O,inds,segs]=edgesDetect(I,model); toc
gtWidth = size(segs,1);
imageWidth_S = size(segs,3);
imageHeight_S = size(segs,4);
nTrees = size(segs,5);
nClasses = 255;
stride = 2;
rg = gtWidth/2;
votes = zeros(nClasses ,imageWidth_S*stride, imageHeight_S*stride);
tic,
for t=1:nTrees
    w=1;
    for i=1:imageWidth_S
        h=1;
        for j=1:imageHeight_S
            for p_i=1:gtWidth
                if w-rg+p_i >= 1 && w-rg+p_i <= size(votes,2)
                    for p_j=1:gtWidth
                        if h-rg+p_j >= 1 && h-rg+p_j <= size(votes,3)
                            votes(segs(p_i,p_j,i,j,t)+1,w-rg+p_i, h-rg+p_j) = votes(segs(p_i,p_j,i,j,t)+1,w-rg+p_i, h-rg+p_j) + 1;
                        end
                    end
                end
            end
            h=h+stride;
        end
        w=w+stride;
    end
end
toc
%pred = ones(size(votes,2),size(votes,3));
tic, [M,In] = max(votes,[],1); toc


%figure(1); im(I); figure(2); im(1-E);
