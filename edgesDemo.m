% Demo for Structured Edge Detector (please see readme.txt first).

%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
%opts.modelDir='models/';          % model will be in models/forest
%opts.modelFnm='modelBsds';        % model name
opts.modelDir='/media/data1/work/results/SF_edges_k_30_matrix_z_1_cluster/';          % model will be in models/forest
opts.modelFnm='model';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=0;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=7;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results
%I = imread('peppers.png');
I = imread('/media/data1/work/datasets/CamVid/extracted_testing/Seq05VD_f01860_colors.png');
%I = imread('/media/data1/work/datasets/CamVid/extracted_training/0001TP_006690_colors.png');
%I = imread('/media/data1/work/datasets/CamVid/extracted_testing/0001TP_008550_colors.png');

tic, [E,O,inds,segs, votes]=edgesDetect(I,model); toc

% tic, [E,O,inds,segs]=edgesDetect(I,model); toc
% gtWidth = size(segs,1);
% imageWidth_S = size(segs,3);
% imageHeight_S = size(segs,4);
% nTrees = size(segs,5);
% nClasses = 30;
% stride = 2;
% rg = gtWidth/2;
% votes = zeros(nClasses ,imageWidth_S*stride, imageHeight_S*stride);
% % tic,
% % for t=1:nTrees
% %     w=1;
% %     for i=1:100%imageWidth_S
% %         h=1;
% %         for j=1:100%imageHeight_S
% %             for p_i=1:gtWidth
% %                 if w-rg+p_i >= 1 && w-rg+p_i <= size(votes,2)
% %                     for p_j=1:gtWidth
% %                         if h-rg+p_j >= 1 && h-rg+p_j <= size(votes,3)
% %                             votes(segs(p_i,p_j,i,j,t)+1,w-rg+p_i, h-rg+p_j) = votes(segs(p_i,p_j,i,j,t)+1,w-rg+p_i, h-rg+p_j) + 1;
% %                         end
% %                     end
% %                 end
% %             end
% %             h=h+stride;
% %         end
% %         w=w+stride;
% %     end
% % end
% % toc
% 
% 
% tic,
% p_size = gtWidth*gtWidth*nTrees;
% 
% %for t=1:nTrees
%     w=1;
%     for i=1:imageWidth_S
%         h=1;
%         if (w>=rg) %(w-rg>=1) %(w>=rg)
%             start_w = w-rg+1; %w-rg; %w-rg+1;
%             start_u = 1;
%         else
%             start_w = 1;
%             start_u = gtWidth-rg-w+1; %rg-w+1;
%         end
%         if (w+rg<=size(votes,2))
%             end_w = w+rg;
%             end_u = gtWidth;
%         else
%             end_w = size(votes,2);
%             end_u = gtWidth-(w+rg-size(votes,2)); %size(votes,2)-w+rg;
%         end
%         for j=1:imageHeight_S
%             if (h>=rg) %(h-rg>=1); %(h>=rg)
%                 start_h = h-rg+1; %h-rg; %h-rg+1;  %h-rg+1;
%                 start_v = 1; %1;
%             else
%                 start_h = 1; %h;
%                 start_v = gtWidth-rg-h+1; %rg;
%             end
%             if (h+rg<=size(votes,3))
%                 end_h = h+rg; %h+rg;
%                 end_v = gtWidth;  %gtWidth;
%             else
%                 end_h = size(votes,3); %h;
%                 end_v = gtWidth-(h+rg-size(votes,3)); %rg;
%             end    
%             hist = zeros(nClasses, p_size);
%             x = double(segs(:,:,i,j,:))+1;
%             hist(sub2ind(size(hist),reshape(x,1,p_size),1:p_size)) = 1;
%             hist = reshape(hist,nClasses, gtWidth,gtWidth, nTrees);  
%             hist = sum(hist,4);
%             
%             bhist = hist(:,start_u:end_u,start_v:end_v);
%             votes(:,start_w:end_w, start_h:end_h) = votes(:,start_w:end_w, start_h:end_h) + bhist;
%             h=h+stride;
%         end
%         w=w+stride;
%     end
% %end
% toc

ignored_labels = [11;29];
confusion_matrix = zeros(30,30);

[~,predicted] = max(votes,[],1);
predicted = squeeze(predicted)-1;
groundtruth_file = load('/media/data1/work/datasets/CamVid/SF_edges/testing/Seq05VD_f01860_colors.mat');
gt = double(groundtruth_file.groundTruth{1}.Segmentation);
mask = ~ismember(gt,ignored_labels);
idx = find(mask);
accuracy = sum(sum(gt(idx)==predicted(idx)))/length(idx)

c_idx = sub2ind(size(confusion_matrix),gt+1,predicted+1);
classes_hist = histc(c_idx(:),1:900);
confusion_matrix(:) = confusion_matrix(:) + h;

save('/media/data1/work/results/pred_labels.mat','predicted');
convert_labels_to_colors(predicted);


%figure(1); im(I); figure(2); im(1-E);
