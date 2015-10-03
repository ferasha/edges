function convert_labels_to_colors(labels)
%CONVERT_LABELS_TO_COLORS Summary of this function goes here
%   Detailed explanation goes here

    %saved_var = load('/media/data1/work/results/pred_labels.mat');
    %labels = saved_var.In;
    colors = [[128,64,128]; [128,0,0]; [128,128,128]; [128,128,0]; [0,0,192]; [64,0,128]; [192,192,128]; [192,128,128]; ... 
    [64,64,128]; [64,64,0]; [0,128,192]; [0,0,0]; [64,128,64]; [192,0,128]; [0,128,64]; [64,0,192]; [192,128,64]; ...
    [128,0,192]; [192,0,64]; [128,128,64]; [192,0,192]; [128,64,64]; [64,192,128]; [128,128,192]; [64,128,192]; ...
    [0,0,64]; [0,64,64]; [192,128,192]; [192,192,0]; [64,192,0]];
    
    image_width = size(labels,1);
    image_height = size(labels, 2);
    color_image = zeros(image_width, image_height, 3);
    for i=1:image_width
        for j = 1:image_height
            color_image(i,j,:) = colors(labels(i,j)+1,:,:)/255;
        end
    end
    imwrite(color_image, '/media/data1/work/results/pred.png');
end

