function [ ] = convert_to_struct_forest_edges_format( )
%CONVERT_TO_STRUCT_FOREST_EDGES_FORMAT Summary of this function goes here
%   Detailed explanation goes here

dir_path = '/media/data1/work/datasets/CamVid/extracted_training/';
files_to_find = '*_L.png';
imagefiles = dir([dir_path files_to_find]);      
nfiles = length(imagefiles);    % Number of files found

for ii=1:nfiles
   filename = imagefiles(ii).name;
   image_label = uint16(imread(['/media/data1/work/datasets/CamVid/extracted_training/' filename]));
   image_edges = false(size(image_label));
   for i=1:size(image_label,1) 
       for j=1:size(image_label,2) 
           if ((j<size(image_label,2) && image_label(i,j)~=image_label(i,j+1)) ||(i<size(image_label,1) && image_label(i,j)~=image_label(i+1,j))) 
               image_edges(i,j)=true;
           end
       end
   end
   groundTruth = {struct('Segmentation',image_label,'Boundaries',image_edges)};
   var_name = strcat('/media/data1/work/datasets/CamVid/extracted_training/SF_edges_format/',filename(1:end-6));
   save(strcat(var_name,'.mat'), 'groundTruth');
end
 
end

