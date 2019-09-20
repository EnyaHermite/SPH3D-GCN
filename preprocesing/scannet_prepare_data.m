function scannet_prepare_data(binRoot,asciiDir)

all_class_names = {'wall','floor','cabinet','bed','chair',...
                   'sofa','table','door','window','bookshelf',...
                   'picture','counter','blinds','desk','shelves',...
                   'curtain','dresser','pillow','mirror','floor mat',...
                   'clothes','ceiling','books','refridgerator','television',...
                   'paper','towel','shower curtain','box','whiteboard',...
                   'person','nightstand','toilet','sink','lamp',...
                   'bathtub','bag','otherstructure','otherfurniture','otherprop'};
subset_labelid = [1:12 14 16 24 28 33 34 36 39];
subset_class_names = all_class_names(subset_labelid);

voxel_size = 3; align = false;
load('labelcolor_labelid');
color_labelId = double(color_labelId(2:41,1:3)); % get labels in the correct range 1~40

% ScanNet data preparation
% rewrite binary PLY files to ASCII PLY files
for phase = ["train","test"]
    phase = char(phase);
    
    if align && strcmp('train',phase)
        phaseFolder = strcat(phase,'-align');
    else
        phaseFolder = phase;
    end
    
    dataDir = sprintf('%s/scannet_%s',binRoot,phase);
    writeDir = sprintf('%s/ScanNet/%s',asciiDir,phaseFolder);
    writeVoxelDir = sprintf('%s/ScanNet-%dcm/%s',asciiDir,voxel_size,phaseFolder);
    if ~exist(writeDir)
        mkdir(writeDir);
    end
    if ~exist(writeVoxelDir)
        mkdir(writeVoxelDir);
    end
    
    scene_set = dir(dataDir);
    scene_set = scene_set(3:end);
    isDirFLAGS = [scene_set.isdir];
    scene_set = scene_set(isDirFLAGS);
    
    scannet_stats.(phase) = ones(length(scene_set),1);
    for i = 1:length(scene_set)
        ptName = sprintf('%s_vh_clean_2.ply',scene_set(i).name);
        pt_path = fullfile(dataDir,scene_set(i).name,ptName);
        Cloud = pcread(pt_path);
        
        if strcmp(phase,'train')        
            labelName = sprintf('%s_vh_clean_2.labels.ply',scene_set(i).name);
            label_path = fullfile(dataDir,scene_set(i).name,labelName);
            [labelCloud,alpha,label] = scannet_plyread(label_path); % alpha is not used, 0 in label means 'unlabelled'          
            
            alignName = strcat(scene_set(i).name,'.txt');
            align_path = fullfile(dataDir,scene_set(i).name,alignName);
            matrix_info = '';
            fid = fopen(align_path,'r');
            while ~feof(fid)
                tline = fgetl(fid);
                if strfind(tline,'axisAlignment')
                    matrix_info = tline;
                    break;
                end
            end
            fclose(fid);
            if ~isempty(matrix_info)
                matrix_info = strrep(matrix_info,'axisAlignment = ','');
                T = sscanf(matrix_info,'%f');
                T = reshape(T,[4,4]);
                T = T';
            end
        end
        
        if strcmp(phase,'train')
            index = (label>=1) & (label<=40); 
            label = label(index);        
            
            %% use the 20 labels for benchmark, others are 0, 
            % there will be 21 classes for network prediction
            new_label = zeros(size(label));
            for k = 1:numel(subset_labelid)
                id = subset_labelid(k);
                new_label(label==id) = k;
            end
            
            label = single(new_label);
            Cloud = pointCloud(Cloud.Location(index,:),'Color',Cloud.Color(index,:));
        end
        xyz = Cloud.Location;
        color = single(Cloud.Color);
        
        if strcmp(phase,'train')
                data = [xyz color label];
            else
                data = [xyz color];
        end 
        dlmwrite(fullfile(writeDir,strcat(scene_set(i).name,'.txt')),data);  
        
        if voxel_size>0
            sampleCloud = pcdownsample(Cloud,'gridAverage',voxel_size/100);
            sample_xyz = sampleCloud.Location;
            color = single(sampleCloud.Color);
            
            if strcmp(phase,'train')
                IDX = knnsearch(xyz,sample_xyz);
                label = label(IDX);
                xyz = sample_xyz;
                data = [xyz color label];
            else
                xyz = sample_xyz;
                data = [xyz color];
            end    
            dlmwrite(fullfile(writeVoxelDir,strcat(scene_set(i).name,'.txt')),data);
        end        
        scannet_stats.(phase)(i) = size(data,1);
    end
end
if voxel_size>0
    stats_name = sprintf('scannet_stats_%dcm.mat',voxel_size);
else
    stats_name = 'scannet_stats.mat';
end
save(stats_name,'scannet_stats');
