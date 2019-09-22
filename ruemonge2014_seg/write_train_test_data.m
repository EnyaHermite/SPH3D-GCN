clc;clear;close all;

dataDir = '/media/huanlei/Data/Datasets/RueMonge2014';

load('pcl_split.mat');
ptCloud.train = pcread('pcl_gt_train.ply');
ptCloud.test = pcread('pcl_gt_test.ply');

isTrain = sum(ptCloud.train.Color,2)>0;
isTest = sum(ptCloud.test.Color,2)>0;

split.train = isTrain;
split.test = isTest;

data = load('pcl.txt');
xyz = data(:,[1,3,2]); % make height in the z-axis
xyz(:,3) = -xyz(:,3); %height (from ground to sky)
normals = data(:,[4,6,5]);
rgb = uint8(data(:,7:9));
fullCloud = pointCloud(xyz,'color',rgb,'normal',normals);

distThresh = 0.3;

plotColor =struct('train','r','test','g');

%% write large training split blocks
IDset = unique(splitLabels);
IDset = IDset(2:end)'; % 0 represents the unlabelled data, we do not use them in the traing and test stages
skipID = []; % store the id of splits which have merged with their small neighbor splits, no need to process separately again
for phase= {'train', 'test'}
    phase = phase{:};
    disp(phase);
    
    if ~exist(fullfile(dataDir,phase))
       mkdir(fullfile(dataDir,phase)); 
    end
    
    figure,
    for i = IDset
        if any(i==skipID)
            continue;
        end
        
        index = splitLabels==i & split.(phase);
        if sum(index)>2000
            pt = fullCloud.Location(index,:);
            color = fullCloud.Color(index,:);
            normal = fullCloud.Normal(index,:);
            label_color = ptCloud.(phase).Color(index,:);            
            gt_label = rgb2label(label_color);      
            fprintf('%s_%d, pt:%d\n',phase,i,size(pt,1));   
            
            scatter3(pt(:,1),pt(:,2),pt(:,3),8,double(label_color)/255,'filled'),hold on
           
            feature = [double(pt) double(color) double(normal) double(gt_label)];
            dlmwrite(fullfile(dataDir,sprintf('%s/%s_%d.txt',phase,phase,i)),feature);       
        else
            if sum(index)>0
                pt = fullCloud.Location(index,:);
                color = fullCloud.Color(index,:);
                normal = fullCloud.Normal(index,:);
                label_color = ptCloud.(phase).Color(index,:);
                gt_label = rgb2label(label_color);  
                
                prev_path = fullfile(dataDir,sprintf('%s/%s_%d.txt',phase,phase,i-1));
                if exist(prev_path)
                    prev_feature = load(prev_path);
                    prev_pt = feature(:,1:3);
                    prev_color = prev_feature(:,4:6);
                    prev_normal = prev_feature(:,7:9);
                    prev_gt_label = prev_feature(:,10);
                    if size(prev_pt,1)>2000
                        [IDX1,D1] = knnsearch(prev_pt,pt);
                        
                        merge_pt = [prev_pt;pt(D1<distThresh,:)];
                        merge_color = [prev_color;color(D1<distThresh,:)];
                        merge_normal = [prev_normal;normal(D1<distThresh,:)];
                        merge_gt_label = [prev_gt_label;gt_label(D1<distThresh)];
                        fprintf('%s_%d, prev_pt: %d, merged_prev_pt: %d.\n',phase,(i-1),size(prev_pt,1),size(merge_pt,1));
                        
                        scatter3(merge_pt(:,1),merge_pt(:,2),merge_pt(:,3),8,label2rgb(merge_gt_label)/255,'filled'), hold on 
                        
                        feature = [double(merge_pt) double(merge_color) double(merge_normal) double(merge_gt_label)];
                        dlmwrite(fullfile(dataDir,sprintf('%s/%s_%d.txt',phase,phase,i-1)),feature);
                    end
                end                
                
                next_index = splitLabels==(i+1) & split.(phase);
                next_pt = fullCloud.Location(next_index,:);
                next_color = fullCloud.Color(next_index,:);
                next_normal = fullCloud.Normal(next_index,:);
                next_label_color = ptCloud.(phase).Color(next_index,:);
                if size(next_pt,1)>2000
                    [IDX2,D2] = knnsearch(next_pt,pt);
                    
                    merge_pt = [pt(D2<distThresh,:);next_pt];
                    merge_color = [color(D2<distThresh,:);next_color];
                    merge_normal = [normal(D2<distThresh,:);next_normal];
                    merge_label_color = [label_color(D2<distThresh,:);next_label_color];
                    merge_gt_label = [gt_label(D2<distThresh);rgb2label(next_label_color)];
                    fprintf('%s_%d, next_pt: %d, merged_next_pt: %d.\n',phase,(i+1),size(next_pt,1),size(merge_pt,1));
                    
                    scatter3(merge_pt(:,1),merge_pt(:,2),merge_pt(:,3),8,label2rgb(merge_gt_label)/255,'filled'), hold on 
                    
                    feature = [double(merge_pt) double(merge_color) double(merge_normal) double(merge_gt_label)];                    
                    dlmwrite(fullfile(dataDir,sprintf('%s/%s_%d.txt',phase,phase,i+1)),feature);
                    
                    skipID = [skipID, i+1];
                end
            end
        end
    end
    xlabel('x'),
    ylabel('y')
end



