function scannet_merge(fullDir,voxelDir)

classes = {'other20','wall','floor','cabinet','bed','chair',...
           'sofa','table','door','window','bookshelf',...
           'picture','counter','desk','curtain','refridgerator',...
           'shower curtain','toilet','sink','bathtub','otherfurniture'};
num_cls = length(classes);
labelid_set = [40 1:12 14 16 24 28 33 34 36 39]; % 0 to 40
       
% compute evaluation metric by removing overlapping between blocks
baseDir = pwd;
baseDir(baseDir=='\') == '/';
str = split(pwd,'/');
sph3dgcnDir = join(str(1:end-1),'/');
sph3dgcnDir = sph3dgcnDir{:};

dataFolder = 'scannet-3cm';
which_epoch = 112;
resultFolder = sprintf('test_results_augment_%d_%s',which_epoch,dataFolder);
indexFolder = sprintf('test_block_index_%s',dataFolder);
test_folder = 'test';

scene_names = textread(fullfile(voxelDir,'scannetv2_test.txt'),'%s');
for i = 1:numel(scene_names)
    scene = scene_names{i};
    voxelCloud = load(fullfile(voxelDir,test_folder,strcat(scene,'.txt')));
    fullCloud = load(fullfile(fullDir,test_folder,strcat(scene,'.txt')));
    [IDX, D] = knnsearch(voxelCloud(:,1:3),fullCloud(:,1:3));
    
    predictions = zeros(size(voxelCloud,1),numel(classes));
    
    %% merge the predictions
    pred_files = dir(fullfile(sph3dgcnDir,'log_scannet',resultFolder,sprintf('%s_*.mat',scene)));
    index_files = dir(fullfile(sph3dgcnDir,'log_scannet',indexFolder,sprintf('%s_*.mat',scene)));
    if isempty(pred_files)
        continue;
%         error('scene not found');
    end
    for k = 1:numel(pred_files)
        load(fullfile(pred_files(k).folder,pred_files(k).name));
        load(fullfile(index_files(k).folder,index_files(k).name));

        in_index = data(:,8)==1;
        inner_pt = data(in_index,1:3);
        pred_logits = data(in_index,9:end);
        pred_logits = pred_logits./sqrt(sum(pred_logits.^2,2)); % normlize to unit vector
        pred_logits = exp(pred_logits)./sum(exp(pred_logits),2); % further normlize to probability/confidence

        block2full_index = index(in_index)+1;
        predictions(block2full_index,:) = predictions(block2full_index,:) + pred_logits;
    end
    [~,pred_label] = max(predictions,[],2);
    pred_label = pred_label - 1; 
    pred_label_40 = labelid_set(pred_label+1);
    pred_label_40 = pred_label_40(IDX(:)); % pred_label in the original full point cloud    
   
    figure(1);clf;visualize(fullCloud(:,1:3),pred_label_40)
    figure(2);clf;scatter3(fullCloud(:,1),fullCloud(:,2),fullCloud(:,3),...
                           4,fullCloud(:,4:6)/255,'filled'),hold on 
    testpredFolder = sprintf('test_pred_%d',which_epoch);
    if ~exist(testpredFolder)
        mkdir(testpredFolder);
    end
%     dlmwrite(sprintf('%s/%s.txt',testpredFolder,scene),pred_label_40(:));
end
