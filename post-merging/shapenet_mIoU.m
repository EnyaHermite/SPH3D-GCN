function shapenet_mIoU(dataDir,resultDir)

[shapenames,folders,numparts,cumtotals] = textread(fullfile(dataDir,'class_info_all.txt'),'%s %s %d %d\n');
for k = 1:numel(folders)
    field = shapenames{k};
    dict.(field) = [numparts(k) cumtotals(k)];    
    shape_ious.(field) = [];
    dict_names.(field) = shapenames{k};
end
    
folders = dir(resultDir);
folders = folders(3:end);
for k = 1:length(folders)
    shape_folder = folders(k).name;
    str = split(shape_folder,'_');
    fname = str{1};
    files = dir(fullfile(resultDir,shape_folder,'*.txt'));
    for i = 1:numel(files)        
        respath = fullfile(files(i).folder,files(i).name);
        res = load(respath);
        pred = res(:,1);
        gt = res(:,2);
        
        field = fname;
        IoU = evaluateIoU(pred,gt,dict.(field)(1),0);
     
        shape_ious.(fname) = [shape_ious.(fname);IoU'];
    end
end

IoU_all = [];
IoU_separate = [];
for k = 1:numel(folders)
    shape_folder = folders(k).name;
    str = split(shape_folder,'_');
    fname = str{1};
    class_IoU = mean(shape_ious.(fname),2);
    IoU_all = [IoU_all;class_IoU];
    fprintf('%s: %.2f%%\n',fname,mean(class_IoU)*100);
    IoU_separate(k) = mean(class_IoU);
end
fprintf('total: %.2f%%\n',mean(IoU_all)*100);
fprintf('mean: %.2f%%\n',mean(IoU_separate)*100);
