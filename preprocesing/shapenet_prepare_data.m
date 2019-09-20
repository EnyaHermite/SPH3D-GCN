function shapenet_prepare_data(dataDir,writeDir)

% preprocess the point cloud
% remove singular points 

radius = 0.3;
if ~exist(writeDir)
    mkdir(writeDir);
end

[shapenames,folders] = textread(fullfile(dataDir,'synsetoffset2category.txt'),'%s %s\n');
record = fopen(sprintf('record_%02d.txt',floor(radius*10)),'w');

processed = cell(length(folders),1);
check = [];

C = [1,0,0;0,0,1;0,1,1;1,0,1;1,1,0.5;0.5,1,1];
totalParts = 0;
for i = 1:length(folders)
    files = dir(fullfile(dataDir,folders{i},'points','*.pts'));
    if ~exist(fullfile(writeDir,folders{i}))
        mkdir(fullfile(writeDir,folders{i}));
    end
    
    processed{i} = zeros(length(files),1);
    numParts = 0;
    for j = 1:length(files)
        pts_path = fullfile(dataDir,folders{i},'points',files(j).name);
        seg_path = fullfile(dataDir,folders{i},'points_label',strrep(files(j).name,'.pts','.seg'));        
        pt = load(pts_path);        
        label = load(seg_path);
        
        % normalize the points 
        pt = pt - mean(pt);
        scale = sqrt(sum(pt.^2,2));  
        pt = pt/max(scale);
        
        tab = tabulate(label);   
        sz = tab(:,2);
        sz = sz(sz>0);
        check = [check;min(sz)];
        
        for k = 1:numel(sz)
           if sz(k)<=10
               IDX = rangesearch(pt,pt,radius);
               L = zeros(numel(IDX),1);
               for m = 1:numel(IDX)
                   temp = label(IDX{m})==label(m);
                   L(m,1) = sum(temp);
               end
               
               if min(L)==1
                   processed{i}(j) = 1;
               end
               pt(L==1,:) = [];
               label(L==1,:) = [];
               fprintf(record, sprintf('%s/%s: num of singular points #%d\n',shapenames{i},files(j).name(1:end-4),sum(L==1)));                         
               %figure(1);clf;scatter3(pt(:,1),pt(:,2),pt(:,3),30,C(label,:),'filled','MarkerEdgeColor','k')
           else
               continue;
           end
        end
        numParts = max(numParts,max(label));
        
        data = [pt,label,label+totalParts];
        dlmwrite(fullfile(writeDir,folders{i},strrep(files(j).name,'.pts','.txt')),data);
    end
    totalParts = totalParts + numParts; 
end

A = cell2mat(processed);
disp([sum(A==1),sum(check==1)]);
save(sprintf('processed_%02d.mat',floor(radius*10)),'processed');