function s3dis_merge_6Areas()

baseDir = pwd;
baseDir(baseDir=='\') == '/';
str = split(pwd,'/');
sph3dgcnDir = join(str(1:end-1),'/');
sph3dgcnDir = sph3dgcnDir{:};

total_intersect = zeros(13,1);
total_union = zeros(13,1);
total_seen = zeros(13,1);

merged_correct = 0;
merged_seen = 0;
Area = {'Area_1','Area_2','Area_3','Area_4','Area_5','Area_6'};
for i = 1:6
    disp(sprintf('%s/s3dis_seg/%s_metric.mat',sph3dgcnDir,Area{i}));
    data = load(sprintf('%s/s3dis_seg/%s_metric.mat',sph3dgcnDir,Area{i}));
    merged_correct = merged_correct + data.merged_correct;
    merged_seen = merged_seen + data.merged_seen;
    total_intersect = total_intersect + data.total_intersect;
    total_union = total_union + data.total_union;
    total_seen = total_seen + data.total_seen;
    
    OA = merged_correct./(merged_seen+eps);
    class_iou = total_intersect./(total_union+eps);
    class_acc = total_intersect./(total_seen+eps);
    
    if i==5
        OA_5 = data.merged_correct./(data.merged_seen+eps);
        class_iou_5 = data.total_intersect./(data.total_union+eps);
        class_acc_5 = data.total_intersect./(data.total_seen+eps);
        
        fprintf('==================================(Area_5: OA,mAcc,mIoU)==================================\n')
        fprintf('OA: %.2f%%, mAcc: %.2f%%, mIoU: %.2f%%\n', OA_5*100, mean(class_acc_5)*100, mean(class_iou_5)*100);
        fprintf('=====================================end=====================================\n')
        fprintf('==================================class_iou==================================\n')
        disp(class_iou_5');
        fprintf('=====================================end=====================================\n')
        fprintf('==================================class_acc==================================\n')
        disp(class_acc_5');
        fprintf('=====================================end=====================================\n\n\n');
    end
    
end
fprintf('==================================(OA,mAcc,mIoU)==================================\n')
fprintf('OA: %.2f%%, mAcc: %.2f%%, mIoU: %.2f%%\n', OA*100, mean(class_acc)*100, mean(class_iou)*100);
fprintf('=====================================end=====================================\n')
fprintf('==================================class_iou==================================\n')
disp(class_iou');
fprintf('=====================================end=====================================\n')
fprintf('==================================class_acc==================================\n')
disp(class_acc');
fprintf('=====================================end=====================================\n');