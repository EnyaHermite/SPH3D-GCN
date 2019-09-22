function IoU = evaluateIoU(pred,gt,numparts,addnum)

for l=1:numparts
    label = l+addnum-1;
    intersect = (pred==label & gt==label);
    union = (pred==label | gt==label);
    if sum(union)==0
        IoU(l,1) = 1.0;
    else
        IoU(l,1) = sum(intersect)/(sum(union)+eps);
    end
end