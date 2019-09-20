function label = rgb2label(rgb)

rgb = uint8(rgb);
dict = [  0     0   255   % blue
          0   255     0   % green
        128     0   255   % purple
        128   255   255   % cyan: sky
        255     0     0   % red
        255   128     0   % orange
        255   255     0]; % yellow
dict = uint8(dict);
    
for i = 1:size(rgb,1)
    idx = [];
    for j = 1:size(dict,1)
       if isequal(rgb(i,:),dict(j,:)) 
           idx = j;
           break;
       end
    end
    if isempty(idx)
        error('label not found!');
    else
        label(i,1) = idx-1;
    end
end