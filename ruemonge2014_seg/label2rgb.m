function rgb = label2rgb(label)


dict = [  0     0   255   % blue
          0   255     0   % green
        128     0   255   % purple
        128   255   255   % cyan: sky
        255     0     0   % red
        255   128     0   % orange
        255   255     0]; % yellow
    
rgb = dict(label+1,:);