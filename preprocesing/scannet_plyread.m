function [ptCloud,alpha,label] = my_plyread(filename)
%pcread Read a 3-D point cloud from PLY or PCD file.
%   ptCloud = PCREAD(filename) reads a point cloud from the PCD or PLY file
%   specified by the string filename. If the file is not in the current
%   directory, or in a directory on the MATLAB path, specify the full 
%   pathname. The return value ptCloud is a pointCloud object. 
%
%   Notes
%   -----
%   - PLY or PCD files can contain numerous data entries. pcread loads only
%     the following properties: point locations, colors, normals and
%     intensities.
%
%   Example : Read a point cloud from a PLY file
%   --------------------------------------------
%   ptCloud = pcread('teapot.ply');
%   pcshow(ptCloud)
%
%   See also pointCloud, pcwrite, pcshow
 
%  Copyright 2015-2016 The MathWorks, Inc.

% Validate the input
if isstring(filename)
    filename = char(filename);
end

if ~ischar(filename)
    error(message('vision:pointcloud:badFileName'));
end

% Validate the file type
idx = find(filename == '.');
if (~isempty(idx))
    extension = lower(filename(idx(end)+1:end));
else
    extension = '';
end

% Validate the file extension.
if(~(strcmpi(extension,'pcd') || strcmpi(extension,'ply')))
    error(message('vision:pointcloud:unsupportedFileExtension'));
end

% Verify that the file exists.
fid = fopen(filename, 'r');
if (fid == -1)
    if ~isempty(dir(filename))
        error(message('MATLAB:imagesci:imread:fileReadPermission', filename));
    else
        error(message('MATLAB:imagesci:imread:fileDoesNotExist', filename));
    end
else
    % File exists.  Get full filename.
    filename = fopen(fid);
    fclose(fid);
end

if( strcmpi(extension,'ply') )
    % Read properties of 'Vertex'
    elementName = 'vertex';
    requiredProperties = {'x','y','z'};
    % Alternative names are specified in a cell array within the main cell array.
    optionalProperties = {{'r','red'},{'g','green'},{'b','blue'},'nx','ny','nz','intensity','alpha','label'};
    properties = visionPlyRead(filename,elementName,requiredProperties,optionalProperties);

    % Get location property
    x = properties{1};
    y = properties{2};
    z = properties{3};
    if isa(x,'double') || isa(y,'double') || isa(z,'double')
        loc = [double(x), double(y), double(z)];
    else
        loc = [single(x), single(y), single(z)];
    end

    % Get color property
    r = properties{4};
    g = properties{5};
    b = properties{6};
    color = [im2uint8(r), im2uint8(g), im2uint8(b)];

    % Get normal property
    nx = properties{7};
    ny = properties{8};
    nz = properties{9};
    if isa(nx,'double') || isa(ny,'double') || isa(nz,'double')
        normal = [double(nx), double(ny), double(nz)];
    else
        normal = [single(nx), single(ny), single(nz)];
    end
    
    % Get intensity property
    intensity = properties{10};
    
    alpha = properties{11};
    label = properties{12};
   
    
    ptCloud = pointCloud(loc, 'Color', color, 'Normal', normal, ...
        'Intensity', intensity);
end

