function [X,y,meta] = load_data(dataPath,cfg,requireLabels)
% Strict training filenames: man_PERSON_PHOTO or woman_PERSON_PHOTO.
% Unknown test filenames are permitted; labels are then NaN.
if nargin<3, requireLabels=true; end
assert(exist(dataPath,'dir')==7,'Dataset directory does not exist.');
ds=imageDatastore(dataPath,'IncludeSubfolders',true, ...
    'FileExtensions',{'.png','.jpg','.jpeg','.bmp','.tif','.tiff'});
paths=sort(ds.Files); N=numel(paths);
assert(N>0,'No supported images found.');
X=zeros(N,prod(cfg.image_size)); y=nan(N,1); persons=cell(N,1);
for i=1:N
    [~,name]=fileparts(paths{i});
    token=regexp(lower(name),'^(man|male|woman|female)_(\d+)_(\d+)$','tokens','once');
    if isempty(token)
        assert(~requireLabels,'Cannot parse training filename: %s',paths{i});
        persons{i}='';
    else
        female=ismember(token{1},{'woman','female'}); y(i)=double(female);
        if female, prefix='female'; else, prefix='male'; end
        persons{i}=sprintf('%s_%d',prefix,str2double(token{2}));
    end
    [I,map]=imread(paths{i});
    if ~isempty(map), I=ind2gray(I,map); end
    if ndims(I)==3
        assert(size(I,3)==3,'Unsupported channel count: %s',paths{i});
        I=rgb2gray(I);
    end
    if ~isequal(size(I),cfg.image_size)
        if cfg.resize
            I=imresize(I,cfg.image_size);
        else
            error('Expected %dx%d but found %dx%d: %s',cfg.image_size(1), ...
                cfg.image_size(2),size(I,1),size(I,2),paths{i});
        end
    end
    I=im2double(I);
    assert(all(isfinite(I(:))) && all(I(:)>=0 & I(:)<=1),'Invalid pixel values.');
    X(i,:)=I(:)';
    if mod(i,25)==0 || i==N, fprintf('Loaded %d / %d\n',i,N); end
end
if requireLabels, assert(isequal(unique(y),[0;1]),'Both classes are required.'); end
meta.files=paths; meta.person_id=persons; meta.image_size=cfg.image_size;
end
