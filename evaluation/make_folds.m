function foldID=make_folds(y,personID,cfg)
rng(cfg.seed,'twister'); k=cfg.num_folds;
assert(k>=2 && k==floor(k),'Invalid fold count.');
if strcmp(cfg.mode,'person_grouped')
    [units,~,group]=unique(personID); unitY=zeros(numel(units),1);
    for g=1:numel(units)
        assert(~isempty(units{g}),'Missing person ID.');
        v=unique(y(group==g)); assert(numel(v)==1,'Conflicting labels for a person.');
        unitY(g)=v;
    end
elseif strcmp(cfg.mode,'image_stratified')
    group=(1:numel(y))'; unitY=y;
else
    error('Unknown CV mode.');
end
unitFold=zeros(numel(unitY),1);
for c=0:1
    ix=find(unitY==c); assert(numel(ix)>=k,'Not enough units per class.');
    ix=ix(randperm(numel(ix)));
    unitFold(ix)=mod((0:numel(ix)-1)',k)+1;
end
foldID=unitFold(group);
for f=1:k
    assert(numel(unique(y(foldID==f)))==2,'A validation fold lacks a class.');
    if strcmp(cfg.mode,'person_grouped')
        assert(isempty(intersect(personID(foldID~=f),personID(foldID==f))), ...
            'Person overlap detected.');
    end
end
end
