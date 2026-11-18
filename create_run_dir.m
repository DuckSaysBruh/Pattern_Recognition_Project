function outDir = create_run_dir(parentDir, tag)
% Readable directory name; never intentionally overwrite an earlier run.
if ~exist(parentDir,'dir'), mkdir(parentDir); end
tag=regexprep(char(tag),'[^a-zA-Z0-9_-]','_');
baseName=[tag '_' datestr(now,'yyyymmdd_HHMMSS')];
outDir=fullfile(parentDir,baseName); index=2;
while exist(outDir,'file') || exist(outDir,'dir')
    outDir=fullfile(parentDir,sprintf('%s_%02d',baseName,index));
    index=index+1;
end
[ok,msg]=mkdir(outDir); assert(ok,'Cannot create output directory: %s',msg);
end
