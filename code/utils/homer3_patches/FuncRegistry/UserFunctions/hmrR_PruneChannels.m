% SYNTAX:
% mlActAuto = hmrR_PruneChannels(data, probe, mlActMan, tInc, dRange, SNRthresh, SDrange)
%
% UI NAME:
% Prune_Channels
%
% DESCRIPTION:
% Prune channels from the measurement list if their signal is too weak, too
% strong, or their standard deviation is too great. This function
% updates MeasListAct based on whether data 'd' meets these conditions
% as specified by'dRange' and 'SNRthresh'.
%
% INPUTS:
% d - SNIRF object containing time course data (nTpts x nChannels )
% probe - SNIRF object describing the probe - optode positions and wavelengths.
% mlActMan - 
% dRange - if mean(d) < dRange(1) or > dRange(2) then it is excluded as an
%      active channel
% SNRthresh - if mean(d)/std(d) < SNRthresh then it is excluded as an
%      active channel
% SDrange - will prune channels with a source-detector separation <
%           SDrange(1) or > SDrange(2)
%
% OUTPUTS:
% mlAct - cell array of all data blocks - each data block is an array
%         of true/false for all channels in the contanining data block
%         specifying active/inactive status. (# of data blocks x # of Channels)
%
% USAGE OPTIONS:
% Prune_Channels: mlActAuto = hmrR_PruneChannels(data, probe, mlActMan, tIncMan, dRange, SNRthresh, SDrange)
%
% PARAMETERS:
% dRange: [1e4, 1e7]
% SNRthresh: 2
% SDrange: [0.0, 45.0]
%
% TO DO:
% consider Conc as well as wavelength data
%
function mlActAuto = hmrR_PruneChannels(data, probe, mlActMan, tIncMan, dRange, SNRthresh, SDrange)

% Init output 
mlActAuto = cell(length(data),1);

% Check input args
if nargin<7
    disp( 'USAGE: hmrR_PruneChannels(data, probe, mlActMan, tIncMan, dRange, SNRthresh, SDrange)' )
    return
end
if isempty(tIncMan)
    tIncMan = cell(length(data),1);
end
if isempty(mlActMan)
    mlActMan = cell(length(data),1);
end

for iBlk = 1:length(data)
    
    d        = data(iBlk).GetDataTimeSeries();
    t        = data(iBlk).GetTime();
    MeasList = data(iBlk).GetMeasList();
    Lambda   = probe.GetWls();
    SrcPos   = probe.GetSrcPos();
    DetPos   = probe.GetDetPos();    
    
    mlActMan{iBlk} = mlAct_Initialize(mlActMan{iBlk}, MeasList);

    if isempty(tIncMan{iBlk})
        tIncMan{iBlk} = ones(length(t),1);
    end
    tInc = tIncMan{iBlk};
        
    lstInc = find(tInc==1);
    d = d(lstInc,:);

    % check for dRange and SNRthresh
    dmean = mean(d,1);
    dstd = std(d,[],1);
    
    idxs1 = [];
    idxs2 = [];
    idxs3 = [];
    nLambda = length(Lambda);
    lst1 = find(MeasList(:,4)==1);    

    % Start by including all channels
    chanList = ones(size(MeasList,1),1);
    
    lst = {};

    for ii = 1:nLambda
        
        rhoSD = [];
        for jj = 1:length(lst1)
            lst{ii}(jj) = find(MeasList(:,1)==MeasList(lst1(jj),1) & ...
                           MeasList(:,2)==MeasList(lst1(jj),2) & ...
                           MeasList(:,4)==ii);
            rhoSD(jj) = norm( SrcPos(MeasList(lst1(jj),1),:) - DetPos(MeasList(lst1(jj),2),:) );
        end
                
        % dRange exclusion criteria
        idxs1 = [idxs1, find(dmean(lst{ii})<=dRange(1) | dmean(lst{ii})>=dRange(2))];
        
        % SNRthresh exclusion criteria
        idxs2 = [idxs2, find((dmean(lst{ii})./dstd(lst{ii})) <= SNRthresh)];

        % SDrange exclusion criteria
        idxs3 = [idxs3, find(rhoSD<SDrange(1) | rhoSD>SDrange(2))];
        
    end
    
    idxsExcl = unique([idxs1(:)', idxs2(:)', idxs3(:)']);
    
    for ii = 1:nLambda
        chanList(lst{ii}(idxsExcl)) = 0;
    end
    
    idxs4 = find(mlActMan{iBlk}(:,3) == 0);
    
    if ~isempty(idxs1)
        fprintf('hmrR_PruneChannels:  excluded channels [ %s ] based on dRange=[ %s ]\n', num2str(unique(idxs1(:)')), num2str(dRange));
    end
    if ~isempty(idxs2)
        fprintf('hmrR_PruneChannels:  excluded channels [ %s ] based on SNRthresh=%0.1f\n', num2str(unique(idxs2(:)')), SNRthresh);
    end
    if ~isempty(idxs3)
        fprintf('hmrR_PruneChannels:  excluded channels [ %s ] based on SDrange=[ %s ]\n', num2str(unique(idxs3(:)')), num2str(SDrange));
    end
    if ~isempty(idxs4)
        fprintf('hmrR_PruneChannels:  manually excluded channels [ %s ]\n', num2str(idxs4(:)'));
    end

    chanList = chanList(:) & mlActMan{iBlk}(:,3);
    
    % update MeasListAct  
    mlActAuto{iBlk} = mlAct_Initialize(chanList, MeasList);
end
