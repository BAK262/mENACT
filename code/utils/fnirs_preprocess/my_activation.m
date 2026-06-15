function [cfg, dataset] = my_activation(cfg)
% Create the classification dataset by a sliding window specified by `cfg`
%   and return it in a `table` object.
%
% [cfg, dataset] = my_activation(cfg)
%
% Required:
%   cfg.nSubs           - integer.
%   cfg.pathData        - str.
%   cfg.secsRetain      - the time (in second) to retain at the end of each 
%                       trial, set Inf if retain all available data.
%   cfg.secsWindow      - the length (in second) of the sliding window, set
%                       Inf or equal to secsRetain if not to segment.
%   cfg.secsStep        - the length (in second) of the sliding step, set Inf 
%                       if not to segment.
%   cfg.neutralSubtract - if to subtract the [neutral] condition as the new
%                       baseline.
%   cfg.neutralExclude  - if to exclucde samples of the [neutral] condition


    fprintf(['\nCreating dataset by a ',num2str(cfg.secsWindow),' ses(s) sliding window' ...
        'and ',num2str(cfg.secsStep),' sec(s) step on the final ',num2str(cfg.secsRetain),' sec(s)' ...
        ' data for each trial...\n'])

    % Default settings of this dataset
    cfg.emotions    = {'disgust','fear','anger','sadness', ...
        'neutral','inspiration','amusement','tenderness','joy'};
    if cfg.neutralExclude
        cfg.emotions= cfg.emotions([1:4 6:9]);
    end
    cfg.tasks       = {'Perception';'Production';'Performance'};
    cfg.nEmotions   = length(cfg.emotions);
    cfg.nTasks      = length(cfg.tasks);
    cfg.nSubs       = 53;
    cfg.nExps       = 3;
    cfg.fs          = 11; % sampling frequency (point/sec)
    cfg.secsBefore  = 3; % fixation time to exclude from the trial

    % Check sliding window parameters
    assert(cfg.secsRetain>=0); if cfg.secsRetain==0; cfg.secsRetain=Inf; end
    assert(cfg.secsWindow>=0); if cfg.secsWindow==0; cfg.secsWindow=Inf; end
    assert(cfg.secsStep>=0); if cfg.secsStep==0; cfg.secsStep=Inf; end
    assert(~xor(isinf(cfg.secsWindow), isinf(cfg.secsStep)), ...
        'The length of the window and the step must be both finite/infinite(includ 0) numbers.')
    
    % Translate setting durations in sec to in point
    doSegment = true;
    if isinf(cfg.secsRetain) && isinf(cfg.secsWindow)
        doSegment = false;
    end
    nPtsBefore  = round(cfg.fs * cfg.secsBefore);
    nPtsRetain  = min(round(cfg.fs * cfg.secsRetain), cfg.fs * 1200);
    nPtsWindow  = min(round(cfg.fs * cfg.secsWindow), nPtsRetain);
    nPtsStep    = min(round(cfg.fs * cfg.secsStep), nPtsRetain);

    % Get number and onset of sliding windows
    %   ** relative to the end of each trial **
    cfg.onsetWins   = (1-nPtsRetain):nPtsStep:(1-nPtsWindow);
    cfg.nWins       = max(1, ...
        1+round((nPtsRetain-nPtsWindow)/nPtsStep));
    fprintf(['Got no more than ',num2str(cfg.nWins),' sample(s) per trial.\n'])

    % Translate back
    fprintf(['Set [secsWindow] = ',num2str(cfg.secsWindow),' -> ',num2str(nPtsWindow/cfg.fs),'\n'])
    if ~isinf(cfg.secsWindow)
        cfg.secsWindow  = nPtsWindow /cfg.fs;
    end
    fprintf(['Set [secsStep] = ',num2str(cfg.secsStep),' -> ',num2str(nPtsStep/cfg.fs),'\n'])
    if ~isinf(cfg.secsStep)
        cfg.secsStep    = nPtsStep /cfg.fs;
    end

    % Initialize a table for storing the dataset
    dataset = table();

    % Go through all expriments
    for expNo = 1:cfg.nExps
        fprintf(['Now segmenting data from [Experiment ',num2str(expNo),']...\n'])
        if expNo == 3
            dataVars = {'task' 'targetEmotion' 'feel_disgust' 'feel_fear' ...
                'feel_anger' 'feel_sadness' 'feel_inspiration' 'feel_amusement' ...
                'feel_tenderness' 'feel_joy' 'self_valence' 'self_arousal'};
        else
            dataVars = {'task' 'targetEmotion' 'disgust' 'fear' 'anger' ...
                'sadness' 'inspiration' 'amusement' 'tenderness' 'joy' 'valence' 'arousal'};
        end

        % Go through all subjects
        for subNo = 1:cfg.nSubs
            subPath = [cfg.pathData, filesep, num2str(subNo)];
            fileName = [subPath, filesep, 'trialsDC_exp',num2str(expNo),'.mat'];
            if ~isfile(fileName)
                continue
            end
            load(fileName); % -> trialsDC
            nTrials     = length(trialsDC.trial);
            nWins       = max(1, cfg.nWins);
            tempTbl     = trialsDC.trialinfo(:,dataVars);
            tempTbl.subject     = ones(nTrials,1)*subNo;
            tempTbl.missingFeat = false(nTrials,1);
            if expNo == 3
                tempTbl = renamevars(tempTbl, ...
                    {'feel_disgust' 'feel_fear' 'feel_anger' 'feel_sadness' ...
                    'feel_inspiration' 'feel_amusement' 'feel_tenderness' 'feel_joy' ...
                    'self_valence' 'self_arousal'}, ...
                    {'disgust' 'fear' 'anger' 'sadness' 'inspiration' ...
                    'amusement' 'tenderness' 'joy' 'valence' 'arousal'});
            end
            tempTbl{:,end+1:end+nWins} = cell(nTrials,nWins);

            % Go through all trials
            for trialNo = 1:nTrials
                tempPoints  = length(trialsDC.time{trialNo});
                maxExclude  = max(nPtsBefore, tempPoints-nPtsRetain);

                % Segment the trial data by sliding windows
                if doSegment
                    tempOnsets  = tempPoints + cfg.onsetWins;
                    tempOnsets  = tempOnsets(tempOnsets>maxExclude);
                    nTempWins   = length(tempOnsets);
                    for winNo = 1:nTempWins
                        ptBeg   = tempOnsets(winNo);
                        ptEnd   = ptBeg + nPtsStep -1;
                        tempData = trialsDC.trial{trialNo}(:,ptBeg:ptEnd);
                        tempFeat = mean(tempData,2,'omitnan')';
                        tempTbl{trialNo,end-nWins+winNo} = {tempFeat};
                    end
                else
                    tempData = trialsDC.trial{trialNo}(:,maxExclude:end);
                    tempFeat = mean(tempData,2,'omitnan')';
                    tempTbl{trialNo,end} = {tempFeat};
                end
            end

            % Reformat the table to treat each sliding window as a sample
            tempTbl = stack(tempTbl,width(tempTbl)-nWins+1:width(tempTbl), ...
                'NewDataVariableName',{'features'}, ...
                'IndexVariableName',{'idxWin'});
            tempTbl.idxWin = double([tempTbl.idxWin]);

            % Remove empty windows and indicate if there is missing feature(s)
            loc            = cellfun('isempty', tempTbl{:,'features'});
            tempTbl(loc,:) = [];
            loc            = cellfun(@(d) any(isnan(d)), tempTbl{:,'features'});
            tempTbl{loc,'missingFeat'} = true;

            % Subtract the neutral condition as the comparison baseline
            if cfg.neutralSubtract
                loc = strcmp(tempTbl.targetEmotion,"neutral");
                if sum(loc)==0
                    continue
                end
                neutralFeatures = mean( ...
                    cell2mat(tempTbl{loc,'features'}), ...
                    1,"omitnan");
                f = @(x) x-neutralFeatures;
                tempTbl.features = cellfun(f, tempTbl{:,'features'}, ...
                    'UniformOutput', false);
            end

            % Exclude samples from the neutral condition
            if cfg.neutralExclude
                loc = strcmp(tempTbl.targetEmotion,"neutral");
                tempTbl(loc,:) = [];
            end

            % Concatenate the tables
            tempTbl.subject = categorical(tempTbl.subject);
            tempTbl.task    = categorical(tempTbl.task,cfg.tasks);
            tempTbl.targetEmotion = categorical(tempTbl.targetEmotion,cfg.emotions);
            dataset( ...
                height(dataset)+1:height(dataset)+height(tempTbl),: ...
                ) = tempTbl;
        end
    end

    % Save the dataset
    cfg.dataLabel   = trialsDC.label;
    cfg.dataOpto    = trialsDC.opto;
    save([cfg.pathSave,filesep,'dataset.mat'],"cfg","dataset")
    fprintf('Successfully made and saved this dataset!\n\n')

end