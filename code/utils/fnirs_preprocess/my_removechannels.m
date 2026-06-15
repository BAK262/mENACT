function snirf = my_removechannels(snirf, invilidSource, invilidDetector)
% DESCRIPTION
%   This function remove nirs channels involved `invilidSource` and 
%   `invilidDetector` from SNIRF class and return the updated SNIRF class.
%
% snirf = my_removechannels(snirf, invilidSource, invilidDetector)
    delSource = find(contains(snirf.probe.sourceLabels, invilidSource));
    newSource = find(~contains(snirf.probe.sourceLabels, invilidSource));
    delDetector = find(contains(snirf.probe.detectorLabels, invilidDetector));
    newDetector = find(~contains(snirf.probe.detectorLabels, invilidDetector));
    snirf.probe.sourceLabels(delSource) = [];
    snirf.probe.sourcePos2D(delSource,:) = [];
    snirf.probe.sourcePos3D(delSource,:) = [];
    snirf.probe.detectorLabels(delDetector) = [];
    snirf.probe.detectorPos2D(delDetector,:) = [];
    snirf.probe.detectorPos3D(delDetector,:) = [];
    delChannel = [];
    for i = 1:length(snirf.data.measurementList)
        if ismember( ...
                snirf.data.measurementList(1,i).sourceIndex, ...
                delSource ...
                ) || ismember( ...
                snirf.data.measurementList(1,i).detectorIndex, ...
                delDetector ...
                )
            delChannel = [delChannel, i];
        else
            snirf.data.measurementList(1,i).sourceIndex = find( ...
                newSource == snirf.data.measurementList(1,i).sourceIndex);
            snirf.data.measurementList(1,i).detectorIndex = find( ...
                newDetector == snirf.data.measurementList(1,i).detectorIndex);
        end
    end
    snirf.data.cache.measurementListMatrix = [];
    snirf.data.measurementList(delChannel) = [];
    snirf.data.dataTimeSeries(:,delChannel) = [];
end