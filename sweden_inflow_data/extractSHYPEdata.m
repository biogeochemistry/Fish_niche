

startdate = datenum([2001 1 1]);
enddate   = datenum([2010 12 31]);

%[data_num, data_txt, data_raw] = xlsread('Sjölista till Raoul.xlsx', 'Sjölista');
%subIDs = data_num(:, 5);
%flow_scalings = data_num(:, 7);

%subIDs = [5049, 5129];
subIDs = [3326,13834,21309,40488,10702,31358,23177,20344,40622]
flow_scalings = [1, 1];

for ii = 1:length(subIDs)
    try
        subID = subIDs(ii);
        fprintf(1, 'Trying to extract data for subID %d\n', subID);
        Data = extractData(subID, startdate, enddate);
        Data.Q = Data.Q * flow_scalings(ii);
        
        datasetname = sprintf('/%d/Q', subID);
        h5create('sweden_inflow_data_20010101_20101231.h5', datasetname, length(Data.Q));
        h5write ('sweden_inflow_data_20010101_20101231.h5', datasetname, Data.Q);
        
        datasetname = sprintf('/%d/T', subID);
        h5create('sweden_inflow_data_20010101_20101231.h5', datasetname, length(Data.T));
        h5write ('sweden_inflow_data_20010101_20101231.h5', datasetname, Data.T);
        
        datasetname = sprintf('/%d/TP', subID);
        h5create('sweden_inflow_data_20010101_20101231.h5', datasetname, length(Data.TP));
        h5write ('sweden_inflow_data_20010101_20101231.h5', datasetname, Data.TP);
        
        datasetname = sprintf('/%d/DOP', subID);
        h5create('sweden_inflow_data_20010101_20101231.h5', datasetname, length(Data.DOP));
        h5write ('sweden_inflow_data_20010101_20101231.h5', datasetname, Data.DOP);
        
    catch ME
        fprintf(1, '%s %s\n', ME.identifier, ME.message);
    end
end

function Data = extractData(subID, startdate, enddate)
    filename = sprintf('SHYPEdata/%d.xls', subID);
    
    [data_num, data_txt, data_raw] = xlsread(filename, 'Dygnsvärden');
    
    dates_txt = data_txt(4:end, 1);
    dates = datenum(dates_txt);
    
    firstidx = startdate - dates(1) + 1;
    lastidx  = enddate   - dates(1) + 1;
    outsetlength = enddate-startdate+1;
    
    Data.Q = data_num((3 + firstidx):(3 + lastidx), 1) * 86400; % 86400 seconds per day
    if(size(data_num, 2) > 4)
        Data.T = data_num((3 + firstidx):(3 + lastidx), 5);
    else
        Data.T = ones(outsetlength, 1) * 10;    % Dummy water temperature value (make better one?)
        fprintf(1, 'Missing temperature data\n');
    end
    
    [data_num, data_txt, data_raw] = xlsread(filename, 'Månadsvärden');
    TPmonth = data_num(1:end, 10);
    DOPmonth = data_num(1:end, 12);
    monthdates_txt = data_txt(2:end, 1);
    monthdates = datenum(monthdates_txt);
    monthdatevec = datevec(monthdates);
    
    Data.TP = zeros(outsetlength, 1);
    Data.DOP = zeros(outsetlength, 1);
    
    for ii = 1:outsetlength
        dnum = startdate + ii - 1;
        dvec = datevec(dnum);
        idx = (dvec(1) - monthdatevec(1,1))*12 + (dvec(2) - monthdatevec(1,2)) + 1;
        Data.TP(ii) = TPmonth(idx);
        Data.DOP(ii) = DOPmonth(idx);
    end
    
end