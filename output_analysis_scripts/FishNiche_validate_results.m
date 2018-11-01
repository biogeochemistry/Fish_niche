

lakelistfile  = '..\lakes\2017SwedenList_only_validation_lakes.csv';
outputdir = '..\output';

% The following are for scenario 2 only, it should be automated based on scenario choice.
m1 = 'ICHEC-EC-EARTH';
m2 = 'r3i1p1_DMI-HIRHAM5_v1_day';
exA = 'historical';
y1A = 2001;
exB = 'rcp45';
y1B = 2006;
y2A = y1A + 4;
y2B = y1B + 4;

modelstart = datenum([y1A, 1, 1]);
modelend   = datenum([y2B, 12, 31]);
ntimesteps = modelend-modelstart+1;

lakes = readLakeListCsv(lakelistfile);

nlakes = length(lakes);

%lakeswehavedatafor = [698, 67035, 19167, 31985, 310, 32276, 31895, 59185, 4810];
lakeswehavedatafor = [698, 67035, 19167, 31895, 310, 32276, 99045, 99516, 6950];

for lakenum = 1:nlakes
    
    curlake = lakes(lakenum);
    
    if any(curlake.lake_id == lakeswehavedatafor)
        outputdir2 = getOutputPathFromEbHex(outputdir, curlake.ebhex);
        d4 = sprintf('EUR-11_%s_%s-%s_%s_%d0101-%d1231', m1, exA, exB, m2, y1A, y2B);

        lakedir = strcat(outputdir2, d4, '\');
        disp(lakedir);

        O2model   = csvread(strcat(lakedir, 'O2zt.csv'))';
        Tmodel    = csvread(strcat(lakedir, 'Tzt.csv'))';

        filename = '../sweden_inflow_data/Validation_data_for_lookup.xlsx';
        worksheet = sprintf('%d', curlake.lake_id);

        [data_num, data_txt, data_raw] = xlsread(filename, worksheet);

        dates = datenum(data_txt(2:end, 1));
        depths = data_num(:, 1);
        O2raw = data_num(:, 2) * 1000;
        Traw= data_num(:, 3);

        lakedepth = ceil(curlake.depth);
        O2data = zeros(lakedepth, ntimesteps) * NaN;
        Tdata = zeros(lakedepth, ntimesteps) * NaN;

        for ii=1:length(dates)
            dnum = datenum(dates(ii));
            dateindex = dnum - modelstart + 1;
            if dateindex >= 1 && dateindex <= ntimesteps
                depth = ceil(depths(ii));
                O2data(depth, dateindex) = O2raw(ii);
                Tdata(depth, dateindex) = Traw(ii);
            end
        end


        subplotwidth = 6;
        ndepthswithdata = sum(any(~isnan(Tdata')));
        if ndepthswithdata > 0
            datearray = datetime(modelstart:modelend, 'ConvertFrom', 'datenum');
            plotidx = 1;
            figure('name', sprintf('%s (%d) - temperature', curlake.name, curlake.lake_id));
            if ndepthswithdata > subplotwidth
                subplotdepth = ceil(ndepthswithdata / subplotwidth);
            else
                subplotdepth = 1;
                subplotwidth = ndepthswithdata;
            end
            for ii=1:curlake.depth
                if any(~isnan(Tdata(ii,:)))
                    ax = subplot(subplotdepth, subplotwidth, plotidx);
                    hold(ax, 'on');
                    plot(datearray, Tmodel(ii,:));
                    plot(datearray, Tdata(ii,:), 'o');
                    xlabel(sprintf('Depth %d', ii));

                    hold(ax, 'off');
                    plotidx = plotidx + 1;
                end
            end
        end
        
    end
end

