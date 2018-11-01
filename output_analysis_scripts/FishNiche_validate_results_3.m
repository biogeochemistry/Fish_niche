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

fig = figure();

ax_t = subplot(1, 2, 1);
ax_o = subplot(1, 2, 2);
        
hold(ax_t, 'on');
hold(ax_o, 'on');

T_model_samples = [];
T_data_samples = [];

O2_model_samples = [];
O2_data_samples = [];


for lakenum = 1:nlakes
    
    curlake = lakes(lakenum);
    
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

    try
        for ii=1:length(dates)
            dnum = datenum(dates(ii));
            dateindex = dnum - modelstart + 1;
            if dateindex >= 1 && dateindex <= ntimesteps
                depth = ceil(depths(ii));
                O2_data = O2raw(ii);
                T_data = Traw(ii);

                O2_model = O2model(depth, dateindex);
                T_model = Tmodel(depth, dateindex);

                if ~isnan(T_data)
                    plot(ax_t, T_data, T_model, 'b.');

                    T_model_samples = [T_model_samples T_model];
                    T_data_samples = [T_data_samples T_data];
                end

                if ~isnan(O2_data)
                    plot(ax_o, O2_data, O2_model, 'b.');

                    O2_model_samples = [O2_model_samples O2_model];
                    O2_data_samples = [O2_data_samples O2_data];
                end
            end

        end
    catch err
        disp(sprintf('Error when handling lake %s (%d)', curlake.name, curlake.lake_id));
        %disp(err);
    end

end


RMSE_T = sqrt(mean( (T_model_samples - T_data_samples).^2 ));
xlabel(ax_t, sprintf('Temperatures. RMSE: %.2f', RMSE_T));
RMSE_O = sqrt(mean( (O2_model_samples - O2_data_samples).^2 ));
xlabel(ax_o, sprintf('DO. RMSE: %.2f', RMSE_O));

axi = axis(ax_t);
tempmax = max(axi);
plot(ax_t, [0, tempmax], [0, tempmax], 'r');

axi = axis(ax_o);
tempmax = max(axi);
plot(ax_o, [0, tempmax], [0, tempmax], 'r');

hold(ax_t, 'off');
hold(ax_o, 'off');
