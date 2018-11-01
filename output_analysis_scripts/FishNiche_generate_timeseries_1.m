
path(path,'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes')%2018-5-11 MC
path(path,'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\output')%2018-5-11 MC

lakelistfile  = 'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\2017SwedenList_only_validation_lakes.csv';%2018-5-11 MC
outputdir = 'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\output';%2018-5-11 MC

% The following are for scenario 2 only, it should be automated based on scenario choice.
m1 = 'MOHC-HadGEM2-ES';% 2018-05-25 MC model 4
m2 = 'r1i1p1_SMHI-RCA4_v1_day'% 2018-05-25 MC model 4
exA = 'historical';
y1A = 2001;
exB = 'rcp45';
y1B = 2006;
y2A = y1A + 4;
y2B = y1B + 4;

startdate = [y1A, 1, 1];
datastart = datenum(startdate);

lakes = readLakeListCsv(lakelistfile);

nlakes = length(lakes);

timeseries_records = cell(nlakes+1, 9);
timeseries_records{1,1} = 'lakeid';
timeseries_records{1,2} = 'Average O2 above max T gradient';
timeseries_records{1,3} = 'Average O2 below max T gradient';
timeseries_records{1,4} = 'Average T above max T gradient';
timeseries_records{1,5} = 'Average T below max T gradient';
timeseries_records{1,6} = 'Volume with T < 15 C';
timeseries_records{1,7} = 'Volume with O2 > 3000';
timeseries_records{1,8} = 'Volume with Attn > 1% of surface Attn';
timeseries_records{1,9} = 'Volume satisfying all three previous';
timeseries_records{1,10} = 'Depth of maximal T gradient';

for lakenum = 1:nlakes
    ebhex = lakes(lakenum).ebhex;
    outputdir2 = getOutputPathFromEbHex(outputdir, ebhex);
    d4 = sprintf('EUR-11_%s_%s-%s_%s_%d0101-%d1231', m1, exA, exB, m2, y1A, y2B);

    lakedir = strcat(outputdir2, d4, '\');
    disp(lakedir);
    
    O2   = csvread(strcat(lakedir, 'O2zt.csv'));
    T    = csvread(strcat(lakedir, 'Tzt.csv'));
    Attn = csvread(strcat(lakedir, 'Attn_zt.csv'));
    
    surface_area = lakes(lakenum).area;
    max_depth    = lakes(lakenum).depth;

    area_at_depth = @(depth)( surface_area .* (depth - max_depth).^2 ./ max_depth.^2 );

    T_grad = diff(T')';
    [maxgrad, maxgrad_depth] = max(abs(T_grad'));
    maxgrad_depth = maxgrad_depth';
    tlen = size(T, 1);
    zlen = size(T, 2);

    avg_O2_above = zeros(tlen, 1);
    avg_O2_below = zeros(tlen, 1);
    avg_T_above  = zeros(tlen, 1);
    avg_T_below  = zeros(tlen, 1);

    volume_O2_above_3000 = zeros(tlen, 1);
    volume_T_below_15    = zeros(tlen, 1);
    volume_Attn_above_1_percent_of_surface = zeros(tlen, 1);
    volume_all_three     = zeros(tlen, 1);

    for time = 1:tlen
        avg_O2_above(time) = mean(O2(time, 1:maxgrad_depth(time)));
        avg_O2_below(time) = mean(O2(time, (maxgrad_depth(time)+1):end));
        avg_T_above(time)  = mean(T(time, 1:maxgrad_depth(time)));
        avg_T_below(time)  = mean(T(time, (maxgrad_depth(time)+1):end));

        surface_attn = Attn(time, 1);
        for depth = 1:zlen
            test = area_at_depth(depth);
            if O2(time, depth) > 3000
                volume_O2_above_3000(time) = volume_O2_above_3000(time) + area_at_depth(depth);
            end
            if T(time, depth) < 15
                volume_T_below_15(time) = volume_T_below_15(time) + area_at_depth(depth);
            end
            if Attn(time, depth) >= 0.01 * surface_attn
                volume_Attn_above_1_percent_of_surface(time) = volume_Attn_above_1_percent_of_surface(time) + area_at_depth(depth);
            end
            if O2(time, depth) > 3000 && T(time, depth) < 15 && Attn(time, depth) >= 0.01 * surface_attn
                volume_all_three(time) = volume_all_three(time) + area_at_depth(depth);
            end
        end
    end
    
    timeseries_records{1+lakenum, 1} = lakes(lakenum).lake_id;
    timeseries_records{1+lakenum, 2} = avg_O2_above;
    timeseries_records{1+lakenum, 3} = avg_O2_below;
    timeseries_records{1+lakenum, 4} = avg_T_above;
    timeseries_records{1+lakenum, 5} = avg_T_below;
    timeseries_records{1+lakenum, 6} = volume_T_below_15;
    timeseries_records{1+lakenum, 7} = volume_O2_above_3000;
    timeseries_records{1+lakenum, 8} = volume_Attn_above_1_percent_of_surface;
    timeseries_records{1+lakenum, 9} = volume_all_three;
    timeseries_records{1+lakenum, 10} = maxgrad_depth;
    
end

