function exportVolumeTimeseries(exportfile,timeseries_records,startdate)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here   
datastart = datenum(startdate);

nlakes = size(timeseries_records, 1) - 1;

fid = fopen(exportfile, 'w', 'n', 'UTF-8');

fprintf(fid, '%s,Date,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n',...
    timeseries_records{1, 1}, timeseries_records{1, 2}, timeseries_records{1, 3}, timeseries_records{1, 4},...
    timeseries_records{1, 5}, timeseries_records{1, 6}, timeseries_records{1, 7}, timeseries_records{1, 8},...
    timeseries_records{1, 9}, timeseries_records{1, 10},timeseries_records{1, 11});

for lakenum = 1:nlakes
   
    lake_id = timeseries_records{lakenum + 1, 1};
    
    fprintf(1, 'Outputting for lake with id %d\n', lake_id);
    
    avg_O2_above = timeseries_records{1+lakenum, 2};
    avg_O2_below = timeseries_records{1+lakenum, 3};
    avg_T_above = timeseries_records{1+lakenum, 4};
    avg_T_below = timeseries_records{1+lakenum, 5};
    volume_T_below_15 = timeseries_records{1+lakenum, 6};
    volume_O2_above_3000 = timeseries_records{1+lakenum, 7};
    volume_Attn_above_1_percent_of_surface = timeseries_records{1+lakenum, 8};
    volume_all_three = timeseries_records{1+lakenum, 9};
    maxgrad_depth = timeseries_records{1+lakenum, 10};
    total_volume = timeseries_records{1+lakenum, 11};
    
    ndays = length(avg_O2_above);
    
    for line = 1:ndays
        dn = datastart+line-1;
        outdate = datevec(dn);
        year = outdate(1);
        month = outdate(2);
        day = outdate(3);
        
        fprintf(fid, '%d,%02d.%02d.%04d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f\n',...
            lake_id, day, month, year, avg_O2_above(line), avg_O2_below(line), avg_T_above(line), avg_T_below(line),...
            volume_T_below_15(line), volume_O2_above_3000(line), volume_Attn_above_1_percent_of_surface(line),...
            volume_all_three(line), maxgrad_depth(line),total_volume(line));
    end
    
end

fclose(fid);
end

