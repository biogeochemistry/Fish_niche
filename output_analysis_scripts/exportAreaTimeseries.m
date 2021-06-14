function exportAreaTimeseries(exportfile,timeseries_records,startdate)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here   

%datastart = datenum(startdate);

nlakes = size(timeseries_records, 1) - 1;

fid = fopen(exportfile, 'w', 'n', 'UTF-8');

fprintf(fid, '%s,%s,%s,%s\n',...
    timeseries_records{1, 1}, timeseries_records{1, 2},timeseries_records{1, 3},timeseries_records{1, 5});

for lakenum = 1:nlakes
   
    lake_id = timeseries_records{lakenum + 1, 1};
    
%     fprintf(1, 'Outputting for lake with id %d\n', lake_id);
    
    avg_area= timeseries_records{1+lakenum, 2};
    avg_areabyTotal= timeseries_records{1+lakenum, 3};
    ntg= timeseries_records{1+lakenum, 5};
    
    fprintf(fid, '%d,%.2f,%.2f,%.2f\n',...
             lake_id, avg_area,avg_areabyTotal,ntg);
    
%     ndays = length(avg_area);
%     
%     for line = 1:ndays
%         dn = datastart+line-1;
%         outdate = datevec(dn);
%         year = outdate(1);
%         month = outdate(2);
%         day = outdate(3);
%         
%         fprintf(fid, '%d,%02d.%02d.%04d,%.2f\n',...
%             lake_id, day, month, year, avg_area(line));
%     end
    
end

fclose(fid);
end

