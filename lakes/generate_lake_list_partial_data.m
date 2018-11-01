% Generates a lake list csv from "Sjölista till Raoul.xlsx" which can later
% be used to fetch the ebhex value and create a finished lake list csv file
% to be used with FishNiche


[data_num, data_txt, data_raw] = xlsread('../sweden_inflow_data/Sjölista till Raoul.xlsx', 'Sjölista');

PikeIDs = data_num(:, 1);
Names = data_txt(2:end, 2);
SubIDs = data_num(:, 5);
MaxDepths = data_num(:, 6);
Latitudes = data_num(:, 10);
Longitudes = data_num(:, 11);
Areas = data_num(:, 12) * 10000; %Conversion from ha to m^2


fid = fopen('2017SwedenList_incomplete_data.csv', 'w');
fprintf(fid, 'lake_id,subid,name,area,depth,longitude,latitude\n');
for ii=1:length(PikeIDs)
    fprintf(fid, '%d,%d,%s,%f,%f,%f,%f\n',PikeIDs(ii),SubIDs(ii),Names{ii},Areas(ii),MaxDepths(ii),Longitudes(ii),Latitudes(ii));
end

fclose(fid);