function lakes = readLakeListCsv(filename)

fid = fopen(filename, 'r', 'n');
fgetl(fid); % One trash line

line = fgetl(fid);
lakenum = 1;
while ischar(line)
    values = strsplit(line, ',');
    lakes(lakenum).lake_id = str2num(values{1});
    lakes(lakenum).subid = str2num(values{2});
    lakes(lakenum).name = strrep(values{3}, '"', '');
    lakes(lakenum).ebhex = strrep(values{4}, '"', '');
    lakes(lakenum).area = str2num(values{5});
    lakes(lakenum).depth = str2double(values{6});
    lakes(lakenum).longitude = str2double(values{7});
    lakes(lakenum).latitude = str2double(values{8});

    line = fgetl(fid);
    lakenum = lakenum + 1;
end

fclose(fid);