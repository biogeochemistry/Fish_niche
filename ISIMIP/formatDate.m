function formatDate(file)

path(path, '../../MyLake_O_simple')%change directory from mylake folder to Mylake_O_simple
path(path, '../sediments')

warning('off', 'all') 

T = readtable(file);
rows = height(T);
T.Var1=num2str(T.Var1);
for row = 1:rows
datenum = T{row,1};
date = datenum(1:4) +"-"+datenum(5:6)+"-"+datenum(7:8)+", 00:00:00";
T{row,1} = date;

end

end 