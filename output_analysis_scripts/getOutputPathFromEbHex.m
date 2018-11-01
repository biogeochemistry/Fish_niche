function path = getOutputPathFromEbHex(outputbasedirectory, ebhex)

ebh = ebhex;
if strcmp(ebh(1:2), '0x')
    ebh = ebh(3:end);
end
while length(ebh) < 6
    ebh = strcat('0', ebh);
end
d1 = ebh(1:2);
d2 = ebh(1:4);
d3 = ebh(1:6);

path = strcat(outputbasedirectory, '\', d1, '\', d2, '\', d3, '\');