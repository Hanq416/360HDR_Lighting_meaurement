% MATLAB retrieve Exposure value of a Radiance hdr
% Hankun Li, KU-LRL lab use
% 10/20/2020
function exposure = getExpValue(filename)
fid = fopen(filename); ct = 0;
while ct < 16
    line = fgetl(fid);
    if contains(line, 'EXPOSURE')
        line = erase(line, ' '); break
    end
end
fclose(fid); exposure = str2double(erase(line, 'EXPOSURE='));
end