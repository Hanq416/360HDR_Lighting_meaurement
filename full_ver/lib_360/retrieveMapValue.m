% use this function to retrieve any data map value, hk.
function value = retrieveMapValue(data_map)
[hva,vva] = ui_input(); 
x = (hva + 180) + 1; y = vva + 90 + 1;
data_map = imresize(data_map, [181, 361]);
value = data_map(y, x);
fprintf('horizontal viewing angle = %d deg\n',hva);
fprintf('Vertical viewing angle = %d deg\n',vva);
end

function [hva,vva] = ui_input()
prompt = {'Horizontal viewing angle     [-180 to 180 degree]',...
    'Vertical viewing angle     [-90 to 90 degree]'};
dlgtitle = 'User Input'; dims = [1 50];
definput = {'0','0'};
answer = str2double(inputdlg(prompt,dlgtitle,dims,definput));
if isempty(answer)
    hva = 0; vva = 0;
else
    hva = answer(1); vva = answer(2);
end
end
