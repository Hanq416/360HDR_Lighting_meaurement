function view = findView(pano, size)
[hva,vva] = ui_input();
IF = imequ2fish_hdr(pano, vva, hva, 0);
view = imresize(IF, [size, size]);
view = imrotate(view, -90);
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