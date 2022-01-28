%dev version of angle-based illuminance calibraiton function
%2 reference method
%09/20/2020, Hankun Li
function [cfh,cfv0,cfv90,cfv180,cfv270] = getluxCF_2ref(pano,gcf,sx,sy,f)
[Ef,Er] = reflux_dialog();
E0 = Ef; E180 = Er;
illu_map = zeros([2,2]);
illu_map(2,2) = 180;
for z = 1: size(illu_map,1)
    IF_hdr = imequ2fish_hdr(pano,illu_map(z,1),illu_map(z,2),90);
    [hy,hx] = size(IF_hdr(:,:,1));
    if hy ~= hx
        IF_hdr = imresize(IF_hdr,[hy,hy]);
    end
    L = LuminanceRetrieve(IF_hdr.*gcf,hy);%temporary global CF function!
    illu_map(z,3) = PerPixel_Fequisolid(hy,hy,sx,sy,f,L);
end
cfv0 = round(E0/illu_map(1,3),2); cfv180 = round(E180/illu_map(2,3),2);
cfv270 = cfv0*0.5 + cfv180*0.5; cfv90 = cfv0*0.5 + cfv180*0.5;
cfh = cfv0*0.5 + cfv180*0.5;
end

function [Ef,Er] = reflux_dialog()
prompt = {'Meter measured illuminance (front camera)',...
    'Meter measured illuminance (rear camera) '};
dlgtitle = 'Reference Lux Input'; dims = [1 50];
definput = {'0','0'};
answer = str2double(inputdlg(prompt,dlgtitle,dims,definput));
if isempty(answer)
    Ef = 0; Er = 0;
else
    Ef = answer(1); Er = answer(2);
end
end