%dev version of angle-based illuminance calibraiton function
%09/20/2020, Hankun Li
function [cfh,cfv0,cfv90,cfv180,cfv270] = getluxCF(pano,gcf,sx,sy,f)
[Eh,E0,E90,E180,Eneg90] = reflux_dialog();
illu_map = zeros([5,2]);
illu_map(1,1) = -90; illu_map(3,2) = 90; illu_map(4,2) = 180; illu_map(5,2) = -90;
for z = 1: size(illu_map,1)
    IF_hdr = imequ2fish_hdr(pano,illu_map(z,1),illu_map(z,2),90);
    [hy,hx] = size(IF_hdr(:,:,1));
    if hy ~= hx
        IF_hdr = imresize(IF_hdr,[hy,hy]);
    end
    L = LuminanceRetrieve(IF_hdr.*gcf,hy);%temporary global CF function!
    illu_map(z,3) = PerPixel_Fequisolid(hy,hy,sx,sy,f,L);
end
cfh = round(Eh/illu_map(1,3),2); cfv0 = round(E0/illu_map(2,3),2);
cfv90 = round(E90/illu_map(3,3),2); cfv180 = round(E180/illu_map(4,3),2);
cfv270 = round(Eneg90/illu_map(5,3),2);
end

function [Eh,E0,E90,E180,Eneg90] = reflux_dialog()
prompt = {'Meter measured horizontal illuminance (Eh)',...
    'Meter measured vertical illuminance @0 (Ev@0) ',...
    'Meter measured vertical illuminance @90 (Ev@90)',...
    'Meter measured vertical illuminance @180 (Ev@180)',...
    'Meter measured vertical illuminance @-90 (Ev@-90)'};
dlgtitle = 'Reference Lux Input'; dims = [1 50];
definput = {'1','1','1','1','1'};
answer = str2double(inputdlg(prompt,dlgtitle,dims,definput));
if isempty(answer)
    Eh = 1; E0 = 1; E90 = 1; E180 = 1; Eneg90 = 1;
else
    Eh = answer(1); E0 = answer(2); E90 = answer(3); E180 = answer(4);
    Eneg90 = answer(5);
end
end