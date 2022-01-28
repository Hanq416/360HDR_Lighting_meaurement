% plot pano luminance map
function Luminance_map(I,figName,fcode,gm,gcf)
lpano = (I(:,:,1).*0.265 + I(:,:,2).*0.670 + I(:,:,3).*0.065).*179.*gcf;
lpano(lpano<0) = 0;
%% auto gamma
cv = std(std(lpano))/mean(mean(lpano));
if  (1<cv)&&(cv<10)
    gm1 = round(1/cv,2);
elseif cv > 10
    gm1 = 0.09;
else
    gm1 = 1;
end
fprintf('\n\nauto gamma= %.2f \n',gm1);
fprintf('manual gamma= %.2f \n',gm);
yn = yn_dialog('using auto-calculated gamma? [check command window info]');
if ismember(yn, ['Yes', 'yes'])
    gm = gm1;
end
%%
lumimg = (lpano - min(min(lpano)))/(max(max(lpano))-min(min(lpano)));
lumimg = uint8((lumimg.^gm).*256);
rg = max(max(lpano))-min(min(lpano)); crange = jet(256);crange(1,:) = 0;
cb1 = round(rg.*(0.03316.^(1/gm)),4);cb2 = round(rg.*(0.26754.^(1/gm)),2);
cb3 = round(rg.*(0.50191.^(1/gm)),2);cb4 = round(rg.*(0.73629.^(1/gm)),2);
cb5 = round(rg.*(0.97066.^(1/gm)),2);
figure(fcode);imshow(lumimg,'Colormap',crange);
title(['\fontsize{24}\color[rgb]{0 .5 .5}',figName]);
hcb = colorbar('Ticks',[8,68,128,188,248],'TickLabels',{cb1,cb2,cb3,cb4,cb5},...
    'FontSize',14);
title(hcb,'\fontsize{16}cd/m2');
end