function composedMap(I,figName,fcode,gm,gcf,lux_map,ts,as)
%% rotate HDR
Ipano = imresize(stiching(I),0.25);
%%
lpano = (Ipano(:,:,1).*0.265 + Ipano(:,:,2).*0.670 + Ipano(:,:,3).*0.065).*179.*gcf;
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
%% generate luminance map
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
%% generate contour map
[rows,cols] = size(Ipano(:,:,1));
map = imresize(illuminanceMap(lux_map,ts,as),[rows cols]);
hold all;
contour(map,'--k','ShowText','on','LineWidth',3);
%% generate axis
axstep = 30;
x_ticks = -180:axstep:180; y_ticks = -90:axstep:90;  axis on;
xp_ticks = linspace(0.5,size(lumimg,2)+0.5,numel(x_ticks));
yp_ticks = linspace(0.5,size(lumimg,1)+0.5,numel(y_ticks));
Xticklabels = cellfun(@(v) sprintf('%d',v), num2cell(x_ticks),...
    'UniformOutput',false);
Yticklabels = cellfun(@(v) sprintf('%d',v), num2cell(y_ticks),...
    'UniformOutput',false);
set(gca,'XTick',xp_ticks); set(gca,'XTickLabels',Xticklabels);
set(gca,'YTick',yp_ticks); set(gca,'YTickLabels',Yticklabels(end:-1:1));
xlabel('Horizontal viewing direction/ degree');
ylabel('Vertical viewing direction/ degree');
end

function map = illuminanceMap(imap,ts,as)
imap(:,1) = (imap(:,1) + max(imap(:,1)))./ts;
imap(:,2) = imap(:,2)./as;
imap(:,1) = imap(:,1)+1; imap(:,2) = imap(:,2)+1;
map = zeros(max(imap(:,1)),max(imap(:,2)));
for w = 1: size(imap,1)
    map(imap(w,1),imap(w,2)) = imap(w,3);
end
end

function Ipano = stiching(Idf)
h = size(Idf,1); w = size(Idf,2);
c2 = drawcircle('Center',[  w/4,h/2],'Radius',(h/2)*0.98,'Color','red');
c1 = drawcircle('Center',[3*w/4,h/2],'Radius',(h/2)*0.98,'Color','green');
IL = imcrop(Idf,[c1.Center-c1.Radius, c1.Radius*2, c1.Radius*2]);
IR = imcrop(Idf,[c2.Center-c2.Radius, c2.Radius*2, c2.Radius*2]); 
IR = imresize(IR,[size(IL,1), size(IL,2)]); close all;
%camera parameters retrived from multiple attempt, can be improved with
%futher works
fovL  = 185; rollL = -90; tiltL = 0.5; panL = 0; 
fovR  = 185; rollR = -90; tiltR = 0; panR = 180;
EL = imfish2equ_hdr(IL,fovL,rollL,tiltL,panL); ER = imfish2equ_hdr(IR,fovR,rollR,tiltR,panR);
[EL,maskL] = trimImageByFov(EL,fovL,panL); [ER,maskR] = trimImageByFov(ER,fovR,panR);

maskB = maskL & maskR;
stat = regionprops('table',maskB,'Area','PixelIdxList','Image');
alpha = zeros(size(maskB));
idx = stat.PixelIdxList{1};alpha(idx) = 1/size(stat.Image{1},2); 
idx = stat.PixelIdxList{2};alpha(idx) = -1/size(stat.Image{2},2); 
alpha = cumsum(alpha,2);

ELR = alpha.*double(EL) + (1-alpha).*double(ER); Ipano = double(ELR);
end

function [IE2,mask] = trimImageByFov(IE,fov,pan)
w  = int32(size(IE,2)); we = w*(fov/360)/2; ce = mod(w*(0.5+pan/360),w);
idx = [ones(1,we),zeros(1,w-2*we),ones(1,we)]; idx = circshift(idx,ce);
IE2 = IE; IE2(:,~idx,:) = 0; mask = repmat(idx,[size(IE2,1), 1, size(IE2,3)]);
end