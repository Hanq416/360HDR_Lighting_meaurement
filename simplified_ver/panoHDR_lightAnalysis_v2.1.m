clear all; %#ok<*CLALL>
close all;
clc;
% illuminance map with 360 Pano-Camera (Ricoh Theta Z1)
% Type of input: [dual fisheye hdr image]
% Hankun Li, University of Kansas 
% update: 01/28/2022, version 2.1 simplified

% Author Notes:
%[1]illuminance map is to show lux distribution of percieved light.

%[2]CV_map is to show light uniformity at every specified direction
% higher number means less uniformity at specific viewing direction

% Equalrectangular to 180-degree fisheye, [JPEG image version]
% Copyright of some original functions: Kazuya Machida (2020)
% Modified by Hankun Li for KU LRL research use, University of Kansas Aug,18,2020
% Reference: 
% [1] 360-degree-image-processing (https://github.com/k-machida/360-degree-image-processing), GitHub. Retrieved August 18, 2020.
% [2] Tuan Ho, Madhukar Budagavi,  "2DUAL-FISHEYE LENS STITCHING FOR 360-DEGREE IMAGING"

% Camera parameters (Ricoh Theta Z1):
%--------- Camera parameters -----------%
%DO NOT CHANGE FOR THETA Z1 CAMERA!
f = 3; % specify lens focal length ##(INPUT2)##
% sx = sy = f*1.414*2
sx = 8.48; sy = 8.48; % Calculated sensor size (Single 180deg fisheye)
mode = "photopic"; % perception mode: "eml" or "photopic"
%---------Camera parameters end---------%
[fn,pn]=uigetfile('*.hdr','select a dual fisheye 360 hdr image');str=[pn,fn];
[tilt_a,tilt_b,aim_a,aim_b,aim_step,gcf,comr] = initial_dialog();
I = hdrread(str); vmsk = vecorf; I = I./vmsk; exp = getExpValue(str); gcf = gcf./exp;
fprintf('\n Notes: Re-calculation of GCF for old data, exp is now counted! \n');
pano_src = imgstiching_hdr(I); [y1,x1] = size(pano_src(:,:,1));
pano = imresize(pano_src,[round(y1/comr),round(x1/comr)]);
[y,x] = size(pano(:,:,1)); tilt_step = aim_step; clear pn fn str;
i = aim_a; illu_map = []; ct = 1; ram_ts = []; ts_size = 0; ts_ct = 0;
while i <= aim_b
    j = tilt_a; %ta
    while j <= tilt_b
        illu_map(ct,1) = j; illu_map(ct,2) = i + 180;
        ct = ct + 1; j  = j + tilt_step;
    end
    i = i + aim_step; %aa
end
icf_flg = 0; yn = yn_dialog('Apply angel based illuminance calibration?');
if ismember(yn, ['Yes', 'yes'])
    tf = tf_dialog('Reference lux point? [2 or 5]'); %see papaer to understand 2-ref/5-ref methods.
    if ismember(tf, ['2-ref','2-REF'])
        [cfh,cfv0,cfv90,cfv180,cfv270] = getluxCF_2ref(pano,gcf,sx,sy,f, mode);
    else
        [cfh,cfv0,cfv90,cfv180,cfv270] = getluxCF(pano,gcf,sx,sy,f, mode);
    end
    fprintf('\nIlluminance calibration factor: \n');
    fprintf('Eh:%f, E@0:%f, E@90:%f, E@180:%f, E@-90:%f\n', cfh,cfv0,cfv90,cfv180,cfv270);
    icf_flg = 1;
end
% make sure understand 5 reference values before click 'yes' to use
cv_flg = 0; yn = yn_dialog('Generate Coefficient of Variance Map?');
if ismember(yn, ['Yes', 'yes'])
    cv_flg = 1;
end
refHDR = imequ2fish_hdr(pano,0,0,90); [yc,xc] = size(refHDR(:,:,1));
luxmask = equisolidMask(xc,yc,180); clear xc yc;
%%
for z = 1: size(illu_map,1)
    IF_hdr = imequ2fish_hdr(pano,illu_map(z,1),illu_map(z,2),90);
    [hy, hx] = size(IF_hdr(:,:,1));
    if hy ~= hx
        IF_hdr = imresize(IF_hdr, [hy,hy]);
        luxmask = imresize(luxmask, [hy,hy]);
    end
    L = LuminanceRetrieve(IF_hdr.*gcf,hy,mode);
    if icf_flg %illuminance map with angle-based calibration
        raw_lux = FASTequisolid(IF_hdr,luxmask,gcf,mode);
        illu_map(z,3) = luxCalib(raw_lux,illu_map(z,2),illu_map(z,1),cfh,cfv0,cfv90,cfv180,cfv270);
    else
        illu_map(z,3) = FASTequisolid(IF_hdr,luxmask,gcf,mode);%#ok<*SAGROW> %"eml" or "photopic"
    end
    if cv_flg
        illu_map(z,4) = std(L(:,3))/mean(L(:,3)); %CV map, gen source data
    end
    if ts_ct < ts_size
        ram_ts = [ram_ts IF_hdr]; ts_ct = ts_ct + 1;
    end
end

%% OUTPUT FUNCTIONS SET START HERE!



%% [1] illuminance Map plot
[lx_info] = illuminancePlot(illu_map,tilt_step,aim_step,tilt_a,tilt_b,aim_a,aim_b,...
    'illuminance Map',1); %plot illuminance map

%% retrieve lux map value
lx_map_val = retrieveMapValue(lx_info);
fprintf('CV = %.2f \n', lx_map_val);

%% [2] CV Map plot
if cv_flg
    CVPlot(illu_map,tilt_step,aim_step,tilt_a,tilt_b,aim_a,aim_b,...
        'Coefficient of Variation Map',2) %plot cv map
end

%% [3] Generate statistic report
yn = yn_dialog('Generate a light data report?');
if ismember(yn, ['Yes', 'yes'])
    staReport(illu_map,cv_flg,0);
end

%% [4] Generate stichited panoramic luminance map?
yn = yn_dialog('Generate panoramic luminance map?');
if ismember(yn, ['Yes', 'yes'])
    Luminance_map(pano,'Luminance map',10, 0.25, gcf); % source hdri, name, code, gamma value
end

%% [5] Generate composed luminance & illuminance map, optional...
composedMap(I,"Luminance & Illuminance map",10,0.25,...
    gcf,illu_map,tilt_step,aim_step);

%% extra functions

% [a1 retrieve interested HDR image 180 view
size_ = 1000; % define the pixel resolution
Ret_view = findView(pano_src, size_);
gm_ = 0.4; %gamma value, can be changed
hdrGammaShow(Ret_view, gm_); clear gm_;
%% [a1-1] save this view HDR image
save_name = "hdrSaveDefault.hdr"; % change save name here!!
viewSave(Ret_view, save_name);

%% [a2] show panoramic HDR image
gm_ = 0.4; %gamma value, can be changed
hdrGammaShow(pano_src, gm_); clear gm_;

%% MAIN FUNCTION END HERE !!!





















% obj
%% function Library, DO NOT CHANGE.
function viewSave(view, name)
view = real(view);
vR = view(:,:,1); vR = max(vR,0);
vG = view(:,:,2); vG = max(vG,0);
vB = view(:,:,3); vB = max(vB,0);
view = cat(3, vR, vG, vB);
hdrwrite(view, name);
end

function view = findView(pano, size)
[hva,vva] = ui_input();
IF = imequ2fish_hdr(pano, hva, vva, 0);
view = imresize(IF, [size, size]);
view = imrotate(view,-90);
end

function composedMap(I,figName,fcode,gm,gcf,lux_map,ts,as)
Ipano = imresize(stiching(I),0.25);
lpano = (Ipano(:,:,1).*0.265 + Ipano(:,:,2).*0.670 + Ipano(:,:,3).*0.065).*179.*gcf;
lpano(lpano<0) = 0;
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
[rows,cols] = size(Ipano(:,:,1));
map = imresize(illuminanceMap(lux_map,ts,as),[rows cols]);
hold all;
contour(map,'--k','ShowText','on','LineWidth',3);
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

function CVPlot(imap,ts,as,ta,tb,aa,ab,figName,fcode)
imap(:,1) = (imap(:,1) + max(imap(:,1)))./ts;
imap(:,2) = imap(:,2)./as;
imap(:,1) = imap(:,1)+1; imap(:,2) = imap(:,2)+1;
map = zeros(max(imap(:,1)),max(imap(:,2)));
for w = 1: size(imap,1)
    map(imap(w,1),imap(w,2)) = imap(w,4);
end
cv_img = imresize(map,round((ts+as)/2)); %expanssion rate: default 10.
cv_map(cv_img,ta,tb,aa,ab,figName,fcode)
end

function cv_map(lmap,ta,tb,aa,ab,figName,fcode)
lmap(lmap<0) = 0;lumimg = lmap./max(max(lmap));
gm = 1; lumimg = uint8((lumimg.^gm).*256);
rg = max(max(lmap)); crange = jet(256);crange(1,:) = 0;
cb1 = round(rg.*(0.03316.^(1/gm)),3);cb2 = round(rg.*(0.26754.^(1/gm)),3);
cb3 = round(rg.*(0.50191.^(1/gm)),3);cb4 = round(rg.*(0.73629.^(1/gm)),3);
cb5 = round(rg.*(1.^(1/gm)),3);figure(fcode);imshow(lumimg,'Colormap',crange);
title(['\fontsize{16}\color[rgb]{0 .5 .5}',figName]);
hcb = colorbar('Ticks',[8,68,128,188,248],'TickLabels',{cb1,cb2,cb3,cb4,cb5});
title(hcb,'CV values'); axstep = round(abs(tb-ta)/6);
% x_ticks = ta:(round(ta/3)):tb; y_ticks = aa:(round(ta/3)):ab;  axis on;
x_ticks = aa:axstep:ab; y_ticks = ta:axstep:tb;  axis on;
xp_ticks = linspace(0.5,size(lumimg,2)+0.5,numel(x_ticks));
yp_ticks = linspace(0.5,size(lumimg,1)+0.5,numel(y_ticks));
Xticklabels = cellfun(@(v) sprintf('%d',v), num2cell(x_ticks),...
    'UniformOutput',false);
Yticklabels = cellfun(@(v) sprintf('%d',v), num2cell(y_ticks),...
    'UniformOutput',false);
set(gca,'XTick',xp_ticks); set(gca,'XTickLabels',Xticklabels);
set(gca,'YTick',yp_ticks); set(gca,'YTickLabels',Yticklabels(end:-1:1));
xlabel('\fontsize{12}Horizontal viewing direction/ degree');
ylabel('\fontsize{12}Vertical viewing direction/ degree');
end

function luxmask = equisolidMask(xc,yc,fov)
[xf,yf] = meshgrid(1:xc,1:yc); 
xf = (xf - round(xc/2))./round(xc/2);yf = (yf - round(yc/2))./round(yc/2);
phiS = 2*asind(sqrt(yf.^2+xf.^2)*sind(fov/4));bound = boundGen(xc,yc);
sA = 2*pi/sum(sum(bound)); luxmask = cosd(phiS).*bound.*sA;
luxmask = luxmask.*(pi/sum(sum(luxmask)));
% maskShow(luxmask);
end

function maskShow(luxmask) %#ok<*DEFNU>
immsk = uint8((luxmask - min(min(luxmask)))./(max(max(luxmask))-min(min(luxmask))).*255);
f = figure(1); imshow(immsk); uiwait(f); close all;
end

function boundary = boundGen(x,y)
xc = round(x/2); yc = round(y/2);
r = round(mean([xc,yc])); c = zeros(y,x); [L(:,1),L(:,2)] = find(c == 0);
L(:,3) = sqrt((L(:,1) - yc).^2 + (L(:,2) - xc).^2); L(L(:, 3) > r, :) = [];
for i = 1: size(L,1)
   c(y+1-L(i,1),L(i,2)) = 1;
end
boundary = imbinarize(c,0);
end

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

function [cfh,cfv0,cfv90,cfv180,cfv270] = getluxCF(pano,gcf,sx,sy,f,mode)
[Eh,E0,E90,E180,Eneg90] = reflux_dialog();
illu_map = zeros([5,2]);
illu_map(1,1) = -90; illu_map(3,2) = 90; illu_map(4,2) = 180; illu_map(5,2) = -90;
for z = 1: size(illu_map,1)
    IF_hdr = imequ2fish_hdr(pano,illu_map(z,1),illu_map(z,2),90);
    [hy,hx] = size(IF_hdr(:,:,1));
    if hy ~= hx
        IF_hdr = imresize(IF_hdr,[hy,hy]);
    end
    L = LuminanceRetrieve(IF_hdr.*gcf,hy,mode);%temporary global CF function!
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

function [cfh,cfv0,cfv90,cfv180,cfv270] = getluxCF_2ref(pano,gcf,sx,sy,f,mode)
[Ef,Er] = reflux_dialog_2();
E0 = Ef; E180 = Er;
illu_map = zeros([2,2]);
illu_map(2,2) = 180;
for z = 1: size(illu_map,1)
    IF_hdr = imequ2fish_hdr(pano,illu_map(z,1),illu_map(z,2),90);
    [hy,hx] = size(IF_hdr(:,:,1));
    if hy ~= hx
        IF_hdr = imresize(IF_hdr,[hy,hy]);
    end
    L = LuminanceRetrieve(IF_hdr.*gcf,hy,mode);%temporary global CF function!
    illu_map(z,3) = PerPixel_Fequisolid(hy,hy,sx,sy,f,L);
end
cfv0 = round(E0/illu_map(1,3),2); cfv180 = round(E180/illu_map(2,3),2);
cfv270 = cfv0*0.5 + cfv180*0.5; cfv90 = cfv0*0.5 + cfv180*0.5;
cfh = cfv0*0.5 + cfv180*0.5;
end

function [Ef,Er] = reflux_dialog_2()
prompt = {'Meter measured illuminance (front camera, (0, 0) view)',...
    'Meter measured illuminance (rear camera), (180, 0) view'};
dlgtitle = 'Reference Lux Input'; dims = [1 50];
definput = {'0','0'};
answer = str2double(inputdlg(prompt,dlgtitle,dims,definput));
if isempty(answer)
    Ef = 0; Er = 0;
else
    Ef = answer(1); Er = answer(2);
end
end

function hdrGammaShow(imHDR,gamma)
fr = imHDR(:,:,1);fg = imHDR(:,:,2);fb = imHDR(:,:,3);
fr = single(fr).^gamma;fg = single(fg).^gamma;fb = single(fb).^gamma;
fig1 = real(cat(3,fr,fg,fb));imshow(fig1);
end

function [map] = illuminancePlot(imap,ts,as,ta,tb,aa,ab,figName,fcode)
imap(:,1) = (imap(:,1) + max(imap(:,1)))./ts;
imap(:,2) = imap(:,2)./as;
imap(:,1) = imap(:,1)+1; imap(:,2) = imap(:,2)+1;
map = zeros(max(imap(:,1)),max(imap(:,2)));
for w = 1: size(imap,1)
    map(imap(w,1),imap(w,2)) = imap(w,3);
end
lux_img = imresize(map,round((ts+as)/2)); %expanssion rate: default 10.
ilLuminance_map(lux_img,ta,tb,aa,ab,figName,fcode)
end

function ilLuminance_map(lmap,ta,tb,aa,ab,figName,fcode)
lmap(lmap<0) = 0;lumimg = lmap./max(max(lmap));
gm = 1; lumimg = uint8((lumimg.^gm).*256);
rg = max(max(lmap)); crange = jet(256);crange(1,:) = 0;
cb1 = round(rg.*(0.03316.^(1/gm)),3);cb2 = round(rg.*(0.26754.^(1/gm)),3);
cb3 = round(rg.*(0.50191.^(1/gm)),3);cb4 = round(rg.*(0.73629.^(1/gm)),3);
cb5 = round(rg.*(1.^(1/gm)),3);figure(fcode);imshow(lumimg,'Colormap',crange);
title(['\fontsize{16}\color[rgb]{0 .5 .5}',figName]);
hcb = colorbar('Ticks',[8,68,128,188,248],'TickLabels',{cb1,cb2,cb3,cb4,cb5});
title(hcb,'illuminance (lux)'); axstep = round(abs(tb-ta)/6);
% x_ticks = ta:(round(ta/3)):tb; y_ticks = aa:(round(ta/3)):ab;  axis on;
x_ticks = aa:axstep:ab; y_ticks = ta:axstep:tb;  axis on;
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

function imgF = imequ2fish_hdr(imgE,varargin)
p = inputParser;
addRequired(p,'imgE');
addOptional(p,'roll',  0); % defaul value of roll
addOptional(p,'tilt',  0); % defaul value of tilt
addOptional(p,'pan' ,  0); % defaul value of pan
parse(p,imgE,varargin{:});
we = size(imgE,2); he = size(imgE,1); ch = size(imgE,3);
wf = round(we/2); hf = he;
roll = p.Results.roll; tilt = p.Results.tilt; pan  = p.Results.pan;
[xf,yf] = meshgrid(1:wf,1:hf);
xf = 2*((xf-1)/(wf-1)-0.5); yf = 2*((yf-1)/(hf-1)-0.5); 
idx = sqrt(xf.^2+yf.^2) <= 1; xf = xf(idx); yf = yf(idx);
[xe,ye] = fish2equ(xf,yf,roll,tilt,pan);
Xe = round((xe+1)/2*(we-1)+1); Ye = round((ye+1)/2*(he-1)+1); 
Xf = round((xf+1)/2*(wf-1)+1); Yf = round((yf+1)/2*(hf-1)+1); 
Ie = reshape(imgE,[],ch); If = zeros(hf*wf,ch,'double');
idnf = sub2ind([hf,wf],Yf,Xf);idne = sub2ind([he,we],Ye,Xe);
If(idnf,:) = Ie(idne,:);imgF = reshape(If,hf,wf,3);
end

function [xe,ye] = fish2equ(xf,yf,roll,tilt,pan)
fov = 180; thetaS = atan2d(yf,xf);
% phiS = sqrt(yf.^2+xf.^2)*fov/2; % equidistant proj
phiS = 2*asind(sqrt(yf.^2+xf.^2)*sind(fov/4)); % equisolidangle proj
sindphiS = sind(phiS);
xs = sindphiS.*cosd(thetaS); ys = sindphiS.*sind(thetaS); zs = cosd(phiS);
xyzsz = size(xs); xyz = xyzrotate([xs(:),ys(:),zs(:)],[roll tilt pan]);
xs = reshape(xyz(:,1),xyzsz(1),[]); ys = reshape(xyz(:,2),xyzsz(1),[]);
zs = reshape(xyz(:,3),xyzsz(1),[]);
thetaE = atan2d(xs,zs); phiE   = atan2d(ys,sqrt(xs.^2+zs.^2));
xe = thetaE/180; ye = 2*phiE/180;
end

function [xyznew] = xyzrotate(xyz,thetaXYZ)
tX =  thetaXYZ(1); tY =  thetaXYZ(2); tZ =  thetaXYZ(3);
T = [ cosd(tY)*cosd(tZ),- cosd(tY)*sind(tZ), sind(tY); ...
      cosd(tX)*sind(tZ) + cosd(tZ)*sind(tX)*sind(tY), cosd(tX)*cosd(tZ) - sind(tX)*sind(tY)*sind(tZ), -cosd(tY)*sind(tX); ...
      sind(tX)*sind(tZ) - cosd(tX)*cosd(tZ)*sind(tY), cosd(tZ)*sind(tX) + cosd(tX)*sind(tY)*sind(tZ),  cosd(tX)*cosd(tY)];
xyznew = xyz*T;
end

function imgE = imfish2equ_hdr(imgF,varargin)
p = inputParser; addRequired(p,'imgF');
addOptional(p,'fov' ,180); % defaul value of fov
addOptional(p,'roll',  0); % defaul value of roll
addOptional(p,'tilt',  0); % defaul value of tilt
addOptional(p,'pan' ,  0); % defaul value of pan
parse(p,imgF,varargin{:});
%fisheye image size
wf = size(imgF,2); hf = size(imgF,1); ch = size(imgF,3);
%equirectangular image size
we = wf*2; he = hf;

fov  = p.Results.fov; roll = p.Results.roll;
tilt = p.Results.tilt; pan  = p.Results.pan;
[xe,ye] = meshgrid(1:we,1:he);
xe = 2*((xe-1)/(we-1)-0.5); ye = 2*((ye-1)/(he-1)-0.5); 
[xf,yf] = equ2fish(xe,ye,fov,roll,tilt,pan);
idx = sqrt(xf.^2+yf.^2) <=1; 
xf = xf(idx); yf = yf(idx); xe = xe(idx); ye = ye(idx);
Xe = round((xe+1)/2*(we-1)+1); Ye = round((ye+1)/2*(he-1)+1); 
Xf = round((xf+1)/2*(wf-1)+1); Yf = round((yf+1)/2*(hf-1)+1); 
Ie = reshape(imgF,[],ch); If = zeros(he*we,ch,'single');
idnf = sub2ind([hf,wf],Yf,Xf); idne = sub2ind([he,we],Ye,Xe);
If(idne,:) = Ie(idnf,:);imgE = reshape(If,he,we,3);
end

function [xf,yf] = equ2fish(xe,ye,fov,roll, tilt, pan)
thetaE = xe*180; phiE = ye*90; cosdphiE = cosd(phiE); 
xs = cosdphiE.*cosd(thetaE); ys = cosdphiE.*sind(thetaE); zs = sind(phiE);   
xyzsz = size(xs); xyz = xyzrotate([xs(:),ys(:),zs(:)],[roll tilt pan]);
xs = reshape(xyz(:,1),xyzsz(1),[]); ys = reshape(xyz(:,2),xyzsz(1),[]);
zs = reshape(xyz(:,3),xyzsz(1),[]);
thetaF = atan2d(zs,ys); 
r = 2*atan2d(sqrt(ys.^2+zs.^2),xs)/fov; % equidistant proj
% r = 2*(sind(atan2d(sqrt(ys.^2+zs.^2),xs)/2))/(2*sind(fov/4)); % equisolid-angle proj
xf = r.*cosd(thetaF); yf = r.*sind(thetaF);
end

function [ans1,ans2,ans3,ans4,stp,gcf,CR] = initial_dialog()
prompt = {'Lower limit of tilting angle     [-90 to 90 degree]',...
    'Upper limit of tilting angle     [-90 to 90 degree]',...
    'Lower limit of aimming angle     [-180 to 180 degree]',...
    'Upper limit of aimming angle     [-180 to 180 degree]',...
    'Measuring step angle for retriving illuminance     [5, 10(recommended), 15, 30(fast) degree]',...
    'Input Global calibration factor? [No CF = 1, or input a positive number]',...
    'Compression rate? [No: slowest: 1, slow: 2, fast: 4(recommended), fastest: 8]'};
dlgtitle = 'User Input'; dims = [1 50];
definput = {'-90','90','-180','180','10','1','4'};
answer = str2double(inputdlg(prompt,dlgtitle,dims,definput));
if isempty(answer)
    ans1 = -90; ans2 = 90; ans3 = -180; ans4 = 180; stp = 10; gcf = 1; CR = 4;
else
    ans1 = answer(1); ans2 = answer(2); ans3 = answer(3); ans4 = answer(4);
    stp = answer(5); gcf = answer(6); CR = answer(7);
end
end

function Ipano = imgstiching_hdr(Idf)
h = size(Idf,1); w = size(Idf,2);
c1 = drawcircle('Center',[  w/4,h/2],'Radius',(h/2)*0.98,'Color','red');
c2 = drawcircle('Center',[3*w/4,h/2],'Radius',(h/2)*0.98,'Color','green');
IL = imcrop(Idf,[c1.Center-c1.Radius, c1.Radius*2, c1.Radius*2]);
IR = imcrop(Idf,[c2.Center-c2.Radius, c2.Radius*2, c2.Radius*2]); 
IR = imresize(IR,[size(IL,1), size(IL,2)]); close all;
%camera parameters retrived from multiple attempt, can be improved with
%futher works
fovL  = 183; rollL = 0; tiltL = 0; panL = -2.5; 
fovR  = 185; rollR = -0.3; tiltR = -0.5; panR = 180;
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

function Luminance_map(I,figName,fcode,gm,gcf)
lpano = (I(:,:,1).*0.265 + I(:,:,2).*0.670 + I(:,:,3).*0.065).*179.*gcf;
lpano(lpano<0) = 0;
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

function lumi = LuminanceRetrieve(I,y,mode)
R = I(:,:,1); G = I(:,:,2); B = I(:,:,3); lumi = [];
if mode == "eml"
    Lraw = (R.*0.0013 + G.*0.3812 + B.*0.6475).*179; %EML
else
    Lraw = (R.*0.2126 + G.*0.7152 + B.*0.0722).*179; %Inanici, D65-white
end
[lumi(:,2),lumi(:,1)]=find(Lraw);
for i = 1:size(lumi,1)
    lumi(i,3) = Lraw(lumi(i,2),lumi(i,1));
end
lumi(:,2) = y - lumi(:,2);
end

function lux_cal = luxCalib(lux,aa,ta,cfh,cfv0,cfv90,cfv180,cfv270)
if cfv0 == 0
    cfv0 = (cfv90 + cfv270)/2;
end
if cfv180 == 0
    cfv180 = (cfv90 + cfv270)/2;
end
if aa < 0
    aa = 360 + aa;
end
if aa >=0 && aa < 90
    cfv = (90-aa)/90*cfv0 + (aa)/90*cfv90;
elseif aa>=90 && aa <180
    cfv = (180-aa)/90*cfv90 + (aa-90)/90*cfv180;
elseif aa>=180 && aa<270
    cfv = (270-aa)/90*cfv180 + (aa-180)/90*cfv270;
elseif aa>=270 && aa<=360
    cfv = (360-aa)/90*cfv270 + (aa-270)/90*cfv0;
end
if ta <= 0
    lux_cal = lux*cfv;
else
    lux_cal = lux*((90-ta)/90.*cfv + (ta)/90*cfh);
end
end

function staReport(imap,cv_flg,cr_flg)
th = -30; % statisic starting angle range [-90 to 90], 0: defualt
lux_max = max(imap(:,3)); %max lux number
row_pos = (-1.*imap(:,1)>= th); 
lux_avg = mean(nonzeros(imap(:,3).*row_pos));
lux_std = std(nonzeros(imap(:,3).*row_pos));
%CV results
if cv_flg
    cv_max = max(imap(:,4)); %most unnuniform
    ycv = find(imap(:,4) == cv_max);
end
%CR results
if cr_flg
    crln_max = max(imap(:,5)); %highest near background contrast point
    crlf_max = max(imap(:,6)); %highest far background contrast point
    crrn_max = max(imap(:,7)); %highest near background contrast point
    crrf_max = max(imap(:,8)); %highest far background contrast point
    ycrln = find(imap(:,5) == crln_max);
    ycrlf = find(imap(:,6) == crlf_max);
    ycrrn = find(imap(:,7) == crrn_max);
    ycrrf = find(imap(:,8) == crrf_max);
end

ylux = find(imap(:,3) == lux_max);
yev = find(imap(:,1) == 0);
yev(:,2) = imap(yev(:,1),3);
yev(:,3) = find(imap(:,1) == -90);
yev(:,4) = imap(yev(:,3),3);
fprintf('Brief Uniformity Data Evaluation: \n');
fprintf('\n[1] Aiming direction of max illuminance\n');
fprintf('Horizontal : %d(deg); Vertical : %d(deg); Max illuminance: %.2f(lux)\n',...
    imap(ylux(1),2)-180,-1.*imap(ylux(1),1),lux_max);
fprintf('\n[2] Average illuminance of veritcal aiming direction >= %d (deg)\n',th);
fprintf('Average illuminance: %.2f(lux); Standard Deviation: %.2f(lux)\n', lux_avg, lux_std);
if cv_flg
    fprintf('\n[3] Aiming Direction with <MOST Ununiform> lighting condition\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); Coefficient of Variance: %.3f\n',...
        imap(ycv,2)-180,-1.*imap(ycv,1),cv_max);
end
if cr_flg
    fprintf('\n[4-1] Highest near background luminance contrast point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (NEAR)contrast-ratio: %.3f\n',...
        imap(ycrln(1),2)-180,-1.*imap(ycrln(1),1),crln_max);
    fprintf('\n[4-2] Highest far background luminance contrast point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (FAR)contrast-ratio: %.3f\n',...
        imap(ycrlf(1),2)-180,-1.*imap(ycrlf(1),1),crlf_max);
    fprintf('\n[4-3] Highest near background luminance ratio point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (NEAR)contrast-ratio: %.3f\n',...
        imap(ycrrn(1),2)-180,-1.*imap(ycrrn(1),1),crrn_max);
    fprintf('\n[4-4] Highest far background luminance ratio point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (FAR)contrast-ratio: %.3f\n',...
        imap(ycrrf(1),2)-180,-1.*imap(ycrrf(1),1),crrf_max);
end
fprintf('\n[5] Veritcal illuminance (Ev) = %.2f lux \n',mean(yev(:,2)));
fprintf('    Horizontal illuminance (Eh) = %.2f lux \n',mean(yev(:,4)));
end

function VEM = vecorf()
%% parameters of retrived single fisheye image
px = 7296; py = 3648; dfish = 1789;
%%
VEM = single(zeros(py,px)); VEM = VEM + 1;
% y = -0.1349.*x.^(3) + 0.0506.*x.^(2) - 0.0139.*x + 0.9996;
% pre-calculated!! 
%vegnetting curve function
c1x = px/4; c1y = py/2;
c2x = px/4*3; c2y = py/2;
for x = 1:size(VEM,2)
    if x < (px/2) + 1
        for y = 1: size(VEM,1)
            d = ((x - c1x).^2 + (y - c1y).^2).^(0.5);
            if d > dfish %#ok<*BDSCI>
                continue
            end
            d = d/dfish;
            VEM(y,x) = -0.1349.*d.^(3) + 0.0506.*d.^(2) - 0.0139.*d + 0.9996;
        end
    else
        for y = 1: size(VEM,1)
            d = ((x - c2x).^2 + (y - c2y).^2).^(0.5);
            if d > dfish
                continue
            end
            d = d/dfish;
            VEM(y,x) = -0.1349.*d.^(3) + 0.0506.*d.^(2) - 0.0139.*d + 0.9996;
        end
    end
end
end

function tf = tf_dialog(ques)
opts.Interpreter = 'tex'; opts.Default = '5-ref';
tf = questdlg(ques,'Dialog Window',...
    '2-ref','5-ref',opts);
end

function yn = yn_dialog(ques)
opts.Interpreter = 'tex'; opts.Default = 'No';
yn = questdlg(ques,'Dialog Window',...
    'Yes','No',opts);
end

function [lx] = FASTequisolid(hdr,luxmask,CF, perception)
if perception == "eml"
    lmap = (hdr(:,:,1).*0.0013 + hdr(:,:,2).*0.3812 + hdr(:,:,3).*0.6475).*179.*CF;
else
    lmap = (hdr(:,:,1).*0.265 + hdr(:,:,2).*0.670 + hdr(:,:,3).*0.065).*179.*CF;
end
lx = sum(sum(lmap.*luxmask));
end

function [IL] = PerPixel_Fequisolid(x,y,sx,sy,f,L)
cx = max(L(:,1)) - min(L(:,1));
if cx ~= (max(L(:,2)) - min(L(:,2)))
    msgbox('Probe image not aligned!','Error','error');
    error('Error_001: image not aligned!'); return; %#ok<*UNRCH>
end
dis = sqrt(((L(:,1)-x/2).*sx./x).^2+((L(:,2)-y/2).*sy./y).^2);
a1 = 2.*asin(dis./f./2); a1 = (pi/2).*(a1 - min(a1))/(max(a1) - min(a1));
L(:,4) = cos(a1).*(2*pi/size(L,1)); L(:,4) = L(:,4).*(pi/sum(L(:,4)));
MAP = L(:,1:2); MAP(:,3) = L(:,3).*L(:,4);IL = sum(MAP(:,3));
end

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