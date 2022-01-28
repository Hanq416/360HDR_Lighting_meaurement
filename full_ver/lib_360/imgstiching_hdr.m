% Ricoh Theta Z1, 360 Camera image(jpg) stiching
% Copyright of Original Function: Kazuya Machida (2020)

% Modified by Hankun Li, University of Kansas, Aug 18,2020
% For research use of KU Lighting Research Laboratory

% Reference: 
% [1] 360-degree-image-processing (https://github.com/k-machida/360-degree-image-processing), GitHub. Retrieved August 18, 2020.
% [2] Tuan Ho, Madhukar Budagavi,  "2DUAL-FISHEYE LENS STITCHING FOR 360-DEGREE IMAGING"

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

function [IE2,mask] = trimImageByFov(IE,fov,pan)
w  = int32(size(IE,2)); we = w*(fov/360)/2; ce = mod(w*(0.5+pan/360),w);
idx = [ones(1,we),zeros(1,w-2*we),ones(1,we)]; idx = circshift(idx,ce);
IE2 = IE; IE2(:,~idx,:) = 0; mask = repmat(idx,[size(IE2,1), 1, size(IE2,3)]);
end