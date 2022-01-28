% illuminance calculation module for Ricoh Z1 360-degree
% Copyright of Original Function: Kazuya Machida (2020)

% Modified by Hankun Li, University of Kansas, Aug,18,2020
% For research use of KU Lighting Research Laboratory

% Reference: 
% [1] 360-degree-image-processing (https://github.com/k-machida/360-degree-image-processing), GitHub. Retrieved August 18, 2020.
% [2] Tuan Ho, Madhukar Budagavi,  "2DUAL-FISHEYE LENS STITCHING FOR 360-DEGREE IMAGING"


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

function [xyznew] = xyzrotate(xyz,thetaXYZ)
tX =  thetaXYZ(1); tY = thetaXYZ(2); tZ =  thetaXYZ(3);
T = [cosd(tY)*cosd(tZ), -cosd(tY)*sind(tZ), sind(tY); ...
      cosd(tX)*sind(tZ) + cosd(tZ)*sind(tX)*sind(tY), cosd(tX)*cosd(tZ) - sind(tX)*sind(tY)*sind(tZ), -cosd(tY)*sind(tX); ...
      sind(tX)*sind(tZ) - cosd(tX)*cosd(tZ)*sind(tY), cosd(tZ)*sind(tX) + cosd(tX)*sind(tY)*sind(tZ),  cosd(tX)*cosd(tY)];
xyznew = xyz*T;
end
