% Equalrectangular to 180-degree fisheye, [JPEG image version]
% Copyright of Original Function: Kazuya Machida (2020)

% illuminance calculation module for Ricoh Z1 360-degree
% Modified by Hankun Li for KU LRL research use, University of Kansas Aug,18,2020

% Reference: 
% [1] 360-degree-image-processing (https://github.com/k-machida/360-degree-image-processing), GitHub. Retrieved August 18, 2020.
% [2] Tuan Ho, Madhukar Budagavi,  "2DUAL-FISHEYE LENS STITCHING FOR 360-DEGREE IMAGING"


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
% Convert to normalized unit
xf = 2*((xf-1)/(wf-1)-0.5); yf = 2*((yf-1)/(hf-1)-0.5); 
% Get index of valid fisyeye image area
idx = sqrt(xf.^2+yf.^2) <= 1; xf = xf(idx); yf = yf(idx);
[xe,ye] = fish2equ(xf,yf,roll,tilt,pan);
% Convert normalized unit to pixel
Xe = round((xe+1)/2*(we-1)+1); Ye = round((ye+1)/2*(he-1)+1); 
Xf = round((xf+1)/2*(wf-1)+1); Yf = round((yf+1)/2*(hf-1)+1); 
Ie = reshape(imgE,[],ch); If = zeros(hf*wf,ch,'double');
idnf = sub2ind([hf,wf],Yf,Xf);idne = sub2ind([he,we],Ye,Xe);
If(idnf,:) = Ie(idne,:);imgF = reshape(If,hf,wf,3);
end

% Coordinate transform
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

% Aiming direction change
function [xyznew] = xyzrotate(xyz,thetaXYZ)
tX =  thetaXYZ(1); tY =  thetaXYZ(2); tZ =  thetaXYZ(3);
T = [ cosd(tY)*cosd(tZ),- cosd(tY)*sind(tZ), sind(tY); ...
      cosd(tX)*sind(tZ) + cosd(tZ)*sind(tX)*sind(tY), cosd(tX)*cosd(tZ) - sind(tX)*sind(tY)*sind(tZ), -cosd(tY)*sind(tX); ...
      sind(tX)*sind(tZ) - cosd(tX)*cosd(tZ)*sind(tY), cosd(tZ)*sind(tX) + cosd(tX)*sind(tY)*sind(tZ),  cosd(tX)*cosd(tY)];
xyznew = xyz*T;
end
