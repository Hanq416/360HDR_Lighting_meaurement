% Generate a angular map
function luxmask = equisolidMask(xc,yc,fov)
%equisolid projection
[xf,yf] = meshgrid(1:xc,1:yc); 
xf = (xf - round(xc/2))./round(xc/2);
yf = (yf - round(yc/2))./round(yc/2);
phiS = 2*asind(sqrt(yf.^2+xf.^2)*sind(fov/4));
bound = boundGen(xc,yc);
sA = 2*pi/sum(sum(bound)); luxmask = cosd(phiS).*bound.*sA;
luxmask = luxmask.*(pi/sum(sum(luxmask)));
% maskShow(luxmask);
end

function maskShow(luxmask) %#ok<*DEFNU>
immsk = uint8((luxmask - min(min(luxmask)))./(max(max(luxmask))-min(min(luxmask))).*255);
f = figure(1); imshow(immsk);
uiwait(f); close all;
end

function boundary = boundGen(x,y)
xc = round(x/2); yc = round(y/2);
r = round(mean([xc,yc]));
c = zeros(y,x); [L(:,1),L(:,2)] = find(c == 0);
L(:,3) = sqrt((L(:,1) - yc).^2 + (L(:,2) - xc).^2);
L(L(:, 3) > r, :) = [];
for i = 1: size(L,1)
   c(y+1-L(i,1),L(i,2)) = 1;
end
boundary = imbinarize(c,0);
end
