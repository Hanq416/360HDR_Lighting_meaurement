%illuminance calculation
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