%plot CV map
function [cv_img] = CVPlot(imap,ts,as,ta,tb,aa,ab,figName,fcode)
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

%plot ref
%gv_tick=log(gvmin):(log(gvmax)-log(gvmin))/9 : log(gvmax); gv_label1=exp(gv_tick); %10/21/2020