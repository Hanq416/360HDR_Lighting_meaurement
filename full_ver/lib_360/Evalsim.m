% similarity evaluation of lighting model

function Evalsim(Dmap)
[fn,pn]=uigetfile('*.mat','load the reference data file of target space');str=[pn,fn];
ref = load(str); ref = ref.illu_map;
if size(Dmap,1) ~= size(ref,1)
    error('dimension not matched! check both matrix\n');
    return
end
cv_ref = ref(:,4); %read cv map of reference map
cv_tar = Dmap(:,4); %read cv map of target data map
lux_ref = ref(:,3); %read lux map of reference map
lux_tar = Dmap(:,3); %read lux map of reference map
pcv = corrcoef(cv_ref,cv_tar); % pearson correlation on CV
plux = corrcoef(lux_ref,lux_tar); % pearson correlation on illuminance
dlux_avg = mean(abs(lux_tar - lux_ref)); % Average difference of all directional illuminance
dlux_max = max(abs(lux_tar - lux_ref)); % Highest difference of all directional illuminance
y_dlux = find(abs(lux_tar - lux_ref) == dlux_max); % Retrieve Aiming direction with highest lux difference
fprintf('\n Lighting model similarity evaluation report: \n\n');
fprintf('[1] Similarity of luminance distribution (CV map); Pearson correlation = %.3f \n',pcv(1,2));
fprintf('[2] Similarity of illuminance distribution (lux map); Pearson correlation = %.3f \n',plux(1,2));
fprintf('[3] Average difference of illuminance on all direction; AVG_diff = %.2f lux\n',dlux_avg);
fprintf('[4] Highest difference of illuminance on all direction; MAX_diff = %.2f lux\n',dlux_max);
fprintf('    Viewing Direction of highest difference of illuminance; H: %d(deg), V: %d(deg)\n',...
    Dmap(y_dlux(1),2),-1.*Dmap(y_dlux(1),1));
Relative_errorMAP(Dmap, ref)
end

function Relative_errorMAP(TarMAP, RefMAP)
RelMap = TarMAP; RelMap(:,4) = [];
RelMap(:,3) = (TarMAP(:,3) - RefMAP(:,3))./RefMAP(:,3);
fprintf('[5] Relative illuminance error rate: %.2f \n',mean(abs(RelMap(:,3))));
ts = abs(RelMap(1,1) - RelMap(2,1)); as = ts;
ta = min(RelMap(:,1)); tb = max(RelMap(:,1));
aa = min(RelMap(:,2)); ab = max(RelMap(:,2));
relativePlot(RelMap,ts,as,ta,tb,aa,ab)
end

function relativePlot(imap,ts,as,ta,tb,aa,ab)
imap(:,1) = (imap(:,1) + max(imap(:,1)))./ts;
imap(:,2) = (imap(:,2) + max(imap(:,2)))./as;
imap(:,1) = imap(:,1)+1; imap(:,2) = imap(:,2)+1;
map = zeros(max(imap(:,1)),max(imap(:,2)));
for w = 1: size(imap,1)
    map(imap(w,1),imap(w,2)) = imap(w,3).*100;
end
lux_error_img = imresize(map,round((ts+as)/2)); %expanssion rate: default 10.
P_img = lux_error_img; P_img(P_img<0) = 0;
N_img = lux_error_img; N_img(N_img>0) = 0; N_img = - N_img;
relerror_map(P_img,ta,tb,aa,ab,1,3,'Positive Relative illuminance error map (Target > Reference)')
relerror_map(N_img,ta,tb,aa,ab,-1,4,'Negative Relative illuminance error map (Target < Reference)')
end

function relerror_map(lmap,ta,tb,aa,ab,PN,fign,name)
lumimg = (lmap - min(min(lmap)))./(max(max(lmap))- min(min(lmap)));
gm = 1; lumimg = uint8((lumimg.^gm).*256); rg = max(max(lmap)) - min(min(lmap));
cb1 = PN*round(rg.*(0.03316.^(1/gm)),3);cb2 = PN*round(rg.*(0.26754.^(1/gm)),3);
cb3 = PN*round(rg.*(0.50191.^(1/gm)),3);cb4 = PN*round(rg.*(0.73629.^(1/gm)),3);
cb5 = PN*round(rg.*(1.^(1/gm)),3); crange = jet(256); crange(1,:) = 0;
figure(fign); imshow(lumimg,'Colormap',crange);
title(['\fontsize{14}\color[rgb]{0 .5 .5}', name]);
hcb = colorbar('Ticks',[8,68,128,188,248],'TickLabels',{cb1,cb2,cb3,cb4,cb5});
title(hcb,'Error(%)'); axstep = round(abs(tb-ta)/6);
x_ticks = aa:axstep:ab; y_ticks = ta:axstep:tb;  axis on;
xp_ticks = linspace(0.5,size(lumimg,2)+0.5,numel(x_ticks));
yp_ticks = linspace(0.5,size(lumimg,1)+0.5,numel(y_ticks));
Xticklabels = cellfun(@(v) sprintf('%d',v), num2cell(x_ticks),...
    'UniformOutput',false);
Yticklabels = cellfun(@(v) sprintf('%d',v), num2cell(y_ticks),...
    'UniformOutput',false);
set(gca,'XTick',xp_ticks); set(gca,'XTickLabels',Xticklabels);
set(gca,'YTick',yp_ticks); set(gca,'YTickLabels',Yticklabels(end:-1:1));
xlabel('Horizontal aiming direction/ degree');
ylabel('Vertical aiming direction/ degree');
end