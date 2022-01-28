% get the contrast ratio
% ref_equisolid: d = 2sin(theta/2)*focal_length
% notes: cr_l is luminance contrast, cr_m is luminance modulation
% n:near f:far

function [cr_ln, cr_lf, cr_rn, cr_rf] = getContrast(vI, gcf, hy, msk2, msk4, msk10, msk30, msk90)
%% convert HDRi to luminance map
Lmap = (vI(:,:,1).*0.2126 + vI(:,:,2).*0.7152 + vI(:,:,3).*0.0722).*179.*gcf; %Inanici, D65-white
% clear vI;
%% correction & create roi
msk2 = imresize(msk2,[hy,hy]); msk4 = imresize(msk4,[hy,hy]);
msk10 = imresize(msk10,[hy,hy]);msk30 = imresize(msk30,[hy,hy]);
msk90 = imresize(msk90,[hy,hy]);
r2 = Lmap.*msk2; r4 = Lmap.*msk4; r10 = Lmap.*msk10; 
r30 = Lmap.*msk30; r90 = Lmap.*msk90;

%%
cflg = 0; %0 is small obj, 1 is large obj
%% question here! using average value????
ev2 = mean(nonzeros(r2)); ev4 = mean(nonzeros(r4)); ev10 = mean(nonzeros(r10));
evn = mean(nonzeros(r30 - r10)); evf = mean(nonzeros(r90 - r30));
%%
if abs(ev2 - ev4)/ev4 >= 0.05 % 0.95
    cflg = 1;
end
if cflg % large(or close) obj
    cr_ln = abs(ev10 - evn)/max(ev10, evn);
    cr_rn = max(ev10, evn)/min(ev10, evn);
    cr_lf = abs(ev10 - evf)/max(ev10, evf);
    cr_rf = max(ev10, evf)/min(ev10, evf);
else % small(or far) obj
    cr_ln = abs(ev2 - evn)/max(ev2, evn);
    cr_rn = max(ev2, evn)/min(ev2, evn);
    cr_lf = abs(ev2 - evf)/max(ev2, evf);
    cr_rf = max(ev2, evf)/min(ev2, evf);
end
end
