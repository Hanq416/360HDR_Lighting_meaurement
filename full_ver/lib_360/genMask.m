% generate binary mask for calculating contrast
function [msk2,msk4,msk10,msk30,msk90] = genMask(f, sy, iy)
%%
cx = iy/2; cy = iy/2;
rawMsk = single(zeros(iy,iy)); msk2 = imbinarize(rawMsk); 
msk4 = imbinarize(rawMsk); msk10 = imbinarize(rawMsk);
msk30 = imbinarize(rawMsk); msk90 = imbinarize(rawMsk);
r2 = round(2.*sind(2/4).*f./sy.*iy); r4 = round(2.*sind(4/4).*f./sy.*iy);  
r10 = round(2.*sind(10/4).*f./sy.*iy); r30 = round(2.*sind(30/4).*f./sy.*iy);
r90 = round(2.*sind(90/4).*f./sy.*iy);
%%
for x = 1:size(rawMsk,2)
    for y = 1:size(rawMsk,1)
        md = round(((x - cx).^2 + (y - cy).^2).^(0.5));
        if md <= r2
            msk2(y,x) = 1; msk4(y,x) = 1; msk10(y,x) = 1; msk30(y,x) = 1; msk90(y,x) = 1;
        elseif md <= r4
            msk4(y,x) = 1; msk10(y,x) = 1; msk30(y,x) = 1; msk90(y,x) = 1;
        elseif md <= r10
            msk10(y,x) = 1; msk30(y,x) = 1; msk90(y,x) = 1;
        elseif md<= r30
            msk30(y,x) = 1; msk90(y,x) = 1;
        elseif md<= r90
            msk90(y,x) = 1;
        end
    end
end
end
