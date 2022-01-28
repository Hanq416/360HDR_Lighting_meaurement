%Retriving Luminance Map
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