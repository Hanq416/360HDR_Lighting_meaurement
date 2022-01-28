% Vegnetting correction:
% pre-calculation required
function VEM = vecorf()
%% parameters of retrived single fisheye image
px = 7296; py = 3648; dfish = 1789;
%%
VEM = single(zeros(py,px)); VEM = VEM + 1;
% y = -0.1349.*x.^(3) + 0.0506.*x.^(2) - 0.0139.*x + 0.9996;
% pre-calculated!! 
%vegnetting curve function
c1x = px/4; c1y = py/2;
c2x = px/4*3; c2y = py/2;
for x = 1:size(VEM,2)
    if x < (px/2) + 1
        for y = 1: size(VEM,1)
            d = ((x - c1x).^2 + (y - c1y).^2).^(0.5);
            if d > dfish %#ok<*BDSCI>
                continue
            end
            d = d/dfish;
            VEM(y,x) = -0.1349.*d.^(3) + 0.0506.*d.^(2) - 0.0139.*d + 0.9996;
        end
    else
        for y = 1: size(VEM,1)
            d = ((x - c2x).^2 + (y - c2y).^2).^(0.5);
            if d > dfish
                continue
            end
            d = d/dfish;
            VEM(y,x) = -0.1349.*d.^(3) + 0.0506.*d.^(2) - 0.0139.*d + 0.9996;
        end
    end
end
end