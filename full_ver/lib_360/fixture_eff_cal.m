% Author of original function: Siqi He
% modified by HLi

function [fixtEff] = fixture_eff_cal(lx_info, percieveArea, fixture_lumen, plot_key)
%%
[mgaze,~] = gazeMap(plot_key); [Ygaze, Xgaze] = size(mgaze);
%%
lx_info = imresize(lx_info,[Ygaze Xgaze]);
fixtEff = sum(sum(mgaze.*lx_info)).*percieveArea./(fixture_lumen.*sum(sum(mgaze)));
fprintf('Fixture efficiency = %.8f percent \n',fixtEff*100);
end

function [mgaze, time] = gazeMap(pkey)
[fn,pn] = uigetfile('*.csv','load gaze data file.');
fname = [pn,fn]; gaze = readmatrix(fname); clear pn fn fname;
timescale = input("input time scale factor of the gaze map, default = 1\n");
vXEdge = linspace(-180,180,37); vYEdge = linspace(-90,90,19);
mgaze = hist2d (gaze, vYEdge, vXEdge).*timescale;
if pkey
    Plot2dHist(mgaze, vXEdge, vYEdge, 'Horizontal angle (degree)', 'Vertical angle (degree)', 'Gaze Heatmap');
end
time = size(gaze,1)*timescale;
end

function mHist = hist2d (mX, vYEdge, vXEdge)
nCol = size(mX, 2);
if nCol < 2
    error ('mX has less than two columns')
end
nRow = length (vYEdge)-1;nCol = length (vXEdge)-1;
vRow = mX(:,1);vCol = mX(:,2); mHist = zeros(nRow,nCol);
for iRow = 1:nRow
    rRowLB = vYEdge(iRow);rRowUB = vYEdge(iRow+1);
    vColFound = vCol((vRow > rRowLB) & (vRow <= rRowUB));
    if (~isempty(vColFound))
        vFound = histc(vColFound, vXEdge);
        nFound = (length(vFound)-1);
        if (nFound ~= nCol)
            disp([nFound nCol])
            error ('hist2d error: Size Error')
        end
        [nRowFound, nColFound] = size (vFound);
        nRowFound = nRowFound - 1; nColFound = nColFound - 1;  
        if nRowFound == nCol
            mHist(iRow, :)= vFound(1:nFound)';
        elseif nColFound == nCol
            mHist(iRow, :)= vFound(1:nFound);
        else
            error ('hist2d error: Size Error')
        end
    end
end
end

function Plot2dHist(mHist2D, vEdgeX, vEdgeY, strLabelX, strLabelY, strTitle)
nEdgeX = length(vEdgeX)-1; %#ok<*NASGU>
nEdgeY = length(vEdgeY)-1;
rMinX = min(vEdgeX);rMaxX = max(vEdgeX);
rMinY = min(vEdgeY);rMaxY = max(vEdgeY);
rDeltaX = (vEdgeX(2)-vEdgeX(1));rDeltaY = (vEdgeY(2)-vEdgeY(1));
vLabelX = (rMinX+0.5*rDeltaX):rDeltaX:(rMaxX-0.5*rDeltaX);
vLabelY = (rMinY+0.5*rDeltaY):rDeltaY:(rMaxY-0.5*rDeltaY);
pcolor (vLabelX, vLabelY, mHist2D);
shading interp; colorbar; grid on; 
xlabel(strLabelX); ylabel(strLabelY);
xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180]);
yticks([-90,-60,-30,0,30,60,90]);
axis equal;xlim([-180,180]);ylim([-90,90]);
title(strTitle)
end