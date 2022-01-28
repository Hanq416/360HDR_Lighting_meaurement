%selection 2 points or 5 pionts
function tf = tf_dialog(ques)
opts.Interpreter = 'tex'; opts.Default = '5-ref';
tf = questdlg(ques,'Dialog Window',...
    '2-ref','5-ref',opts);
end