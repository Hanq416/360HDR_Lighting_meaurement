function yn = yn_dialog(ques)
opts.Interpreter = 'tex'; opts.Default = 'No';
yn = questdlg(ques,'Dialog Window',...
    'Yes','No',opts);
end