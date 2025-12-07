function make_mfcc_csv(filename)
    % CSVファイルからテキストを読み込み、MFCC特徴量と音声データを生成
    % 入力: train_data.csv（1列目=テキスト、2列目以降=ラベル等）
    % 出力: target/フォルダにMFCCデータ、音声、タイムスタンプを保存
    
    mylib = Mylib();
    csv_data = readtable(filename, 'ReadVariableNames', false);
    
    prepare_output_directories();
    
    mfcc_data = process_all_texts(mylib, csv_data);
    
    save_combined_data(mfcc_data, csv_data, filename);
    
    exit
end

function prepare_output_directories()
    % 出力先ディレクトリを準備
    directories = {'..\target\timestamp\', '..\target\voice\'};
    
    for i = 1:length(directories)
        if ~exist(directories{i}, 'dir')
            mkdir(directories{i});
        end
    end
end

function mfcc_data = process_all_texts(mylib, csv_data)
    % 全テキストを処理してMFCC特徴量を抽出
    mfcc_data = [];
    num_texts = size(csv_data, 1);
    
    for i = 1:num_texts
        text = string(csv_data{i, 1});
        fprintf('処理中 (%d/%d): %s\n', i, num_texts, text);
        
        mfcc_row = process_single_text(mylib, text);
        mfcc_data = [mfcc_data; mfcc_row];
    end
end

function mfcc_row = process_single_text(mylib, text)
    % 1つのテキストを音声に変換し、MFCC特徴量を抽出
    
    % 音声生成とタイムスタンプ取得
    timestamp_table = mylib.make_wave(text);
    
    % タイムスタンプをCSV保存
    save_timestamp(text, timestamp_table);
    
    % MFCC特徴量を計算
    mfcc_row = mylib.calc_mfcc();
    
    % 音声ファイル保存
    save_voice_file(mylib, text);
end

function save_timestamp(text, timestamp_table)
    % タイムスタンプをCSVファイルとして保存
    filepath = sprintf('..\\target\\timestamp\\%s.csv', text);
    try
        writetable(timestamp_table, filepath, ...
                   'Encoding', 'Shift_JIS', ...
                   'WriteVariableNames', false);
    catch ME
        % Shift_JISエラーの場合はUTF-8で保存
        if contains(ME.message, 'encode')
            fprintf('警告: %s はShift_JISで保存できません。UTF-8で保存します。\n', text);
            writetable(timestamp_table, filepath, ...
                       'Encoding', 'UTF-8', ...
                       'WriteVariableNames', false);
        else
            rethrow(ME);
        end
    end
end

function save_voice_file(mylib, text)
    % 音声ファイルを保存
    filepath = sprintf('..\\target\\voice\\%s.wav', text);
    mylib.save_recent_wave(filepath);
end

function save_combined_data(mfcc_data, csv_data, input_filename)
    % MFCC特徴量とラベルを結合して保存
    
    % ラベル列（2列目以降）を取得
    label_columns = table2array(csv_data(:, 2:end));
    
    % MFCC + ラベルを結合
    combined_data = [mfcc_data, label_columns];
    
    % 出力ファイル名を生成（例: train_data.csv -> target/train_data.csv）
    output_filename = sprintf('..\\target\\%s', extract_filename(input_filename));
    
    writematrix(combined_data, output_filename);
    fprintf('結合データを保存: %s\n', output_filename);
end

function filename = extract_filename(filepath)
    % ファイルパスからファイル名を抽出
    [~, name, ext] = fileparts(filepath);
    filename = [name, ext];
end