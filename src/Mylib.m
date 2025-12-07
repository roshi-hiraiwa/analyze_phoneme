classdef Mylib
    properties
        key
    end
    
    properties (Constant)
        MARGIN_SEC = 0.05
        TARGET_LENGTH = 50000
        VOICE_NAME = 'ja-JP-Standard-A'
        API_ENDPOINT = 'https://texttospeech.googleapis.com/v1beta1/text:synthesize'
    end
    
    methods
        function obj = Mylib()
            addpath(genpath('.\MIRtoolbox1.6.1'))
            obj.key = obj.load_api_key();
        end
        
        function time_table = make_wave(obj, text)
            % 日本語テキストを音声に変換し、各音節のタイムスタンプを返す
            
            syllables = obj.parse_japanese_syllables(text);
            ssml_str = obj.create_ssml(syllables);
            resp_json = obj.call_google_tts_api(ssml_str);
            
            obj.save_audio_with_margin(resp_json.audioContent);
            time_table = obj.extract_timestamps(resp_json, syllables);
        end

        function y = calc_mfcc(~)
            % 音声ファイルからMFCC特徴量を抽出
            ceps = mirmfcc('.\voice.wav', 'Frame', .1, 'Rank', 1:6);
            ceps1 = mirgetdata(ceps);
            y = reshape(ceps1, [1, 240]);
        end
        
        function save_recent_wave(~, save_name)
            % 最近生成した音声ファイルを保存
            copyfile('.\voice.wav', save_name);
        end
    end
    
    methods (Access = private)
        function api_key = load_api_key(~)
            % .envファイルからGOOGLE_API_KEYを読み込む
            env_file = '.env';
            
            % プロジェクトルートの.envファイルを探す
            if ~exist(env_file, 'file')
                env_file = '..\\.env';
            end
            
            if ~exist(env_file, 'file')
                error('.envファイルが見つかりません。プロジェクトルートに.envファイルを作成してGOOGLE_API_KEYを設定してください。');
            end
            
            % .envファイルを読み込む
            fid = fopen(env_file, 'r');
            if fid == -1
                error('.envファイルを開けませんでした: %s', env_file);
            end
            
            api_key = '';
            try
                while ~feof(fid)
                    line = fgetl(fid);
                    if ischar(line) && contains(line, 'GOOGLE_API_KEY')
                        % コメント行をスキップ
                        if startsWith(strtrim(line), '#')
                            continue;
                        end
                        % GOOGLE_API_KEY=値 の形式を解析
                        parts = strsplit(line, '=');
                        if length(parts) >= 2
                            api_key = strtrim(strjoin(parts(2:end), '='));
                            % 引用符を削除
                            api_key = strrep(api_key, '"', '');
                            api_key = strrep(api_key, '''', '');
                            break;
                        end
                    end
                end
            catch ME
                fclose(fid);
                rethrow(ME);
            end
            fclose(fid);
            
            if isempty(api_key)
                error('.envファイルにGOOGLE_API_KEYが見つかりませんでした。');
            end
        end
        
        function syllables = parse_japanese_syllables(~, text)
            % 日本語テキストを音節単位に分解（拗音・長音を考慮）
            raw_chars = char(text);
            syllables = cell(0);
            
            % 小さい文字と伸ばし棒（前の文字と結合する文字）
            suffix_chars = ['ぁ','ぃ','ぅ','ぇ','ぉ','ゃ','ゅ','ょ','ゎ',...
                           'ァ','ィ','ゥ','ェ','ォ','ャ','ュ','ョ','ヮ','ー'];
            
            k = 1;
            while k <= length(raw_chars)
                current_unit = raw_chars(k);
                k = k + 1;
                
                % 後続の結合文字をすべて連結
                while k <= length(raw_chars) && ismember(raw_chars(k), suffix_chars)
                    current_unit = [current_unit, raw_chars(k)];
                    k = k + 1;
                end
                
                syllables{end+1, 1} = current_unit;
            end
        end
        
        function ssml_str = create_ssml(~, syllables)
            % 音節リストからSSML文字列を生成
            ssml_str = "<speak>";
            
            for i = 1:length(syllables)
                ssml_str = ssml_str + "<mark name=""" + (i-1) + """/>" + string(syllables{i});
            end
            
            % 終了位置マーク
            ssml_str = ssml_str + "<mark name=""" + length(syllables) + """/></speak>";
        end
        
        function resp_json = call_google_tts_api(obj, ssml_str)
            % Google Text-to-Speech APIを呼び出す
            request = struct(...
                'input', struct('ssml', ssml_str), ...
                'voice', struct('languageCode', 'ja-JP', 'name', obj.VOICE_NAME), ...
                'audioConfig', struct('audioEncoding', 'LINEAR16', 'pitch', '0.00', 'speakingRate', '1.00', 'sampleRateHertz', 24000), ...
                'enableTimePointing', ["SSML_MARK"]);

            obj.write_json('query.json', request);
            
            cmd = sprintf('curl -H "Content-Type: application/json; charset=utf-8" --data-binary @query.json "%s?key=%s" > voice.txt', ...
                         obj.API_ENDPOINT, obj.key);
            
            [status, cmdout] = system(cmd);
            if status ~= 0
                error('Google APIへの接続に失敗しました: %s', cmdout);
            end

            resp_json = obj.parse_api_response('voice.txt');
        end
        
        function resp_json = parse_api_response(~, filename)
            % APIレスポンスを解析してエラーチェック
            resp_txt = fileread(filename);
            
            try
                resp_json = jsondecode(resp_txt);
            catch
                error('JSONの解析に失敗しました。レスポンス: %s', resp_txt);
            end
            
            if isfield(resp_json, 'error')
                error('APIエラー: %s', resp_json.error.message);
            end
            
            if ~isfield(resp_json, 'audioContent')
                error('音声データが含まれていません: %s', resp_txt);
            end
        end
        
        function save_audio_with_margin(obj, audio_content)
            % 音声データをデコードしてマージンとパディングを追加
            raw_audio = matlab.net.base64decode(audio_content);
            temp_file = 'voice_temp.wav';
            obj.write_binary(temp_file, raw_audio);
            
            try
                [y, fs] = audioread(temp_file);
                
                % マージン追加
                margin_samples = round(obj.MARGIN_SEC * fs);
                y_with_margin = [zeros(margin_samples, 1); y];
                
                % 固定長にパディング
                y_final = obj.pad_to_length(y_with_margin, obj.TARGET_LENGTH);
                
                audiowrite('.\voice.wav', y_final, fs);
            catch ME
                % 一時ファイルをクリーンアップ
                if exist(temp_file, 'file')
                    delete(temp_file);
                end
                rethrow(ME);
            end
            
            % 正常終了時も一時ファイルを削除
            if exist(temp_file, 'file')
                delete(temp_file);
            end
        end
        
        function y_padded = pad_to_length(~, y, target_len)
            % 音声データを指定長にパディングまたは切り詰め
            if length(y) < target_len
                y_padded = [y; zeros(target_len - length(y), 1)];
            else
                y_padded = y(1:target_len, 1);
            end
        end
        
        function time_table = extract_timestamps(obj, resp_json, syllables)
            % APIレスポンスからタイムスタンプを抽出してテーブル化
            time_table = table();
            
            if ~isfield(resp_json, 'timepoints')
                return;
            end
            
            tp = struct2table(resp_json.timepoints);
            
            % 時間を数値に変換
            if iscell(tp.timeSeconds) || isstring(tp.timeSeconds)
                tp.timeSeconds = str2double(tp.timeSeconds);
            end
            
            % マージン分を加算
            tp.timeSeconds = tp.timeSeconds + obj.MARGIN_SEC;
            
            % markNameをセル配列に変換
            if istable(tp.markName) || ~iscell(tp.markName)
                mark_names_cell = table2cell(tp(:, 'markName'));
            else
                mark_names_cell = tp.markName;
            end
            
            % 文字列を対応付け
            chars_col = obj.map_syllables_to_marks(mark_names_cell, syllables);
            tp.char = chars_col;
            
            time_table = table(tp.char, tp.timeSeconds, 'VariableNames', {'char', 'timeSeconds'});
        end
        
        function chars_col = map_syllables_to_marks(~, mark_names, syllables)
            % マーク名から音節への対応付け
            chars_col = cell(length(mark_names), 1);
            
            for k = 1:length(mark_names)
                idx = str2double(mark_names{k}) + 1;
                if idx <= length(syllables)
                    chars_col{k} = syllables{idx};
                else
                    chars_col{k} = '';
                end
            end
        end
        
        function write_json(~, filename, data)
            % JSONファイルを書き込む
            fid = fopen(filename, 'w');
            fprintf(fid, "%s", jsonencode(data));
            fclose(fid);
        end
        
        function write_binary(~, filename, data)
            % バイナリファイルを書き込む
            fid = fopen(filename, 'w');
            fwrite(fid, data);
            fclose(fid);
        end
    end
end