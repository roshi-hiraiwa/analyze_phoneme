import os
import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont


# ==========================================
# 設定・定数定義
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# パス設定
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "train_data.csv")
RESULT_CSV_PATH = os.path.join(PROJECT_ROOT, "target", "result.csv")
IMG_DIR = os.path.join(PROJECT_ROOT, "target", "img")
TIMESTAMP_DIR = os.path.join(PROJECT_ROOT, "target", "timestamp")
VALIDATION_DIR = os.path.join(PROJECT_ROOT, "target", "validation")
HEATMAP_DIR = os.path.join(PROJECT_ROOT, "target", "heatmap")

# ヒートマップ設定
CROP_X_START, CROP_X_END = 80, 478
CROP_Y_START, CROP_Y_END = 213, 273
HEATMAP_WIDTH_PX = CROP_X_END - CROP_X_START

# 音声の長さ (50000サンプル / 24000Hz = 2.08333...秒)
AUDIO_DURATION_SEC = 50000 / 24000

# 赤色の定義 (HSV)
LOWER_RED = np.array([0, 100, 100])
UPPER_RED = np.array([5, 255, 255])


# ==========================================
# 関数定義
# ==========================================
def load_data():
    """CSVデータを読み込み、統合してTP/TNのデータを返す"""
    df_train = pd.read_csv(
        TRAIN_DATA_PATH,
        encoding="shift_jis",
        header=None,
        names=["name", "true_label"],
        usecols=[0, 1],
    )

    df_result = pd.read_csv(
        RESULT_CSV_PATH,
        encoding="shift_jis",
        header=None,
        names=["name", "prob_0", "prob_1"],
        usecols=[0, 1, 2],
    )

    # 列を横方向に結合（行番号順）
    df = pd.concat([df_train, df_result[["prob_0", "prob_1"]]], axis=1)

    # 予測ラベルの決定 (確率が高い方)
    df["pred_label"] = (df["prob_1"] > df["prob_0"]).astype(int)

    # TPとTNを抽出
    df_tp = df.loc[(df["true_label"] == 1) & (df["pred_label"] == 1), ["name"]].copy()
    df_tn = df.loc[(df["true_label"] == 0) & (df["pred_label"] == 0), ["name"]].copy()

    df_tp["type"] = "TP"
    df_tn["type"] = "TN"

    return pd.concat([df_tp, df_tn])


def crop_and_save_images(df_tp_tn):
    """ヒートマップ画像を切り取ってheatmapフォルダに保存"""
    # 出力先ディレクトリを作成
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    processed_count = 0

    for _, row in df_tp_tn.iterrows():
        name = row["name"]
        result_type = row["type"]

        # TP=1, TN=0のサフィックスで画像を検索
        suffix = "1" if result_type == "TP" else "0"
        search_pattern = os.path.join(IMG_DIR, f"*{name}*-{suffix}.png")
        image_files = glob.glob(search_pattern)

        if not image_files:
            print(f"警告: {name} ({result_type}) の画像が見つかりません")
            continue

        # 最初に見つかった画像を処理
        image_path = image_files[0]

        # 画像を読み込み（日本語パス対応）
        try:
            n = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"エラー: {image_path} の読み込みに失敗 - {e}")
            continue

        if img is None:
            print(f"エラー: {image_path} のデコードに失敗しました")
            continue

        # 指定座標で切り取り
        cropped = img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

        # heatmapフォルダに保存
        output_filename = f"{result_type}_{name}.png"
        heatmap_path = os.path.join(HEATMAP_DIR, output_filename)

        # 日本語パス対応で保存
        is_success, encoded_img = cv2.imencode(".png", cropped)
        if is_success:
            encoded_img.tofile(heatmap_path)
            processed_count += 1
        else:
            print(f"エラー: {output_filename} の保存に失敗しました")

    return processed_count


def detect_red_timestamps(df_tp_tn):
    """heatmap画像から赤い領域を検出し、時間を計算"""
    results = []

    for _, row in df_tp_tn.iterrows():
        name = row["name"]
        result_type = row["type"]

        # heatmap画像のパス
        image_filename = f"{result_type}_{name}.png"
        image_path = os.path.join(HEATMAP_DIR, image_filename)

        if not os.path.exists(image_path):
            print(f"警告: {image_filename} が見つかりません")
            continue

        # 画像を読み込み（日本語パス対応）
        try:
            n = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"エラー: {image_path} の読み込みに失敗 - {e}")
            continue

        if img is None:
            print(f"エラー: {image_path} のデコードに失敗")
            continue

        # HSV変換して赤色を検出
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)

        # 輪郭抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        time_ranges = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:  # 小さすぎる領域を無視
                continue

            # 輪郭のバウンディングボックスを取得
            x, y, w, h = cv2.boundingRect(cnt)

            # 開始位置と終了位置のX座標
            x_start = x
            x_end = x + w

            # ピクセル位置から時間に変換
            time_start = (x_start / HEATMAP_WIDTH_PX) * AUDIO_DURATION_SEC
            time_end = (x_end / HEATMAP_WIDTH_PX) * AUDIO_DURATION_SEC

            time_ranges.append(
                {
                    "start": time_start,
                    "end": time_end,
                    "duration": time_end - time_start,
                }
            )

        # 開始時間でソート
        time_ranges.sort(key=lambda x: x["start"])

        # 結果を保存
        if time_ranges:
            range_strs = [f"{tr['start']:.3f}-{tr['end']:.3f}" for tr in time_ranges]
            results.append(
                {
                    "name": name,
                    "type": result_type,
                    "time_ranges": time_ranges,
                    "range_str": ", ".join(range_strs),
                }
            )

    return results


def match_timestamps_with_phonemes(timestamp_results):
    """赤い領域の時間とタイムスタンプCSVの文字を照合"""
    matched_results = []

    for result in timestamp_results:
        name = result["name"]
        result_type = result["type"]
        time_ranges = result["time_ranges"]

        # タイムスタンプCSVを読み込み
        timestamp_csv = os.path.join(TIMESTAMP_DIR, f"{name}.csv")
        if not os.path.exists(timestamp_csv):
            print(f"警告: {name} のタイムスタンプCSVが見つかりません")
            continue

        try:
            df_ts = pd.read_csv(
                timestamp_csv, encoding="shift_jis", header=None, names=["char", "time"]
            )
        except Exception as e:
            print(f"エラー: {timestamp_csv} の読み込みに失敗 - {e}")
            continue

        # デバッグ用の可視化画像を作成
        os.makedirs(VALIDATION_DIR, exist_ok=True)

        # heatmap画像を読み込み
        heatmap_img_path = os.path.join(HEATMAP_DIR, f"{result_type}_{name}.png")
        try:
            n = np.fromfile(heatmap_img_path, np.uint8)
            vis_img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        except:
            vis_img = None

        if vis_img is not None:
            # 画像の高さを拡張してタイムスタンプ表示用のスペースを追加
            timeline_height = 100
            h, w, _ = vis_img.shape
            canvas = np.ones((h + timeline_height, w, 3), dtype=np.uint8) * 255
            canvas[0:h, 0:w] = vis_img

            # 赤い領域を赤色で描画
            for tr in time_ranges:
                red_start = tr["start"]
                red_end = tr["end"]

                x_start = int((red_start / AUDIO_DURATION_SEC) * HEATMAP_WIDTH_PX)
                x_end = int((red_end / AUDIO_DURATION_SEC) * HEATMAP_WIDTH_PX)

                # 赤い領域を赤色の半透明で描画
                cv2.rectangle(
                    canvas, (x_start, h + 10), (x_end, h + 40), (0, 0, 255), -1
                )
                cv2.rectangle(canvas, (x_start, h + 10), (x_end, h + 40), (0, 0, 0), 1)

            # タイムスタンプの区間を描画
            for i in range(len(df_ts) - 1):
                char = df_ts.iloc[i]["char"]
                char_start = float(df_ts.iloc[i]["time"])
                char_end = float(df_ts.iloc[i + 1]["time"])

                if pd.isna(char) or char == "":
                    continue

                # 時間からピクセル位置に変換
                x_start = int((char_start / AUDIO_DURATION_SEC) * HEATMAP_WIDTH_PX)
                x_end = int((char_end / AUDIO_DURATION_SEC) * HEATMAP_WIDTH_PX)

                # タイムスタンプ区間を青色で描画
                cv2.rectangle(
                    canvas, (x_start, h + 50), (x_end, h + 80), (255, 0, 0), -1
                )
                cv2.rectangle(canvas, (x_start, h + 50), (x_end, h + 80), (0, 0, 0), 1)

                # canvasをPIL Imageに変換
                pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                # 日本語フォントを読み込み（システムフォントを使用）
                try:
                    font = ImageFont.truetype("msgothic.ttc", 14)
                except:
                    try:
                        font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 14)
                    except:
                        font = ImageFont.load_default()

                # 文字を描画
                draw.text((x_start + 2, h + 55), char, font=font, fill=(255, 255, 255))

                # PIL ImageをOpenCV形式に戻す
                canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 凡例を追加
            cv2.putText(
                canvas,
                "Red: Red Region",
                (5, h + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                canvas,
                "Blue: Timestamp",
                (150, h + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            # 保存
            comparison_filename = f"comparison_{result_type}_{name}.png"
            comparison_path = os.path.join(VALIDATION_DIR, comparison_filename)
            cv2.imencode(".png", canvas)[1].tofile(comparison_path)

        # 各赤い領域に対して文字を照合
        matched_chars = []
        for tr in time_ranges:
            red_start = tr["start"]
            red_end = tr["end"]

            # この範囲内の文字を検索
            chars_in_range = []
            for i in range(len(df_ts) - 1):
                char = df_ts.iloc[i]["char"]
                char_start = float(df_ts.iloc[i]["time"])
                char_end = float(df_ts.iloc[i + 1]["time"])

                # 空文字をスキップ
                if pd.isna(char) or char == "":
                    continue

                # 重なり判定: 赤い範囲と文字の時間範囲が重なっているか
                if not (char_end <= red_start or char_start >= red_end):
                    chars_in_range.append(
                        {
                            "char": char,
                            "char_start": char_start,
                            "char_end": char_end,
                            "red_start": red_start,
                            "red_end": red_end,
                        }
                    )

            if chars_in_range:
                matched_chars.extend(chars_in_range)

        # 重複を除去して文字列化
        unique_chars = []
        seen = set()
        for match in matched_chars:
            char = match["char"]
            if char not in seen:
                unique_chars.append(char)
                seen.add(char)

        matched_results.append(
            {
                "name": name,
                "type": result_type,
                "matched_chars": "".join(unique_chars),
                "matched_details": matched_chars,
                "range_str": result["range_str"],
            }
        )

    return matched_results


if __name__ == "__main__":
    # TPとTNのデータを取得
    df_tp_tn = load_data()

    # 結果を表示
    print(f"TPデータ数: {len(df_tp_tn[df_tp_tn['type'] == 'TP'])}")
    print(f"TNデータ数: {len(df_tp_tn[df_tp_tn['type'] == 'TN'])}")
    print(f"合計: {len(df_tp_tn)}")

    # 画像を切り取って保存
    print(f"\n画像処理を開始...")
    processed = crop_and_save_images(df_tp_tn)
    print(f"{processed}枚の画像を保存: {HEATMAP_DIR} ")

    # 赤い領域の時間を検出
    print(f"赤色領域の時間検出を開始...")
    timestamp_results = detect_red_timestamps(df_tp_tn)

    # 赤い領域と音声の文字を照合
    print(f"文字照合を開始...")
    matched_results = match_timestamps_with_phonemes(timestamp_results)
    print(f"比較画像を保存: {VALIDATION_DIR}")
    print("\n[照合結果]")
    for result in matched_results:
        print(f"{result['type']} - {result['name']}: {result['matched_chars']}")
