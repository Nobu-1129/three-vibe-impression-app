import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import streamlit.components.v1 as components
import csv
import io
from datetime import datetime
import pandas as pd
import json
import base64
import os
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

plt.rcParams["axes.unicode_minus"] = False

USE_MOCK_DATA = False

SECOND_PROJECT_PROMPT = """
あなたは、一枚絵の印象と映え感を、遊び感覚で可視化する評価アシスタントです。
入力された画像について、芸術作品としての優劣や真面目な審査ではなく、画像から受けるノリ・雰囲気・印象・刺さり方を評価してください。

【評価対象】
- 写真
- イラスト
- 一枚絵
- 画像作品全般

【評価の基本方針】
- 芸術性の優劣を断定しないでください。
- 上手い / 下手、優れている / 劣っている という言い方は避けてください。
- 「どんなふうに見えるか」「どう刺さりそうか」「どんな空気を持っているか」を重視してください。
- 見た人が遊び感覚で楽しめる結果にしてください。
- 必ず、画像の良いところをどこかで拾ってください。
- からかいすぎたり、強く否定したりしないでください。
- コメントは少し面白くても良いですが、悪意のある表現にはしないでください。
- 評価は毎回できるだけ安定させ、気分で極端にぶらさないでください。
- 派手さだけで高くせず、落ち着いた作品にも別の良さがあることを認めてください。
- 同じような画像を見たときには、極端に違う評価を出さないでください。

【画像を見る順番】
必ず以下の順番で画像を観察し、そのうえで最終評価を行ってください。

1. ファーストインプレッション（直感）
- まずは細かい分析をせず、パッと見た瞬間に感じる印象を確認してください。
- 何を感じたか：楽しい、不気味、静か、圧倒される、親しみやすい、クール、エモい、など
- どこに最初に目が行くか：アイキャッチ、視線が吸い寄せられる場所

2. 全体の構成とレイアウト（構図）
- 視線誘導：視線、手足、配置、遠近感、線の流れなどが見る人の目をどこへ導くか
- シルエット：主役の形がはっきりしていて、一目で印象がつかめるか
- 余白：描き込みすぎず、休める空間があるか
- 画面全体のまとまりや安定感があるか

3. 光と色（ライティング・配色）
- 光源：どこから光が来ているように見えるか、その影の付き方に一貫性があるか
- 色のバランス：メインカラー、サブカラー、アクセントカラーの調和
- コントラスト：明暗差や色の差で見せたい部分が立っているか
- 色や光が雰囲気づくりにどう効いているか

4. 描写の技術とディテール（質感）
- 質感の描き分け：肌、布、金属、木、ガラスなどの違いが感じられるか
- 細部の見どころ：拡大して見ると発見があるか
- 線や形の整理：線の強弱、形の取り方、細部の丁寧さ
- 写実寄りの作品では、手足や関節などに強い違和感がないかを軽く確認する

5. 文脈とストーリー性（テーマ）
- 状況設定：表情、持ち物、背景、小物からどんな場面が想像できるか
- オリジナリティ：作者らしいこだわり、癖、独特さがあるか
- ただ綺麗なだけでなく、何か少し想像を広げたくなる要素があるか

【最終評価の考え方】
- 上の5段階を踏まえて、最終的に「この画像はどんな印象に見えるか」を8軸でまとめてください。
- 評価は、ファーストインプレッションだけで決めず、構図・光・ディテール・文脈も加味してください。
- ただし、重すぎる講評にはせず、遊び感のある診断としてまとめてください。
- 落ち着いた画像を「弱い」と決めつけないでください。
- 派手な画像を「良い」と短絡しないでください。
- 8軸のスコアは、見た目の勢いだけでなく、構図・色・質感・文脈から総合的に判断してください。

【8つの評価軸】
各軸を -5 〜 +5 の整数で評価してください。
-5 は左側の印象がかなり強い
0 はどちらともいえない / 中立
+5 は右側の印象がかなり強い

1. 地味 ↔ 映える
- 地味：落ち着いていて主張が控えめ
- 映える：目を引く、見せたくなる、印象に残りやすい

2. ゆるい ↔ キマってる
- ゆるい：力が抜けていてラフ、親しみやすい
- キマってる：まとまりがあり、雰囲気や見せ方がハマっている

3. 素直 ↔ クセつよ
- 素直：わかりやすく受け取りやすい
- クセつよ：独特で引っかかりがあり、忘れにくい

4. やさしい ↔ 圧が強い
- やさしい：柔らかく穏やか
- 圧が強い：迫力があり、存在感が強い

5. しっとり ↔ テンション高い
- しっとり：静かで落ち着いた空気がある
- テンション高い：にぎやかで勢いがあり、アガる感じがある

6. 親しみやすい ↔ 近寄りがたい
- 親しみやすい：気軽に好きと言いやすい
- 近寄りがたい：距離感があり、簡単には近づけない魅力がある

7. 王道 ↔ 尖ってる
- 王道：定番の強さがあり、わかりやすい
- 尖ってる：独自性が強く、刺さる人には強く刺さる

8. インパクト派 ↔ ディテール派
- インパクト派：一発で持っていく強さがある
- ディテール派：見るほど細部の作り込みや発見がある

【登場キャラクター】
この画像を、次の3人がそれぞれの視点で見ます。
3人とも、まずは画像の良いところを拾ってからコメントしてください。

1. おじさん
- 情緒の目利きおじさん
- 心にグッとくるか、余韻、人の気配を重視する
- 基本は丁寧語
- 感動すると少し熱が入る
- コメントはやさしく、少しだけ古いネタ感が混じってもよい

2. ギャル
- 映えとエモの直感ギャル
- 第一印象、映え、ノリ、見せたくなる感じを重視する
- 王道ギャルっぽい明るい口調
- 人を下げる言い方はしない
- ノリは良いが、ちゃんと褒める

3. モデラー
- 意匠の観測職人
- 作り込み、個性、熱量、見せ方の整理を重視する
- 作り手目線で、構図や細部や説得力を見る
- 基本は少し理屈っぽいが、嫌味ではない
- 作り込みが強い作品には熱くなりやすい

【キャラごとのコメント方針】
- おじさん：余韻や空気感、じわっと来るポイントを褒める
- ギャル：映え、エモさ、第一印象の強さを前向きに褒める
- モデラー：構図、細部、質感、こだわりを見てコメントする
- 3人とも、短くてもいいので「良いところ」をちゃんと入れてください
- 3人のコメントの方向性は少しずつ変えてください

【追加アドバイス】
- 各キャラは、その画像について「もしもう一歩こういう方向に寄せたいなら」という提案型の短いアドバイスを1つ考えてください。
- アドバイスは改善点の指摘ではなく、方向性の提案にしてください。
- 強い否定やダメ出しはしないでください。
- 「もっと映えさせるなら」「もう少し全体をまとめるなら」「少しプロっぽく見せるなら」などの言い回しで、やわらかく提案してください。
- アドバイスは各キャラらしい視点で2文で作ってください。
- 今の画像の良さを壊さない提案にしてください。

【シェアしたくなるタイトル】
- 画像を見て、少し斜めで印象に残る一言タイトルをつけてください
- 大げさすぎず、でも少しだけ人に見せたくなる感じにしてください
- タイトルは短めにしてください
- タイトルは画像全体の印象を踏まえてください

【どの層にウケそうか】
- どんな人に刺さりそうかを、2〜4個の短い語句で返してください
- 例：『映え好き』『クセ強好き』『しっとり派』『動物好き』など
- 悪意のある表現は使わないでください

【出力ルール】
- 必ずJSONのみを返してください
- JSON以外の文章は一切出力しないでください
- 数値は整数で返してください
- character_comments は各キャラ 2〜3文で返してください
- share_title は短めにしてください
- appeal_targets は2〜4個の短い語句で返してください
- キー名は以下と完全一致させてください

【出力フォーマット】
{
  "axis_scores": {
    "地味 ↔ 映える": 0,
    "ゆるい ↔ キマってる": 0,
    "素直 ↔ クセつよ": 0,
    "やさしい ↔ 圧が強い": 0,
    "しっとり ↔ テンション高い": 0,
    "親しみやすい ↔ 近寄りがたい": 0,
    "王道 ↔ 尖ってる": 0,
    "インパクト派 ↔ ディテール派": 0
  },
  "character_comments": {
    "おじさん": "",
    "ギャル": "",
    "モデラー": ""
  },
  "character_advice": {
    "おじさん": "",
    "ギャル": "",
    "モデラー": ""
  },
  "share_title": "",
  "appeal_targets": ["", "", ""]
}
"""

CHARACTER_WEIGHTS = {
    "おじさん": {
        "地味 ↔ 映える": 10,
        "ゆるい ↔ キマってる": 15,
        "素直 ↔ クセつよ": 15,
        "やさしい ↔ 圧が強い": 20,
        "しっとり ↔ テンション高い": 10,
        "親しみやすい ↔ 近寄りがたい": 25,
        "王道 ↔ 尖ってる": 10,
        "インパクト派 ↔ ディテール派": 15,
    },
    "ギャル": {
        "地味 ↔ 映える": 25,
        "ゆるい ↔ キマってる": 20,
        "素直 ↔ クセつよ": 10,
        "やさしい ↔ 圧が強い": 15,
        "しっとり ↔ テンション高い": 25,
        "親しみやすい ↔ 近寄りがたい": 10,
        "王道 ↔ 尖ってる": 10,
        "インパクト派 ↔ ディテール派": 5,
    },
    "モデラー": {
        "地味 ↔ 映える": 10,
        "ゆるい ↔ キマってる": 10,
        "素直 ↔ クセつよ": 20,
        "やさしい ↔ 圧が強い": 10,
        "しっとり ↔ テンション高い": 10,
        "親しみやすい ↔ 近寄りがたい": 10,
        "王道 ↔ 尖ってる": 25,
        "インパクト派 ↔ ディテール派": 25,
    }
}

CHARACTER_DISPLAY_NAMES = {
    "おじさん": "ジンさん",
    "ギャル": "レイナ",
    "モデラー": "タクミ",
}

CHARACTER_SHORT_NAMES = {
    "おじさん": "ジン",
    "ギャル": "レイ",
    "モデラー": "タク",
}

CHARACTER_EMOJIS = {
    "おじさん": "🧔",
    "ギャル": "💖",
    "モデラー": "🛠️",
}

CHARACTER_ICON_PATHS = {
    "おじさん": "assets/icon_jin.png",
    "ギャル": "assets/icon_reina.png",
    "モデラー": "assets/icon_takumi.png",
}

CHARACTER_COMMENT_NAMES = {
    "おじさん": "ジンさん",
    "ギャル": "レイナ",
    "モデラー": "タクミ",
}

def compress_image_for_ai(uploaded_file, max_size_kb=500, max_width=1280, max_height=1280, quality=85):
    """
    uploaded_file を読み込み、API送信用の軽量JPEG bytesを返す
    目標サイズ: max_size_kb KB 以下
    """
    image = Image.open(uploaded_file)

    # RGBAやPモードはJPEG保存できるRGBへ変換
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # リサイズ
    image.thumbnail((max_width, max_height))

    # 品質を下げながら目標サイズ以下に近づける
    current_quality = quality
    output = None

    while current_quality >= 40:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=current_quality, optimize=True)
        size_kb = buffer.tell() / 1024

        output = buffer.getvalue()

        if size_kb <= max_size_kb:
            break

        current_quality -= 5

    return output

def calculate_character_scores(axis_scores: dict) -> tuple[dict, int, int]:
    """
    axis_scores: AIが返した8軸スコア（-5～+5）
    return:
        character_scores: 各キャラの表示用スコア（合計が three_vis になる）
        true_score: 内部用 0～360点
        three_vis: 0～1000
    """
    raw_character_scores = {}

    for character, weights in CHARACTER_WEIGHTS.items():
        total = 0.0

        for axis_name, weight in weights.items():
            raw_score = axis_scores.get(axis_name, 0)

            # 印象の強さだけを見る
            strength = abs(raw_score) / 5.0   # 0.0～1.0
            axis_point = weight * strength
            total += axis_point

        raw_character_scores[character] = total

    raw_total = sum(raw_character_scores.values())
    true_score = round(raw_total)
    three_vis = round((raw_total / 360) * 1000)

    # 表示用キャラスコアは、合計が three_vis になるように再配分
    if raw_total == 0 or three_vis == 0:
        character_scores = {character: 0 for character in raw_character_scores}
        return character_scores, true_score, three_vis

    scaled_scores_float = {
        character: (score / raw_total) * three_vis
        for character, score in raw_character_scores.items()
    }

    # まず整数部分を入れる
    character_scores = {
        character: int(score)
        for character, score in scaled_scores_float.items()
    }

    # 端数の大きい順に残りを配る
    allocated = sum(character_scores.values())
    remainder = three_vis - allocated

    if remainder > 0:
        fractions = sorted(
            scaled_scores_float.items(),
            key=lambda x: x[1] - int(x[1]),
            reverse=True
        )
        for i in range(remainder):
            character_scores[fractions[i][0]] += 1

    return character_scores, true_score, three_vis

def get_top_character_name(character_scores: dict) -> str:
    top_key = max(character_scores, key=character_scores.get)
    return CHARACTER_DISPLAY_NAMES.get(top_key, top_key)

def plot_8axis_radar(axis_scores: dict):
    labels = list(axis_scores.keys())
    values = list(axis_scores.values())

    # レーダーチャート用に最初の値を最後に追加
    values_closed = values + values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # 角度方向のラベル
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)

    # 半径方向の範囲
    ax.set_ylim(-5, 5)
    ax.set_yticks([-5, -3, -1, 0, 1, 3, 5])
    ax.set_yticklabels(["-5", "-3", "-1", "0", "1", "3", "5"], fontsize=9)

    # 0の円を強調
    ax.axhline(0, color="gray", linewidth=0.8)

    # プロット
    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.15)

    ax.set_title("8軸レーダーチャート", fontsize=16, pad=20)

    return fig

def analyze_image_with_ai(image_bytes):
    if USE_MOCK_DATA:
        return get_mock_analysis_data()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{base64_image}"

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SECOND_PROJECT_PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    response_text = response.output_text
    data = json.loads(response_text)

    axis_scores = data["axis_scores"]
    character_comments = data["character_comments"]
    character_advice = data["character_advice"]
    share_title = data["share_title"]
    appeal_targets = data["appeal_targets"]

    return axis_scores, character_comments, character_advice, share_title, appeal_targets

def plot_16axis_radar_from_8axis(axis_scores: dict):
    # 8軸ペア
    axis_pairs = [
        ("地味 ↔ 映える", "地味", "映える"),
        ("ゆるい ↔ キマってる", "ゆるい", "キマってる"),
        ("素直 ↔ クセつよ", "素直", "クセつよ"),
        ("やさしい ↔ 圧が強い", "やさしい", "圧が強い"),
        ("しっとり ↔ テンション高い", "しっとり", "テンション高い"),
        ("親しみやすい ↔ 近寄りがたい", "親しみやすい", "近寄りがたい"),
        ("王道 ↔ 尖ってる", "王道", "尖ってる"),
        ("インパクト派 ↔ ディテール派", "インパクト派", "ディテール派"),
    ]

    # まず左右それぞれの値を作る
    pair_values = {}
    for axis_name, left_label, right_label in axis_pairs:
        raw_score = axis_scores.get(axis_name, 0)

        if raw_score < 0:
            left_value = abs(raw_score)
            right_value = 0
        elif raw_score > 0:
            left_value = 0
            right_value = raw_score
        else:
            left_value = 0
            right_value = 0

        pair_values[left_label] = left_value
        pair_values[right_label] = right_value

    # 対極に来るように並べる
    labels = [
        "地味",
        "ゆるい",
        "素直",
        "やさしい",
        "しっとり",
        "親しみやすい",
        "王道",
        "インパクト派",
        "映える",
        "キマってる",
        "クセつよ",
        "圧が強い",
        "テンション高い",
        "近寄りがたい",
        "尖ってる",
        "ディテール派",
    ]

    values = [pair_values[label] for label in labels]

    # 表示用に +1 して中心に小さい円を残す
    display_values = [v + 1 for v in values]

    values_closed = display_values + display_values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)

    # 0～5 を 1～6 にずらして描く
    ax.set_ylim(0, 6)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["", "", "", "", "", ""])

    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.15)

    # ax.set_title("16項目レーダーチャート", fontsize=16, pad=20)

    return fig

def make_compare_table_html(df, first_col_width="260px"):
    headers = df.columns.tolist()

    html = '<table style="border-collapse: collapse; width: 100%; font-size: 16px; margin-bottom: 16px;">'
    html += '<thead><tr>'

    for i, header in enumerate(headers):
        if i == 0:
            html += (
                f'<th style="border: 1px solid #ddd; padding: 8px; text-align: left; '
                f'width: {first_col_width}; background-color: #f7f7f7;">{header}</th>'
            )
        else:
            html += (
                f'<th style="border: 1px solid #ddd; padding: 8px; text-align: center; '
                f'background-color: #f7f7f7;">{header}</th>'
            )

    html += '</tr></thead><tbody>'

    for _, row in df.iterrows():
        html += '<tr>'
        for i, value in enumerate(row):
            if i == 0:
                html += (
                    f'<td style="border: 1px solid #ddd; padding: 8px; text-align: left; '
                    f'width: {first_col_width};">{value}</td>'
                )
            else:
                html += (
                    f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{value}</td>'
                )
        html += '</tr>'

    html += '</tbody></table>'
    return html

def rgba_str(rgb, alpha=1.0):
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

def mpl_rgba(rgb, alpha=1.0):
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, alpha)

def render_8axis_bar_chart(axis_scores: dict):
    axis_pairs = [
        ("地味 ↔ 映える", "地味", "映える"),
        ("ゆるい ↔ キマってる", "ゆるい", "キマってる"),
        ("素直 ↔ クセつよ", "素直", "クセつよ"),
        ("やさしい ↔ 圧が強い", "やさしい", "圧が強い"),
        ("しっとり ↔ テンション高い", "しっとり", "テンション高い"),
        ("親しみやすい ↔ 近寄りがたい", "親しみやすい", "近寄りがたい"),
        ("王道 ↔ 尖ってる", "王道", "尖ってる"),
        ("インパクト派 ↔ ディテール派", "インパクト派", "ディテール派"),
    ]

    strength_words = {
        0: "どちらともいえない",
        1: "ちょっと",
        2: "やや",
        3: "けっこう",
        4: "かなり",
        5: "めっちゃ",
    }

    html_parts = []

    for axis_name, left_label, right_label in axis_pairs:
        score = axis_scores.get(axis_name, 0)
        marker_left = ((score + 5) / 10) * 100
        strength_text = strength_words.get(abs(score), "")

        # 「どちらともいえない」だけ少し左へ補正して、見た目の中央を合わせる
        if abs(score) == 0:
            label_left = f"calc({marker_left}% - 54px)"
            label_width = "108px"
        else:
            label_left = f"calc({marker_left}% - 38px)"
            label_width = "76px"

        html_parts.append(f"""
<div style="margin-bottom:30px;">
  <div style="font-size:16px; font-weight:700; margin-bottom:8px;">
    {left_label}
    <span style="float:right;">{right_label}</span>
  </div>

  <div style="
    position:relative;
    height:20px;
    border-radius:999px;
    background:linear-gradient(to right, #dbeafe 0%, #f3f4f6 50%, #fee2e2 100%);
    border:1px solid #d1d5db;
    overflow:visible;
  ">
    <!-- つまみ -->
    <div style="
      position:absolute;
      left:calc({marker_left}% - 6px);
      top:2px;
      width:12px;
      height:16px;
      border-radius:8px;
      background:#2f2f2f;
    "></div>

    <!-- つまみ下の表現 -->
    <div style="
      position:absolute;
      left:{label_left};
      top:24px;
      width:{label_width};
      text-align:center;
      font-size:16px;
      color:#4b5563;
      font-weight:600;
      line-height:1;
      white-space:nowrap;
    ">
      {strength_text}
    </div>
  </div>
</div>
""")

    full_html = "".join(html_parts)
    components.html(full_html, height=700, scrolling=False)

def is_admin_mode() -> bool:
    st.markdown("### 管理者メニュー")

    entered_password = st.text_input(
        "管理者パスワード",
        type="password",
        key="admin_password_input"
    )

    if not entered_password:
        return False

    return entered_password == st.secrets.get("ADMIN_PASSWORD", "")

def get_mock_analysis_data():
    axis_scores = {
        "地味 ↔ 映える": 3,
        "ゆるい ↔ キマってる": 2,
        "素直 ↔ クセつよ": 0,
        "やさしい ↔ 圧が強い": 1,
        "しっとり ↔ テンション高い": 1,
        "親しみやすい ↔ 近寄りがたい": 2,
        "王道 ↔ 尖ってる": 4,
        "インパクト派 ↔ ディテール派": 2,
    }

    character_comments = {
        "おじさん": "いやあ、これは実にいい焼き色ですな。切り口のピンクの残り方もきれいで、見ているだけで「うむ、今日は当たりだな」と思わせる余韻があります。食卓のあたたかい気配まで伝わってきますね。",
        "ギャル": "うわ、これは普通に見せたくなるやつ！ 表面のこんがり感と中のレア感のコントラストがめっちゃ映えてて、肉の説得力が強い〜。お皿の上でちゃんと主役してるのが最高。",
        "モデラー": "焼き面、断面、脂のツヤの三段構成がしっかり効いていて、かなり情報が伝わりやすいです。余計なものを入れすぎず、肉の存在感を前に出しているので、視線の着地が素直で気持ちいいですね。",
    }

    character_advice = {
        "おじさん": "もっと食欲の余韻を出すなら、湯気やカトラリーの置き方を少しだけ見せると、食卓の物語がふっと立ち上がります。今の素直なおいしさはそのままに、ひと呼吸ある雰囲気に寄せるとまた良いですな。",
        "ギャル": "もっと映えさせるなら、湯気やカトラリーの置き方を少しだけ見せると、食卓の物語がふっと立ち上がります。今の素直なおいしさはそのままに、ひと呼吸ある雰囲気に寄せるとまた良いですな。",
        "モデラー": "少しプロっぽく見せるなら、湯気やカトラリーの置き方を少しだけ見せると、食卓の物語がふっと立ち上がります。今の素直なおいしさはそのままに、ひと呼吸ある雰囲気に寄せるとまた良いですな。",
    }

    share_title = "肉、正面から来た"
    appeal_targets = ["肉好き", "飯テロ好き", "映え好き", "こってり派"]

    return axis_scores, character_comments, character_advice, share_title, appeal_targets

def prepare_image_for_app(uploaded_file, max_size_kb=500, max_width=1280, max_height=1280, quality=85):
    """
    uploaded_file を1回だけ圧縮して、
    画面表示・API送信・保存に使えるJPEG bytesを返す
    """
    image = Image.open(uploaded_file)

    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    image.thumbnail((max_width, max_height))

    current_quality = quality
    output_bytes = None

    while current_quality >= 40:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=current_quality, optimize=True)
        size_kb = buffer.tell() / 1024
        output_bytes = buffer.getvalue()

        if size_kb <= max_size_kb:
            break

        current_quality -= 5

    return output_bytes

def image_file_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

st.image("assets/hero_characters.png", use_container_width=True)


st.markdown("""
<div style="margin-top: 10px; margin-bottom: 0px; font-size: 18px; color: #333;">
    AIキャラに見せる画像を読み込んでください
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stFileUploader"] {
    margin-top: -40px;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    prepared_image_bytes = prepare_image_for_app(uploaded_file, max_size_kb=500)
    display_image = Image.open(io.BytesIO(prepared_image_bytes))

    st.image(display_image, caption="読み込んだ画像", use_container_width=True)

    if st.button("AIで評価する"):
        try:
            axis_scores, character_comments, character_advice, share_title, appeal_targets = analyze_image_with_ai(prepared_image_bytes)
            character_scores, true_score, three_vis = calculate_character_scores(axis_scores)

            st.success("AI評価を実行しました。")

        except Exception as e:
            st.error(f"AI評価に失敗しました: {e}")
            st.stop()

        st.session_state["prepared_image_bytes"] = prepared_image_bytes

        # session_state に保存
        st.session_state["axis_scores"] = axis_scores
        st.session_state["character_comments"] = character_comments
        st.session_state["character_advice"] = character_advice
        st.session_state["share_title"] = share_title
        st.session_state["appeal_targets"] = appeal_targets
        st.session_state["character_scores"] = character_scores
        st.session_state["true_score"] = true_score
        st.session_state["three_vis"] = three_vis
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["prepared_image_bytes"] = prepared_image_bytes

    if "axis_scores" in st.session_state:
        axis_scores = st.session_state["axis_scores"]
        character_comments = st.session_state["character_comments"]
        character_advice = st.session_state["character_advice"]
        share_title = st.session_state["share_title"]
        appeal_targets = st.session_state["appeal_targets"]
        character_scores = st.session_state["character_scores"]
        true_score = st.session_state["true_score"]
        three_vis = st.session_state["three_vis"]

        top_character_name = get_top_character_name(character_scores)

        st.divider()

        # タイトル表示
        st.markdown(f"""
        <div style="text-align:center; margin-top: 8px; margin-bottom: 22px;">
            <div style="font-size: 16px; color: #444; margin-bottom: 10px;">
                この画像につけられたタイトルは
            </div>
            <div style="font-size: 34px; font-weight: 800; line-height: 1.4; margin-bottom: 10px;">
                {share_title}
            </div>
            <div style="font-size: 16px; color: #555;">
                名付け親：{top_character_name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 印象値表示
        score_card_html = f"""
<div style="
    border: 1px solid #d1d5db;
    border-radius: 24px;
    padding: 18px 16px 14px 16px;
    margin-bottom: 24px;
    background: #fafafa;
    text-align: center;
">
    <div style="
        font-size: 16px;
        color: #222;
        margin-bottom: 10px;
        line-height: 1.5;
        font-weight: 600;
    ">
        この画像に3人のAIキャラが勝手につけた印象値
    </div>

    <div style="
        font-size: 44px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
        color: #111;
    ">
        {three_vis}
    </div>

    <div style="
        font-size: 16px;
        color: #444;
        line-height: 1.4;
        font-weight: 600;
    ">
        Your Three Vibe Impression Score
    </div>
</div>
"""
        components.html(score_card_html, height=160, scrolling=False)
        # 好きそうな人たち
        tags_html = "".join([
            f'<span style="display:inline-block; background:#f2f4f7; border-radius:14px; padding:6px 10px; margin:4px 6px 4px 0; font-size:16px;">{t}</span>'
            for t in appeal_targets
        ])

        st.markdown(f"""
        <div style="margin-top: 6px; margin-bottom: 20px;">
            <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px;">
                この画像が好きそうな人たち
            </div>
            <div>{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)
    
        # 3人コメント（吹き出し風）
        st.markdown("""
        <div style="font-size: 18px; font-weight: 700; margin-top: 18px; margin-bottom: 10px;">3人のコメント</div>
        """, unsafe_allow_html=True)

        comment_styles = {
            "おじさん": {
                "icon_bg": "#cfe0ff",
                "bubble_border": "#9bbcf7",
                "bubble_bg": "#f3f8ff",
                "reverse": False,
            },
            "ギャル": {
                "icon_bg": "#ffd6e7",
                "bubble_border": "#f5a8c8",
                "bubble_bg": "#fffafb",
                "reverse": True,
            },
            "モデラー": {
                "icon_bg": "#d9f5d6",
                "bubble_border": "#9ed49a",
                "bubble_bg": "#fbfffb",
                "reverse": False,
            },
        }

        comment_html_parts = []

        for name, comment in character_comments.items():
            icon_path = CHARACTER_ICON_PATHS.get(name, "")
            icon_data_uri = image_file_to_data_uri(icon_path) if icon_path else ""
            display_name = CHARACTER_DISPLAY_NAMES.get(name, name)
            score_value = character_scores.get(name, 0)

            style = comment_styles.get(name, {
                "icon_bg": "#eeeeee",
                "bubble_border": "#cccccc",
                "bubble_bg": "#ffffff",
                "reverse": False,
            })

            if style["reverse"]:
                comment_html_parts.append(f"""
<div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:18px; flex-direction:row-reverse;">
    <div style="
        min-width:90px;
        width:90px;
        flex-shrink:0;
        margin-top:2px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:flex-start;
    ">
        <div style="
            width:90px;
            height:90px;
            border-radius:50%;
            overflow:hidden;
            background:{style["icon_bg"]};
            display:flex;
            align-items:center;
            justify-content:center;
        ">
            <img src="{icon_data_uri}" style="width:180%; height:180%; object-fit:cover; object-position:center 20%;">
        </div>

        <div style="
            margin-top:6px;
            font-size:16px;
            font-weight:700;
            color:#444;
            line-height:1.2;
            text-align:center;
            white-space:nowrap;
        ">
            {display_name}
        </div>
    </div>
  <div style="
    position:relative;
    flex:1;
    border:1px solid {style["bubble_border"]};
    border-radius:16px;
    background:{style["bubble_bg"]};
    padding:14px 16px;
  ">
    <div style="
      position:absolute;
      right:-8px;
      top:16px;
      width:14px;
      height:14px;
      background:{style["bubble_bg"]};
      border-right:1px solid {style["bubble_border"]};
      border-top:1px solid {style["bubble_border"]};
      transform:rotate(45deg);
    "></div>

    <div style="
      font-size:16px;
      font-weight:700;
      color:#555;
      margin-bottom:8px;
      line-height:1.2;
    ">
      SCORE: {score_value}pt
    </div>

    <div style="
      font-size:16px;
      line-height:1.75;
      color:#333;
    ">
      {comment}
    </div>
  </div>
</div>
""")
            else:
                comment_html_parts.append(f"""
<div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:18px;">
  <div style="
    min-width:90px;
    width:90px;
    flex-shrink:0;
    margin-top:2px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:flex-start;
  ">
    <div style="
        width:90px;
        height:90px;
        border-radius:50%;
        overflow:hidden;
        background:{style["icon_bg"]};
        display:flex;
        align-items:center;
        justify-content:center;
    ">
        <img src="{icon_data_uri}" style="width:180%; height:180%; object-fit:cover; object-position:center 20%;">
    </div>

    <div style="
        margin-top:6px;
        font-size:16px;
        font-weight:700;
        color:#444;
        line-height:1.2;
        text-align:center;
        white-space:nowrap;
    ">
        {display_name}
    </div>
  </div>

  <div style="
    position:relative;
    flex:1;
    border:1px solid {style["bubble_border"]};
    border-radius:16px;
    background:{style["bubble_bg"]};
    padding:14px 16px;
  ">
    <div style="
      position:absolute;
      left:-8px;
      top:16px;
      width:14px;
      height:14px;
      background:{style["bubble_bg"]};
      border-left:1px solid {style["bubble_border"]};
      border-bottom:1px solid {style["bubble_border"]};
      transform:rotate(45deg);
    "></div>

    <div style="
      font-size:16px;
      font-weight:700;
      color:#555;
      margin-bottom:8px;
      line-height:1.2;
    ">
      SCORE: {score_value}pt
    </div>

    <div style="
      font-size:16px;
      line-height:1.75;
      color:#333;
    ">
      {comment}
    </div>
  </div>
</div>
""")

        full_comment_html = "".join(comment_html_parts)
        components.html(full_comment_html, height=600, scrolling=False)

        show_advice = st.checkbox("アドバイスも見る")

        if show_advice:
            lowest_character_key = min(character_scores, key=character_scores.get)
            advice_display_name = CHARACTER_DISPLAY_NAMES.get(lowest_character_key, lowest_character_key)
            advice_text = character_advice.get(lowest_character_key, "")

            st.markdown(f"""
            <div style="
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 14px 16px;
                margin-top: 10px;
                margin-bottom: 18px;
                background: #fffdf7;
            ">
                <div style="
                    font-size: 16px;
                    font-weight: 700;
                    margin-bottom: 8px;
                    color: #222;
                ">
                    もう一歩だけ寄せるなら
                </div>
                <div style="
                    font-size: 16px;
                    color: #555;
                    margin-bottom: 6px;
                    font-weight: 600;
                ">
                    {advice_display_name}からのひとこと
                </div>
                <div style="
                    font-size: 16px;
                    line-height: 1.7;
                    color: #333;
                ">
                    {advice_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 16項目レーダーチャート
        st.markdown("""
        <div style="font-size: 18px; font-weight: 700; margin-top: 18px; margin-bottom: 10px;">
            16項目レーダーチャート
        </div>
        """, unsafe_allow_html=True)

        radar_fig = plot_16axis_radar_from_8axis(axis_scores)
        st.pyplot(radar_fig, use_container_width=True)

        render_8axis_bar_chart(axis_scores)

        st.divider()

        st.subheader("保存")

        if st.button("結果を保存"):
            save_file = "results_2nd.csv"
            image_save_dir = "saved_images_2nd"

            os.makedirs(image_save_dir, exist_ok=True)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_base_name, _ = os.path.splitext(st.session_state["uploaded_file_name"])
            saved_image_name = f"{image_base_name}_{timestamp_str}.jpg"
            saved_image_path = os.path.join(image_save_dir, saved_image_name)

            try:
                with open(saved_image_path, "wb") as img_f:
                    img_f.write(st.session_state["prepared_image_bytes"])

                row_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": st.session_state["uploaded_file_name"],
                    "saved_image_path": saved_image_path,
                    "share_title": share_title,
                    "title_named_by": top_character_name,
                    "three_vis": three_vis,
                    "appeal_targets": " / ".join(appeal_targets),

                    "ジンさん_score": character_scores["おじさん"],
                    "レイナ_score": character_scores["ギャル"],
                    "タクミ_score": character_scores["モデラー"],

                    "ジンさん_comment": character_comments["おじさん"],
                    "レイナ_comment": character_comments["ギャル"],
                    "タクミ_comment": character_comments["モデラー"],
                }

                for axis_name, score in axis_scores.items():
                    row_data[axis_name] = score

                file_exists = os.path.exists(save_file)

                with open(save_file, mode="a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=row_data.keys())

                    if not file_exists:
                        writer.writeheader()

                    writer.writerow(row_data)

                st.success(f"保存しました：{save_file}")
                st.info(f"画像も保存しました：{saved_image_path}")

            except PermissionError:
                st.error(f"保存できませんでした。{save_file} がExcelなどで開かれていないか確認してください。")

            except Exception as e:
                st.error(f"保存中にエラーが発生しました: {e}")

        st.divider()

        if is_admin_mode():
            st.subheader("保存履歴")

            history_file = "results_2nd.csv"

            if os.path.exists(history_file):
                history_df = pd.read_csv(history_file, encoding="utf-8-sig")
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("まだ保存履歴はありません。")

        st.divider()