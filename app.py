import streamlit as st
from PIL import Image, ImageOps
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
from supabase import create_client, Client
from uuid import uuid4


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_SERVICE_ROLE_KEY"],
)

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

【3VISスコアの考え方】
- 3VISは単なる画像の整い方や情報量ではなく、「人に見せたときにどれだけ印象に残るか」「ギャラリーに並べたときに見てもらいたくなるか」「投稿者がその一枚を見せたいと思った理由が伝わるか」を重視してください。
- 3人それぞれの character_scores は 0〜100 点の整数で返してください。
- 10点刻み、25点刻み、50点刻みのような丸い点数に寄せないでください。
- 画像ごとの微妙な差を反映し、1点単位で自然にばらつく点数にしてください。
- ただし、わざと不自然な端数にするのではなく、観察した印象の差から自然に決めてください。
- 地図、メニュー表、案内図、フロアマップ、説明資料のような情報画像は、情報が整理されていても、それだけで高得点にしないでください。
- 情報画像で鑑賞性、感情反応、驚き、面白さ、見せたくなる力が弱い場合はスコアを抑えてください。
- 料理写真、手作り料理、記録写真、日常の一枚は、プロの写真のように整っていなくても、食欲、驚き、温かみ、投稿者らしさ、見せたい意図が伝わる場合はきちんと評価してください。
- 「写真として綺麗か」だけでなく、「人に見せたときに反応が返ってきそうか」「コメントしたくなる要素があるか」「記憶に残る主題があるか」を重視してください。
- おじさんは、情緒、空気感、記憶に残る感じを重視して採点してください。
- ギャルは、映え、第一印象、誰かに見せたくなる感じを重視して採点してください。
- モデラーは、構図、質感、主題の見え方を重視して採点してください。
- 全体としては、ギャラリーに並べたときの鑑賞性と見せたくなる力を重視してください。

【投稿者が見てほしいポイントについて】
- 投稿者から「ココ見てほしい」が指定されている場合は、その内容を補助情報として軽く参考にしてください。
- ただし、コメントの主題にしすぎないでください。
- 画像そのものから読み取れる印象を最優先してください。
- 投稿者の意図をそのままなぞるのではなく、画像から自然に確認できる範囲で少しだけ反映してください。
- 3人のコメントすべてに「ココ見てほしい」の内容を入れる必要はありません。
- タイトル生成では、投稿者コメントよりも画像全体の印象を優先してください。
- スコアは画像自体の印象を基本とし、投稿者コメントに引っ張られすぎないようにしてください。

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
- ギャルのコメントでは、明るさやテンションを出すために絵文字を使ってもよいです。
- ただし、絵文字は1コメントにつき0〜2個までにしてください。
- 絵文字だらけにせず、大人向けアプリとして読みにくくならない範囲にしてください。
- 使う絵文字は、画像の印象に合うものを自然に選んでください。
- ギャルのコメントは、ジンさんやタクミより少しだけ明るく、親しみやすい温度感にしてください。
- ただし、過度な若者言葉、ネットスラング、強すぎるギャル語は避けてください。
- ジンさんとモデラーのコメントでは、基本的に絵文字を使わない

【追加アドバイス】
- 各キャラは、その画像について「もしもう一歩こういう方向に寄せたいなら」という提案型の短いアドバイスを1つ考えてください。
- アドバイスは改善点の指摘ではなく、方向性の提案にしてください。
- 強い否定やダメ出しはしないでください。
- 「もっと映えさせるなら」「もう少し全体をまとめるなら」「少しプロっぽく見せるなら」などの言い回しで、やわらかく提案してください。
- アドバイスは各キャラらしい視点で2文で作ってください。
- 今の画像の良さを壊さない提案にしてください。

【シェアしたくなるタイトル】
- 3人のキャラクターそれぞれの視点で、短い一言タイトルをつけてください
- おじさん：余韻、情緒、空気感を拾った落ち着いたタイトル
- ギャル：映え、エモさ、第一印象を拾った明るいタイトル
- モデラー：構図、質感、作り込みを拾った観察者らしいタイトル
- ただし全体として、今の表紙の世界観に合わせて、遊びすぎない大人向けのタイトルにしてください
- 大げさすぎるタイトル、ネットミームっぽいタイトル、ふざけすぎたタイトルは避けてください
- 短めで、人に見せたくなる一言にしてください
- タイトルは画像全体の印象を踏まえてください
- タイトルは分析メモや資料名のようにしすぎず、画廊に並べても自然な一言にしてください。

【この画像が好きそうな人たち】
- appeal_targets は、必ず下の候補タグ一覧から3〜4個だけ選んでください。
- 候補タグ一覧にない語句は使わないでください。
- 表記ゆれを避けるため、文字は候補タグと完全一致させてください。
- 画像の種類ではなく、「見たい人が探しやすい分類」として選んでください。
- 迷った場合は、より広く探されやすいタグを優先してください。

候補タグ一覧：
- 写真好き
- イラスト好き
- 料理好き
- 肉好き
- 風景好き
- 動物好き
- 乗り物好き
- ガンプラ好き
- キャラ好き
- かわいい好き
- かっこいい好き
- きれい好き
- 映え好き
- エモい好き
- しっとり派
- にぎやか派
- クセ強好き
- 王道好き
- 尖り好き
- ディテール好き
- 手作り好き
- 日常好き
- 作品づくり好き
- ネタ好き
- ファンタジー好き
- 魔法世界好き
- 冒険好き
- 幻想的好き
- 世界観好き

【出力ルール】
- 必ずJSONのみを返してください
- JSON以外の文章は一切出力しないでください
- 数値は整数で返してください
- character_scores は各キャラ 0〜100 の整数で返してください
- character_scores は丸い点数に寄せず、1点単位の自然な差を反映してください
- character_comments は各キャラ 2〜3文で返してください
- character_titles は各キャラごとに短めのタイトルを返してください
- share_title は互換性のために短めのタイトルを1つ返してもよいですが、最終表示では character_titles から選びます
- appeal_targets は候補タグ一覧から3〜4個だけ選び、候補タグと完全一致する文字列で返してください
- 候補タグ一覧にない語句は絶対に返さないでください
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
  "character_scores": {
    "おじさん": 0,
    "ギャル": 0,
    "モデラー": 0
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
  "character_titles": {
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
    "おじさん": "assets/icon_jin.jpg",
    "ギャル": "assets/icon_reina.jpg",
    "モデラー": "assets/icon_takumi.jpg",
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

def normalize_character_scores(character_scores: dict) -> dict:
    normalized_scores = {}

    for character in CHARACTER_WEIGHTS.keys():
        raw_score = character_scores.get(character, 0) if isinstance(character_scores, dict) else 0

        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = 0

        normalized_scores[character] = max(0, min(100, score))

    return normalized_scores

def calculate_fallback_character_scores(axis_scores: dict) -> dict:
    character_scores = {}

    for character, weights in CHARACTER_WEIGHTS.items():
        weighted_total = 0.0
        max_total = sum(weights.values())
        signed_nuance = 0.0

        for index, (axis_name, weight) in enumerate(weights.items(), start=1):
            raw_score = axis_scores.get(axis_name, 0)
            strength = abs(raw_score) / 5.0
            weighted_total += weight * strength
            signed_nuance += raw_score * index

        if max_total == 0:
            character_scores[character] = 0
            continue

        base_score = (weighted_total / max_total) * 100
        nuance_adjustment = signed_nuance * 0.13
        character_scores[character] = round(base_score + nuance_adjustment)

    return normalize_character_scores(character_scores)

def calculate_character_scores(axis_scores: dict, ai_character_scores: dict = None) -> tuple[dict, int, int]:
    """
    axis_scores: AIが返した8軸スコア（-5～+5）
    ai_character_scores: AIが返した3キャラ別スコア（0～100）
    return:
        character_scores: 各キャラの表示用スコア（0～100）
        true_score: 内部用 0～300点
        three_vis: 表示・保存用 0～300
    """
    if isinstance(ai_character_scores, dict) and ai_character_scores:
        character_scores = normalize_character_scores(ai_character_scores)
    else:
        character_scores = calculate_fallback_character_scores(axis_scores)

    true_score = sum(character_scores.values())
    three_vis = true_score

    return character_scores, true_score, three_vis

def get_top_character_key(character_scores: dict) -> str:
    return max(character_scores, key=character_scores.get)

def get_top_character_name(character_scores: dict) -> str:
    top_key = get_top_character_key(character_scores)
    return CHARACTER_DISPLAY_NAMES.get(top_key, top_key)

def choose_share_title(character_titles: dict, character_scores: dict, fallback_title: str = "") -> str:
    if not isinstance(character_titles, dict):
        character_titles = {}

    top_character_key = get_top_character_key(character_scores)
    selected_title = character_titles.get(top_character_key, "")
    return selected_title or fallback_title or "無題の一枚"

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

def analyze_image_with_ai(image_bytes, focus_point=""):
    if USE_MOCK_DATA:
        return get_mock_analysis_data()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{base64_image}"

    extra_focus_prompt = ""
    if focus_point and focus_point.strip():
        extra_focus_prompt = f"""

【投稿者が見てほしいポイント】
{focus_point.strip()}
"""

    full_prompt = SECOND_PROJECT_PROMPT + extra_focus_prompt

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": full_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    response_text = response.output_text
    data = json.loads(response_text)

    axis_scores = data["axis_scores"]
    ai_character_scores = data.get("character_scores", {})
    character_comments = data["character_comments"]
    character_advice = data["character_advice"]
    character_titles = data.get("character_titles", {})
    if not isinstance(character_titles, dict):
        character_titles = {}
    share_title = data.get("share_title", "")
    appeal_targets = data["appeal_targets"]

    return axis_scores, ai_character_scores, character_comments, character_advice, character_titles, share_title, appeal_targets

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


def save_result_to_supabase(
    image_bytes: bytes,
    uploaded_file_name: str,
    share_title: str,
    top_character_name: str,
    three_vis: int,
    appeal_targets,
    character_scores: dict,
    character_comments: dict,
    axis_scores: dict,
    poster_name: str,
    poster_profile: str,
    focus_point: str,
):
    file_ext = uploaded_file_name.split(".")[-1].lower() if "." in uploaded_file_name else "jpg"
    file_name = f"{uuid4()}.{file_ext}"
    storage_path = f"uploads/{file_name}"

    # 画像を Storage に保存
    supabase.storage.from_("evaluation-images").upload(
        path=storage_path,
        file=image_bytes,
        file_options={"content-type": "image/jpeg"}
    )

    # DB に保存
    row_data = {
        "share_title": share_title,
        "three_vis": three_vis,
        "appeal_targets": appeal_targets,
        "comment_jin": character_comments["おじさん"],
        "comment_reina": character_comments["ギャル"],
        "comment_takumi": character_comments["モデラー"],
        "axis_scores_json": axis_scores,
        "image_path": storage_path,
        "is_public": False,
        "is_featured": False,
        "lang": "ja",
        "poster_name": poster_name,
        "poster_profile": poster_profile,
        "focus_point": focus_point,
    }

    supabase.table("image_evaluations").insert(row_data).execute()

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

    character_scores = {
        "おじさん": 72,
        "ギャル": 84,
        "モデラー": 79,
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

    character_titles = {
        "おじさん": "食卓に残る焼き色",
        "ギャル": "この焼き目、見せたい",
        "モデラー": "断面で魅せる主役感",
    }
    share_title = "肉、正面から来た"
    appeal_targets = ["料理好き", "肉好き", "映え好き", "手作り好き"]

    return axis_scores, character_scores, character_comments, character_advice, character_titles, share_title, appeal_targets

def prepare_image_for_app(
    uploaded_file,
    rotation_angle=0,
    max_size_kb=500,
    max_width=1280,
    max_height=1280,
    quality=85
):
    """
    uploaded_file を1回だけ圧縮して、
    画面表示・API送信・保存に使えるJPEG bytesを返す
    rotation_angle: 0 / 90 / 180 / 270
    """
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    if rotation_angle:
        # PILのrotateは反時計回りなので、見た目として右回転にするためマイナス指定
        image = image.rotate(-rotation_angle, expand=True)

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

    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    return f"data:{mime};base64,{encoded}"

st.image("assets/hero_impression_ja_mobile.jpg", use_container_width=True)
GALLERY_URL = "https://three-vibe-gallery-ydrzzjnrcfd989aciulbxh.streamlit.app"

st.markdown(
    f"""
    <div style="margin: 12px 0 22px; text-align: center;">
      <a href="{GALLERY_URL}" target="_blank" style="
          display: inline-block;
          width: 100%;
          box-sizing: border-box;
          padding: 13px 16px;
          border-radius: 999px;
          background: #ffffff;
          border: 1px solid #d0d5dd;
          color: #1f2937;
          font-size: 16px;
          font-weight: 700;
          text-decoration: none;
          box-shadow: 0 4px 14px rgba(31, 41, 55, 0.08);
      ">
        公開作品ギャラリーを見る
      </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style="margin-top: 10px; margin-bottom: 0px; font-size: 18px; color: #333;">
    AIキャラに見せる画像を読み込んでください
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stFileUploader"] {
    margin-top: -10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background:#fff7ed;
    border:1px solid #fed7aa;
    border-left:5px solid #f97316;
    border-radius:12px;
    padding:12px 14px;
    margin:10px 0 16px;
    color:#333;
    font-size:15px;
    line-height:1.7;
">
  <div style="font-weight:800; margin-bottom:6px;">投稿前のお願い</div>
  <div>
    公開しても問題ない画像だけをアップロードしてください。<br>
    他人の顔がはっきり写っている画像、住所・氏名・電話番号・車のナンバーなど個人情報が読める画像、
    他人の著作物を無断で使った画像、社会的に公開するのが不適切な画像は投稿しないでください。<br>
    ギャラリー公開前に内容を確認し、不適切と判断したものは非公開または削除する場合があります。
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

poster_name = st.text_input(
    "投稿者名",
    placeholder="例：Taro、Jony、photo_life、illustration_mika など"
)

poster_profile = st.text_area(
    "投稿者プロフィール（任意・公開されます）",
    placeholder=(
        "例：\n"
        "X：@sample_creator\n"
        "Instagram：@sample_gallery\n"
        "note：https://note.com/sample\n"
        "感想や制作相談はXのDMまでお願いします。\n\n"
        "※電話番号・住所・本名・個人メールなど、公開したくない情報は入力しないでください。"
    ),
    height=150
)

focus_point = st.text_area(
    "この画像の「ココ見てほしい！」を入力してください\n※空欄でも印象値には影響しません",
    placeholder="例：アピールポイント、全体の配色、手作り感、高級感、かわいさ など",
    height=80
)

if uploaded_file is not None:
    current_upload_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("current_upload_key") != current_upload_key:
        st.session_state["current_upload_key"] = current_upload_key
        st.session_state["has_evaluated_current_image"] = False
        st.session_state["has_entered_current_image"] = False

        # 前の画像の評価結果を消す
        for key in [
            "axis_scores",
            "character_comments",
            "character_advice",
            "share_title",
            "appeal_targets",
            "character_scores",
            "true_score",
            "three_vis",
            "uploaded_file_name",
            "focus_point",
            "poster_name",
            "poster_profile",
            "prepared_image_bytes",
        ]:
            st.session_state.pop(key, None)

    rotation_label = st.radio(
        "画像の向き",
        ["そのまま", "右に90度", "180度", "左に90度"],
        horizontal=True,
        help="スマホ写真が横向きに表示される場合は、ここで向きを調整してください。"
    )

    rotation_map = {
        "そのまま": 0,
        "右に90度": 90,
        "180度": 180,
        "左に90度": 270,
    }

    rotation_angle = rotation_map[rotation_label]

    prepared_image_bytes = prepare_image_for_app(
        uploaded_file,
        rotation_angle=rotation_angle,
        max_size_kb=500
    )

    display_image = Image.open(io.BytesIO(prepared_image_bytes))

    st.image(display_image, caption="読み込んだ画像", use_container_width=True)

    already_evaluated = st.session_state.get("has_evaluated_current_image", False)

    if st.button(
        "3人に見てもらう",
        disabled=already_evaluated,
        use_container_width=True,
    ):
        try:
            axis_scores, ai_character_scores, character_comments, character_advice, character_titles, fallback_share_title, appeal_targets = analyze_image_with_ai(prepared_image_bytes, focus_point)
            character_scores, true_score, three_vis = calculate_character_scores(axis_scores, ai_character_scores)
            share_title = choose_share_title(character_titles, character_scores, fallback_share_title)

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
        st.session_state["focus_point"] = focus_point
        st.session_state["poster_profile"] = poster_profile
        st.session_state["prepared_image_bytes"] = prepared_image_bytes
        st.session_state["poster_name"] = poster_name
        st.session_state["has_evaluated_current_image"] = True

    if already_evaluated:
        st.info("この画像はすでに評価済みです。もう一度評価したい場合は、別の画像を読み込んでください。")

    if "axis_scores" in st.session_state:
        axis_scores = st.session_state["axis_scores"]
        character_comments = st.session_state["character_comments"]
        character_advice = st.session_state["character_advice"]
        share_title = st.session_state["share_title"]
        appeal_targets = st.session_state["appeal_targets"]
        character_scores = st.session_state["character_scores"]
        true_score = st.session_state["true_score"]
        three_vis = st.session_state["three_vis"]
        focus_point = st.session_state.get("focus_point", "")

        top_character_name = get_top_character_name(character_scores)

        st.divider()

        st.markdown("""
        <div style="
            background:#fff7ed;
            border:1px solid #fed7aa;
            border-left:5px solid #f97316;
            border-radius:12px;
            padding:12px 14px;
            margin:8px 0 18px;
            color:#333;
            font-size:15px;
            line-height:1.7;
        ">
          <strong>保存について</strong><br>
          評価結果の下に「ギャラリーにエントリーする」ボタンがあります。ボタンを押すと、この画像と評価結果がギャラリー候補に登録されます。<br>
          登録後、管理人が画像を確認してからギャラリーに公開します。
        </div>
        """, unsafe_allow_html=True)


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
        この画像に3人のAIキャラがつけた印象値
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
        components.html(score_card_html, height=230, scrolling=False)
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
        components.html(full_comment_html, height=1200, scrolling=False)
  
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

        st.subheader("ギャラリーへのエントリー")

        st.info("この評価結果をギャラリー候補に登録するには、下の「ギャラリーにエントリーする」を押してください。公開は管理人の確認後に行われます。")

        already_entered = st.session_state.get("has_entered_current_image", False)

        if st.button(
            "ギャラリーにエントリーする",
            disabled=already_entered,
            use_container_width=True,
        ):

            try:
                save_result_to_supabase(
                    image_bytes=st.session_state["prepared_image_bytes"],
                    uploaded_file_name=st.session_state["uploaded_file_name"],
                    share_title=share_title,
                    top_character_name=top_character_name,
                    three_vis=three_vis,
                    appeal_targets=appeal_targets,
                    character_scores=character_scores,
                    character_comments=character_comments,
                    axis_scores=axis_scores,
                    poster_name=st.session_state.get("poster_name", ""),
                    poster_profile=st.session_state.get("poster_profile", ""),
                    focus_point=st.session_state.get("focus_point", ""),
                )

                st.session_state["has_entered_current_image"] = True

                st.success("ギャラリー候補にエントリーしました。管理人の確認後に公開されます。")
                st.info("画像と評価結果を保存しました。")

            except Exception as e:
                st.error(f"保存中にエラーが発生しました: {e}")

        if already_entered:
            st.info("この画像はすでにギャラリー候補にエントリー済みです。")

        st.divider()

def render_contact_footer():
    st.markdown("""
<div style="
    margin-top: 28px;
    padding: 14px 12px;
    border-top: 1px solid #e5e7eb;
    color: #555;
    font-size: 14px;
    line-height: 1.7;
">
  <strong>お問い合わせ</strong><br>
  公開画像の削除依頼、不適切な画像の報告、その他のお問い合わせは、管理人までご連絡ください。<br>
  連絡先：threevibe.gallery@gmail.com
</div>
""", unsafe_allow_html=True)


render_contact_footer()