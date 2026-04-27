# Three Vibe Impression App

3人のAIキャラクターが、アップロードされた画像の「印象値」を遊び感覚で評価する Streamlit アプリです。

画像を OpenAI API で解析し、8つの印象軸、3人のキャラクター別コメント、タイトル、刺さりそうな層、レーダーチャートなどを表示します。

## 公開URL

[Three Vibe Impression App](https://three-vibe-impression-app-amy8zqpimaqbtkf4grxqj6.streamlit.app/)

## できること

- PNG / JPG / JPEG 画像をアップロード
- OpenAI API による画像の印象分析
- 8つの印象軸によるスコア表示
- 3人のAIキャラクターによるコメント表示
- 画像につけられたタイトルと、刺さりそうな層の表示
- 16項目レーダーチャートと8軸バー表示
- 評価結果のCSV保存と画像保存
- 保存履歴の表示

## アプリの構成

```text
.
├── app.py
├── requirements.txt
└── assets/
```


app.py : Streamlit アプリ本体
requirements.txt : Community Cloud / ローカル実行用の依存パッケージ
assets/ : アプリ内で使うキャラクター画像など
必要なもの
Python 3.10 以上を推奨
OpenAI API キー
Streamlit Community Cloud またはローカル実行環境
セットアップ
pip install -r requirements.txt

ローカルで実行する場合は、.streamlit/secrets.toml に OpenAI API キーを設定します。

OPENAI_API_KEY = "your-api-key"

Streamlit Community Cloud では、アプリの Settings から Secrets に同じ値を登録してください。

実行方法
streamlit run app.py

起動後、ブラウザで表示された Streamlit 画面から画像をアップロードし、AI評価を実行します。

Streamlit Community Cloud での注意

このアプリには、評価結果を results_2nd.csv に追記し、画像を saved_images_2nd/ に保存する機能があります。

ただし、Streamlit Community Cloud のファイルシステムは永続保存先としては扱えません。
アプリの再起動、再デプロイ、環境のリセットなどにより、保存したCSVや画像が失われる可能性があります。

公開環境で結果を長期保存したい場合は、外部ストレージやデータベースへの保存に切り替える必要があります。

requirements.txt の確認

現在の依存パッケージは、アプリで使っている主要ライブラリに対応しています。

streamlit : アプリUI
pillow : アップロード画像の読み込みと圧縮
matplotlib : グラフ描画
numpy : レーダーチャートなどの数値処理
pandas : 保存履歴CSVの読み込みと表示
openai : OpenAI API 呼び出し
japanize-matplotlib : matplotlib の日本語表示
開発メモ

app.py には UI 表示、プロンプト、スコア計算、保存処理がまとまっています。
今後の変更では、次のように分けると機能追加や保守がしやすくなります。

AI解析処理を services/ などに分離
スコア計算ロジックを専用モジュールに分離
グラフ描画処理を専用モジュールに分離
保存処理をローカル保存 / 外部保存で切り替えられる形に整理
プロンプト本文を設定ファイル化し、変更履歴を追いやすくする
今後追加しやすい機能候補
Community Cloud でも消えない保存先への対応
Google Sheets
Supabase
SQLite + 外部永続ストレージ
S3互換ストレージ
保存履歴の検索、絞り込み、並び替え
結果の共有用画像の生成
評価結果のCSVダウンロード
複数画像の一括評価
キャラクターや評価軸の設定切り替え
OpenAI API エラー時のリトライやわかりやすい案内
アプリ本体のモジュール分割と簡単なテスト追加
ライセンス

必要に応じて、このリポジトリにライセンスファイルを追加してください。