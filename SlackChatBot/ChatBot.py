# Slackライブラリ
from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
# 正規表現
import re
# rinna株式会社日本語版GPT-2事前学習モデル
import RinnaJapaneseGPT2 as rinna

# Bot User OAuth Token
SLACK_BOT_TOKEN = "Bot User OAuth Tokenを記述"
# Socket Mode のToken
SLACK_APP_TOKEN = "Socket Mode のTokenを記述"

# botを動かすチャンネルのID
CHANNEL_ID = "botを動かすチャンネルIDを記述"
# チャンネルIDをchannelに格納
channel=CHANNEL_ID

# トークンをtokenに格納
token = WebClient(SLACK_BOT_TOKEN)

# ログイン時にメッセージを送信する処理
token.chat_postMessage(channel=channel, text="ログインしました。このチャンネルで話しかけてください。")

# bot初期化
app = App(token=SLACK_BOT_TOKEN)

# 投稿されたメッセージに対してrinnaで生成された文章で返信
@app.message(re.compile("(.*)"))
def reply(message, context, say):
    # 正規表現のマッチ結果がcontext.matchesに設定される
    greeting = context['matches'][0]
    # 投稿されたメッセージをrinnaに投げて生成された文章を受け取る
    response = rinna.nlp(greeting)
    # メッセージを投稿したユーザーを判定
    user = message["user"]
    # メンション付きメッセージを指定チャンネルに返す
    text = f'<@{user}>{response}'
    say(text=text, channel=channel)

# bot起動
if __name__ == "__main__":
  print('start slackbot')
  SocketModeHandler(app, SLACK_APP_TOKEN).start()
