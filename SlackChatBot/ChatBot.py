# Slackライブラリ
from slack_sdk import WebClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
# 正規表現
import re
#処理の状態管理
from ProcessPool import ProcessPool
# # rinna株式会社日本語版GPT-2事前学習モデル
# import RinnaJapaneseGPT2 as rinna

# rinna株式会社日本語特化GPT言語モデル
import RinnaJapaneseGPT1b as rinna

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
token.chat_postMessage(text="ログインしました。このチャンネルで話しかけてください。", channel=channel)

# bot初期化
app = App(token=SLACK_BOT_TOKEN)

# 投稿されたメッセージに対してrinnaで生成された文章で返信
@app.message(re.compile("(.*)"))
def reply(message, context, say):
    # メッセージを投稿したユーザーを判定
    user = message["user"]

    # メッセージが投稿された時に処理の状態がロックされていない(文章生成中ではない)ならロックする
    if ProcessPool.is_lock() == False:
        ProcessPool.lock()
    # メッセージが投稿された時に処理の状態がロックされている(文章生成中)ならメッセージを返して無効にする
    else:
      response = "現在文章生成中です。文章の生成が完了してから話しかけてください。"
      # メンション付きメッセージを指定チャンネルに返す
      text = f'<@{user}>{response}'
      say(text=text, channel=channel)
      return
    
    # 文章生成開始時にメッセージを送信する処理
    token.chat_postMessage(text="文章生成中…", channel=channel)
    
    # 正規表現のマッチ結果がcontext.matchesに設定される
    greeting = context['matches'][0]
    # 投稿されたメッセージをrinnaに投げて生成された文章を受け取る
    response = rinna.nlp(greeting)
    # メンション付きメッセージを指定チャンネルに返す
    text = f'<@{user}>{response}'
    say(text=text, channel=channel)
    # ロックを解除する
    ProcessPool.unlock()

# bot起動
if __name__ == "__main__":
  print('start slackbot')
  SocketModeHandler(app, SLACK_APP_TOKEN).start()
