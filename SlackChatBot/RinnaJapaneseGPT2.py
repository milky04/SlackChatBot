# Transformerライブラリを介してrinna株式会社日本語特化GPT言語モデルを呼び出し
from transformers import T5Tokenizer, AutoModelForCausalLM
# 正規表現
import re

# rinna株式会社日本語特化GPT言語モデル(パラメータ数3.3億)
model_name = 'rinna/japanese-gpt2-medium'
# tokenizerとmodelの生成
# tokenizerで文字列をトークンにエンコード/デコード
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 受け取ったメッセージに対して文章を生成する処理
# input_messageにチャットボットに返答させる文章を格納
def nlp(input_message):
    # 受け取ったメッセージをtokenizerでエンコード
    # 発言者(ユーザーとAI)を区別することで対話文と認識させる
    input_ids = tokenizer.encode(
        "私: " + input_message + "\nAI: ",
        # スペシャルトークンの自動追加を無効(自分で追加するため)
        add_special_tokens=False,
        # PyTorchのテンソル型で返す(Transformerモデルの入力でテンソル形式に変換する必要があるため)
        # 機械学習におけるテンソルは多次元配列とほぼ同義(=行列。GPUとテンソルの計算の相性が良い)
        return_tensors="pt"
    )

    # generate関数で文章生成
    output_sequences = model.generate(
        # エンコードしたメッセージ
        input_ids=input_ids,
        # 文章生成文字数上限
        max_length=50,
        # 文章生成文字数下限
        min_length=10,
        # 値が1に近づくほどクリエイティブな文章、0に近づくと論理的で正確な返答になる…らしい
        temperature=0.9,
        # 確率の高い上位k個の候補の単語からランダムに選択
        top_k=50,
        # 確率の高い上位候補の単語の確率の合計がpを超えるような最小個数の候補を動的に選択(1>p>=0)
        top_p=0.95,
        # 1に近づけると同じ文章の繰り返しを減少する効果
        repetition_penalty=1.0,
        # サンプリングを有効化
        do_sample=True,
        # 一度の実行で生成する文章の数を指定
        num_return_sequences=1,
        # モデルの入力に必要な情報となるスペシャルトークンを指定
        # PAD(Padding)は使用されていない部分を埋めるためのトークン
        pad_token_id=tokenizer.pad_token_id,
        # BOS(Beggin Of Sentence)はシーケンス(文章)の始まりを表すトークン
        bos_token_id=tokenizer.bos_token_id,
        # EOS(End Of Sentence)はシーケンス(文章)の終わりを表すトークン
        eos_token_id=tokenizer.eos_token_id,
        # 生成が許可されていないトークンIDリストを設定(禁止用語等の設定で使用)
        # UNK(Unknown)は未知の語彙を表すトークン
        bad_word_ids=[[tokenizer.unk_token_id]]
    )

    # トークンの状態の生成文章が二次元リストで格納されているので二次元リスト0番目のリストを取得
    output_sequence = output_sequences.tolist()[0]
    # 生成された文章をtokenizerでデコード
    text = tokenizer.decode(
        output_sequence,
        # 余分に生成されたスペースを削除
        clean_up_tokenization_spaces=True
        )

    # 最初に与えた入力文の長さを取得
    input_ids_length = len(tokenizer.decode(
        input_ids[0],
        # 余分に生成されたスペースを削除
        clean_up_tokenization_spaces=True
        ))
    # デコードされた文章から最初に与えた入力文を除去
    total_text = (text[input_ids_length:])

    # 生成文にpatternにある文字列が含まれていた場合それ以降の文字列を除去。含まれていなかった場合はそのまま出力
    pattern = r"AI:|私:|俺:|僕:|あなた:|<unk>"
    if re.search(pattern, total_text) != None:
        # 文章中のpatternにある文字列の位置を全て抽出
        match_position = [match.span() for match in re.finditer(pattern, total_text)]
        # 一番最初に出現したpatternにある文字列の位置を抽出
        first_match_position = match_position[0][0]
        #  一番最初に出現したpatternにある文字列以降の文字列を除去
        edited_text = total_text[:first_match_position]
        # 生成された文章を返す
        return edited_text
    else:
        # 生成された文章を返す
        return total_text
