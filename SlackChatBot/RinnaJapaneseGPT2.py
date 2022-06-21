# Transformerライブラリを介してrinna株式会社日本語版GPT-2事前学習モデルを呼び出し
from transformers import T5Tokenizer, AutoModelForCausalLM
# 正規表現
import re

# rinna株式会社日本語版GPT-2事前学習モデル
model_name = 'rinna/japanese-gpt2-medium'
# tokenizerとmodelの生成
# tokenizerで文字列をトークンにエンコード/デコード
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 受け取ったメッセージに対して文章を生成する処理
# input_messageにチャットボットに返答させる文章を格納
def nlp(input_message):
    # tokenizerでエンコード
    # 発言者(ユーザーとAI)を区別することで対話文と認識させる
    input_ids = tokenizer.encode(
        "私: " + input_message + "\nAI: ",
        return_tensors="pt"
    )
    # generate関数で文章生成
    output_sequences = model.generate(
        # エンコードしたメッセージ
        input_ids=input_ids,
        # 文章生成文字数上限
        max_length=50,
        # 値が1に近づくほどクリエイティブな文章、0に近づくと論理的で正確な返答になる…らしい
        temperature=0.9,
        # top_kとtop_pの値は排他で、片方を>0に設定したら片方を0にする必要がある
        # 確率の上位候補の絞り込み数を設定
        top_k=0,
        # 確率が一定以上の単語に絞る。下限となる確率を0～1の範囲で設定
        top_p=0.9,
        # 1に近づけると同じ文章の繰り返しを減少する効果
        repetition_penalty=1.0,
        # num_return_sequencesを複数回指定する場合Trueにする
        do_sample=True,
        # 一度の実行で生成する文章の数を指定
        num_return_sequences=1)

    # トークンの状態の生成文章が二次元リストで格納されているので二次元リスト0番目のリストを取得
    output_sequence = output_sequences.tolist()[0]
    # tokenizerでデコード
    text = tokenizer.decode(
        output_sequence, clean_up_tokenization_spaces=True)

    # 最初に与えた入力文の長さを取得
    input_ids_length = len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True))
    # デコードされた文章から最初に与えた入力文を除去
    total_text = (text[input_ids_length:])

    # 生成文にpatternにある文字列が含まれていた場合それ以降の文字列を除去。含まれていなかった場合はそのまま出力
    pattern = r"AI:|私:|俺:|僕:|あなた:|<unk>|:"
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
