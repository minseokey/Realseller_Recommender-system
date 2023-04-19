import pandas as pd
import re
from gensim.models import Word2Vec
from konlp.kma.klt2023 import klt2023


df = pd.read_csv("data/review_data.csv")
k = klt2023()


# 한글만 남기기
def text_clearing(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', text)
    return result

# 리뷰가 들어올때, 단어벡터 학습 모델 생성 하는 함수 128차원으로 만들었다.
def modelmaker(dataFrame):
    token = []
    for i in range(len(df)):
        token.append(dataFrame["review"][i])
    print(token)
    model = Word2Vec(sentences=token, vector_size=128, window=5, min_count=2, workers=4, sg=0)
    model_file = 'word2vec-review_data.model'
    model.save(model_file)
    print(f"--> Model file <{model_file}> was created!\n")
    return model


# 각각의 리뷰를 형태소로 분해.
df["review"] = df["review"].apply(lambda x: text_clearing(x))
df["review"] = df["review"].apply(lambda x: k.nouns(x))

model = modelmaker(df)
print(model.wv.most_similar("고기"))

