# preprocess_single_jieba.py

import pandas as pd
import re
import jieba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# ===== 載入資料 =====
df1 = pd.read_excel("113索資.xlsx")[['索取資料題目', '承辦機關']]
df2 = pd.read_excel("114索資.xlsx")[['索取資料題目', '承辦機關']]
df = pd.concat([df1, df2]).dropna()
df.columns = ['text', 'label']
df['text'] = df['text'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()
df = df[(df['text'] != '') & (df['label'] != '')]

# ===== 贅詞移除 =====
redundant_words = [
    "有", "是", "與", "的", "在", "從", "到", "自", "由", "往", "朝", "於", "對", "跟", "與", "給", "為",
    "因", "因為", "由於", "由", "因應", "靠", "以", "用", "替", "為了", "關於", "對於", "就", "有關",
    "自從", "直到", "至", "依", "據", "憑", "循", "藉由", "藉", "趁", "比", "如", "按照", "根據", "依照",
    "照", "沿著", "沿", "順著", "隨", "隨著", "除了", "除了以外", "不如", "］", "?", "？", "［", "。",
    "提供", "制表", "彙整後", "貴單位", "主旨", "：", ":", "懇請", "（", "）", "一、", "二、", "三、", "四、",
    "五、", "六、", "1.", "2.", "3.", "4.", "5.", "1、", "2、", "3、", "4、", "5、", "承上", "敬請", "茲問政需要，",
    "惠予", "_x000D_", "惠請", "【", "】"
] # 同前完整列表
redundant_words = sorted(set(redundant_words), key=lambda x: -len(x))

def remove_redundant_words(text):
    for word in redundant_words:
        text = text.replace(word, "")
    return text.strip()

df['text'] = df['text'].apply(remove_redundant_words)

# ===== Label 映射與正規化 =====
label_mapping = {
"市政大樓公共事務管理中心": "秘書處",

    "殯葬管理處": "民政局", 
    "孔廟管理委員會": "民政局",
    
    "稅捐稽徵處": "財政局", 
    "動產質借處": "財政局",
    
    "圖書館": "教育局", 
    "動物園": "教育局", 
    "教師研習中心": "教育局", 
    "教研中心":"教育局",
    "天文科學教育館": "教育局", 
    "家庭教育中心": "教育局",
    "青少年發展暨家庭教育中心": "教育局",

    "停車管理工程處": "交通局", 
    "交通管制工程處": "交通局", 
    "公共運輸處": "交通局", 
    "交通事件裁決所": "交通局",

    "陽明教養院": "社會局", 
    "浩然敬老院": "社會局", 
    "家庭暴力暨性侵害防治中心": "社會局",

    "勞動檢查處": "勞動局", 
    "就業服務處": "勞動局", 
    "勞動力重建運用處": "勞動局", 
    "職能發展學院": "勞動局",

    "警察局保安警察大隊": "警察局", 
    "警察局刑事警察大隊": "警察局", 
    "警察局交通警察大隊": "警察局", 
    "警察局少年警察隊": "警察局",
    "警察局婦幼警察隊": "警察局", 
    "警察局捷運警察隊": "警察局", 
    "警察局通信隊": "警察局",
    "警察局大同分局": "警察局",
    "警察局萬華分局": "警察局",
    "警察局中山分局": "警察局",
    "警察局大安分局": "警察局",
    "警察局中正第一分局": "警察局",
    "警察局中正第二分局": "警察局",
    "警察局松山分局": "警察局",
    "警察局信義分局": "警察局",
    "警察局士林分局": "警察局",
    "警察局北投分局": "警察局",
    "警察局文山第一分局": "警察局",
    "警察局文山第二分局": "警察局",
    "警察局南港分局": "警察局",
    "警察局內湖分局": "警察局",

    "聯合醫院": "衛生局",

    "環境保護局環保稽查大隊": "環境保護局", 
    "環境保護局內湖垃圾焚化廠": "環境保護局", 
    "環境保護局木柵垃圾焚化廠": "環境保護局", 
    "環境保護局北投垃圾焚化廠": "環境保護局",

    "國樂團": "文化局", 
    "交響樂團": "文化局", 
    "美術館": "文化局", 
    "中山堂管理所": "文化局", 
    "文獻館": "文化局", 
    "藝文推廣處": "文化局",

    "捷運工程局第一區工程處": "捷運工程局", 
    "捷運工程局第二區工程處": "捷運工程局", 
    "捷運工程局機電系統工程處": "捷運工程局",

    "臺北廣播電臺": "觀光傳播局",

    "地政局土地開發總隊": "地政局",

    "臺北自來水事業處工程總隊": "臺北自來水事業處",

    "松山區戶政事務所": "民政局",
    "信義區戶政事務所": "民政局",
    "大安區戶政事務所": "民政局",
    "中山區戶政事務所": "民政局",
    "中正區戶政事務所": "民政局",
    "大同區戶政事務所": "民政局",
    "南港區戶政事務所": "民政局",
    "內湖區戶政事務所": "民政局",
    "士林區戶政事務所": "民政局",
    "北投區戶政事務所": "民政局",
    "文山區戶政事務所": "民政局",
    "萬華區戶政事務所": "民政局",    

    "松山地政事務所": "地政局",
    "大安地政事務所": "地政局",
    "中山地政事務所": "地政局",
    "古亭地政事務所": "地政局",
    "士林地政事務所": "地政局",
    "建成地政事務所": "地政局",

    "松山區公所": "民政局",
    "信義區公所": "民政局",
    "大安區公所": "民政局",
    "中山區公所": "民政局",
    "中正區公所": "民政局",
    "大同區公所": "民政局",
    "南港區公所": "民政局",
    "內湖區公所": "民政局",
    "士林區公所": "民政局",
    "北投區公所": "民政局",
    "文山區公所": "民政局",
    "萬華區公所": "民政局",

}
valid_labels = {
    "臺北市政府",
    "秘書處", "民政局", "財政局", "教育局", "產業發展局", "工務局", "交通局", "社會局", "勞動局", 
    "警察局", "衛生局", "環境保護局", "都市發展局", "文化局", "消防局", "捷運工程局", "臺北翡翠水庫管理局", "觀光傳播局", 
    "地政局", "兵役局", "體育局", "資訊局", "法務局", "青年局", "主計處", "人事處", "政風處", 
    "公務人員訓練處", "研究發展考核委員會", "都市計畫委員會", "原住民族事務委員會", "客家事務委員會", "臺北自來水事業處", 
    "臺北大眾捷運股份有限公司", "工務局新建工程處", "工務局水利工程處", "工務局公園路燈工程管理處", 
    "工務局衛生下水道工程處", "工務局大地工程處", "市場處", "商業處", "動物保護處", "都市更新處", "建築管理工程處",
}

df['label'] = df['label'].replace(label_mapping)

def normalize_label(label):
    return label if label in valid_labels else "其他"

df['label'] = df['label'].apply(normalize_label)

# ===== 正則清理流水號格式 =====
pattern = r'\b[A-Za-z]\d{5}-\d{8}\b'
df['text'] = df['text'].apply(lambda x: re.sub(pattern, '', str(x)))

# ===== 單標籤模式 =====
grouped = df.groupby('text')['label'].first().reset_index()
grouped = grouped[grouped['label'].isin(valid_labels)]

df = shuffle(df, random_state=42)

# ===== 分割資料 =====
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(grouped['label'])

X_temp, X_test, y_temp, y_test = train_test_split(grouped['text'], y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

# ===== jieba 分詞 =====
X_train = X_train.apply(lambda x: " ".join(jieba.cut_for_search(x)))
X_val = X_val.apply(lambda x: " ".join(jieba.cut_for_search(x)))
X_test = X_test.apply(lambda x: " ".join(jieba.cut_for_search(x)))

# ===== 儲存資料 =====
X_train.to_csv("train_texts_single_jieba.csv", index=False, encoding='utf-8-sig')
X_val.to_csv("val_texts_single_jieba.csv", index=False, encoding='utf-8-sig')
X_test.to_csv("test_texts_single_jieba.csv", index=False, encoding='utf-8-sig')
pd.Series(y_train).to_csv("train_labels_single_jieba.csv", index=False)
pd.Series(y_val).to_csv("val_labels_single_jieba.csv", index=False)
pd.Series(y_test).to_csv("test_labels_single_jieba.csv", index=False)

import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

print("前處理完成，輸出檔案已儲存。")
