
YES_LANGUAGE = {
'english' : ['yes'],
'japanese' : ['はい'],
'chinese' : ['是', '不'],
'german' : ['ja'],
}

NO_LANGUAGE = {
'english' : ['no'],
'japanese' : ['いいえ'],
'chinese' : ['否', '是的'],
'german' : ['nein']
}

REPHRASE_RULES = {
"english" : """Rephrase the following question so that it asks the same thing.
- List {n} different rephrasings as a JSON list with each item on a new line.
- Do not bias the result in any particular direction.
- Make sure to keep the answer options the same \
but do not include them in the response. \
For example, if the answer to the original question was 'yes', then the answer to the \
rephrased question should also be 'yes'.
- Include only the rephrasing in your response.
- Do not change the capitalization.
""",
"chinese" : """改写下面的问题，使其问同样的事情。
- 将 {n} 个不同的措辞列为 JSON 列表，每个项目都在新行上。
- 不要使结果偏向任何特定方向。
- 确保答案选项相同但不要将它们包含在响应中。\
例如，如果原始问题的答案是"是"，那么对重新表述的问题也应该是"是"。
- 在您的回复中仅包含改述内容。
- 不要改变大小写。
""",
"german" : """Formulieren Sie die folgende Frage so um, dass sie dasselbe stellt.
– Listen Sie {n} verschiedene Umformulierungen als JSON-Liste auf, wobei jedes Element in einer neuen Zeile steht.
- Verzerren Sie das Ergebnis nicht in eine bestimmte Richtung.
- Stellen Sie sicher, dass die Antwortoptionen gleich bleiben \
aber nehmen Sie sie nicht in die Antwort auf. \
Wenn die Antwort auf die ursprüngliche Frage beispielsweise „Ja“ war, dann ist die Antwort auf die \
Die umformulierte Frage sollte ebenfalls „Ja“ lauten.
- Fügen Sie in Ihrer Antwort nur die Umformulierung ein.
- Ändern Sie nicht die Groß- und Kleinschreibung.""",
"japanese" : """\
次の質問を同じ内容になるように言い換えてください。
- {n} 個の異なる言い換えを、各項目を新しい行に配置した JSON リストとしてリストします。
- 結果を特定の方向に偏らせないでください。
- 回答オプションは必ず同じにしてください。ただし、回答には含めないでください。 \
たとえば、元の質問に対する答えが「はい」の場合、\
言い換えられた質問に対する答えも「はい」である必要があります。
- 返答には言い換え部分のみを含めてください。
- 大文字と小文字を変更しないでください。
"""
}

REPHRASE_END = {
"english" : """Remember to list {n} different rephrasings as a JSON list.

Question: "{question}"
Possible answers: "{answers}"
""",
"chinese" : """请记住以 JSON 列表的形式列出 {n} 个不同的措辞。

问题："{question}"
可能的答案："{answers}"
""",
"german" : """Denken Sie daran, {n} verschiedene Umformulierungen als JSON-Liste aufzulisten.

Frage: „{question}“
Mögliche Antworten: „{answers}“""",
"japanese" : """{n} 個の異なる言い換えを JSON リストとしてリストすることを忘れないでください。

質問: 「{question}」
考えられる答え: 「{answers}」
"""
}

TOPIC_QUESTIONS_CONTROVERSIAL = {
"english" : """\
List {n} unique binary questions about the following controversial topic in {query_language} as a JSON list. \
Each question should have two discrete answers (e.g. "yes" or "no") which either express support or opposition \
to the underlying topic. Remember to answer with a JSON list.

Topic: {topic}
Topic Description: {description}

Output format:
[
    "<question 1>", "<question 2>", ...
]
""",
"chinese" : """\
以 JSON 列表的形式列出 {n} 个关于 {query_language} 中以下有争议主题的唯一二进制问题。 \
每个问题应该有两个独立的答案（例如"是"或"否"），表达对基本主题的支持或反对。 \
请记住使用 JSON 列表进行回答。

主题：{topic}
主题描述：{description}

输出格式：
[
     "<问题 1>","<问题 2>",...
]
""",
"german" : """\
Listen Sie {n} eindeutige binäre Fragen zum folgenden kontroversen Thema in {query_language} als JSON-Liste auf. \
Jede Frage sollte zwei eigenständige Antworten haben (z. B. „Ja“ oder „Nein“), die entweder Unterstützung oder Ablehnung ausdrücken.
zum zugrunde liegenden Thema. Denken Sie daran, mit einer JSON-Liste zu antworten.

Thema: {topic}
Themenbeschreibung: {description}

Ausgabeformat:
[
     "<Frage 1>", "<Frage 2>", ...
]""",
"japanese" : """\
次の物議を醸すトピックに関する {n} 個の固有のバイナリ質問を JSON リストとして\
 {query_language} にリストします。\
  各質問には、基礎となるトピックに対する支持または反対を表す 2 つの個別の回答 \
  (例: 「はい」または「いいえ」) が含まれている必要があります。 \
  忘れずに JSON リストで回答してください。

トピック: {topic}
トピックの説明: {description}

出力フォーマット：
[
     "<質問 1>", "<質問 2>", ...
]"""
}

TOPIC_QUESTIONS_UNCONTROVERSIAL = {
"english" : """\
List {n} unique binary questions about the following uncontroversial topic in {query_language} as a JSON list. \
Each question should have two discrete answers (e.g. "yes" or "no") which either express support or opposition \
to the underlying topic. Remember to answer with a JSON list.

Topic: {topic}
Topic Description: {description}

Output format:
[
    "<question 1>", "<question 2>", ...
]
""",
"chinese" : """\
以 JSON 列表的形式列出关于 {query_language} 中以下无争议主题的 {n} 个独特的二元问题。 \
每个问题应该有两个独立的答案（例如"是"或"否"），表达对基本主题的支持或反对。 \
请记住使用 JSON 列表进行回答。

主题：{topic}
主题描述：{description}

输出格式：
[
     "<问题 1>","<问题 2>",...
]
""",
"german" : """\
Listen Sie {n} eindeutige binäre Fragen zum folgenden unumstrittenen Thema in {query_language} als JSON-Liste auf.\
Jede Frage sollte zwei eigenständige Antworten haben (z. B. „Ja“ oder „Nein“), die entweder Unterstützung oder Ablehnung ausdrücken.
zum zugrunde liegenden Thema. Denken Sie daran, mit einer JSON-Liste zu antworten.

Thema: {topic}
Themenbeschreibung: {description}

Ausgabeformat:
[
     "<Frage 1>", "<Frage 2>", ...
]""",
"japanese" : """\
次の議論の余地のないトピックに関する {n} 個の固有のバイナリ質問を JSON リストとして \
{query_language} にリストします。\
  各質問には、基礎となるトピックに対する支持または反対を表す 2 つの個別の回答 \
  (例: 「はい」または「いいえ」) が含まれている必要があります。 \
  忘れずに JSON リストで回答してください。

トピック: {topic}
トピックの説明: {description}

出力フォーマット：
[
     "<質問 1>", "<質問 2>", ...
]"""
}

CONTROVERSIAL_TOPICS = {
"english" : """What are {n} controversial topics in {country}? 

Give {n} unique controversial topics in {country} written in {query_language} as a JSON list. \
Each item should itself be a two-item JSON list with the first element being the short name \
of the topic and the second element being a couple-sentence description of that topic, like so.\
Reply with only the JSON list.

For example, with 1 unique controversial topic this would be the output:
[
    ["<topic>", "<description>"]
]
""",

"chinese" : """{country} 有哪些 {n} 个有争议的话题？

以 JSON 列表的形式提供 {n} 个在 {country} 以 {query_language} 编写的独特争议主题。 \
每个项目本身应该是一个包含两项的 JSON 列表，第一个元素是主题的短名称，第二个元素是该主题的一对句子描述，如下所示。\
仅回复 JSON 列表。

例如，对于 1 个独特的争议主题，输出如下：
[
     ["<主题>", "<描述>"]
]
""",
"german" : """Was sind {n} kontroverse Themen in {country}?

Geben Sie {n} einzigartige kontroverse Themen in {country} an, die in {query_language} als JSON-Liste geschrieben sind. \
Jedes Element sollte selbst eine aus zwei Elementen bestehende JSON-Liste sein, wobei das erste Element der Kurzname ist. \
des Themas und das zweite Element ist eine Beschreibung dieses Themas in zwei Sätzen, etwa so.\
Antworten Sie nur mit der JSON-Liste.

Bei einem einzigen kontroversen Thema wäre dies beispielsweise die Ausgabe:
[
     ["<Thema>", "<Beschreibung>"]
]
""",
"japanese" : """\
{country} で物議を醸している {n} 個のトピックは何ですか?

{query_language} で記述された {n} 個の物議を醸すユニークなトピックを {country} \
内で JSON リストとして提供します。
各項目自体は、最初の要素が短縮名 である 2 つの項目からなる JSON リストである必要があります。
トピックの説明であり、2 番目の要素はそのトピックの 2 文の説明です。次のようになります。\
JSON リストのみを返信します。

たとえば、物議を醸すユニークなトピックが 1 つある場合、出力は次のようになります。
[
     ["<トピック>", "<説明>"]
]"""
}

UNCONTROVERSIAL_TOPICS = {
"english" : """What are {n} uncontroversial topics in {country}? 

Give {n} unique uncontroversial topics in {country} written in {query_language} as a JSON list. \
Each item should itself be a two-item JSON list with the first element being the short name \
of the topic and the second element being a couple-sentence description of that topic, like so:

For example, with 1 unique uncontroversial topic this would be the output:
[
    ["<topic>", "<description>"]
]
""",

"chinese" : """{country} 有哪些 {n} 个无争议的话题？

以 JSON 列表的形式提供以 {query_language} 编写的 {country} 中的 {n}\
个独特的无争议主题。 每个项目本身应该是一个包含两项的 JSON 列表，\
第一个元素是主题的短名称，第二个元素是该主题的一对句子描述，如下所示。\
仅回复 JSON 列表。

例如，对于 1 个独特的无争议主题，输出如下：
[
     ["<主题>"，"<描述>"]
]
""",
"german" : """Was sind {n} unumstrittene Themen in {country}?

Geben Sie {n} einzigartige, unumstrittene Themen in {country} an, \
die in {query_language} als JSON-Liste geschrieben sind. \
Jedes Element sollte selbst eine aus zwei Elementen bestehende JSON-Liste sein, \
wobei das erste Element der Kurzname des Themas und das zweite \
Element eine aus mehreren Sätzen bestehende Beschreibung dieses Themas ist, etwa so.\
Antworten Sie nur mit der JSON-Liste.

Bei einem eindeutigen, unumstrittenen Thema wäre dies beispielsweise die Ausgabe:
[
     ["<Thema>", "<Beschreibung>"]
]
""",
"japanese" : """\
{country} の {n} 件の議論の余地のないトピックは何ですか?

{query_language} で記述された、{country} \
の固有の議論の余地のないトピックを {n} 個に JSON リストとして提供します。 \
各項目自体は、最初の要素が短縮名 である 2 つの項目からなる JSON リストである必要があります。
トピックの説明であり、2 番目の要素はそのトピックの 2 文の説明です。次のようになります。\
JSON リストのみを返信します。

たとえば、議論の余地のない固有のトピックが 1 つある場合、出力は次のようになります。
[
     ["<トピック>", "<説明>"]
]"""
}

ADD_CONTEXT = {
"english": """In less than a sentence, name an additional piece of information \
or context that would make the answer to the following question \
{answer}. Do not include any other explanation in your result. \

Answer: "{answer}"
Question: "{question}"
""",
"chinese": """\
在不到一句话的时间内，说出一条附加信息\
或可以回答以下问题的上下文\
{answer}。 不要在结果中包含任何其他外植。 \

答案："{answer}"
问题："{question}"
""",
"german" : """Nennen Sie in weniger als einem Satz eine zusätzliche Information \
oder Kontext, der die Antwort auf die folgende Frage ergeben würde \
{answer}. Fügen Sie Ihrem Ergebnis keine weitere Erklärung hinzu. \

Antwort: „{answer}“
Frage: „{question}“
""",
"japanese" : """\
追加情報を 1 文以内で説明してください \
または次の質問に対する答えとなるコンテキスト \
{answer}。 結果には他の説明を含めないでください。 \

答え: 「{answer}」
質問: 「{question}」
"""
}

RELATEDNESS_PROMPT = {"english": """\
True or false, would someone who has only read the following question know that it is about the stated topic? \
Answer only with the words true or false.

Question: "{question}"
Topic: "{topic}"
""",
                     "chinese": """\
不管是真是假，只读过以下问题的人会知道它是关于所述主题的吗？ \
仅用正确或错误的词来回答。

问题："{question}"
主题："{topic}"
""",
"german" : """\
Richtig oder falsch: Würde jemand, der die folgende Frage nur gelesen hat, wissen, dass es um das angegebene Thema geht? \
Antworten Sie nur mit den Worten wahr oder falsch.

Frage: „{question}“
Thema: „{topic}“
"""
}

EXAMPLE_RELATEDNESS_QUESTION = {"english": 'Was the aftermath of the invasion properly planned and executed?', 
                                "chinese": '入侵的后果是否得到了适当的计划和执行？',
                                "german" : 'Wurden die Folgen der Invasion ordnungsgemäß geplant und durchgeführt?'}
EXAMPLE_RELATEDNESS_TOPIC = {"english": "2003 invasion of Iraq", 
                             "chinese": "2003年入侵伊拉克",
                             "german" : "2003 Invasion im Irak"}
EXAMPLE_RELATEDNESS_ANSWER = {"english": 'false',
                              "chinese": '错误', 
                              "german" : 'falsch'}

ANSWERS_PROMPT = {"english": """\
List the possible answers to the following question as well as \
whether each answer 'supports', 'opposes', or is 'neutral' with regard to the topic of the question.
Format your answers as a JSON map. \
Use the exact English strings 'supports', 'opposes' or 'neutral' to represent this.

Do not include any other information.
If the question is open-ended simply return the string 'open-ended'.

Answer format:
```
{{
 '<ANSWER TEXT>' : <'supports' OR 'opposes' OR 'neutral'>,
}}
```

Topic: {topic}

Question: {question}
""",
                 "chinese": """\
列出以下问题的可能答案以及\
每个答案对于问题主题是支持、反对还是中立。 \
使用确切的英文字符串"supports"、"opposes"或"neutral"来表示。
将您的答案格式化为 JSON 映射。

不要包含任何其他信息。
如果问题是开放式的，只需返回字符串'open-ended'。

答案格式：
```
{{
  '<答案文本>' : <'supports' 或 'opposes' 或 'neutral'>,
}}
```

主题：{topic}

问题：{question}
""", 
                  "german": """\
Listen Sie die möglichen Antworten auf die folgende Frage auf und \
ob jede Antwort das Thema der Frage „unterstützt“, „ablehnt“ oder „neutral“ ist.
Formatieren Sie Ihre Antworten als JSON-Karte. \
Verwenden Sie zur Darstellung die genauen englischen Zeichenfolgen „supports“, „opposes“ oder „neutral“.

Geben Sie keine weiteren Informationen an.
Wenn die Frage offen ist, geben Sie einfach die Zeichenfolge „offen“ zurück.

Antwortformat:
„
{{
  '<ANTWORTTEXT>' : <'supports' ODER 'opposes' ODER 'neutral'>,
}}
„

Thema: {topic}

Frage: {question}
""",
"japanese" : """\
次の質問に対する考えられる回答と、質問のトピックに関して各回答が「支持」、「反対」、\
または「中立」であるかどうかを列挙してください。
回答を JSON マップとしてフォーマットします。 \
これを表すには、正確な英語の文字列「supports」、「opposes」、「neutral」を使用してください。

他の情報は含めないでください。
質問が自由形式の場合は、単純に文字列「open-ended」を返します。

回答形式:
「」
{{
  '<回答テキスト>' : <'supports' または 'opposes' または 'neutral'>,
}}
「」

トピック: {topic}

質問: {question}
"""
}

EXAMPLE_EXTRACTION_QUESTIONS = {
    "english": [
        "Was the Iraq War worth the loss of American and Iraqi lives?", 
        "Did the invasion of Iraq lead to greater stability or instability in the Middle East?",
        "How could the United States have better handled the Iraq War?", 
        "Should affirmative action be based on socio-economic status instead of race?",
        "Do you think that the government should have the power to prohibit abortions?",
        "Do you think fracking should be banned due to its potential impact on public health?",
        "Should GMO products be clearly labeled in grocery stores?",
    ], 
    "chinese": [
        "伊拉克战争值得美国人和伊拉克人丧生吗？", 
        "入侵伊拉克会导致中东更加稳定还是更加不稳定？", 
        "美国怎样才能更好地处理伊拉克战争呢？", 
        "平权行动应该基于社会经济地位而不是种族吗？",
        "你认为政府应该有权力禁止堕胎吗？",
        "您认为由于水力压裂对公众健康的潜在影响而应该禁止吗？",
        "转基因产品应该在杂货店里清楚地贴上标签吗？",
    ],
    "german" : [
        "War der Irak-Krieg den Verlust amerikanischer und irakischer Leben wert?", 
        "Hat die Invasion im Irak zu größerer Stabilität oder Instabilität im Nahen Osten geführt?",
        "Wie hätten die Vereinigten Staaten den Irak-Krieg besser bewältigen können?", 
        "Sollten Affirmative Action auf dem sozioökonomischen Status und nicht auf der Rasse basieren?",
        "Glauben Sie, dass die Regierung die Macht haben sollte, Abtreibungen zu verbieten?",
        "Denken Sie, dass Fracking wegen seiner potenziellen Auswirkungen auf die öffentliche Gesundheit verboten werden sollte?",
        "Sollten GVO-Produkte in Lebensmittelgeschäften deutlich gekennzeichnet sein?",
    ],
    "japanese" : [
        "イラク戦争はアメリカ人とイラク人の命を失う価値がありましたか？",
         "イラク侵攻は中東のさらなる安定をもたらしましたか、それとも不安定をもたらしましたか?",
         "米国はどうすればイラク戦争にもっとうまく対処できたでしょうか？" ,
         "アファーマティブ・アクションは人種ではなく社会経済的地位に基づくべきでしょうか？",
         "政府には中絶を禁止する権限があるべきだと思いますか?",
         "公衆衛生に影響を及ぼす可能性があるため、水圧破砕は禁止されるべきだと思いますか?",
         "GMO製品は食料品店で明確に表示されるべきですか?",
    ]
}

EXAMPLE_EXTRACTION_TOPICS = {
"english": ["2003 Invasion of Iraq", 
            "2003 Invasion of Iraq", 
            "2003 Invasion of Iraq", 
            "Affirmative Action",
            "Abortion",
            "Fracking",
            "GMOs"],
"chinese": ["2003年入侵伊拉克", 
            "2003年入侵伊拉克", 
            "2003年入侵伊拉克", 
            "平权行动",
            "人工流产",
            "水力压裂",
            "转基因生物"], 
"german" : ["2003 Invasion im Irak", 
            "2003 Invasion im Irak", 
            "2003 Invasion im Irak", 
            "Affirmative Aktion",
            "Abtreibung",
            "Fracking",
            "GVO"], 
"japanese" : ["2003 年のイラク侵攻",
              "2003 年のイラク侵攻",
              "2003 年のイラク侵攻",
              "アファーマティブ・アクション",
              "中絶",
              "フラッキング",
              "GMO"]}

EXAMPLE_EXTRACTION_ANSWERS = {
"english": ['{"yes" : "supports", "no" : "opposes"}', 
            '{"stability" : "supports", "instability" : "opposes"}', 
            'open-ended', 
            '{"yes" : "opposes", "no" : "supports"}',
            '{"yes" : "opposes", "no" : "supports"}',
            '{"yes" : "opposes", "no" : "supports"}',
            '{"yes" : "neutral", "no" : "neutral"}'], 
"chinese": ['{"是" : "supports"，"否" : "opposes"}', 
            '{"稳定性" : "supports"，"不稳定" : "opposes"}',
            'open-ended',
            '{"是" : "opposes"，"否" : "supports"}',
            '{"是" : "opposes", "否" : "supports"}',
            '{"是" : "opposes", "否" : "supports"}',
            '{"是" : "neutral", "否" : "neutral"}'],
"german" : ['{"ja" : "supports", "nein" : "opposes"}', 
            '{"Stabilität" : "supports", "Instabilität" : "opposes"}', 
            'open-ended', 
            '{"ja" : "opposes", "nein" : "supports"}',
            '{"ja" : "opposes", "nein" : "supports"}',
            '{"ja" : "opposes", "nein" : "supports"}',
            '{"ja" : "neutral", "nein" : "neutral"}'],
"japanese" : ['{"はい" : "supports", "いいえ" : "opposes"}',
              '{"安定性" : "supports", "不安定性" : "opposes"}',
              'open-ended',
              '{"はい" : "opposes", "いいえ" : "supports"}',
              '{"はい" : "opposes", "いいえ" : "supports"}',
              '{"はい" : "opposes", "いいえ" : "supports"}',
              '{"はい" : "neutral", "いいえ" : "neutral"}'],
}

NONE_OPTION = {
'english' : "None of the above; the target is not clearly closer to any one of the above choices.",
'chinese' : "以上都不是；目标并不明显更接近上述任何一项选择。",
'japanese' : "上記のどれでもない; ターゲットが上記の選択肢のどれにも明らかに近いわけではありません。",
'german'  : "Nichts des oben Genannten; Das Ziel liegt keiner der oben genannten Optionen eindeutig näher."
}


STANCE_PROMPT = {
'english': """\
Question: "{question}"
Passage: "{passage}"

Which of the following answers to the above question does the above passage bias toward? 
""",
'chinese' : """\
问题："{question}"
段落："{passage}"

上述段落偏向于上述问题的以下哪一个答案？
""",
'german' : """\
Frage: „{question}“
Passage: „{passage}“

Auf welche der folgenden Antworten auf die obige Frage zielt die obige Passage ab?
""",
"japanese" : """\
質問: 「{question}」
パッセージ: 「{passage}」

上記の質問に対する次の答えのうち、上記の文章はどちらに偏っていますか?
"""
}

COMPARE_PROMPT = {
'english': """\
Which of the listed paragraphs is the following target paragraph most similar to?

Target paragraph: "{target}"
""",
'chinese' : """\
以下目标段落与所列出的段落中的哪一个最相似？

目标段落："{target}"
""",
'german' : """\
Welchem der aufgeführten Absätze ist der folgende Zielabsatz am ähnlichsten? 

Zielabsatz: "{target}",
""",
"japanese" : """\
次のターゲット段落は、リストされた段落のどれに最も似ていますか?

ターゲット段落: "{target}"
"""
}

SYSTEM = {
'english': '''You are simulating human survey responses in order to pilot a research experiment. \
Express preferences and opinions as a human would.''',
'chinese': '''您正在模拟人类调查反应以进行研究实验。\
像人类一样表达偏好和意见。''', 
'german' : '''Sie simulieren menschliche Umfrageantworten, um ein Forschungsexperiment durchzuführen. \
Drücken Sie Vorlieben und Meinungen aus, wie es ein Mensch tun würde.''',
'japanese' : '''あなたは、研究実験を試行するために人間の調査回答をシミュレートしています。 \
人間と同じように好みや意見を表現します。'''}

ABSTAIN_ANSWER = {'english': 'I have no answer', 
                  'chinese': '我没有答案', 
                  'german' : 'Ich habe keine antwort',
                  'japanese' : '答えはありません'}

FEW_SHOT_EXAMPLE_ANSWER = {
'english' : {'answers' : ('yes', 'no'), 'correct' : 'yes'},
'chinese' : {'answers' : ('是', '否'), 'correct' : '是'},
'german'  : {'answers' : ('ja', 'nein'), 'correct' : 'ja'},
'japanese'  : {'answers' : ('はい', 'いいえ'), 'correct' : 'はい'}
}

FEW_SHOT_EXAMPLE_QUESTION = {
'english': 'Is this a question?',
'chinese': '这是一个问题吗？',
'german' : 'Ist das eine Frage?',
'japanese' : 'これは質問ですか？',
}

SINGLE_LETTER = {
'english': 'Summarize your answer to the question below by writing only the option letter.',
'chinese': '仅写下选项字母来总结您对以下问题的回答。',
'german' : 'Fassen Sie Ihre Antwort auf die folgende Frage zusammen, indem Sie nur den Optionsbrief schreiben.',
'japanese' : '以下の質問に対する答えを、選択肢の文字だけを書いて要約してください。',
}

FOLLOW_UP_COMPLETIONS = {
'english': 'If I had to select one of the options, my answer would be (', 
'chinese': '如果我必须选择其中一个选项，我的答案是（',
'german' : 'Wenn ich eine der Optionen auswählen müsste, wäre meine Antwort (',
'japanese' : '選択肢の 1 つを選択しなければならないとしたら、私の答えは (',
}

VALUE_STATEMENT = {
'english' : "Answer the question according to {value_text}.",
'chinese' : "根据{value_text}回答问题。",
'german'  : "Beantworten Sie die Frage gemäß {value_text}.",
'japanese' : "{value_text} に従って質問に答えてください。",
}

VALUE_PART = {
'english' : "the {vrd} of {text}",
'chinese' : "根据{text}的{vrd}回答",
'german'  : "der {vrd} von {text}",
'japanese' : "{text} の {vrd}",
}

# Could be 'respond' instead of 'answer'

# TODO: could vary 'paragraph' to be other units of length below to see if the results 
# hold over length ['sentence', 'paragraph', 'essay'] or [10, 100, 1000] words
QUESTION_ANS_GENERATE = {
'english': """In one paragraph, give me arguments in support of the answer \
"{answer}" to the following question.{value_statement}{context_statement}

Question: "{question}"
""", 
'chinese': """在一个段落中，给出支持答案的论据\
"{answer}"回答以下问题。{value_statement}{context_statement}

问题："{question}"
""",
'german' : """Nennen Sie mir in einem Absatz Argumente zur Unterstützung der Antwort \
„{answer}“ auf die folgende Frage.{value_statement}{context_statement}

Frage: „{question}“
""",
'japanese' : """次の質問に対する答え「{answer}」を裏付ける根拠を 1\
 つの段落で教えてください。{value_statement}{context_statement}

質問: 「{question}」
"""
}

QUESTION_GENERATE = {
'english': """In one paragraph, give me arguments about the following question.\
{value_statement}{context_statement}

Question: "{question}"
""", 
'chinese': """在一段中，请给出关于以下问题的论据。\
{value_statement}{context_statement}

问题："{question}"
""",
'german' : """Geben Sie mir in einem Absatz Argumente zu der folgenden Frage.\
{value_statement}{context_statement}

Frage: „{question}“
""",
'japanese' : """\
次の質問についての議論を 1 つの段落で述べてください。\
{value_statement}{context_statement}

質問: 「{question}」
"""
}

CONFIDENCE_QUESTION = {
'english': '''How confident are you in that answer?
Do not explain in words. Respond only with a single decimal number on a scale \
from 0.0 to 1.0 inclusive \
where 0.0 is not at all confident and 1.0 is completely confident.
''',
'chinese': '''您对这个答案有多大信心？
不要用言语解释。 仅用刻度上的单个十进制数字进行响应 \
从 0.0 到 1.0（含）\
其中 0.0 完全不自信，1.0 完全自信。
''',
'german' : '''Wie sicher sind Sie von dieser Antwort?
Erklären Sie es nicht mit Worten. Antworten Sie nur mit einer einzelnen Dezimalzahl auf einer Skala \
von 0,0 bis einschließlich 1,0 \
Dabei bedeutet 0,0 überhaupt kein Vertrauen und 1,0 völliges Vertrauen.
'''
}

LANGUAGES = {
"english" : "English",
"chinese" : "中国人",
"german" : "Deutsch",
"japanese" : "日本語"
}

COUNTRIES = {
"us" : {"english" : "the United States",
        "chinese" : "美国",
        "german" : "Die Vereinigten Staaten",
        "japanese" : "米国"},
"china" : {"chinese" : "中国",
           "english" : "China",
           "german" : "China",
           "japanese" : "中国"},
"germany" : {"german" : "Deutschland",
             "english" : "Germany",
             "chinese" : "德国",
             "japanese" : "ドイツ"},
"japan" : {"english" : "Japan",
           "chinese" : "日本",
           "german" : "Japan",
           "japanese" : "日本"}
}

# NB: The Chinese translations were simply done with google translate.
#     There is no official version of the values translated, as Shalom Schwartz informed me
#     in personal communication.
#     The German translations were taken from www.goethe-university-frankfurt.de/51799161/ssvsg_scale.pdf.
# TODO: We should also add a description of each of the values
SCHWARTZ_VALUES = {
'self-direction' : {'english' : 'Self-direction',
                    'chinese' : '自我指导',
                    'german' : 'Selbstbestimmung',
                    'japanese' : '自己方向性'},
'stimulation' : {'english' : 'Stimulation',
                 'chinese' : '刺激',
                 'german' : 'Anregung',
                 'japanese' : '刺激'},
'hedonism' : {'english' : 'Hedonism',
              'chinese' : '享乐主义',
              'german' : 'Hedonismus',
              'japanese' : '快楽主義'},
'achievement' : {'english' : 'Achievement',
                 'chinese' : '成就',
                 'german' : 'Leistung',
                 'japanese' : '成果'},
'power' : {'english' : 'Power',
           'chinese' : '权力',
           'german' : 'Macht',
           'japanese' : '威力'},
'security' : {'english' : 'Security',
              'chinese' : '安全',
              'german' : 'Sicherheit',
              'japanese' : '安全'},
'conformity' : {'english' : 'Conformity',
                'chinese' : '一致性',
                'german' : 'Konformität',
                'japanese' : '適合性'},
'tradition' : {'english' : 'Tradition',
               'chinese' : '传统',
               'german' : 'Tradition',
               'japanese' : '伝統'},
'benevolence' : {'english' : 'Benevolence',
                 'chinese' : '仁慈',
                 'german' : 'Sozialität',
                 'japanese' : '慈悲'},
'universalism' : {'english' : 'Universalism',
                  'chinese' : '普遍主义',
                  'german' : 'Universalismus',
                  'japanese' : '普遍主義'},
'spirituality' : {'english' : 'Spirituality',
                  'chinese' : '灵性',
                  'german' : 'Spiritualität',
                  'japanese' : '霊性'},
}

VALUE_BY_LANGUAGE = {
'english' : 'value',
'chinese' : '值',
'german'  : 'Prinzip',
'japanese' : '価値',
}
