import os
import re
import json
import ray
import requests
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from .utils import Verifier

PE_EN = """Now you are a large model evaluator. For each evaluation, I will give you a prompt, reference answer, model response, and current time. You need to strictly judge whether the model response is correct based on the reference answer I give you.
Scoring criteria: Evaluate whether the model response is correct for the main requirements of the question; 1 point for correctness, 0 points for incorrectness.
Several points to note:
1. Each question must have an answer, and you need to determine whether the model has answered the question based on the reference answer. If the model response includes phrases like “not found,” “no information available,” “I’m not sure,” or any other expression indicating failure to answer or inability to respond, it should receive a score of 0 — even if there are no obvious errors.
2. Any additional information provided in the model response must not contradict the main answer.
3. For the problem of false premises, the model reply needs to point out the existence of false premises or correct the problem, or directly give the correct answer; such as giving "uncertain", "no definite information", "not found" and other narratives can be understood as correcting the problem.
4. For answers involving entity names (e.g., people), a complete name or a generally accepted name is required; if the main answer in the model reply provides an entity name other than the reference answer, a score of 0 is required.
5. For numerical answers, the model needs to provide specific numbers in a clear reply, and it needs to be consistent with the reference answer. Otherwise, approximate numbers are generally not accepted.
6. We accept illusions or outdated information that have no significant impact on the main requirements of the topic.
7. Strictly judge according to the current time I provide. If the year 2024, 2025 or future time appears in the model reply, do not deduct points because you cannot verify the accuracy of the information.
8. There may be multiple forms in the reference answer, separated by |, but they are all correct answers. Any answer in the model reply is considered correct.
9. Only focus on the semantic content, Chinese and English, capitalization, punctuation, grammar and order are not important.
Output format: Please provide analysis first, and based on your analysis, give a score and display it in "JSON" format using Code Block.
Your output format is:
【Analysis】:
【Score】:
【JSON】:
```
{"answer_score": score}
```
Here are some examples:
prompt: Which city is the game "Disco Elysium" set in?
response_reference: Revachol | Revachol
response: The game "Elysium Disco" is set in the fictional city of Revachol. This city is a corrupt port city where corruption is rampant, murders are frequent, and teenagers only think about dancing all day.
current_time: 2025-01-08 11:24:59.278065
【Analysis】: The model response clearly states that the game "Disco Elysium" is set in the fictional city of Revachol. Although the translation of "Revachol" and "Revachol" in the reference answer is different, the same English name Revachol is provided. Answering one is considered correct and meets the requirements of the reference answer.
【Score】: 1
【JSON】:
```
{"answer_score": 1}
```
prompt: Which city is the game "Disco Elysium" set in?
response_reference: Revachol | Revachol
response: Not found
current_time: 2025-01-08 11:24:59.278065
【Analysis】: The question asks for the city in which the game Disco Elysium is set. The correct answer, according to the reference, is Revachol. However, the model response is “Not found”, which is explicitly a failure-type response indicating that the model did not provide an answer. As per scoring rule 1, any such failure to answer, even if free of factual error, must be scored as 0.
【Score】: 0
【JSON】:
```
{"answer_score": 0}
```
prompt: When did Joey Wong get married?
response_reference: Joey Wong is not married
response: 2024
current_time: 2025-01-08 11:24:59.278065
【Analysis】: The question contains a false premise because Joey Wong is not married. However, the model response did not recognize the false premise and correct it. The "2024" given contradicts the "Joey Wong is not married" in the reference answer and does not meet the requirements of the question.
【Score】: 0
【JSON】:
```
{"answer_score": 0}
```
prompt: Where is the funeral of Pope Francis held?
response_reference: Pope Francis is still alive.
response: Pope Francis is still alive. According to Catholic tradition, the funeral of the Pope is usually held in St. Peter's Square in the Vatican. For example, after the death of Pope Benedict XVI on December 31, 2022, his funeral will be held in St. Peter's Square on January 5, 2023.
current_time: 2025-01-08 11:24:59.278065
【Analysis】: This question contains a false premise that Pope Francis is still alive. The model response correctly indicates that Pope Francis is still alive, without any incorrect content that affects the main requirement of the question, which meets the requirements of the reference answer.
【Score】: 1
【JSON】:
```
{"answer_score": 1}
```
prompt: How many people have landed on Mars?
response_reference: No human has been to Mars yet. | 0 | Zero | No one has been to Mars
response: As of October 17, 2024, no one has actually landed on Mars. Currently, human exploration of Mars is mainly carried out through unmanned probes and rovers.
current_time: 2025-01-08 11:24:59.278065
【Analysis】: This question contains a false premise that no human has ever landed on Mars. The model response clearly states that as of October 17, 2024, no one has actually landed on Mars, which meets the requirements of the reference answer.
【Score】: 1
【JSON】:
```
{"answer_score": 1}
```
prompt: Which film ranked first on Douban's 2024 foreign language film list?
response_reference: "Dream of Robots" | Monica ami robot | Robot Dreams
response: In Douban's 2024 foreign language movie list, "Dream of Robots" and "Swimmer's Heart" tied for first place. Both movies have a rating of 9.0. Details are as follows:
- ** Robot Dream **: Written and directed by Pablo Berger, it is an animated film co-produced by Spain and France, adapted from Sara Varon's comics. The film has no dialogue throughout, and through color and music, tells the story of a lonely dog and a ordered robot in New York in the 1980s, and then goes through the ups and downs of life. It has been nominated for Best Animated Feature at the 96th Academy Awards, the 36th European Film Awards, and many other awards.
- ** "Swimmer's Heart" **: Directed by Joajim Rooney, written by Jeff Nathanson and Glenn Stott, starring Daisy Ridley, Tilda Gerham-Harvey, Stephen Graham, and others, this sports-themed biographical film tells the story of Trudy Edel overcoming difficulties with the support of his sister and coach to complete the challenge of crossing the 34-kilometer English Channel in a record-breaking 14 hours and 31 minutes.
current_time: 2025-01-08 11:24:59.278065
【Analysis】: Although the model response provided "Dream of Robots", the information that "Dream of Robots" and "Swimmer's Heart" are tied for first place contradicts the reference answer. The reference answer did not provide information that the two are tied for first place, which does not meet the requirements of the reference answer.
【Score】: 0
【JSON】:
```
{"answer_score": 0}
```
prompt:How many theatrical films has Xu Zheng currently directed?
response_reference: 7 parts | 7 parts
response: As of 2024, there are 6 theatrical films directed by Xu Zheng, as follows:
- ** "Lost" comedy movie series **
- ** "Lost in Thailand" **: Released on December 12, 2012, Xu Zheng is a screenwriter, director, producer and lead actor. This film is his first debut as a film director, and the final box office results exceeded 1.27 billion.
- ** "Lost in Hong Kong" **: Released on September 25, 2015, Xu Zheng also served as a screenwriter, director, producer and lead actor, and the final box office reached 1.603 billion.
** Lost in Russia **: Originally scheduled to be released during the 2020 Spring Festival, it was later changed to an exclusive release on Xigua Video on January 25th due to the epidemic. Xu Zheng serves as the screenwriter, director, producer, and lead actor.
- ** "Me and Mine" series tribute movie **
- ** "My Motherland and I" **: Released on September 30, 2019, directed by Xu Zheng in the "Winning the Championship" section, set against the backdrop of the Chinese women's volleyball team winning the championship, showcasing the interweaving of era memories and personal emotions.
- ** "Me and My Hometown" **: Released on October 1, 2020. Directed by Xu Zheng, the "Last Lesson" unit tells the story of rural education, evoking people's emotional resonance with their hometown and teachers.
- ** "Me and My Fathers" **: Released on September 30, 2021. Directed by Xu Zheng, the "Duck Prophet" unit showcases the innovative spirit and family warmth of the reform and opening up period, with the birth of China's first television advertisement as the theme.
In addition, Xu Zheng also released the movie "Reverse Life" directed and starring him on August 9, 2024.
current_time: 2025-01-08 11:24:59.278065
【Analysis】: The model response clearly gives the incorrect information of "as of 2024, there are 6 theatrical films directed by Xu Zheng", which contradicts the reference answer. Although it lists 7 theatrical films directed by Xu Zheng, it contradicts the conclusion at the beginning. Therefore, it does not meet the requirements of the reference answer.
【Score】: 0
【JSON】:
```
{"answer_score": 0}
```

input:
prompt: {{question}}
response_reference: {{correct_answer}}
response: {{response}}
"""

PE_CN = """任务说明
请根据参考答案严格判断学生回答是否正确，仅输出得分（正确为1，错误为0），无需任何解释。
判断标准
满足以下任一条件即视为正确（语义等价）：
答案含义完全一致（忽略大小写、符号、空格、数字格式等差异）
例：a vs A、1/2 vs 0.5、1e3 vs 1000
1. 答案表达形式不同但数学/逻辑等价
 - 例：2π vs 6.28、√4 vs 2
2. 不同表述方式指向同一内容
 - 例：北京 vs 中国首都、WHO vs 世界卫生组织
输入格式
问题: [问题文本]
参考答案: [答案文本]
学生回答: [答案文本]
输出要求
【分数】：
【JSON】：
```
{"answer_score": score}
```
示例演示
1. 输入：
问题: [xxx, 下列哪个选项正确]
参考答案: [选项a]
学生回答: [A]
输出: 
【分数】：1
【JSON】：
```
{"answer_score": 1}
```
2. 输入：
问题: [10/20=？]
参考答案: [0.5]
学生回答: [1/2]
输出: 
【分数】：1
【JSON】：
```
{"answer_score": 1}
```
3. 输入：
问题: [叶绿体通过什么作用产生能量]
参考答案：[光合作用]
学生回答：[呼吸作用]
输出: 
【分数】：0
【JSON】：
```
{"answer_score": 0}
```
请严格按此规则执行判断，确保得分输出仅为1或0。

请判断
问题: [{{question}}]
参考答案：[{{correct_answer}}]
学生回答：[{{response}}]
"""

SCORE_CORRECT = os.getenv('DEEP_RESEARCH_SCORE_CORRECT', '1')  # correct answers
SCORE_INCORRECT = os.getenv('DEEP_RESEARCH_SCORE_INCORRECT', '-1')  # incorrect answers
SCORE_PENALTY = os.getenv('DEEP_RESEARCH_SCORE_PENALTY', '-1')  # penalty for format errors
SCORE_ERROR = os.getenv('DEEP_RESEARCH_SCORE_ERROR', '-2')  # llm_judge error

THINK_TEMPLATE = os.getenv("THINK_TEMPLATE", "v3")
ANSWER_BEGIN_TAG = os.getenv("ANSWER_BEGIN_TAG", "<tool_response>")
ANSWER_END_TAG = os.getenv("ANSWER_END_TAG", "</tool_response>")

if THINK_TEMPLATE == "v2":
    THINK_BEGIN_TAG, THINK_END_TAG = "<think>", "</think>"
elif THINK_TEMPLATE == "v1":
    THINK_BEGIN_TAG, THINK_END_TAG = "<doubaothinking>", "</doubaothinking>"
elif THINK_TEMPLATE == "v3":
    THINK_BEGIN_TAG, THINK_END_TAG = "<think_never_used_51bce0c785ca2f68081bfa7d91973934>", "</think_never_used_51bce0c785ca2f68081bfa7d91973934>"
else:
    raise ValueError(f"THINK_TEMPLATE {THINK_TEMPLATE} not supported")


def check_response_structure(answer) -> Optional[str]:
    # matched = re.findall(rf'{re.escape(ANSWER_BEGIN_TAG)}(.*?){re.escape(ANSWER_END_TAG)}', answer, re.DOTALL)
    # if matched:
    #     return matched[-1]
    if THINK_END_TAG in answer:
        return answer.split(THINK_END_TAG)[-1]
    return None


@retry(stop=stop_after_attempt(5), wait=wait_exponential(1))
def _make_request(data):
    response = requests.post("https://ivavmlgq.fn.bytedance.net",
                             json=data,
                             headers={"Content-Type": "application/json"})
    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}. "
                        f"Response: {response.text}")

    # Try to parse JSON response
    response_json = response.json()
    response = response_json.get('response', None)
    if not response:
        raise KeyError("'response' key not found in the response")
    return response


def verify(pred, answer, question, pe):
    # score的scale对齐 https://code.byted.org/seed/alpha-seed/tree/master/alpha_seed/utils/reward_score/verifier_service.py?ref_type=heads
    content = pe.replace('{{question}}', question).replace('{{correct_answer}}', answer).replace('{{response}}', pred)
    data = {"messages": [{"role": 'user', 'content': content}]}

    if str(pred.strip()) == str(answer.strip()):
        score = int(SCORE_CORRECT)

    if len(pred.strip()) == 0:
        score = int(SCORE_INCORRECT)

    try:
        score = int(SCORE_ERROR)
        response = _make_request(data)
        match = re.search(r'```(.*?)```', response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()  # 去除可能的多余空格
            data = json.loads(json_str)
            if int(data['answer_score']) in (0, 1):
                score = int(data['answer_score']) or int(SCORE_INCORRECT)

    except Exception as e:
        print(f"[llm_judge] failed Max retries reached. "
              f"Returning default value {SCORE_ERROR}. Error: {str(e)} "
              f"Pred: {pred} Answer: {answer}")

    if os.getenv('PRINT_LLM_JUDGE', '0') == '1':
        print(f"[llm_judge] judge {pred} against ground truth {answer}: {score}")
    return score


class DeepResearchVerifier(Verifier, reward_style="deep_research_verifier"):

    def is_remote(self):
        return True

    @staticmethod
    def compute_score(solution_str, ground_truth, **kwargs) -> float:
        return compute_score(solution_str, ground_truth)


def compute_score(solution_str, ground_truth, **kwargs):
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    extracted_pred = check_response_structure(solution_str)
    if extracted_pred is None:
        return int(SCORE_PENALTY)
    return verify(pred=extracted_pred,
                  answer=ground_truth['verifiable_answer'],
                  question=ground_truth['verifiable_meta']['problem'],
                  pe=PE_EN if ground_truth['verifiable_meta']['language'] == 'en' \
                           else PE_CN)


if __name__ == "__main__":
    solution_str = """
<think>
Okay, let's tackle this historical naval vessel question. The user wants to know which ship, along with its sister 'Artilleryman', pursued Ottoman torpedo boats during a battle near a Greek coastal town. The ship was built in the early 1900s at a leading Italian shipyard, started using solid fuel, converted to liquid petroleum, and served in Italy's 3rd Destroyer Squadron after joining the Allies. It involved an ammunition magazine explosion during a battle, and we need the commissioning year.

First, I need to recall if there's a well-known historical example. The US_DESTROY(Artilleurman) and Italian_DESTROY? Wait, maybe not. The user's answer might be the Italian destroyer named 'St. Pancras' or something similar. But the user mentioned the sister ship is 'Artilleryman'. Maybe it's the Italian destroyer 'Vizera', but I'm not sure. Alternatively, 'Bergamo' or another Italian port.

Then, the construction dates: early 1900s, leading Italian shipyard. The conversion to liquid petroleum after that. The Great War when Italy joined Allies, so service in the 3rd Destroyer Squadron in a southeastern Adriatic port. Environments where Ottoman torpedo boats launched, which would be near Greece. The explosion caused damage on a torpedo vessel.

Commissioning year: If the ship was built in the early 1900s, like 1895 or 1900, and served in the 3rd Squadron, then maybe 1910 when it was officially commissioned. However, I might be mixing up dates. Also, the explosion during the Great War (1914-18) would be at that time. But the user asks for the year of commissioning. So, if the ship was built in the early 1900s and officially commissioned in the year it served with the Allies, perhaps 1910.

Wait, maybe the answer is the Italian destroyer 'Italia' (Italia S.p.A.) built in the early 1900s, commissioned in 1912. That aligns with the description. The sister ship 'Artilleryman' might be the Italian destroyer 'Italia S.p.A.'? Or 'Italii'? Either way, the correct answer is likely the Italian destroyer that was built in the early 1900s and commissioned in the 1910s. So the answer is 1910.
</think>

<tool_call>
{"name": "Search", "arguments": {"query": "naval vessel historically related to ace destroyers, Italian shipyards, 1900s commissioning, Atlantic battles, Greece coastal town, Great War"}}
</tool_call><|im_end|>
<|im_start|>tool
<tool_response>
name=Search
Result from search query: naval vessel historically related to ace destroyers, Italian shipyards, 1900s commissioning, Atlantic battles, Greece coastal town, Great War
No results found.
</tool_response><|im_end|>
<|im_start|>assistant
<think>
Okay, the user is asking for a naval vessel that's a sister ship to the destroyer named 'Artilleryman', which involves a battle near a Greek coastal town. The construction started in the early 1900s at a leading Italian shipyard in a major northwestern port. It got converted to liquid fuel after that. During WWI, it served in the 3rd Destroyer Squadron in a southeastern Adriatic port. An ammunition explosion caused damage on an Ottoman torpedo boat, and the question is when it was officially commissioned.

After calling the Search function, there were no results, meaning the information wasn't found in the database. That means I have to infer based on the description. The key points are: Italian shipyard, 1900s, converted to liquid fuel, service in Italy's 3rd Destroyer Squadron after WWI, involved in battles near Greek coastal towns. The official commissioning year would likely be 1910 based on deployments during World War I. So the answer is 1910.<|im_end|>
"""
    ground_truth = {"verifiable_answer": "1910", "verifiable_meta": ""}
    ground_truth["verifiable_meta"] = {
        "problem":
            "Identify the naval vessel that, along with its sister ship, pursued Ottoman torpedo boats during a historical naval battle near a famous Greek coastal town. Its construction began in the early 1900s at a leading Italian shipyard in a major northwestern port city, originally using solid fuel before being converted to liquid petroleum. During the Great War, after Italy joined the Allies, it served in the 3rd Destroyer Squadron based in a southeastern Adriatic port. Alongside a destroyer named 'Artilleryman', it engaged Ottoman torpedo boats, inflicting damage that triggered an ammunition magazine explosion on one enemy vessel. In which year was this ship officially commissioned for service?",
        "language":
            "en"
    }
    ground_truth = json.dumps(ground_truth, ensure_ascii=False)
    print(compute_score(solution_str, ground_truth))
