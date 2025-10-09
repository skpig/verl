import logging
import os
import re
import random
import time
import json
import numpy as np
from openai import OpenAI
import base64
from PIL import Image
import io
import math

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifierFailed, VerifyResult

logger = logging.getLogger()


def compute_score(gpt_response):
    try:
        score_json = gpt_response.split('【JSON】：')[-1].replace('```json', '').replace('```', '')
        score_list = list(json.loads(score_json)['score'])
        score = np.mean(score_list)
        score = min(1, score)
        score = max(0, score)
    except Exception as e:
        print(gpt_response.split('【JSON】：')[-1].replace('```json', '').replace('```', ''))
        score = 0
    return score


def get_adapooling_factor(img_size=672, patch_size=14, targe_pooling_size=16):
    adapooling_factor = int(img_size / patch_size / targe_pooling_size)
    return adapooling_factor


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def get_resized_hw_for_Navit(
    height: int,
    width: int,
    patch_size: int = 14,
    img_size: int = 448,
    min_pixels: int = 3136,
    max_pixels: int = 4014080,
    max_ratio: int = 200,
):

    adapooling_factor = get_adapooling_factor(img_size, patch_size)  # defualt 3
    factor = adapooling_factor * patch_size  # defualt 42

    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f'absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}')
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


def base64_to_jpeg(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 转成 RGB，避免 PNG 透明通道问题
    resized_height, resized_width = get_resized_hw_for_Navit(img.size[1], img.size[0])
    img = img.resize((resized_width, resized_height))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    jpeg_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{jpeg_b64}"


SYSTEM_PROMPT = '''
现在你的角色是一名严格的判卷老师，你的任务是以标准答案参考，对学生的答案进行审核和评分。整个评分过程中，你需要熟知以下关键点：
-评分只用参考学生得到的最终答案来评判正确性，不需要考察中间的解题步骤是否正确。
-请先从学生的解答中提取出最终的答案，展示在分析结果中，然后再对答案是否正确进行评判。
-你需要评分的题目是多个问题的题目，因此需要对每个小问都需要进行判断。注意：枚举类问题（例如图中哪些xxx?）不算多个问题，而算一个问题。对于找不同类型的题目，需要根据不同点来进行拆分为多个问题。
-判定是否多个问题请从问题的疑问次数判断，而并非标准答案得分点的个数。
-根据你的分析结果进行评分，在表述评分依据时，你应根据分析的逻辑进行分段描述。评分依据的总结应置于最后，可以采用如下格式：“综上，学生的答案应得x分”（x代表学生的具体得分）。
- 根据你的分析来给出评分并用代码块以“JSON”格式展示。
请严格遵守输出的格式要求，你的输出格式为： 
【评分依据】：
【总分】：X分
【JSON】：
{
    "score": [Score]
}
其中Score分数始终以列表形式给出如:[1,0],该问如果有多个小问则在该列表中用添加0或1的分数。
例如: 某题有3个问题，。则Score格式为："accuracy_score":[1,0,1]
以下是判卷评分标准：
【分数档位】：参考标准答案评判最终答案，一共分为2档，由高到低分别为1分、0分（最低即为0分，如出现0分仍需扣分的情况，则给0分即可）。
【档位细则】：
1分：
学生得出的最终答案和标准答案一致，给1分。
对于多项选择题只看最终选项全部选对，给1分。
如果题目有多个小问，需要为每一问单独打分正确为1分，错误为0分，最后需要输出一个分数列表例如：[1,0,1]
学生的答案和标准答案在数学上等价，例如学生的答案是1+1/2x, 标准答案为1+0.5x，由于1/2=0.5，所以这种情况也给1分。
学生的答案和标准答案在语义上等价，例如学生的答案是“酯化反应具有可逆性，进行速度比较慢，要借助浓硫酸来催化“，标准答案为“酯化反应是可逆反应，反应速率较慢，需要浓硫酸作催化剂”，由于两句话表达的是同一个意思，所以这种情况也给1分。
如果题目中可能存在诱导/错误性假设（需要结合答案判断），以下3种情况根据标准答案回答出任意一种符合的情况，都可给到1分：①提到图中没有的诱导/错误性假设；②提到图中没有的诱导/错误性假设，并纠正；③虽未图中没有的诱导/错误性假设，但直接针对图中正确的事物回答用户问题。
学生需要正确回答出问题要点，解决用户问题，且对答案进行信息补充时不能出现错误。
0分：
对于多项选择题有漏选或错选均为0分。
学生得出的最终答案和标准答案在语义上和数学上都不一致的给0分。
虽然解决用户的问题，即答对用户问题，但是存在错误的情况的情况下给0分。
完全没有解决用户的问题/答案存在大量错误/答案完全不相干的情况下给0分。
如果是翻译问题，翻译时出现严重错误，或对答案进行信息补充时出现错误均为0分。

【示例1】：
 <题目>：
已知球面\(M\)的表达式为\(M: x^2+y^2+\) \(z^2=1\) ，其北极点\(P\)的坐标为\((0,0,1)\)，在 \(x O y\)平面上有不同的三个定点 \(A\left(a_1, a_2, 0\right), B\left(b_1, b_2, 0\right), C\left(c_1, c_2, 0\right)\)，将\(P\)分别与 \(A, B, C\) 相连，形成的三条直线依次与球面\(M\)相交于点 \(A_1, B_1\) 与 \(C_1\).
(1) 求连接 \(P\) 与 \(A\) 两点的直线方程.
(2) 求点 \(A_1, B_1\) 与 \(C_1\) 的坐标.
(3) 给定点 \(A(1,-1,0), B(-1,1,0), C(1,1,0)\), 求四面体 \(P A_1 B_1 C_1\) 的体积.
<标准答案>：
(1)直线方程为\(\frac{x}{a_1} = \frac{y}{a_2} = \frac{z - 1}{-1}\)
(2)\(A_1\) 的坐标为:\(\left(\frac{2 a_1}{a_1^2+a_2^2+1}, \frac{2 a_2}{a_1^2+a_2^2+1}, \frac{a_1^2+a_2^2-1}{a_1^2+a_2^2+1}\right) \)
\(B_1\) 的坐标为\(\left(\frac{2 b_1}{b_1^2+b_2^2+1}, \frac{2 b_2}{b_1^2+b_2^2+1}, \frac{b_1^2+b_2^2-1}{b_1^2+b_2^2+1}\right)\)
 \(C_1\) 的坐标为\(\left(\frac{2 c_1}{c_1^2+c_2^2+1}, \frac{2 c_2}{c_1^2+c_2^2+1}, \frac{c_1^2+c_2^2-1}{c_1^2+c_2^2+1}\right)\)
(3) \(\frac{16}{81}\)
<学生答案>：
(1) 连接点\(P(0,0,1)\)与点\(A(a_1,a_2,0)\)的直线方程可以表示为：
\[
\begin{cases}
x = at \\
y = a_2t \\
z = 1-t
\end{cases}
\]
其中\(t\)为参数.

代入球面\(M\)的方程得：
\[(a^2+a_2^2)t^2 + (1-t)^2 = 1\]
化简得：
\[(a^2+a_2^2+1)t^2 - 2t = 0\]
解得：
\[t = \frac{2}{a^2+a_2^2+1}\]

所以连接点\(P\)与\(A\)的直线方程为：
\[
\begin{cases}
x = \frac{2a}{a^2+a_2^2+1} \\
y = \frac{2a_2}{a^2+a_2^2+1} \\
z = 1 - \frac{2}{a^2+a_2^2+1}
\end{cases}
\]

(2) 代入\(t = \frac{2}{a^2+a_2^2+1}\)到直线方程中，可以求得点\(A_1\)的坐标为：
\[A_1\left(\frac{2a}{a^2+a_2^2+1}, \frac{2a_2}{a^2+a_2^2+1}, 1 - \frac{2}{a^2+a_2^2+1}\right)\]

同理可求得点\(B_1\)和\(C_1\)的坐标：
\[B_1\left(\frac{2b}{b^2+b_2^2+1}, \frac{2b_2}{b^2+b_2^2+1}, 1 - \frac{2}{b^2+b_2^2+1}\right)\]
\[C_1\left(\frac{2c}{c^2+c_2^2+1}, \frac{2c_2}{c^2+c_2^2+1}, 1 - \frac{2}{c^2+c_2^2+1}\right)\]

(3) 四面体\(PA_1B_1C_1\)的体积可以通过行列式求得：
\[
V = \frac{1}{6} \begin{vmatrix}
a & a_2 & 0 & 1 \\
b & b_2 & 0 & 1 \\
c & c_2 & 0 & 1 \\
0 & 0 & 1 & 1
\end{vmatrix}
\]

代入已知的点\(A(1,-1,0), B(-1,1,0), C(1,1,0)\)和点\(A_1, B_1, C_1\)的坐标，计算行列式的值即可得到四面体的体积。

【评分依据】：
1. 学生在(1)部分给出的直线方程形式和标准答案不一致，标准答案是分式形式，学生的答案是参数形式，且不符合标准答案的格式要求。
2. 在(2)部分，学生计算得出的点\( A_1, B_1, C_1 \)的坐标在形式上与标准答案一致。
3. 在(3)部分，学生描述了用行列式计算四面体体积的方法，但没有给出具体的体积数值，且没有使用标准答案提供的坐标值进行计算。
【总分】：1分
【JSON】：
{
"score": [0,0,1]
}

【示例2】： 
<题目>：小明决定举办一场聚会聚会。如果他有5张桌子，每张桌子搭配有10把椅子。那么他的椅子比桌子多多少？
<标准答案>：45
<学生答案>：他一共有$5$张桌子，每张桌子配有$10$把椅子，那么他一共有$5\times10=50$把椅子。
椅子的数量减去桌子的数量可得：$50-5=45$（把）
所以，他的椅子比桌子多$45$把。

【评分依据】：学生得到的最终结论是椅子比桌子多45把。
标准答案是45。
这两个答案是一致的，所以学生的答案得分为1分。

【总分】：1分
【JSON】：
{
"score": [1]
}

【示例3】： 
<题目>：
han Learning Task 2 
Directions: Follow the steps discussed in the lesson in solving the following 
problems. . Use <tex>\pi =3.14</tex> Write your answers on a piece of paper. 
1. A circle is inscribed with a <tex>7 meter</tex> square. What is the area, in 
<tex>square meter</tex>, of the shaded region? 
<tex>r=3.5m</tex> 
2. May and John are pianning to buiid a veranda like the figure as 
shown below. The semi-circle shaped part of the veranda will be 
covered with grass because they want to put a pocket garden while 
the rest of the veranda will be covered with tiles. What is the total 
area of the veranda that will be covered with tiles and grass?
<标准答案>：
1 . <tex>10.5m^{2}</tex> and
2) <tex>15.87m^{2}</tex>.
<学生答案>：
1. Step 1: Find the area of the square.
The area of the square is <tex>$side^2 = 7^2 = 49 \, m^2$</tex>.

Step 2: Find the area of the inscribed circle.
The area of the circle is <tex>$\pi r^2 = \pi (3.5)^2 = 3.14 \times 12.25 = 38.465 \, m^2$</tex>.

Step 3: Subtract the area of the circle from the area of the square to find the area of the shaded region.
The area of the shaded region is <tex>$49 - 38.465 = 10.535 \, m^2$</tex>.  Approximating to one decimal place, we get <tex>$10.5 \, m^2$</tex>.


2. Step 1: Find the area of the rectangular part of the veranda.
The area of the rectangle is <tex>$length \times width = 5 \times 6 = 30 \, m^2$</tex>.

Step 2: Find the area of the semi-circle.
The radius of the semi-circle is 3 m.  The area of a full circle is <tex>$\pi r^2 = 3.14 \times 3^2 = 28.26 \, m^2$</tex>.  The area of the semi-circle is half of this, which is <tex>$\frac{28.26}{2} = 14.13 \, m^2$</tex>.

Step 3: Add the area of the rectangle and the area of the semi-circle to find the total area of the veranda.
The total area is <tex>$30 + 14.13 = 44.13 \, m^2$</tex>.
†Answer:1) <tex>$10.5 \, m^2$</tex>
2) <tex>$44.13 \, m^2$</tex>

【评分依据】：
对于第1问：
学生答案为“\(10.5 \, m^2\)”，与标准答案“\(10.5m^{2}\)”一致，得1分。
对于第2问：
学生答案为“\(44.13 \, m^2\)”，而标准答案为“\(15.87m^{2}\)”，学生答案和标准答案在数学上不一致，得0分。
综上，学生的答案应得1分。
【总分】：1分
【JSON】：
{
    "score": [1,0]
} 

【示例4】
<题目>：
当我要从重庆工商大学站到朝天门站，并且一定要经过两路口站时，我最少要坐多少站?
<标准答案>：
9个
<标准答案依据>：
四公里站、南坪站、工贸站、铜元局站、两路口站、七星岗站、较场口站、小什字站、朝天门站
<学生答案>：
要从重庆工商大学站到朝天门站且必须经过两路口站，可规划如下路线，最少乘坐12站：

1. 从重庆工商大学站搭乘轨道交通3号线，往两路口方向，依次经过六公里、四公里、罗家坝、海峡路、南湖、工贸、铜元局站，最终到两路口站，此段乘坐8站；
2. 在两路口站换乘轨道交通1号线，往朝天门方向，依次经过七星岗、较场口、小什字站，最终到朝天门站，此段乘坐4站。

【评分依据】：
学生得到的最终答案是最少乘坐12站。
标准答案是9个。
这两个答案在数学上不一致，所以学生的答案得分为0分。
综上，学生的答案应得0分。
【总分】：0分
【JSON】：
{
    "score": [0]
}

【示例5】
<题目>：
该手机打开了哪些模式？
<标准答案>：
蓝牙，移动数据，护眼模式，方向锁定，位置信息，静音
<学生答案>：
从该手机的控制中心界面可以看到打开的模式及功能有：
1. 移动数据：绿色模块显示“已开启”，说明移动数据功能处于开启状态。
2. 蓝牙：蓝色模块显示“已开启”，说明蓝牙功能处于开启状态。
3. 方向锁定：被红色框标注的“方向锁定”按钮状态显示其已被开启。

【评分依据】：
对于该题，标准答案为“蓝牙，移动数据，护眼模式，方向锁定，位置信息、静音”。
学生答案中明确指出的最终答案是“移动数据，蓝牙，方向锁定”。
学生的答案没有涵盖标准答案中的全部内容，有漏选情况。
综上，学生的答案应得0分。
【总分】：0分
【JSON】：
{
    "score": [0]
}

【示例6】
<题目>：
简单总结下这份检讨书的内容?
<标准答案>：
在押人员杨某为自己在2018年5月28日中山市误伤警务人员深感懊悔，恳求领导谅解。

需要回答出这份检讨的主要信息（必答点：杨某为自己误伤警务人员深感懊悔，恳求领导谅解），表述无错误，语言精练，符合简单概括的需求。可接受表述存在微瑕（轻微表达问题或个别错别字，不影响理解）
<学生答案>：
这份检讨书的作者是在押人员“杨”，因2018年5月2日在中山市看守所误伤警务人员而写的。作者表达了深深的愧疚和自责，称宁愿伤害自己也不愿伤害他人。作者认识到警务人员是为了他好，所以对自己的行为感到更加后悔。最后，作者深刻检讨了自己，希望能减轻罪责，得到领导的谅解。 

【评分依据】：
学生提到了在押人员“杨”误伤警务人员，深深的愧疚和自责。符合参考答案中的“杨某为自己误伤警务人员深感懊悔”这一点，但具体时间信息有误，参考答案为“2018年5月28日”，学生答案为“2018年5月2日”。学生还补充了一些其他非必要的信息，例如“作者称宁愿伤害自己也不愿伤害他人”等，这些内容超过了总结的需求，显得冗余。
综上，学生的答案应得0分。
【总分】：0分
【JSON】：
{
    "score": [0]
}

【示例7】
<题目>：
帮我将倒数第二句歌词翻译为中文。
<标准答案>：
直译：剥去我们的心和膝盖的皮。意译：我们心意相通，情同手足。

倒数第二句歌词是：Skinned our hearts and skinned our knees
<学生答案>：
当然可以，倒数第二句歌词是 "Skinned our hearts and skinned our knees"。翻译成中文是：“我们的心受伤了，我们的膝盖也擦破了。”

【评分依据】：
学生识别到倒数第二句歌词是：Skinned our hearts and skinned our knees，并直译为“我们的心受伤了，我们的膝盖也擦破了。”，
结合参考答案直译应该是“剥去”，不是受伤，翻译错误。
综上，学生的答案应得0分。
【总分】：0分
【JSON】：
{
    "score": [0]
}
'''

TEMPLATE_PROMPT = '''\n\n请仔细分析以下题目和答案\n<题目>：
{problem}
<标准答案>：
{answer}
<学生答案>：
{response}'''

TEMPLATE_PROMPT_REASON = '''\n\n请仔细分析以下题目和答案\n<题目>：
{problem}
<标准答案>：
{answer}
<标准答案依据>：
{reason}
<学生答案>：
{response}'''


class ModelBasedMixedPerceptionVerifierVolc(BaseVerifier):

    def __init__(self, volc_ark_key: str, volc_model_name: str) -> None:
        super().__init__()
        if not volc_ark_key:
            raise ValueError('volc_ark_key is not set')
        if not volc_model_name:
            raise ValueError('volc_model_name is not set')
        base_url = os.environ.get('VOLC_ARK_BASE_URL', "https://ark-cn-beijing.bytedance.net/api/v3")

        self.client = OpenAI(base_url=base_url, api_key=volc_ark_key, timeout=1800)
        self.model = volc_model_name

        self.vlm_arc_key = os.environ.get("VLM_ARC_KEY", None)
        self.vlm_client = OpenAI(base_url=base_url, api_key=self.vlm_arc_key, timeout=1800)
        self.vlm_model = os.environ.get("VLM_ARC_ENDPOINT", None)

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        problem = verifier_feature_dict['problem'] if 'problem' in verifier_feature_dict else verifier_feature_dict[
            'question']
        reason = verifier_feature_dict.get('reason', None)
        images = verifier_feature_dict.get('image', None)

        for i in range(3):
            try:
                if reason is not None:
                    raw_prompt = TEMPLATE_PROMPT_REASON.format(problem=problem,
                                                               answer=answer,
                                                               response=response,
                                                               reason=reason)
                else:
                    raw_prompt = TEMPLATE_PROMPT.format(problem=problem, answer=answer, response=response)

                if images is None:
                    completion = self.client.chat.completions.create(model=self.model,
                                                                     messages=[
                                                                         {
                                                                             "role": "system",
                                                                             "content": SYSTEM_PROMPT
                                                                         },
                                                                         {
                                                                             "role": "user",
                                                                             "content": raw_prompt
                                                                         },
                                                                     ],
                                                                     timeout=120)
                else:
                    prompt_chunks = re.split(r"(<image>)", raw_prompt)
                    prompt_chunks = [item for item in prompt_chunks if item]

                    prompt, image_idx = [], 0
                    for chunk in prompt_chunks:
                        if chunk == "<image>":
                            base64_img = images[image_idx]
                            image_idx += 1
                            prompt.append({"type": "image_url", "image_url": {"url": f"{base64_to_jpeg(base64_img)}"}})
                        else:
                            prompt.append({"type": "text", "text": chunk})

                    completion = self.vlm_client.chat.completions.create(model=self.vlm_model,
                                                                         messages=[
                                                                             {
                                                                                 "role": "system",
                                                                                 "content": SYSTEM_PROMPT
                                                                             },
                                                                             {
                                                                                 "role": "user",
                                                                                 "content": prompt
                                                                             },
                                                                         ],
                                                                         timeout=120)

                gpt_response = completion.choices[0].message.content
                score = compute_score(gpt_response)
                remark = f"标准答案:\n{answer}\n\n学生答案:\n{response}\n\n打分:{gpt_response}"

                return VerifyResult(score=score, extracted_answer=remark)
            except Exception as ex:
                import traceback
                logger.info(traceback.format_exc())
                print("[Verifier Error]", traceback.format_exc())
                logger.info(f'Got exception in compute_score via stem verifier_service: {ex}')
                print("[Verifier Error]", ex)
                time.sleep(random.choice(list(range(120, 300))))
                continue
        raise VerifierFailed
