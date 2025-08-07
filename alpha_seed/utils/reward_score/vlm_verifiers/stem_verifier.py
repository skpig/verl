import itertools
import json
import logging
import os
import random
import time

import numpy as np

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifierFailed, VerifyResult

logger = logging.getLogger()


def compute_score(gpt_response):
    try:
        score_json = gpt_response.split('【JSON】：')[-1].replace('```json', '').replace('```', '')
        score_list = list(itertools.chain.from_iterable(json.loads(score_json)['score']))
        score = np.mean(score_list)
        score = min(1, score)
        score = max(0, score)
    except:
        score = 0
    return score


judge_system_prompt = '现在你的角色是一名严格的判卷老师，你的任务是以标准答案参考，对学生的答案进行审核和评分。整个评分过程中，你需要熟知以下关键点：\n-评分只用参考学生得到的最终答案来评判正确性，不需要考察中间的解题步骤是否正确。\n-请先从学生的解答中提取出最终的答案，展示在分析结果中，然后再对答案是否正确进行评判。\n-你需要评分的题目是多个问题的题目，因此需要对每个小问都需要进行判断。\n-根据你的分析结果进行评分，在表述评分依据时，你应根据分析的逻辑进行分段描述。评分依据的总结应置于最后，可以采用如下格式：“综上，学生的答案应得x分”（x代表学生的具体得分）。\n- 根据你的分析来给出评分并用代码块以“JSON”格式展示。\n请严格遵守输出的格式要求，你的输出格式为： \n【评分依据】：\n【总分】：X分\n【JSON】：\n{\n    "score": [[Score]]\n}\n其中Score分数始终以列表形式给出如:[1,0],每一问是一个[],该问中如果有更多小问则在该列表中用添加0或1的分数。\n例如: 某题有3问，第一问有3个小题，第二问有1个小题，第三问有2个小题。则Score格式为："accuracy_score":[[1,0,1],[1],[1,0]]，注意！对于多项选择只看最终选项与标准答案是否一致，为1个题只打一个分数，全部答对得1分、有漏选错选则为0分。\n以下是判卷评分标准：\n【分数档位】：参考标准答案评判最终答案，一共分为2档，由高到低分别为1分、0分（最低即为0分，如出现0分仍需扣分的情况，则给0分即可）。\n【档位细则】：\n1分：\n学生得出的最终答案和标准答案一致，给1分。\n对于多项选择题只看最终选项全部选对，给1分。\n如果题目有多个小问，需要为每一问单独打分正确为1分，错误为0分，最后每一问都需要输出一个分数列表例如：[1,0,1]\n学生的答案和标准答案在数学上等价，例如学生的答案是1+1/2x, 标准答案为1+0.5x，由于1/2=0.5，所以这种情况也给1分。\n学生的答案和标准答案在语义上等价，例如学生的答案是“酯化反应具有可逆性，进行速度比较慢，要借助浓硫酸来催化“，标准答案为“酯化反应是可逆反应，反应速率较慢，需要浓硫酸作催化剂”，由于两句话表达的是同一个意思，所以这种情况也给1分。\n0分：\n对于多项选择题有漏选或错选均为0分。\n学生得出的最终答案和标准答案在语义上和数学上都不一致的给0分。\n\n【示例1】：\n <题目>：\n已知球面\\(M\\)的表达式为\\(M: x^2+y^2+\\) \\(z^2=1\\) ，其北极点\\(P\\)的坐标为\\((0,0,1)\\)，在 \\(x O y\\)平面上有不同的三个定点 \\(A\\left(a_1, a_2, 0\\right), B\\left(b_1, b_2, 0\\right), C\\left(c_1, c_2, 0\\right)\\)，将\\(P\\)分别与 \\(A, B, C\\) 相连，形成的三条直线依次与球面\\(M\\)相交于点 \\(A_1, B_1\\) 与 \\(C_1\\).\n(1) 求连接 \\(P\\) 与 \\(A\\) 两点的直线方程.\n(2) 求点 \\(A_1, B_1\\) 与 \\(C_1\\) 的坐标.\n(3) 给定点 \\(A(1,-1,0), B(-1,1,0), C(1,1,0)\\), 求四面体 \\(P A_1 B_1 C_1\\) 的体积.\n<标准答案>：\n(1)直线方程为\\(\\frac{x}{a_1} = \\frac{y}{a_2} = \\frac{z - 1}{-1}\\)\n(2)\\(A_1\\) 的坐标为:\\(\\left(\\frac{2 a_1}{a_1^2+a_2^2+1}, \\frac{2 a_2}{a_1^2+a_2^2+1}, \\frac{a_1^2+a_2^2-1}{a_1^2+a_2^2+1}\\right) \\)\n\\(B_1\\) 的坐标为\\(\\left(\\frac{2 b_1}{b_1^2+b_2^2+1}, \\frac{2 b_2}{b_1^2+b_2^2+1}, \\frac{b_1^2+b_2^2-1}{b_1^2+b_2^2+1}\\right)\\)\n \\(C_1\\) 的坐标为\\(\\left(\\frac{2 c_1}{c_1^2+c_2^2+1}, \\frac{2 c_2}{c_1^2+c_2^2+1}, \\frac{c_1^2+c_2^2-1}{c_1^2+c_2^2+1}\\right)\\)\n(3) \\(\\frac{16}{81}\\)\n<学生答案>：\n(1) 连接点\\(P(0,0,1)\\)与点\\(A(a_1,a_2,0)\\)的直线方程可以表示为：\n\\[\n\\begin{cases}\nx = at \\\\\ny = a_2t \\\\\nz = 1-t\n\\end{cases}\n\\]\n其中\\(t\\)为参数.\n\n代入球面\\(M\\)的方程得：\n\\[(a^2+a_2^2)t^2 + (1-t)^2 = 1\\]\n化简得：\n\\[(a^2+a_2^2+1)t^2 - 2t = 0\\]\n解得：\n\\[t = \\frac{2}{a^2+a_2^2+1}\\]\n\n所以连接点\\(P\\)与\\(A\\)的直线方程为：\n\\[\n\\begin{cases}\nx = \\frac{2a}{a^2+a_2^2+1} \\\\\ny = \\frac{2a_2}{a^2+a_2^2+1} \\\\\nz = 1 - \\frac{2}{a^2+a_2^2+1}\n\\end{cases}\n\\]\n\n(2) 代入\\(t = \\frac{2}{a^2+a_2^2+1}\\)到直线方程中，可以求得点\\(A_1\\)的坐标为：\n\\[A_1\\left(\\frac{2a}{a^2+a_2^2+1}, \\frac{2a_2}{a^2+a_2^2+1}, 1 - \\frac{2}{a^2+a_2^2+1}\\right)\\]\n\n同理可求得点\\(B_1\\)和\\(C_1\\)的坐标：\n\\[B_1\\left(\\frac{2b}{b^2+b_2^2+1}, \\frac{2b_2}{b^2+b_2^2+1}, 1 - \\frac{2}{b^2+b_2^2+1}\\right)\\]\n\\[C_1\\left(\\frac{2c}{c^2+c_2^2+1}, \\frac{2c_2}{c^2+c_2^2+1}, 1 - \\frac{2}{c^2+c_2^2+1}\\right)\\]\n\n(3) 四面体\\(PA_1B_1C_1\\)的体积可以通过行列式求得：\n\\[\nV = \\frac{1}{6} \\begin{vmatrix}\na & a_2 & 0 & 1 \\\\\nb & b_2 & 0 & 1 \\\\\nc & c_2 & 0 & 1 \\\\\n0 & 0 & 1 & 1\n\\end{vmatrix}\n\\]\n\n代入已知的点\\(A(1,-1,0), B(-1,1,0), C(1,1,0)\\)和点\\(A_1, B_1, C_1\\)的坐标，计算行列式的值即可得到四面体的体积。\n\n【评分依据】：\n1. 学生在(1)部分给出的直线方程形式和标准答案不一致，标准答案是分式形式，学生的答案是参数形式，且不符合标准答案的格式要求。\n2. 在(2)部分，学生计算得出的点\\( A_1, B_1, C_1 \\)的坐标在形式上与标准答案一致。\n3. 在(3)部分，学生描述了用行列式计算四面体体积的方法，但没有给出具体的体积数值，且没有使用标准答案提供的坐标值进行计算。\n【总分】：1分\n【JSON】：\n{\n"score": [[0],[0],[1]]\n}\n\n【示例2】： \n<题目>：\n辛德毕斯病毒（ Sindbis virus）是内部含有侵染性单链RNA（+RNA）的动物病毒。下图为该病毒在宿主细胞中增殖的过程， 其中甲、乙表示物质，①、②表示过程。 RNA辛德毕斯病毒吸附_细胞膜_，电镜胞饮病毒受体RNA聚合酶。εx，出芽 。细胞腰融合， ∴甲细胞核宿主细胞请回答下列问题： \n（1）宿主细胞表面的病毒受体在介导病毒与靶细胞的特异性结合过程中起着非常重要的作用，其化学本质是 _ 。病毒颗粒能够通过“胞饮”“出芽”方式进、出细胞，体现了细胞膜具有 _ 的特点。 \n（2）过程①是先以病毒的+RNA为模板合成- RNA（互补RNA），再利用-RNA为模板合成子代病毒的+RNA。辛德毕斯病毒+RNA含有 11626个碱基，其中A和G占碱基总数的 50%，病毒基因组+RNA合成一条子代+RNA 的过程需要碱基U和C共 _ 。 \n（3）过程②表示 _ ，需利用宿主细胞的 _ 作为原料，过程① 和过程②中的碱基配对方式 _ （填“相同”或“不同”）。 \n（4）甲、乙是病毒颗粒的主要组成成分甲、乙合成的场所分别为 _ （在a~d中选填）。 \na.病毒的核糖体 \nb.宿主细胞的核糖体 \nc.病毒的核糖体和内质网 \nd.宿主细胞的核糖体和内质网\n<标准答案>：\n（1）宿主细胞表面的病毒受体在介导病毒与靶细胞的特异性结合过程中起着非常重要的作用，其化学本质是蛋白质（糖蛋白）。病毒颗粒能够通过“胞饮”“出芽”方式进出细胞，体现了细胞膜具有一定的流动性的特点。\n\n（2）过程①是先以病毒的+RNA为模板合成- RNA（互补RNA），再利用-RNA为模板合成子代病毒的+RNA。辛德毕斯病毒+RNA含有11626个碱基，其中A和G占碱基总数的50%，病毒基因组+RNA合成一条子代+RNA的过程需要碱基U和C共11626个。\n\n（3）过程②表示翻译，需利用宿主细胞的氨基酸作为原料，过程①和过程②中的碱基配对方式相同。\n\n（4）甲、乙是病毒颗粒的主要组成成分，甲的合成场所为b.宿主细胞的核糖体，乙的合成场所为d.宿主细胞的核糖体和内质网。\n<学生答案>：\n### 关键信息分析：\n图片展示辛德毕斯病毒的增殖过程：病毒通过表面受体吸附到宿主细胞膜，经胞饮进入细胞，释放+RNA。过程①为RNA复制，先以+RNA为模板合成-RNA，再以-RNA为模板合成子代+RNA；过程②为翻译，合成病毒蛋白质（甲、乙）。病毒通过出芽方式从宿主细胞释放，体现膜的流动性。\n\n\n#### 1. 病毒受体化学本质与细胞膜特性  \n- 宿主细胞表面的病毒受体负责特异性结合病毒，其化学本质是**糖蛋白（或蛋白质）**。  \n- 病毒通过“胞饮”“出芽”进出细胞，依赖细胞膜的形态变化，体现了细胞膜具有**流动性**的特点。  \n\n#### 2. 合成子代+RNA所需碱基计算  \n- 病毒+RNA含11626个碱基，A+G=50%（5813个），则U+C=50%（5813个）。  \n- 过程①包括两步合成：+RNA→-RNA→+RNA。  \n  - 第一步（+RNA→-RNA）：-RNA的U+C等于+RNA的A+G（5813个）。  \n  - 第二步（-RNA→+RNA）：+RNA的U+C等于-RNA的A+G，而-RNA的A+G等于+RNA的U+C（5813个）。  \n- 总需U和C：5813 + 5813 = **11626个**。  \n\n#### 3. 过程②的本质、原料及碱基配对方式  \n- 过程②是**翻译**，以RNA为模板合成蛋白质，需宿主细胞的**氨基酸**作为原料。  \n- 过程①（RNA复制）和过程②（翻译）均为RNA间的碱基配对，配对方式均为A-U、U-A、G-C、C-G，故**相同**。  \n\n#### 4. 甲、乙的合成场所  \n- 甲、乙是病毒蛋白质，病毒无核糖体，合成需依赖宿主细胞。蛋白质在**宿主细胞的核糖体**上合成，若涉及加工（如包膜蛋白）可能需要内质网，但题目强调“合成场所”，核心是核糖体。结合选项，选**b（宿主细胞的核糖体）**。  \n\n### 最终答案：  \n1. \\boxed{糖蛋白（或蛋白质）}；\\boxed{流动性}  \n2. \\boxed{11626}  \n3. \\boxed{翻译}；\\boxed{氨基酸}；\\boxed{相同} \n4. \\boxed{b}\n\n【评分依据】：\n对于第（1）问：\n学生答案中病毒受体化学本质为“糖蛋白（或蛋白质）”，与标准答案“蛋白质（糖蛋白）”在语义上等价，得1分。\n学生答案中细胞膜特点为“流动性”，标准答案为“一定的流动性”，虽然学生答案表述相对简略，但在语义上是一致的，得1分。\n对于第（2）问：\n学生答案为“11626”，与标准答案“11626个”在数学上等价（都表示数量为11626），得1分。\n对于第（3）问：\n学生答案中过程②表示“翻译”，与标准答案一致，得1分。\n学生答案中原料为“氨基酸”，与标准答案一致，得1分。\n学生答案中碱基配对方式为“相同”，与标准答案一致，得1分。\n对于第（4）问：\n学生答案只给出了“b”，而标准答案为甲的合成场所为“b.宿主细胞的核糖体”，乙的合成场所为“d.宿主细胞的核糖体和内质网”，学生答案不完整，没有答出乙的合成场所，对于此问得0分。\n综上，学生的答案应得7分。\n【总分】：7分\n【JSON】：\n{\n"score": [[1,1],[1],[1,1,1],[0]]\n}\n\n【示例3】： \n<题目>：\n如图所示，半径为$R$的半圆形光滑轨道$ABC$固定在光滑水平桌面上，$AOC$为半圆形轨道的直径，一质量为$m$的小球放置在$A$点，某时刻，给小球施加一方向垂直$AC$、大小为$F$的水平恒力，在以后的运动过程中，下列说法正确的是（   ） \nA. 小球能够越过$C$点达到$AC$的左侧 \nB. 小球运动过程中的最大速度为$\\dfrac{\\sqrt{2\\mathit{F}\\mathit{R}\\mathit{m}}}{\\mathit{m}}$ \nC. 小球运动过程中对轨道压力的最大值为$3F$ \nD. 小球运动过程中对桌面的压力先增大后减小\n<标准答案>：\nBC\n<学生答案>：\nBC\n\n【评分依据】：\n对于该选择题，学生最终答案为“BC”，与标准答案“BC”一致。\n\n综上，学生的答案应得1分。\n【总分】：1分\n【JSON】：\n{\n"score": [[1]]\n}\n\n【示例4】： \n<题目>：\n3.On his fishing trip, Mike wok the boat <tex>13 km</tex> south. The fish are not \nbiting so he went <tex>5 km</tex> back north. He followed a school of fish <tex>2 km</tex> \nt distance did he cover? What was the displacement? \n4. Amy walked to her school which is <tex>2 km</tex> east from his house. She \ntravelled <tex>1 km</tex> east before he realized she forgot his pack lunch and then \nwalked back home to get it. After picking up her food, she headed back \nto school. What distance did she cover? \nWhat was the displacement?\n<标准答案>：\n<tex>3. 6</tex>\n<tex>4. 4 2</tex>\n<学生答案>：\n3.\nStep 1: Calculate the total southward distance: 13 km.\nStep 2: Calculate the total northward distance: 5 km + 2 km = 7 km.\nStep 3: Calculate the net southward distance: 13 km - 7 km = 6 km.\nStep 4: The distance covered is the sum of all distances traveled: 13 km + 5 km + 2 km = 20 km.\nStep 5: The displacement is the net southward distance: 6 km.\n\n4.\nStep 1: Amy initially walks 1 km east.\nStep 2: She then walks 1 km west back home.\nStep 3: Finally, she walks 2 km east to school.\nStep 4: Total distance covered: 1 km + 1 km + 2 km = 4 km.\nStep 5: Displacement is the final distance from her starting point: 2 km.\n†Answer:3. 6 km\n4. 4 km, 2 km\n\n【评分依据】：\n对于第3问：\n学生答案中位移为“6 km”，与标准答案“6”（这里标准答案虽没写单位，但根据题目情境可知是km，学生答案与之在数学和语义上等价）一致，得1分。\n对于第4问：\n学生答案中距离为“4 km”，与标准答案“4”（同样没写单位，但根据情境可知是km，学生答案与之在数学和语义上等价）一致，得1分。\n学生答案中位移为“2 km”，与标准答案“2”（单位为km，学生答案与之在数学和语义上等价）一致，得1分。\n综上，学生的答案应得3分。\n【总分】：3分\n【JSON】：\n{\n    "score": [[1],[1,1]]\n} \n\n\n【示例5】： \n<题目>：\nhan Learning Task 2 \nDirections: Follow the steps discussed in the lesson in solving the following \nproblems. . Use <tex>\\pi =3.14</tex> Write your answers on a piece of paper. \n1. A circle is inscribed with a <tex>7 meter</tex> square. What is the area, in \n<tex>square meter</tex>, of the shaded region? \n<tex>r=3.5m</tex> \n2. May and John are pianning to buiid a veranda like the figure as \nshown below. The semi-circle shaped part of the veranda will be \ncovered with grass because they want to put a pocket garden while \nthe rest of the veranda will be covered with tiles. What is the total \narea of the veranda that will be covered with tiles and grass?\n<标准答案>：\n1 . <tex>10.5m^{2}</tex> and\n2) <tex>15.87m^{2}</tex>.\n<学生答案>：\n1. Step 1: Find the area of the square.\nThe area of the square is <tex>$side^2 = 7^2 = 49 \\, m^2$</tex>.\n\nStep 2: Find the area of the inscribed circle.\nThe area of the circle is <tex>$\\pi r^2 = \\pi (3.5)^2 = 3.14 \\times 12.25 = 38.465 \\, m^2$</tex>.\n\nStep 3: Subtract the area of the circle from the area of the square to find the area of the shaded region.\nThe area of the shaded region is <tex>$49 - 38.465 = 10.535 \\, m^2$</tex>.  Approximating to one decimal place, we get <tex>$10.5 \\, m^2$</tex>.\n\n\n2. Step 1: Find the area of the rectangular part of the veranda.\nThe area of the rectangle is <tex>$length \\times width = 5 \\times 6 = 30 \\, m^2$</tex>.\n\nStep 2: Find the area of the semi-circle.\nThe radius of the semi-circle is 3 m.  The area of a full circle is <tex>$\\pi r^2 = 3.14 \\times 3^2 = 28.26 \\, m^2$</tex>.  The area of the semi-circle is half of this, which is <tex>$\\frac{28.26}{2} = 14.13 \\, m^2$</tex>.\n\nStep 3: Add the area of the rectangle and the area of the semi-circle to find the total area of the veranda.\nThe total area is <tex>$30 + 14.13 = 44.13 \\, m^2$</tex>.\n†Answer:1) <tex>$10.5 \\, m^2$</tex>\n2) <tex>$44.13 \\, m^2$</tex>\n\n【评分依据】：\n对于第1问：\n学生答案为“\\(10.5 \\, m^2\\)”，与标准答案“\\(10.5m^{2}\\)”一致，得1分。\n对于第2问：\n学生答案为“\\(44.13 \\, m^2\\)”，而标准答案为“\\(15.87m^{2}\\)”，学生答案和标准答案在数学上不一致，得0分。\n综上，学生的答案应得1分。\n【总分】：1分\n【JSON】：\n{\n    "score": [[1],[0]]\n} \n'


class ModelBasedStemVerifierVolc(BaseVerifier):

    def __init__(self, volc_ark_key: str, volc_model_name: str) -> None:
        super().__init__()
        if not volc_ark_key:
            raise ValueError('volc_ark_key is not set')
        if not volc_model_name:
            raise ValueError('volc_model_name is not set')
        base_url = os.environ.get('VOLC_ARK_BASE_URL', "https://ark-cn-beijing.bytedance.net/api/v3")

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=volc_ark_key, timeout=1800)
        self.model = volc_model_name

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        answer = verifier_feature_dict['answer']
        problem = verifier_feature_dict['problem']

        tb = ''
        for i in range(3):
            try:
                prompt_template = '<题目>：\n{problem}\n<标准答案>：\n{answer}\n<学生答案>：\n{response}\n\n'
                prompt = prompt_template.format(problem=problem, answer=answer, response=response)
                completion = self.client.chat.completions.create(model=self.model,
                                                                 messages=[
                                                                     {
                                                                         "role": "system",
                                                                         "content": judge_system_prompt
                                                                     },
                                                                     {
                                                                         "role": "user",
                                                                         "content": prompt
                                                                     },
                                                                 ],
                                                                 timeout=120)
                gpt_response = completion.choices[0].message.content
                score = compute_score(gpt_response)
                remark = f"题目:\n{problem}\n\n学生答案:\n{response}\n\n打分:{gpt_response}"

                return VerifyResult(score=score, extracted_answer=remark)
            except Exception as ex:
                import traceback
                logger.info(traceback.format_exc())
                logger.info(f'Got exception in compute_score via stem verifier_service: {ex}')
                time.sleep(random.choice(list(range(10, 25))))
                continue
        raise VerifierFailed
