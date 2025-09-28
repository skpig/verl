import ast
import re

from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult


class GUIORMVerifier(BaseVerifier):
    """
    Verifies the correctness and format of a GUI ORM model response.
    The verification checks for strict adherence to a predefined structure,
    content logic, and tag uniqueness.
    """

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        """
        Verifies the model response based on a set of rules.

        Args:
            response (str): The model's generated response string.
            verifier_feature_dict (dict): A dictionary containing features for verification.

        Returns:
            VerifyResult: An object containing the score and extracted answer.
        """
        gt_content = verifier_feature_dict['answer']
        is_correct = verify_format_and_content(gt_content, response)

        if is_correct:
            return VerifyResult(score=1.0, extracted_answer=response)
        else:
            return VerifyResult(score=0.0, extracted_answer=response)


def verify_format_and_content(pred_content: str, gt_content: str) -> bool:
    """
    Performs a comprehensive check on the model's output format and content logic.

    Args:
        pred_content (str): The model's generated response string.

    Returns:
        bool: True if the format and content are correct, False otherwise.
    """
    # Step 1: Check for tag uniqueness and structure
    start_tag = "<gui_orm_think>"
    end_tag = "</gui_orm_think>"
    judgment_tag = "## Judgment:"

    # Use re.findall to get a robust count of non-overlapping occurrences
    if len(re.findall(re.escape(start_tag), pred_content)) != 1 or \
       len(re.findall(re.escape(end_tag), pred_content)) != 1:
        print("Verification Failed: Tags are not unique or are missing.")
        return False

    start_index = pred_content.find(start_tag)
    end_index = pred_content.find(end_tag)

    if not (0 <= start_index < end_index + 3):
        print("Verification Failed: Tag order is incorrect.")
        return False

    # Step 2: Extract and validate the judgment part
    thinking_process = pred_content[start_index + len(start_tag):end_index].strip()
    judgment_part = pred_content[end_index + len(end_tag):].strip()

    if len(re.findall(re.escape(judgment_tag), judgment_part)) != 1:
        print("Verification Failed: Judgement Tags are not unique .")
        return False

    # Step 3: Validate the thinking process content (required headers)
    required_sections = [
        "#### Analysis of Execution",
        "#### Key Factors for Outcome",
        "#### Suggestion for revision",
        "#### Correctness of Answer",
        "## Task Breakdown",
    ]

    section_flag = False
    for section in required_sections:
        if section in thinking_process:
            section_flag = True
    if not section_flag:
        print(f"Verification Failed: Missing required section header: {section}")
        return False

    # All format checks passed
    if "### Judgment:" not in judgment_part:
        return False
    if judgment_part.split("### Judgment:")[-1].strip().lower() == gt_content.split(
            "### Judgment:")[-1].strip().lower():
        return True
    else:
        return False


if __name__ == "__main__":
    print('##################Starting GUI ORM Verifier Test##################')

    # Example test cases
    test_cases = [
        # Correct and successful output
        # '<gui_orm_think>#### Analysis of Execution:\n代理首先输入 "早街村委会2010年农村经济总收入"进行搜索，然后在AI Overview部分看到了明确的信息，得到了最终答案，关键词使用清晰明确。\n#### Key Factors for Outcome\n输入 "早街村委会2010年农村经济总收入"进行搜索至关重要。代理关键词使用清晰明确。\n#### Suggestion for revision\nNone\n#### Correctness of Answer\n##### Correctness: true\n##### Correct Answer:None</gui_orm_think>### Judgment: SUCCESS',

        # Correct and failed output
        '<gui_orm_think>#### Analysis of Execution:\n代理执行搜索但未能找到正确答案。\n#### Key Factors for Outcome\n搜索词不精确，或信息不存在于可信源。\n#### Suggestion for revision\n建议使用更精确的搜索词。\n#### Correctness of Answer\n##### Correctness: false\n##### Correct Answer:666</gui_orm_think>### Judgment: FAILURE',
        '<gui_orm_think>#### Correctness of Answer\n##### Correctness: false\n##### Correct Answer:666</gui_orm_think>### Judgment: FAILURE',
        '<gui_orm_think>### Judgment: FAILURE#### Analysis of Execution:\n代理执行搜索但未能找到正确答案。\n#### Key Factors for Outcome\n搜索词不精确，或信息不存在于可信源。\n#### Suggestion for revision\n建议使用更精确的搜索词。\n#### Correctness of Answer\n##### Correctness: false\n##### Correct Answer:666</gui_orm_think>### Judgment: FAILURE',
        '<gui_orm_think></gui_orm_think>### Judgment: FAILURE',
        '<gui_orm_think>#### Analysis of Execution:\n代理执行搜索但未能找到正确答案。\n#### Key Factors for Outcome\n搜索词不精确，或信息不存在于可信源。\n#### Suggestion for revision\n建议使用更精确的搜索词。\n#### Correctness of Answer\n##### Correctness: false\n##### Correct Answer:666</gui_orm_think>### Judgment: FAIL',
        '<gui_orm_think>#### Analysis of Execution:\n代理执行搜索但未能找到正确答案。\n#### Key Factors for Outcome\n搜索词不精确，或信息不存在于可信源。\n#### Suggestion for revision\n建议使用更精确的搜索词。\n#### Correctness of Answer\n##### Correctness: false\n##### Correct Answer:666</gui_orm_think>### Judgment:### Judgment: FAILURE',

        # Incorrect - extra tag
        # '<gui_orm_think><gui_orm_think>Content</gui_orm_think>### Judgment: SUCCESS',

        # # Incorrect - missing tag
        # '<gui_orm_think>Content### Judgment: SUCCESS',

        # # Incorrect - missing think process
        # '<gui_orm_think></gui_orm_think>### Judgment: SUCCESS',

        # # Incorrect - format error
        # '<gui_orm_think></gui_orm_think>Content### Judgment: SUCCESS',

        # # Incorrect - inconsistent judgment and correctness
        # '<gui_orm_think>#### Task Breakdown and Analysis of Execution:\n1.打开文件并定位单元格：代理成功定位到桌面上的\"data.xlsx\"文件并使用LibreOffice Calc将其打开，成功查看到文件内容，定位至指定单元格，该子任务完成。\n2.使用COUNTIF函数统计人数：代理对要求的语文，数学，英语分别使用COUNTIF函数进行了统计，并按照题目要求分别填写进指定单元格，使用COUNTIF函数时，仅统计已展示部分单元格数据，不全面，该子任务部分完成。\n3.保存文件：代理使用快捷键Ctrl+S将结果保存至文档中，完成了任务。\n总之，第一个子任务中，代理定位单元格正确，输入坐标与标出位置一致，顺利输入COUNTIF函数；第二个子任务中，代理对其理解到位，对三个科目使用COUNTIF作了对应的合格或不合格人数统计，但仅统计了已展示部分单元格中的数据，不全面，导致任务未完成。\n#### Suggestion for revision\n在使用COUNTIF函数时，需要考虑该科目全部的数据，寻找更有效快捷的选取待算单元格的方式，可通过点击目标列的标题选取整列数据以减少计算误差，如：=COUNTIF(B:B,\"合格\")，以指代B列全部数据来计算其合格人数。</gui_orm_think>### Judgment: SUCCESS',
    ]
    gt_content = '<gui_orm_think>#### Analysis of Execution:\n代理执行搜索但未能找到正确答案。\n#### Key Factors for Outcome\n搜索词不精确，或信息不存在于可信源。\n#### Suggestion for revision\n建议使用更精确的搜索词。\n#### Correctness of Answer\n##### Correctness: false\n##### Correct Answer:666</gui_orm_think>### Judgment: FAILURE'
    for test_case in test_cases:
        print(f"Test Case: {test_case}")
        result = verify_format_and_content(test_case, gt_content)
        print(f"Result: {result}\n")
    print('##################Ending GUI ORM Verifier Test##################')
