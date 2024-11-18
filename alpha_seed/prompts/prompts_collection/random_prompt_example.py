from alpha_seed.prompts.base import PromptBase
import random


class RandomPromptExample(PromptBase):
    '''A prompt can be random prompt itself!
    Create by converstion to o1 like this conversation:
    https://chatgpt.com/share/67369e23-b75c-8001-b307-193422bb15bb
    '''

    def __init__(
        self,
    ):
        super().__init__()
        self.system_prompt = "You are good at long-time reflective thinking."
        self.thinking_styles = [
            """## Effective Thinking Style Guidelines

### 1. Understand the Problem

- **Carefully read the question** to grasp what is being asked.
- **Identify key information** and determine the main objectives.
- **Restate the problem in your own words** to ensure comprehension.

### 2. Analyze the Components

- **Break down the problem** into smaller, manageable parts.
- **Determine knowns and unknowns**, and how they relate.
- **Identify any constraints or special conditions** that apply.

### 3. Develop a Plan

- **Consider various strategies** that could be applied.
- **Select the most appropriate method** based on the problem type.
- **Outline the steps** needed to reach a solution.

### 4. Execute the Plan

- **Work through each step methodically**, keeping calculations clear.
- **Show all workings**, including intermediate steps and logic.
- **Check for consistency** in units, signs, and terms.

### 5. Reflect and Verify

- **Review each step** to ensure accuracy.
- **Verify the solution** by substituting it back into the original problem, if possible.
- **Consider alternative methods** to confirm the result.

### 6. Communicate the Solution

- **Present your answer clearly and concisely**.
- **Include explanations** to demonstrate understanding.
- **Highlight the final answer**, ensuring it directly addresses the question.

""", """
## Analytical Problem-Solving Framework

### 1. **Clarify the Objective**

- **Identify the main goal** of the problem.
- **Determine what is being asked** explicitly and implicitly.
- **List out the desired outcomes** to stay focused.

### 2. **Gather Information**

- **Collect all relevant data** provided in the problem.
- **Note down given variables**, equations, and constants.
- **Research additional information** if necessary and applicable.

### 3. **Decompose the Problem**

- **Break the problem into smaller parts** to simplify complexity.
- **Analyze each component individually** to understand their roles.
- **Establish relationships** between different parts of the problem.

### 4. **Formulate a Strategy**

- **Consider multiple approaches** to find the most effective one.
- **Apply relevant theories or principles** that pertain to the problem.
- **Create a step-by-step plan** outlining the path to the solution.

### 5. **Implement the Solution**

- **Execute calculations carefully**, ensuring each step follows logically.
- **Use diagrams or charts** if they help visualize the problem.
- **Keep track of units and signs** to maintain accuracy.

### 6. **Review and Reflect**

- **Double-check calculations** and reasoning at each step.
- **Assess whether the solution makes sense** in the context of the problem.
- **Reflect on the method used** and consider if there was a more efficient path.

### 7. **Document the Findings**

- **Summarize the solution clearly**, highlighting key results.
- **Explain the reasoning** behind each major step.
- **Prepare to communicate the solution** to others, if needed.

### 8. **Continuous Improvement**

- **Evaluate what was learned** during the problem-solving process.
- **Identify areas for improvement** in methodology or understanding.
- **Apply these insights** to future problems to enhance proficiency.
""", """
## **Dynamic Problem-Solving Framework**

### **I. Comprehensive Understanding**

1. **Initial Reading**
   - **Read the problem carefully** to grasp all aspects.
   - **Highlight key terms and data** that are crucial.

2. **Restate in Your Own Words**
   - **Paraphrase the problem** to ensure you understand it fully.
   - **Identify the main question** and what is being asked.

3. **Determine Objectives**
   - **Establish clear goals** for what needs to be achieved.
   - **Recognize any sub-questions** or secondary objectives.

### **II. Information Organization**

1. **List Known Variables**
   - **Catalog all given information**, constants, and variables.
   - **Note units of measurement** and conversions if necessary.

2. **Identify Unknowns**
   - **Determine what needs to be found** or proven.
   - **Express unknowns using appropriate symbols** or placeholders.

3. **Draw Diagrams or Charts**
   - **Visualize the problem** through sketches or graphs.
   - **Use tables** to organize data systematically.

### **III. Analytical Breakdown**

1. **Decompose the Problem**
   - **Split the problem into smaller parts** or steps.
   - **Analyze each component** individually.

2. **Understand Relationships**
   - **Examine how variables interact** with each other.
   - **Identify direct and inverse relationships**.

3. **Apply Relevant Concepts**
   - **Recall theories, formulas, or principles** that apply.
   - **Consider laws or rules** pertinent to the problem's context.

### **IV. Strategy Formulation**

1. **Brainstorm Approaches**
   - **Think of multiple methods** to tackle the problem.
   - **Consider both conventional and unconventional strategies**.

2. **Evaluate Methods**
   - **Assess the feasibility** of each approach.
   - **Determine which method is most efficient** and effective.

3. **Plan Step-by-Step Actions**
   - **Outline a clear sequence** of steps to follow.
   - **Set milestones** for complex problems.

### **V. Execution of Plan**

1. **Carry Out Calculations**
   - **Perform mathematical operations** carefully.
   - **Show all work** to keep track of your process.

2. **Maintain Logical Flow**
   - **Ensure each step logically leads to the next**.
   - **Use proper notation** and units throughout.

3. **Monitor for Errors**
   - **Watch for common mistakes**, such as sign errors.
   - **Validate results at each significant step**.

### **VI. Critical Evaluation**

1. **Verify Results**
   - **Check answers against the problem statement**.
   - **Use reverse calculations** to confirm results.

2. **Cross-Reference Methods**
   - **Apply alternative methods** to see if they yield the same result.
   - **Compare with known solutions** if available.

3. **Assess Logical Consistency**
   - **Ensure the solution makes sense** in real-world terms.
   - **Consider edge cases** or exceptions.

### **VII. Reflection and Insight**

1. **Analyze the Solution Path**
   - **Reflect on why the chosen method worked**.
   - **Identify any shortcuts** or efficiencies discovered.

2. **Learn from Difficulties**
   - **Note challenges faced** and how they were overcome.
   - **Understand mistakes** to prevent them in the future.

3. **Generalize Findings**
   - **Think about how this problem relates to broader concepts**.
   - **Consider how the solution method can apply elsewhere**.

### **VIII. Communication and Presentation**

1. **Prepare a Clear Solution**
   - **Write the final answer prominently**, with proper units.
   - **Summarize the approach taken** in a few sentences.

2. **Explain Reasoning**
   - **Provide justifications** for each major step.
   - **Anticipate questions** and address potential doubts.

3. **Use Visual Aids**
   - **Include diagrams or graphs** if they enhance understanding.
   - **Highlight key points** and results.

### **IX. Adaptability and Creativity**

1. **Stay Flexible**
   - **Be ready to adjust your approach** if new insights arise.
   - **Embrace creative solutions** that may be unconventional.

2. **Incorporate Intuition**
   - **Trust your instincts** when they guide you toward a solution.
   - **Balance intuition with logical reasoning**.

3. **Explore Further**
   - **Consider what-if scenarios** or extensions of the problem.
   - **Engage in lateral thinking** to expand your perspective.

### **X. Time and Resource Management**

1. **Set Time Limits**
   - **Allocate appropriate time** to each part of the problem.
   - **Prioritize tasks** based on their impact on the solution.

2. **Use Resources Wisely**
   - **Refer to textbooks, formulas, or software tools** as needed.
   - **Seek guidance** when necessary but attempt independently first.

3. **Avoid Procrastination**
   - **Start early** to allow ample time for complex problems.
   - **Maintain focus** to enhance efficiency.

### **XI. Emotional Intelligence**

1. **Stay Calm and Confident**
   - **Manage stress** through deep breathing or short breaks.
   - **Believe in your ability** to solve the problem.

2. **Handle Frustration Positively**
   - **View obstacles as learning opportunities**.
   - **Avoid negative self-talk** that can hinder performance.

3. **Celebrate Progress**
   - **Acknowledge small victories** along the way.
   - **Use positive reinforcement** to maintain motivation.
"""
        ]
        self.prompt2 = """
Now use the above thinking style to solve the question!

Question: {}
"""

    def transform(self, question):
        return [{
            "content": self.system_prompt,
            "role": "system"
        }, {
            "content": random.choice(self.thinking_styles) + self.prompt2.format(question),
            'role': 'user'
        }]
