import os
import importlib
import inspect
from alpha_seed.prompts.base import PromptBase
import random

folder_path = "prompts_collection"
curdir = os.path.abspath(os.path.dirname(__file__))


def load_prompts(prompt_names):
    all_files = os.listdir(os.path.join(curdir, folder_path))
    all_files = [
        x[:-3]
        for x in all_files
        if x.endswith('.py') and x not in [os.path.basename(__file__), 'base.py', '__init__.py']
    ]
    if prompt_names == "all":
        use_files = all_files
    else:
        prompts = prompt_names.split(":")
        include_files = set([x for x in prompts if x[0] != "-"])
        if len(include_files) == 0 or (len(include_files) == 1 and list(include_files)[0] == "all"):
            use_files = set(all_files)
        else:
            use_files = include_files.intersection(set(all_files))
        exclude_files = set([x[1:] for x in prompts if x[0] == "-"])
        use_files = use_files.difference(exclude_files)

    prompt_objects = []

    for module_name in use_files:
        file_name = module_name + ".py"
        file_path = os.path.join(curdir, folder_path, file_name)
        # 从文件路径加载模块规范
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            continue
        # 创建模块对象
        module = importlib.util.module_from_spec(spec)
        try:
            # 执行模块（相当于导入模块）
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"加载模块 {module_name} 时出错：{e}")
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # 检查类是否是 BasePrompt 的子类，且不等于 BasePrompt 自身
            if issubclass(obj, PromptBase) and obj is not PromptBase:
                try:
                    prompt_instance = obj()
                    prompt_objects.append(prompt_instance)
                except Exception as e:
                    print(f"实例化类 {name} 时出错：{e}")
    print("Load Prompts: ")
    for prompt_object in prompt_objects:
        print("=" * 50)
        print(prompt_object)
    print("Load Prompts Finish")
    return prompt_objects


def random_transform(prompts, question):
    return random.choice(prompts).transform(question)


if __name__ == "__main__":
    prompts = load_prompts(prompt_names="all:-random_prompt_example:-action_space_simple")
    msg = random_transform(prompts, "What is the capital of France?")
    print(msg)
