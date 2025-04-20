# 蒙特卡洛树搜索（MCTS）实现类，继承自 Beam Search 基类
class MCTS(BS):
    search_node: Type[BaseNode] = None

    # 用于记录中间结果
    intermediate_metric: Dict = {
        "question": "",
        "gt": "", 
        "answers": [],
        "judgements": [],
        "value_estimate": [],
        "rollout_indexs": [],
    }

    # 校验配置是否合法
    @field_validator("config")
    def validate_config(cls, cfg: Any):
        BaseTree.validate_config(cfg)
        if not cfg.mode == "mcts":
            raise ValueError(f"Wrong value for config mode.")
        if cfg.stop is None:
            raise ValueError(f"Wrong value for config stop, cannot be None")
        return cfg

    # 创建节点方法
    def create_node(self, parent: Optional[Type[MCTSNode]] = None) -> Type[MCTSNode]:
        return MCTSNode(
            parent=parent, 
            additional_state_keys=self.NODE_KEYS,
            c_puct=self.config.c_puct,
        )

    # 从当前或根节点出发进行选择
    def selection(self, from_root=False) -> Optional[Type[MCTSNode]]:
        if from_root:
            start_node = self.root
        else:
            start_node = self.search_node
        node = start_node
        if node is None: return None
        if node.has_children() or node.is_terminal:
            next_node = self.select_child(node)
            if next_node is None:
                node.is_terminal = True
            node = next_node
        return None if (node is None or node.is_terminal) else node

    # 使用 PUCT 选择最佳子节点
    def select_child(self, node: Type[MCTSNode]) -> Optional[Type[MCTSNode]]:
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        return best_childs[0] if best_childs else None

    # 扩展当前节点，生成子节点
    def expand_node(self, outputs: List[CompletionOutput], node: Type[MCTSNode]) -> None:
        for idx, output in enumerate(outputs):
            if not output.stop_reason: output.stop_reason = ""
            step_result, parser_result = self.step_unwrap(output.text + output.stop_reason)
            self.create_child(step_result, parser_result, node, idx)

    # 创建子节点并更新其状态
    def create_child(
        self, 
        step_result: str, 
        parser_result: Dict[str, str], 
        node: Type[MCTSNode],
        idx: int,
    ) -> None:
        new_node = self.create_node(parent=node)
        parent_child_count = len(node.children)
        new_node.tag = f"{node.tag}.{parent_child_count + 1}"
        new_node.depth = node.depth + 1

        # 根据解析结果判断是否为终止节点
        if parser_result is None:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = NO_VALID_CHILD
            self.eval_final_answer(new_node)
        elif parser_result["final_answer"]:
            new_node.is_terminal = True
            new_node.state["text"] = step_result
            new_node.state["final_answer"] = parser_result["final_answer"]
            self.eval_final_answer(new_node)
        elif parser_result["action"]:
            # 执行动作并记录 observation
            observation = code_execution(node, parser_result)
            new_node.state["action"] = parser_result["action"]
            new_node.state["action_input"] = parser_result["action_input"]
            new_node.state["observation"] = observation
            if CODE_END in parser_result["action_input"]:
                observation = self.obs_wrap(observation)
                new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
            else:
                new_node.state["text"] = step_result

            # 错误次数过多，终止该路径
            if "error" in observation.lower():
                new_node.consecutive_errors = node.consecutive_errors + 1
                if new_node.consecutive_errors >= self.config.errors_threshold:
                    observation = self.obs_wrap(observation)
                    step_result = step_result + CODE_END if CODE_END not in step_result else step_result
                    new_node.state["text"] = f"{step_result}{self.config.step_delim}{observation}"
                    new_node.is_terminal = True
                    new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                    self.eval_final_answer(new_node)
        else:
            new_node.state["text"] = step_result

        # 深度超过最大限制，终止
        if not new_node.is_terminal and new_node.depth > self.config.max_depth:
            new_node.is_terminal = True
            new_node.state["final_answer"] = TOO_MANY_STEPS
            self.eval_final_answer(new_node)

        node.children.append(new_node)

    # 评估终止节点的最终答案
    def eval_final_answer(self, node: Type[MCTSNode]) -> None:
        if node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
            node.update(self.config.negative_reward)
            return 
        if self.config.is_sampling:
            final_answer = node.state["final_answer"]
            correct = is_equiv(self.ground_truth, final_answer)
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)
        else:
            self.candidate_nodes.append(node)

    # 记录每次 rollout 的中间指标
    def record_intermediate_metric(self, answer, value_estimate):
        self.intermediate_metric["question"] = self.question
        self.intermediate_metric["gt"] = self.ground_truth
        if self.intermediate_metric["rollout_indexs"] and self.rollout_idx in self.intermediate_metric["rollout_indexs"]:
            index = self.intermediate_metric["rollout_indexs"].index(self.rollout_idx)
            if value_estimate > self.intermediate_metric["value_estimate"][index]:
                self.intermediate_metric["answers"][index] = answer
                self.intermediate_metric["judgements"][index] = is_equiv(self.ground_truth, answer)
                self.intermediate_metric["value_estimate"][index] = value_estimate
        else:
            self.intermediate_metric["answers"].append(answer)
            self.intermediate_metric["judgements"].append(is_equiv(self.ground_truth, answer))
            self.intermediate_metric["value_estimate"].append(value_estimate)
            self.intermediate_metric["rollout_indexs"].append(self.rollout_idx)

    # 根据输出值评估当前候选节点，并选出下一步节点
    def select_next_step(self, outputs=None, from_root=False) -> None:
        self.search_node = self.current_nodes[0] if self.current_nodes else None
        self.current_nodes = []
        if outputs:
            for candidate_node, output in zip(self.candidate_nodes, outputs):
                if candidate_node.is_terminal and self.config.is_sampling:
                    continue
                value_estimate = output.value_estimate if output.value_estimate is not None else self.config.negative_reward
                if output.value_estimate is None:
                    candidate_node.is_terminal = True

                if candidate_node.is_terminal and candidate_node.state["final_answer"]:
                    if candidate_node.state["final_answer"] in [NO_VALID_CHILD, TOO_MANY_STEPS, TOO_MANY_CODE_ERRORS]:
                        candidate_node.update(self.config.negative_reward)
                    else:
                        self.record_intermediate_metric(answer=candidate_node.state["final_answer"], value_estimate=value_estimate)
                        candidate_node.update_recursive(value_estimate, self.root)
                else:
                    if self.config.terminal_sample:
                        pass
                    else:
                        candidate_node.update(value_estimate)

                if self.__class__.is_valid_final_answer_node(candidate_node):
                    self.final_answer_nodes.append(candidate_node)
        selection_node = self.selection(from_root=from_root)
        if selection_node is not None:
            self.current_nodes.append(selection_node)

    # 根据当前节点生成下一步输出
    def generate_next_step(self, outputs: List[RequestOutput]) -> None:
        self.candidate_nodes = []
        for current_node, output in zip(self.current_nodes, outputs):
            value_estimate = output.value_estimate
            if value_estimate is not None:  
                self.expand_node(output.outputs, current_node)
            else:
                value_estimate = self.config.negative_reward
                current_node.is_terminal = True

            if self.config.update_leaf_value:
                for value_node in current_node.children:
                    if value_node not in self.candidate_nodes and value_node.visit_count() < 1:
                        self.candidate_nodes.append(value_node) 

    # 返回整个搜索树中的所有节点状态
    def return_states(self) -> Dict[str, Union[Any, Dict[str, str]]]:
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            states[node.tag] = node.state
            states[node.tag]["value"] = node.value
            states[node.tag]["q_value"] = node.q_value()
            states[node.tag]["visit_count"] = node.visit_count()
            if node.has_children():
                candidates.extend(node.children)
        return states
