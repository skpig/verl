class Solver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: Any
    stop: List[str] = None
    llm: Optional[Callable[[...], List[str]]] = None
    llm_engine: Optional[LLM] = None
    generate_sampling_params: Optional[SamplingParams] = None
    need_value_func: bool = False
    max_agent_steps: int = 1
    reward_model: Optional[Any] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.need_value_func = self.config.need_value_func
        if self.need_value_func:
            self.reward_model = self.create_rm()
        self.llm = self.create_llm()
        if self.config.mode == "mcts":
            self.max_agent_steps = self.config.iterations
            self.config.step_beam_width = 1
            
    def create_llm(self):
        engine, sampling_params = llm_engine(self.config)
        self.llm_engine = engine
        self.generate_sampling_params = sampling_params
        return partial(
            llm_generate,
            engine=self.llm_engine,
        )

        
    @staticmethod
    def processor(agent, output) -> BaseTree:
        agent.generate_next_step(output)
        return agent


    @staticmethod
    def selector(agent, output) -> BaseTree:
        agent.select_next_step(output)
        return agent


    def generate_preprocess(self, agents):
        prompts = []
        rewards = []
        prompts_span = [0]
        valid_agents = []
        invalid_agents = []
        expanded_agents = []

        for agent in agents:
            if agent.should_generate_next():
                if agent.has_expanded():
                    expanded_agents.append(agent)
                else:
                    agent_prompts = agent.create_prompt()
                    rewards.extend(agent.get_rewards())
                    prompts.extend(agent_prompts)
                    prompts_span.append(prompts_span[-1] + len(agent_prompts))
                    valid_agents.append(agent)
            else:
                invalid_agents.append(agent)
        return prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, rewards


    def generate_postprocess(
        self, 
        outputs: List[List[RequestOutput]], 
        valid_agents: List[BaseTree],
    ) -> List[BaseTree]:
        post_agents = []
        #with ProcessPool(max_workers=min(len(valid_agents), os.cpu_count())) as pool:
        with ProcessPool(max_workers=12) as pool:
            future = pool.map(self.__class__.processor, valid_agents, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()
        
        progress_bar = tqdm(total=len(valid_agents), desc="generate_postprocess")  
        while True:
            try:
                result = next(iterator)
                post_agents.append(result)
            except StopIteration:
                break
            except Exception as error:
                print(colored(f"{error}\n", ERROR_COLOR))
                post_agents.append(None)
            progress_bar.update(1) 
        progress_bar.close() 
            
        # update agents
        updated_agents = [
            post_agent if post_agent is not None else valid_agent
            for post_agent, valid_agent in zip(post_agents, valid_agents)
        ]
        return updated_agents
    

    def solve(self, agents: List[BaseTree], saved_jsonl_file: str, cur_data: List[Dict[str, Any]]):
        
        for rollout in tqdm(range(self.max_agent_steps), desc="Rollout Processing"):
            # Initialize the initial search starting point of agents, and the initial point of each rollout is root
            for agent in agents:
                agent.select_next_step(from_root=True)
                agent.rollout_idx = rollout

            for step in range(self.config.max_depth):
                print("-----------------Current Rollout: ", rollout, "-----------------")
                print("-----------------Current Step: ", step, "-----------------")
                prompts, prompts_span, valid_agents, invalid_agents, expanded_agents, valid_rewards = self.generate_preprocess(agents)
                
                if len(valid_agents + expanded_agents) < 1:
                    break
                
                # step expansion
                outputs = self.llm(prompts, self.generate_sampling_params)
                
                for output, reward in zip(outputs, valid_rewards): # attach reward to prevent repeat rewarding
                    output.value_estimate = reward
                reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
                
                # process output and run python code
                valid_agents = self.generate_postprocess(reconstructed_outputs, valid_agents)

                # step evaluation
                prompts, prompts_span = self.value_preprocess(valid_agents)
                if self.need_value_func:
                    outputs = self.reward_model(prompts=prompts)
                    reconstructed_outputs = [outputs[bos_idx : eos_idx] for bos_idx, eos_idx in zip(prompts_span, prompts_span[1:])]
                else:
                    reconstructed_outputs = [None] * (len(prompts_span) - 1)
                
                # selection
                valid_agents = self.value_postprocess(reconstructed_outputs, valid_agents)
                expanded_agents = self.value_postprocess([None] * len(expanded_agents), expanded_agents) # for expanded agents, just do selection step
                
                # keep all agents
                agents = valid_agents + invalid_agents + expanded_agents

            # Save agents internal rollouts
            self.save_intermediate_rollouts(saved_jsonl_file, cur_data, agents, rollout)
            
        return self.output(agents)