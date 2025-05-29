# 这个 MR 做了什么？

简要说明这个 MR 旨在实现或完成的目标。

# ChangeLog:

- Add specific changes and high-level design in this PR.

# 使用方法

- 可以在下方添加一个使用示例。

```python
# Add code snippet or script demonstrating how to use this
```

# 提交前检查

- [ ] 避免了breaking change，如果有的话请说明影响面多大
- [ ] 不需要依赖某个组件的最新版本
- [ ] 已经新添加了unit-test到CI
- [ ] 主要依赖CI中现有的unit-test，且现有的测试逻辑可以覆盖改动的代码
- [ ] 新增加的功能和选项，在代码中添加了对应的docstring
- [ ] 如果涉及强依赖的库，import的时候直接报错抛出原始Exception，而不是try catch 后打印 warning log
