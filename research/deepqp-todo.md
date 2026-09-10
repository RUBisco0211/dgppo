## Idea: 使用 GNN 迁移 DeepQP 到多智能体场景 + CRPO 风格策略更新方法

### 现状

1. DeepQP 从单 agent 迁移到多 agent 场景，目前理论上无法做 CBF-QP，因此原本服务于单 agent 的过于复杂的重参数化技巧和导数网络等组件可能冗余，且迁移到多 agent 场景下算法有部分步骤交代不清
2. GNN 改造的 DeepQP 方法训练出的 CBF 效果还可以 (安全边界比较接近 GCBF+，且在整体类 CRPO 方法训练的策略在 eval 下的部分安全性能要优于 DGPPO baseline) ，但同时也很接近环境定义的约束边界，那么是否还有必要继承这么复杂的理论基础和训练方法
3. DeepQP 原训练方法是 off-policy 的，直接迁移无法集成到 DGPPO 的 on-policy 训练过程，因此目前智能分两个阶段训练，即先训练 CBF，再训练 RL actor-critic
4. 目前无法做分布式 CBF-QP，所以使用类 CRPO 的策略更新方法，对安全性能保证效果不错，但会影响本身任务性能
5. DGPPO 训练出的 CBF 边界明显不安全: 训练不充分？有没有可能是新的 gap?（即学习型 CBF 和真实 CBF 的 gap）



### 可能方案

- 从 HJ Reachability 理论出发，重新构造一个更适合的 CBF 训练方法
    - Pros: HJ Reachability 给出的 HJ Value 函数本身就可以被证明为是离散 CBF，接下来主要就是处理训练方法和损失函数构造的问题
    
    - Cons: 
      - DGPPO 方法本身构造 DGCBF 的方法也和 HJ PDE 有关联，且训练方法是构造 TD-error 形式，已经适合 on-policy 训练
      - 直接用 HJ PDE 来监督，有单智能体论文这么做了，但它是离线训练方法
- 更好的带约束策略更新方法？需要保证实时约束满足而不是长期累积
- 不能做分布式 CBF-QP，那能不能训练时做集中式解出安全动作，然后在策略更新的过程中去监督：有类似方法（GCBF+，HJB-GNN）但具体怎么迁移还需要探究
