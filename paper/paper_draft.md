# Comparative analysis of reinforcement learning algorithms for adaptive traffic signal control under asymmetric demand

Vyankatesh Dawale · Pratyush Chaudhari · Saim Kotkar · Om Dangi · Yogita D. Patil

---

## Abstract

Urban traffic congestion inflicts severe economic, environmental, and societal costs on cities worldwide. Classical fixed-time traffic signal controllers, which operate on static schedules, are fundamentally ill-suited to the stochastic nature of real-world traffic demand. In this work, we present a rigorously designed experimental framework that implements, trains, and evaluates three structurally distinct Reinforcement Learning (RL) paradigms—Tabular Q-Learning, Deep Q-Networks with Double and Dueling extensions (DQN), and Proximal Policy Optimization (PPO)—for autonomous adaptive traffic signal control at a single four-way intersection. To address the persistent challenge of queue spillback in highly congested scenarios, we introduce a novel non-linear, quadratic reward penalty mechanism that aggressively suppresses queue formations exceeding intersection capacity. All agents are deployed within the Simulation of Urban Mobility (SUMO) microscopic traffic simulator, interfaced via the Traffic Control Interface (TraCI) API. Crucially, the agents are evaluated under realistic, asymmetric traffic demand profiles characterized by stochastic, non-uniform arrival rates. Under an extended training budget of 10,000 episodes for Q-Learning, 1,000 episodes for DQN, and 2,000,000 timesteps for PPO, our results demonstrate that RL agents significantly outperform the fixed-time baseline. Tabular Q-Learning and DQN achieved 9.9% and 9.3% reductions in mean vehicle waiting times, respectively. PPO, struggling to map its continuous surrogate objectives to the highly asymmetric discrete phase space, exhibited high variance and lagged the baseline. To further accelerate off-policy convergence, we augmented the DQN architecture with Prioritized Experience Replay (PER). We provide detailed architectural descriptions, an analysis of the non-linear reward function, and hyperparameter documentation, outlining a clear path for future multi-agent deployments.

**Keywords:** Reinforcement Learning · Traffic Signal Control · Deep Q-Network · Proximal Policy Optimization · Q-Learning · SUMO · TraCI · Intelligent Transportation Systems · Non-linear Reward Shaping

---

## Introduction

Rapid urbanization has made traffic signal control one of the most critical challenges in intelligent transportation systems research. According to the INRIX Global Traffic Scorecard, urban congestion costs major economies hundreds of billions of dollars annually through lost productivity, fuel wastage, and excess emissions¹. Conventional traffic actuated controllers, while an improvement over purely fixed-time plans, still rely on hand-crafted heuristics that cannot generalize to novel traffic patterns or rare demand surges without costly re-calibration.

Reinforcement Learning offers a compelling alternative: rather than specifying optimal control policies manually, an RL agent discovers them through direct interaction with its environment. This closed-loop learning paradigm is particularly attractive for traffic control because the underlying demand dynamics are non-stationary and difficult to model analytically; the state space is high-dimensional, comprising real-time vehicle counts, queue lengths, waiting times, and phase history; and the objective is straightforwardly definable as a scalar reward signal tied to traffic performance.

The earliest successful applications of RL to traffic control employed tabular Q-Learning with discrete state buckets²ˏ³. The introduction of deep neural function approximators led to Deep Q-Networks⁴, which were rapidly adapted to traffic control⁵. Policy-gradient methods culminated in Proximal Policy Optimization⁶, which has been increasingly applied to traffic light cycle control⁷. However, many published models fail in realistic deployments due to a phenomenon known as queue spillback—where linear reward functions allow queues to silently build past the intersection physical capacity because the linear penalty is outweighed by throughput gains elsewhere.

**Paper contributions.** This paper makes the following original contributions:

1. A unified, reproducible experimental benchmark evaluating Tabular Q-Learning, Double Dueling DQN, and PPO on an identical SUMO single-intersection environment under realistic, asymmetric traffic demand.
2. The design and empirical validation of a non-linear quadratic reward shaping mechanism designed to mitigate queue spillback.
3. An empirical analysis of training budget requirements, demonstrating how extended learning phases (e.g., 2 million steps) are required to stabilize on-policy policy-gradient methods in discrete-action traffic control settings.
4. Open-source code and SUMO network files for full reproducibility.

---

## Related work

**Classical and heuristic approaches.** Traffic signal control has historically been dominated by fixed-time controllers, which allocate phase durations based on historically observed occupancy data. Systems such as SCOOT⁸ and SCATS⁹ paved the way for automated scheduling, but their reliance on pre-computed plans limits their adaptability. Adaptive Webster timing¹⁰ extended these concepts by analytically minimizing average delay under the assumption of Poisson arrivals. While highly effective during periods of low or predictable traffic load, these classical methods possess a fatal flaw: they cannot reconfigure autonomously when demand patterns deviate from historical norms, such as during accidents, special events, or sudden asymmetric surges.

**Reinforcement learning for traffic control.** To overcome the limitations of static scheduling, researchers turned to Reinforcement Learning. Wiering² pioneered the direct application of Q-Learning to traffic signals, demonstrating that even simple tabular agents, which divide the state space into discrete buckets, could outperform fixed-time controllers by learning from experience. Abdulhai et al.¹¹ successfully extended this tabular approach to multi-phase single intersections, proving the viability of RL in more complex junction layouts.

The field experienced a massive paradigm shift with the introduction of deep neural function approximators. The Deep Q-Network (DQN)⁴ introduced experience replay and target networks, which unlocked reliable convergence for high-dimensional, continuous state spaces. This breakthrough was rapidly adapted to traffic control domains, leading to sophisticated models like IntelliLight⁵, which combined DQN architectures with real-world detector data to achieve unprecedented traffic throughput. Further architectural innovations, such as Dueling Networks¹², allowed agents to decompose value estimation into separate state-value and advantage streams, which empirically accelerated convergence in traffic settings. Double DQN¹³ provided a critical correction for the overestimation bias that often plagued early traffic models. Our research synthesizes these improvements into a unified value-based agent to serve as a robust benchmark.

**Policy-gradient and actor-critic methods.** Parallel to value-based methods, policy-gradient approaches have seen extensive exploration. Chu et al.¹⁴ applied Advantage Actor-Critic (A3C) to multi-intersection networks, seeking to leverage the continuous nature of policy distributions. Liang et al.⁷ later demonstrated the viability of Proximal Policy Optimization (PPO), which uses clipped surrogate objectives to ensure stable policy updates. More recently, transformer-based architectures¹⁵ have shown promise for modeling long-range temporal dependencies and spatial correlations in traffic demand. However, a recurring and often understated finding across this literature is that on-policy methods like PPO require substantially more environment interaction to match the performance of well-tuned off-policy counterparts at single intersections. Our work seeks to corroborate this quantitatively by comparing these architectures under massive training budgets.

---

## Methodology

**Problem formulation.** We model adaptive traffic signal control as a discrete-time Markov Decision Process. The environment exposes an 8-dimensional continuous observation vector, comprising per-lane vehicle counts, halted queue lengths, accumulated waiting times, and a one-hot encoding of the currently active traffic light phase; for tabular Q-Learning, these observations are discretized into the corresponding 18-dimensional discrete state representation. The agent can select from four discrete phase actions at each decision step: a North–South Green phase, an East–West Green phase, an extension of the current phase, or a switch to the next logical phase. The unknown transition dynamics of the environment are handled entirely by the microscopic simulation.

**Reward shaping: non-linear queue penalty.** The reward mechanism is a critical component of our methodology. The reward at each timestep is a delta-based composite signal designed to penalize deteriorating traffic conditions and reward improvement. It negatively weights the change in cumulative vehicle waiting time and the change in total queue length, while positively rewarding the per-step throughput, defined as the number of vehicles successfully clearing the intersection.

To address the critical issue of queue spillback, we fundamentally modified the traditional linear reward model to include an aggressive quadratic penalty. This penalty activates only when a queue exceeds an established capacity threshold, which was set to ten vehicles in our experiments. Once activated, the penalty scales quadratically with the number of excess vehicles (see Fig. 1). This non-linear scaling forces the RL agent to immediately prioritize queue clearing, effectively preventing catastrophic intersection gridlock. For Q-Learning and PPO, the raw reward was clipped to [−1, 1] to improve training stability.

**Simulation environment (SUMO and TraCI).** The Simulation of Urban Mobility (SUMO)¹⁶ is an open-source, continuous-time, microscopic road traffic simulator. Vehicles are modeled as individual agents following the Krauss car-following model¹⁷. We interact with SUMO in real-time via the Traffic Control Interface (TraCI), a TCP-based API that allows the reinforcement learning agents to extract state information and inject phase commands dynamically.

**Asymmetric traffic demand modeling.** Unlike simplistic synthetic environments that utilize perfectly balanced flows, our network routes are programmed with highly asymmetric, stochastic traffic demand designed to represent heterogeneous urban traffic conditions. For instance, incoming flow vectors range dramatically from heavy main arterial traffic (nearly five hundred vehicles per hour) to lighter cross-street turning flows (around sixty vehicles per hour). These rates employ randomized depart speeds and positions to prevent artificial cyclic clustering. This design forces the RL agents to respond to genuine, unpredictable traffic wave propagation rather than memorizing a fixed periodicity. Each episode simulates one hour of traffic, discretized into decision steps of five seconds each.

**Tabular Q-Learning.** Q-Learning¹⁸ is a model-free, off-policy algorithm that maintains a lookup table mapping discretized states and actions to expected future rewards. At each step, after observing the transition, the table is updated using the standard temporal difference learning rule, incorporating the observed reward and the discounted maximum estimated future value. For this expanded study, the Q-Learning budget was extended to ten thousand episodes to ensure deep exploration of the asymmetric traffic permutations.

**Deep Q-Network with Double and Dueling extensions.** Our DQN agent⁴ represents the Q-function as a deep neural network. The implementation incorporates three key enhancements. First, experience replay is utilized to store transitions in a massive buffer, from which random minibatches are sampled to break temporal correlations in the training data. Second, Double DQN¹³ decouples action selection from action evaluation to mitigate the overestimation bias that naturally occurs in standard Q-learning. Finally, a Dueling Architecture¹² factorizes the network output into a state-value stream and an advantage stream. These streams are combined to estimate the final Q-values, which empirically accelerates convergence in traffic settings by allowing the agent to learn the value of a state independent of the specific action taken.

**Proximal Policy Optimization.** PPO⁶ is an advanced on-policy actor-critic algorithm. Because on-policy methods are notoriously sample-inefficient, we dramatically expanded the training budget to two million timesteps in this updated study. PPO uses Generalized Advantage Estimation (GAE)¹⁹ to compute the advantage function, effectively balancing bias and variance in the policy gradient updates. The core mechanism of PPO restricts the magnitude of policy updates via a clipped surrogate objective, preventing the agent from taking destructively large optimization steps:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\epsilon,\ 1+\epsilon)\hat{A}_t\right)\right] \qquad (1)$$

**Experimental setup.** Each agent is trained independently on the same SUMO simulation with stochastic traffic seeding during training to encourage generalization. Training budgets were selected according to the convergence characteristics and sample requirements of each algorithm rather than enforcing an identical episode count. Training reward convergence is illustrated in Fig. 3; note that the horizontal axis represents normalized training progress, so budgets are not directly comparable across algorithms. The per-episode evaluation variance after training is reported in Fig. 5. Table 1 summarizes key implementation details.

**Table 1** | Agent implementation summary.

| Property | Q-Learning | DQN | PPO |
|:---|:---|:---|:---|
| Framework | Custom (tabular) | Custom PyTorch | Stable-Baselines3 |
| Architecture | Q-Table | Dueling MLP | Actor-Critic MLP |
| State space | 18-dim discrete | 8-dim continuous | 8-dim continuous |
| Action space | 4 discrete | 4 discrete | 4 discrete |
| Training budget | 10,000 episodes | 1,000 episodes | 2,000,000 steps |
| Reward clip | Clipped [−1, 1] | Unclipped | Clipped [−1, 1] |
| Key technique | Epsilon-greedy | Exp. Replay + PER | Clipped surrogate |

Following training, all agents are evaluated under a deterministic greedy policy over ten full evaluation episodes at a fixed random seed, ensuring identical stochastic traffic arrival patterns across all methods.

---

## Results and analysis

**Primary traffic performance metrics.** Table 2 presents the aggregate traffic metrics averaged across all ten evaluation episodes. Lower values indicate better traffic management performance. Because the evaluation was conducted across ten paired evaluation episodes with identical stochastic traffic arrival patterns across all agents, we computed 95% confidence intervals using a paired t-test for the performance differences. The performance improvements over the fixed-time baseline were found to be statistically significant for Q-Learning (*t*(9) = 15.3, *p* < 0.001, 95% CI [45.1, 62.1]) and for DQN (*t*(9) = 17.8, *p* < 0.001, 95% CI [46.3, 60.5]). We plan to increase the number of evaluation episodes across multiple random seeds in future work to further strengthen statistical significance.

**Table 2** | Traffic performance benchmarks (10 episodes, seed 42). Lower wait time and queue length indicate better performance.

| Algorithm | Mean Wait Time (s) | Mean Queue Length | Reward Std. Dev. | Wait Δ vs. Baseline | Queue Δ vs. Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| Q-Learning (Tabular) | 519.0 | 26.5 | High | −9.9% | −2.8% |
| DQN (Double+Dueling) | 522.2 | 26.7 | Low | −9.3% | −2.1% |
| Fixed-Time Baseline | 575.8 | 27.3 | Low | — | — |
| PPO (Actor-Critic) | 593.2 | 30.2 | Very High | +3.0% | +10.7% |

Additionally, Table 3 provides a dedicated model evaluation breakdown, highlighting the relative percentage improvements of each agent over the fixed-time baseline. Figure 5 visualizes the per-episode waiting times, highlighting the low variance of the DQN agent.

**Table 3** | Model evaluation: relative improvement versus baseline.

| Algorithm | Wait Time Reduction | Queue Length Reduction |
|:---|:---:|:---:|
| Q-Learning (Tabular) | 9.9% | 2.8% |
| DQN (Double+Dueling) | 9.3% | 2.1% |
| PPO (Actor-Critic) | −3.0% (Degradation) | −10.7% (Degradation) |

**The impact of non-linear queue penalties.** Comparing these results to earlier linear-reward baselines, the introduction of the quadratic queue penalty had a profound impact. Both value-based methods successfully learned to preemptively switch phases when queues approached the capacity threshold. As illustrated in Fig. 7, the agents intelligently adapted their phase selection frequency to match the asymmetric demand. This resulted in a highly robust ten percent reduction in waiting time over the fixed-time controller, vastly outpacing previous attempts in similar environments.

**PPO convergence challenges in discrete action spaces.** Despite quadrupling the training budget to two million steps, PPO continued to struggle with the highly asymmetric, discrete-phase traffic environment, ultimately underperforming the static baseline (see Fig. 6). PPO exhibited the highest reward standard deviation (±28.08) across runs. This strongly indicates that standard policy-gradient methods remain unstable when mapping continuous surrogate objectives to hard, discrete traffic light states. While tuning hyperparameters such as the entropy coefficient (*c*₂ = 0.01) can encourage broader exploration, our findings suggest that value-based architectures (Q-Learning, DQN) are fundamentally better suited for discrete traffic phase selection under stochastic demand.

**Impact of Prioritized Experience Replay on DQN.** The reported DQN results incorporate Prioritized Experience Replay (PER)²⁰, which preferentially samples high-error transitions from the replay buffer. Our empirical analysis confirms that PER significantly accelerates off-policy convergence by focusing the network's updates on the most unexpected traffic dynamics. Furthermore, the inclusion of PER contributed to DQN achieving the lowest evaluation variance (±4.53) among all tested algorithms, indicating a highly robust and stable policy.

**Table 4** | DQN ablation: impact of Prioritized Experience Replay (1,000 episodes). Wait Δ is relative to DQN without PER; baseline fixed-time wait is 575.8 s.

| Architecture | Final Wait Time (s) | Wait Δ (vs. No PER) | Convergence Speed |
|:---|:---:|:---:|:---:|
| DQN (No PER) | 538.4 | — | ~800 episodes |
| DQN (with PER) | 522.2 | −3.0% | ~500 episodes |

Table 4 provides an ablation study comparing DQN with and without PER. The addition of PER improved the final waiting time by 3.0% and significantly accelerated convergence, reaching optimal policies in approximately 500 episodes compared to 800 episodes without PER.

**Limitations.** The evaluation is limited to a single four-way intersection and therefore does not capture coordination effects arising in multi-intersection networks. Traffic demand is generated synthetically within SUMO rather than derived from real-world traffic detector data. Furthermore, evaluation is performed over ten episodes under a fixed random seed, limiting the statistical generalizability of the reported results. Future experiments should evaluate multiple network topologies, demand distributions, and random seeds.

**Future work.** The current framework establishes a rigorous single-intersection baseline and successfully integrates advanced off-policy mechanisms like Prioritized Experience Replay (PER) for rapid convergence. Several high-impact extensions remain planned. First, *multi-intersection MARL*: extending the unified environment to a grid network of multiple intersections. Scaling this single-intersection approach to multi-agent settings introduces significant challenges, particularly environmental non-stationarity as agents adapt simultaneously, and the multi-agent credit assignment problem. We aim to address these by deploying cooperative agents using decentralized algorithms like QMIX²¹ or MAPPO²², leveraging the established quadratic queue penalty to prevent cascading network-wide gridlock. Second, *adaptive reward shaping*: dynamically scaling the non-linear queue penalty thresholds based on temporal factors, such as time-of-day rush hour conditions, to further optimize throughput dynamically.

---

## Conclusions

This work presented a highly-tuned empirical comparison of Tabular Q-Learning, Double Dueling Deep Q-Networks, and Proximal Policy Optimization applied to adaptive traffic signal control under asymmetric, realistic traffic flows. By introducing a novel quadratic penalty for long queues, and significantly extending the algorithmic training budgets, we demonstrated that RL agents can achieve highly stable, robust traffic management. Q-Learning reduced average vehicle waiting times by 9.9% and DQN by 9.3%. Enhanced by Prioritized Experience Replay, DQN demonstrated superior consistency and the lowest variance across all models (±4.53). Conversely, PPO struggled to overcome the discrete action mapping, exhibiting high variance and slightly underperforming the baseline.

Within the evaluated single-intersection environment, the value-based methods demonstrated better performance and lower variance than PPO. Several high-impact extensions are planned for future work, including extending to a grid network of intersections controlled by cooperative agents using MAPPO, and introducing adaptive reward shaping for dynamic traffic conditions.

---

## Declarations

**Acknowledgements.** We extend our gratitude to the developers of Eclipse SUMO for microscopic traffic simulation, Gymnasium (Farama Foundation) for standardized RL API design, and PyTorch for accelerating deep learning models.

**Author contributions.** V.D. implemented the Q-Learning agent, conducted evaluation, and contributed to documentation. P.C. implemented the PPO agent and designed the unified benchmark framework. S.K. implemented the DQN agent with Double/Dueling extensions and PER. O.D. contributed to experimental design and analysis. Y.D.P. supervised the project. All authors reviewed the manuscript.

**Data availability.** The source code, SUMO network configuration files, and trained model weights are publicly available at https://github.com/47combinator/Multi-Agent-Reinforcement-Learning-for-Smart-Traffic-Control.

**Competing interests.** The authors declare no competing interests.

---

## References

1. INRIX Research. INRIX Global Traffic Scorecard. INRIX Inc., Technical Report (2023).
2. Wiering, M. A. Multi-agent reinforcement learning for traffic light control. In *Proc. 17th Int. Conf. Machine Learning (ICML)*, 1151–1158 (2000).
3. Thorpe, T. L. Vehicle traffic light control using SARSA. Master's thesis, Colorado State University (1997).
4. Mnih, V. *et al.* Human-level control through deep reinforcement learning. *Nature* **518**, 529–533 (2015).
5. Wei, H. *et al.* IntelliLight: A reinforcement learning approach for intelligent traffic light control. In *Proc. 24th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2496–2505 (2018).
6. Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. Proximal policy optimization algorithms. Preprint at https://arxiv.org/abs/1707.06347 (2017).
7. Liang, X., Du, X., Wang, G. & Han, Z. A deep reinforcement learning network for traffic light cycle control. *IEEE Trans. Veh. Technol.* **68**, 1243–1253 (2019).
8. Hunt, P. B., Robertson, D. I., Bretherton, R. D. & Winton, R. I. SCOOT—a traffic responsive method of coordinating signals. TRRL Laboratory Report 1014, Transport Research Laboratory (1981).
9. Lowrie, P. R. SCATS, Sydney coordinated adaptive traffic system: a traffic responsive method of controlling urban traffic. Roads and Traffic Authority, Sydney, Australia (1990).
10. Webster, F. V. Traffic signal settings. Road Research Technical Paper No. 39, HMSO, London (1958).
11. Abdulhai, B., Pringle, R. & Karakoulas, G. J. Reinforcement learning for true adaptive traffic signal control. *J. Transportation Engineering* **129**, 278–285 (2003).
12. Wang, Z. *et al.* Dueling network architectures for deep reinforcement learning. In *Proc. 33rd Int. Conf. Machine Learning (ICML)*, 1995–2003 (2016).
13. van Hasselt, H., Guez, A. & Silver, D. Deep reinforcement learning with double Q-learning. In *Proc. 30th AAAI Conf. Artificial Intelligence*, 2094–2100 (2016).
14. Chu, T., Wang, J., Codecà, L. & Li, Z. Multi-agent deep reinforcement learning for large-scale traffic signal control. *IEEE Trans. Intell. Transp. Syst.* **21**, 1086–1095 (2020).
15. Zhang, R. *et al.* Expressa: Adaptive traffic signal control with transformer-based reinforcement learning. In *Proc. IEEE Intelligent Transportation Systems Conference (ITSC)* (2023).
16. Lopez, P. A. *et al.* Microscopic traffic simulation using SUMO. In *Proc. 21st IEEE Int. Conf. Intelligent Transportation Systems (ITSC)*, 2575–2582 (2018).
17. Krauss, S., Wagner, P. & Gawron, C. Metastable states in a microscopic model of traffic flow. *Phys. Rev. E* **55**, 5597–5602 (1997).
18. Watkins, C. J. C. H. & Dayan, P. Q-learning. *Machine Learning* **8**, 279–292 (1992).
19. Schulman, J., Moritz, P., Levine, S., Jordan, M. I. & Abbeel, P. High-dimensional continuous control using generalized advantage estimation. In *Proc. 4th Int. Conf. Learning Representations (ICLR)* (2016).
20. Schaul, T., Quan, J., Antonoglou, I. & Silver, D. Prioritized experience replay. In *Proc. 4th Int. Conf. Learning Representations (ICLR)* (2016).
21. Rashid, T. *et al.* QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. In *Proc. 35th Int. Conf. Machine Learning (ICML)*, 4295–4304 (2018).
22. Yu, C. *et al.* The surprising effectiveness of PPO in cooperative multi-agent games. In *Advances in Neural Information Processing Systems (NeurIPS)* **35** (2022).

---

## Affiliations

**Vyankatesh Dawale, Pratyush Chaudhari** — Department of Computer Science, Vishwakarma University (VU), Pune, India

**Saim Kotkar, Om Dangi** — Department of Artificial Intelligence & ML, Vishwakarma University (VU), Pune, India

**Yogita D. Patil** — Assistant Professor, Vishwakarma University (VU), Pune, India. yogita.patil1@vupune.ac.in

---

## Figure captions

**Fig. 1** | Reward function behaviour as a function of total queue length. The aggressive quadratic penalty curve activates rapidly when the queue exceeds the threshold, forcing the RL agent to prioritize queue clearing and preventing catastrophic intersection gridlock.

**Fig. 2** | Closed-loop SUMO–TraCI–RL agent architecture. The Traffic Environment wrapper mediates state extraction and phase command injection via the TraCI TCP interface at every decision step.

**Fig. 3** | Training reward convergence shown against normalized training progress; training budgets differ across algorithms.

**Fig. 4** | Benchmark comparison: (**a**) Mean waiting time and (**b**) mean queue length across all agents and the fixed-time baseline over 10 evaluation episodes. Lower is better.

**Fig. 5** | Per-episode waiting times across 10 evaluation runs. DQN exhibits the lowest variance, confirming robustness of its learned policy.

**Fig. 6** | Representative simulated intersection snapshot at peak congestion. Green circle indicates the active traffic light phase (N–S Green). Vehicle rectangles illustrate queue accumulation patterns under heavy asymmetric demand.

**Fig. 7** | Phase selection frequency (N–S vs. E–W Green) for each agent over the 10 evaluation episodes. Value-based agents (Q-Learning, DQN) allocate up to 85% of time to the dominant N–S phase, correctly learning the asymmetric demand profile; the fixed-time controller splits equally.
