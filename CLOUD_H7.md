# H7.3 Cloud runner

This branch contains the minimal reproducible runner for the H7.3 online PPO confirmation. It executes one paired seed per Cloud.ru job and writes all logs, statuses, progress events, and result artifacts to `/home/jovyan/rl_muon/h7_online_cloud_r1`.

The scientific comparison is fixed to `raw_muon`, `own_polar_d01`, and `own_polar_d1`. Each route uses an independent trajectory, 20 PPO updates, and evaluations at updates 0, 5, 10, 15, and 20.

