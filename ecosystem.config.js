// PM2 ecosystem config for radar miners.
// Adds restart delay with exponential backoff to avoid tight crash loops
// when DNS or network is temporarily unavailable.
//
// Usage:
//   pm2 start ecosystem.config.js
//   pm2 start ecosystem.config.js --only miner_1

module.exports = {
  apps: [
    {
      name: "miner_1",
      script: "./start_miner.sh",
      args: "--agent_dir agents/frontier_sniper/ --wallet.name miner1 --netuid 87 --subtensor.network finney",
      interpreter: "bash",
      restart_delay: 10000,
      max_restarts: 50,
      exp_backoff_restart_delay: 1000,
    },
    {
      name: "miner_2",
      script: "./start_miner.sh",
      args: "--agent_dir agents/bucket_specialist/ --wallet.name miner2 --netuid 87 --subtensor.network finney",
      interpreter: "bash",
      restart_delay: 10000,
      max_restarts: 50,
      exp_backoff_restart_delay: 1000,
    },
    {
      name: "miner_3",
      script: "./start_miner.sh",
      args: "--agent_dir agents/pareto_hunter/ --wallet.name miner3 --netuid 87 --subtensor.network finney",
      interpreter: "bash",
      restart_delay: 10000,
      max_restarts: 50,
      exp_backoff_restart_delay: 1000,
    },
  ],
};
