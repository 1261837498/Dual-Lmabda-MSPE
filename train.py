import hydra
from omegaconf import DictConfig
import os
import torch
import json
import copy
from hydra import initialize, compose

def _cfg_get(cfg, key, default=None):
    return cfg[key] if key in cfg else default


def _split_file_list(value):
    """
    Supports:
    1) Hydra ListConfig: [a,b,c]
    2) Python list/tuple: [a,b,c]
    3) comma-separated string: "a,b,c"
    4) string representation of list: "['a', 'b', 'c']"
    5) single path string
    """
    if value is None:
        return []

    # Case 1: Hydra ListConfig or Python list/tuple
    if not isinstance(value, str):
        try:
            return [
                str(v).strip().strip("'").strip('"')
                for v in list(value)
                if str(v).strip()
            ]
        except TypeError:
            pass

    # Case 2: string
    value = str(value).strip()

    if not value:
        return []

    # Remove outer quotes
    value = value.strip("'").strip('"')

    # Case 3: string representation of list
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    # Case 4: comma-separated string
    if "," in value:
        return [
            v.strip().strip("'").strip('"')
            for v in value.split(",")
            if v.strip()
        ]

    # Case 5: single path
    return [value.strip().strip("'").strip('"')]


def _infer_context_from_filename(path):
    name = os.path.basename(str(path)).lower()

    scenario_type = "normal"
    if "peak" in name:
        scenario_type = "peak"
    elif "rain" in name:
        scenario_type = "rain"
    elif "event" in name:
        scenario_type = "event"
    elif "incident" in name:
        scenario_type = "incident"

    context = {
        "scenario_type": scenario_type,
        "demand_scale": 1.0,
        "travel_time_scale": 1.0,
        "event_level": 0.0,
        "weather_level": 0.0,
        "incident_level": 0.0,
        "affected_regions": [],
    }

    if scenario_type == "peak":
        context["demand_scale"] = 1.5
    elif scenario_type == "rain":
        context["demand_scale"] = 1.2
        context["travel_time_scale"] = 1.3
        context["weather_level"] = 1.0
    elif scenario_type == "event":
        context["event_level"] = 1.0
    elif scenario_type == "incident":
        context["travel_time_scale"] = 1.5
        context["incident_level"] = 1.0

    return context


def _load_scenario_context(path):
    """
    Load top-level JSON field 'context' if it exists.
    If it does not exist, infer context from filename.
    """
    context = _infer_context_from_filename(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_context = data.get("context", {})
        if isinstance(json_context, dict):
            context.update(json_context)

        if "scenario_id" in data:
            context["scenario_id"] = data["scenario_id"]

    except Exception as e:
        print(f"[WARN] Failed to load context from {path}: {e}")
        print("[WARN] Falling back to filename-inferred context.")

    return context


def _attach_context_to_env(env, scenario_file):
    env.scenario_file = scenario_file
    env.scenario_context = _load_scenario_context(scenario_file)
    return env


def setup_macro(cfg):
    from src.envs.sim.macro_env import Scenario, AMoD, GNNParser
    with open("src/envs/data/macro/calibrated_parameters.json", "r") as file:
        calibrated_params = json.load(file)

    cfg.simulator.cplexpath = cfg.model.cplexpath

    root_cfg = cfg
    sim_cfg = cfg.simulator
    city = sim_cfg.city

    scenario_file = _cfg_get(
        sim_cfg,
        "scenario_file",
        f"src/envs/data/macro/scenario_{city}.json"
    )

    scenario = Scenario(
        json_file=scenario_file,
        demand_ratio=calibrated_params[city]["demand_ratio"],
        json_hr=calibrated_params[city]["json_hr"],
        sd=sim_cfg.seed,
        json_tstep=sim_cfg.json_tsetp,
        tf=sim_cfg.max_steps,
    )
    env = AMoD(scenario, cfg=sim_cfg, beta=calibrated_params[city]["beta"])
    env = _attach_context_to_env(env, scenario_file)
    parser = GNNParser(env, T=sim_cfg.time_horizon, json_file=scenario_file, cfg=root_cfg)
    return env, parser

def setup_model(cfg, env, parser, device):
    model_name = cfg.model.name
    model_cfg = cfg.model
    input_size = getattr(parser, "feature_dim", _cfg_get(model_cfg, "input_size", 0))
    if model_name == "sac" or model_name =="cql":
        from src.algos.sac import SAC
        return SAC(env=env, input_size=input_size, cfg=model_cfg, parser=parser, device=device).to(device)
    elif model_name == "a2c":
        from src.algos.a2c import A2C
        return A2C(env=env, input_size=input_size, cfg=model_cfg, parser=parser, device=device).to(device)
    elif model_name == "bc":
        from src.algos.bc import BC
        return BC(env=env, input_size=input_size, cfg=model_cfg, parser=parser, device=device).to(device)
    else:
        from src.algos.registry import get_model
        model_class = get_model(model_name)
        model_kwargs = {
            "cplexpath": cfg.simulator.cplexpath,
            "directory": _cfg_get(model_cfg, "directory", cfg.simulator.directory),
            "T": cfg.simulator.time_horizon,
            "policy_name": model_name,
        }
        for key, value in model_cfg.items():
            if key not in model_kwargs:
                model_kwargs[key] = value
        return model_class(**model_kwargs)

def setup_env_pool(cfg):

    sim_name = cfg.simulator.name

    if sim_name == "macro":
        files = _split_file_list(_cfg_get(cfg.simulator, "scenario_files", None))
        if not files:
            return None

        pool = []
        for file_path in files:
            cfg_i = copy.deepcopy(cfg)
            cfg_i.simulator.scenario_file = file_path
            env_i, parser_i = setup_macro(cfg_i)
            pool.append((env_i, parser_i))

        print(f"[Context Pool] Loaded {len(pool)} macro scenarios.")
        return pool



    return None

def setup_dataset(cfg, env, device):
    from src.algos.sac import ReplayData
    with open(f"src/envs/data/macro/scenario_{cfg.simulator.city}.json", "r") as file:
        data = json.load(file)

    edge_index = torch.vstack(
        (
            torch.tensor([edge["i"] for edge in data["topology_graph"]]).view(1, -1),
            torch.tensor([edge["j"] for edge in data["topology_graph"]]).view(1, -1),
        )
    ).long()



    Dataset = ReplayData(device=device)
    Dataset.create_dataset(
        edge_index=edge_index,
        memory_path=cfg.model.data_path,
        rew_scale=cfg.model.rew_scale,
        size=cfg.model.samples_buffer,
    )

    return Dataset

def train(config):
    """
    for colab tutorial
    """

    with initialize(config_path="src/config"):
        cfg = compose(config_name="config", overrides= [f"{key}={value}" for key, value in config.items()])  # Load the configuration

    if cfg.simulator.name == "macro":
        env, parser = setup_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {cfg.simulator.name}")

    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = setup_model(cfg, env, parser, device)
    env_pool = setup_env_pool(cfg)
    if env_pool is not None:
        model.env_pool = env_pool
    model.wandb = None
    if _cfg_get(cfg.model, "wandb", False):
        import wandb
        config = {}
        for key in cfg.model.keys():
            config[key] = cfg.model[key]
        wandb = wandb.init(
            project="",
            entity="",
            config=config,
        )
        model.wandb = wandb

    if not hasattr(model, "learn"):
        print(f"Model '{cfg.model.name}' is a non-learning baseline; no training is required. Please run testing.py.")
        return

    model.learn(cfg)

def load_actor_weights(model, path):
    full_model_state = torch.load(f"ckpt/{path}.pth")

    actor_encoder_state = {
        k.replace("actor.", ""): v
        for k, v in full_model_state["model"].items()
        if "actor" in k
    }
    model.actor.load_state_dict(actor_encoder_state)
    return model

@hydra.main(version_base=None, config_path="src/config/", config_name="config")
def main(cfg: DictConfig):
    # Import simulator module based on the configuration
    simulator_name = cfg.simulator.name
    if simulator_name == "macro":
        env, parser = setup_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = setup_model(cfg, env, parser, device)
    env_pool = setup_env_pool(cfg)
    if env_pool is not None:
        model.env_pool = env_pool
    model.wandb = None
    if _cfg_get(cfg.model, "wandb", False):
        import wandb
        config = {}
        for key in cfg.model.keys():
            config[key] = cfg.model[key]
        wandb = wandb.init(
            project="",
            entity="",
            config=config,
        )
        model.wandb = wandb

    if not hasattr(model, "learn"):
        print(f"Model '{cfg.model.name}' is a non-learning baseline; no training is required. Please run testing.py.")
        return

    if "data_path" in cfg.model:
        Dataset = setup_dataset(cfg, env, device)
        model.learn(cfg, Dataset) #offline RL or BC
    else:
        model.learn(cfg) #online RL

if __name__ == "__main__":
    main()
