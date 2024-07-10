from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra

class Config:
    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Config class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing configuration information for the algorithm.
        """
        self.mode = None
        self.device = None
        self.paths = {}
        self.parameters = {}
        
        # Initialize Hydra to load configuration files
        @hydra.main(config_path="config", config_name="config")
        def hydra_init(cfg: DictConfig):
            self.mode = cfg.mode
            self.device = cfg.hyperparameters.device
            self.paths["io"] = Path(cfg.output_path)
            self.io = OmegaConf.to_container(cfg, resolve=True)
            self.configs = {key: Path("/home/iheb/Documents/project/project_algoz_pfe", value) for key, value in self.io.items() if "config_" in key}
            for key, value in self.configs.items():
                self.paths[key] = value
                self.parameters[key] = OmegaConf.to_container(OmegaConf.load(value), resolve=True)

        # Call Hydra initialization function
        hydra_init()

# Example usage
if __name__ == "__main__":
    config = Config()
    print(config.mode)
    print(config.device)
    print(config.paths)
    print(config.parameters)
