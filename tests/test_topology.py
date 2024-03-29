from mucus_rust.config import Config
from mucus_rust.topology import Topology

cfg_path = "tests/data/connected-S-mesh/parameters/param_connected-S-mesh.toml"
cfg = Config.from_toml(cfg_path)
top = Topology(cfg)

# check if every tag index could be reached
n = cfg.number_of_beads
for i in range(n):
    for j in range(n):
        n, m = top.get_tag_indices(i, j)


