import os
import toml
import numpy as np
import rust_mucus as rmc
import h5py
from .config import Config
# from config import Config
from typing import Optional
from pathlib import Path
from io import StringIO
from tqdm import tqdm
from enum import Enum
from dataclasses import dataclass 
import matplotlib.pyplot as plt


class Filetypes(Enum):
    
    TrajectoryGro      = ("snapshots",           "snap",        ".gro" , "trajectory_gro")     # Trajectory          = "trajectory"
    Config             = ("configs",             "cfg",         ".toml", "config")     # TrajectoryGro       = "trajectory_gro"
    Parameters         = ("parameters",          "param",       ".toml", "parameters")     # Forces              = "forces"
    Tags               = ("parameters",          "tags",        ".npy",  "tags")
    Bonds              = ("parameters",          "bonds",       ".npy",  "bonds")
    InitPos            = ("initial_positions",   "xyz",         ".npy" , "init_pos")     # Distances           = "distances"
    Rdf                = ("results",             "rdf",         ".hdf5", "rdf")     # Rdf                 = "rdf"
    StructureFactor    = ("results",             "Sq",          ".hdf5", "structure_factor")     # StructureFactor     = "structure_factor"
    StructureFactorRdf = ("results",             "Sq_rdf",      ".hdf5", "structure_factor_rdf")     # StructureFactorRdf  = "structure_factor_rdf"
    StressTensor       = ("results",             "sigma",       ".hdf5", "stress_tensor")     # StressTensor        = "stress_tensor"
    Msd                = ("results",             "msd",         ".hdf5", "msd")     # Msd                 = "msd"
    Trajectory         = ("results",             "traj",        ".hdf5", "trajectory")     # Results             = "results"
    Forces             = ("results",             "forces",      ".hdf5", "forces")     # Config              = "config"
    Distances          = ("results",             "distances",   ".hdf5", "distances")     # Parameters          = "parameters"
    Results            = ("results",             "results",     ".hdf5", "results")     # InitPos             = "init_pos"

    def __init__(self, subdir, prefix, suffix, key):
        self._subdir = subdir
        self._prefix = prefix
        self._suffix = suffix
        self._key    = key
    
    @property
    def subdir(self):
        return self._subdir
    
    @property
    def prefix(self):
        return self._prefix

    @property
    def suffix(self):
        return self._suffix
    
    @property
    def key(self):
        return self._key

class ParametersKeys(Enum):
    ParticleRadius      = "r_particles"
    ParticleCharge      = "q_particles"
    Mobility            = "mobilities"
    ForceConstant       = "force_constants"
    LjEpsilon           = "epsilon_LJ"
    LjSigma             = "sigma_LJ"
    BondTable           = "bond_table"
    Bonds               = "bonds"
    BondDistance        = "r0_bonds"
    Tags                = "tags"
    TagActiveParticle   = "tag_active_particle"


class ConfigKeys(Enum):
    Steps               = "steps"
    Stride              = "stride"
    NumParticles        = "n_particles"
    Timestep            = "timestep"
    LengthScaleIn_nm    = "r0_nm"
    LjCutoff            = "cutoff_LJ"
    BjerrumLength       = "lB_debye"
    SaltConcentration   = "c_S"
    DebyeCutoff         = "cutoff_debye"
    BoxLength           = "lbox"
    Pbc                 = "pbc"
    PbcCutoff           = "cutoff_pbc"
    WriteTraj           = "write_traj"
    Cunksize            = "chunksize"
    WriteForces         = "write_forces"
    WriteDistances      = "write_distances"
    Cwd                 = "cwd"
    NameSys             = "name_sys"
    DirSys              = "dir_sys"
    SimulationTime      = "simulation_time"
    UsePotBond          = "use_pot_bond"
    UsePotWCA           = "use_pot_WCA"
    UsePotDebye         = "use_pot_debye"

class NatrualConstantsSI(Enum):
    kB = 1.380649e-23   # m^2 kg s^-2 K^-1  Boltzmann constant
    T = 300             # K                 Temperature
    eta_w = 8.53e-4     # Pa*s              Water viscosity
    
# Enum ForceType 
#     None
#     Harmonic{k: f64},
#     DoubleWell{k: f64, a: f64},
#     LinearInterpolation{values: Array<f64>, fe: Array<f64>},
    

# func get_force(force_type: ForceType, r: f64) -> f64 {
#     match force_type {
#         ForceType::None => {
#             0.0
#         },
#         ForceType::Harmonic{k_harm} => {
#             -k_harm * r
#         },
#         ForceType::DoubleWell{k, a} => {
#             -k*(r-a)*(r+a)
#         },
#         ForceType::LinearInterpolation{values, fe} => {
#             -k*(r-a)*(r+a)
#         },
#     }
# }
    
def create_fname(dir_sys, name_sys, filetype: Filetypes, overwrite=False):
    base = Path(dir_sys) / Path(filetype.subdir)
    fname = base / Path(f"{filetype.prefix}_{name_sys}{filetype.suffix}")
    
    if not overwrite and fname.exists():
        raise FileExistsError(f"File {fname} already")
    
    if not os.path.exists(fname.parent):
        os.makedirs(fname.parent)
    
    return fname

def get_path(Config: Config, filetype: Filetypes):
    
    dir_sys = Config.dir_sys
    name_sys = Config.name_sys
    
    fname = create_fname(dir_sys, name_sys, filetype, overwrite=True)
    
    return fname

def get_filetype_from_key(key: str):
    for filetype in Filetypes:
        if filetype.key == key:
            return filetype
    raise ValueError(f"No filetype found with key '{key}'")
    
def _validate_frame_range(frame_range, n_frames):
    if frame_range is None:
        frame_range = [0, n_frames]
    if frame_range[0] == None:
        frame_range[0] = 0
    if frame_range[1] == None:
        frame_range[1] = n_frames
    if frame_range[1] > n_frames:
        frame_range[1] = n_frames
        print(f"Warning: Frame range exceeds trajectory length. Setting frame_range[1] to {n_frames}")
    if frame_range[0] < -n_frames:
        raise ValueError(f"Frame range is not valid: index 0 exceeds trajectory length of {n_frames} frames")
    if frame_range[0] < 0:
        frame_range[0] = n_frames + frame_range[0]
    if frame_range[1] <= -n_frames:
        raise ValueError(f"Frame range is not valid: index 1 exceeds trajectory length of {n_frames} frames")
    if frame_range[1] < 0:
        frame_range[1] = n_frames + frame_range[1]
    if frame_range[0] >= frame_range[1]:
        raise ValueError("Frame range is not valid: index 0 is larger than index 1")
    
    return frame_range

def convert_trajectory(
    config: Optional[Config],
    trajectory: np.ndarray,
    fname: str = None,
    trajtype: str = "gro",
    overwrite: bool = False):
    
    """
    Converts trajectory (3d nd.array) to a specified fieltype
    (For now only type "gro" is possible)
    """
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    if len(trajectory.shape) != 3:
        raise ValueError("trajectory must be a 3d array")
    if trajtype == "gro":
        if fname is None:
            fname = get_path(config, filetype=Filetypes.TrajectoryGro, overwrite=overwrite)
        n_atoms = len(trajectory[0])
        for i, frame in enumerate(trajectory):
            _write_frame_gro(n_atoms, frame, i, fname)


def traj_h5_to_gro(
    config: Optional[Config],
    frame_range: list = None, 
    stride: int = 1,
    fname: str = None,
    overwrite: bool = False):
    """
    Saves trajectory from h5 file as a gro file.
    """
    
    if isinstance(config, str):
         config = Config.from_toml(config)
    
    # get number of frames
    n_frames = get_number_of_frames(config)
    
    # get traj path
    if fname is None:
        fname_traj = get_path(config, filetype=Filetypes.TrajectoryGro)
    
        if frame_range != None:
            frame_range = _validate_frame_range(frame_range, n_frames)
            fname_traj = fname_traj.split(".gro")[0] + f"_frame_{frame_range[0]}_to_{frame_range[1]}_stride{stride}.gro"       
    else:
        fname_traj = fname
            
    # handle frame range input
    frame_range = _validate_frame_range(frame_range, n_frames)
    
    # get h5py filename
    fname_h5 = get_path(config, filetype=Filetypes.Trajectory)
    
    if not os.path.exists(fname_h5):
         raise FileNotFoundError(f"Trajectory file '{fname_h5}' not found")
    
    with h5py.File(fname_h5, "r") as h5_file:
        trajectory = h5_file[Filetypes.Trajectory.key][frame_range[0]:frame_range[1]:stride]
        convert_trajectory(config, trajectory, fname=fname_traj, overwrite=overwrite, trajtype="gro")


    
def _write_frame_gro(n_atoms, coordinates, time, fname, comment="trajectory", box=None, precision=3):
    f = open(fname, "a")
    comment += ', t= %s' % time
    varwidth = precision + 5
    fmt = '%%5d%%-5s%%5s%%5d%%%d.%df%%%d.%df%%%d.%df' % (
            varwidth, precision, varwidth, precision, varwidth, precision)
    lines = [comment, ' %d' % n_atoms]
    for i in range(n_atoms):
        lines.append(fmt % (i+1, "HET", "CA", i+1,
                            coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]))
    lines.append('%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f%10.5f' % (0,0,0,0,0,0,0,0,0))
    f.write('\n'.join(lines))
    f.write('\n')
    f.close()

def get_number_of_frames(config: Optional[Config]):
    return int(config.steps/config.stride)
    

def delete_system(
    config: Optional[Config],
    only_results: bool = False,
    exceptions: list = []):
    """
    Deletes every file of a scpecified system. Use with care.
    """
    
    if isinstance(config, str):
        config = Config.from_toml(config)
    
    if only_results:
        exceptions = ["config", "init_pos", "parameters"] + list(exceptions)
        print(exceptions)

 
    proceed = "y" == input(f"Are you sure you want to delete system {config.name_sys} and all its related files (exceptions: {exceptions})? (y/n) ")
    if proceed:
        dir_dict = _dir_dict()
        for key in dir_dict.keys():
            if key not in exceptions:
                fname = get_path(config, filetype=key)
                if os.path.exists(fname):
                    os.remove(fname)
                    print(f"File {fname} deleted.")
        print(f"System {config.name_sys} deleted.\n")
    else:
        print("Aborted.\n")

def get_timestep_seconds(config: Optional[Config],
                        monomer_tag = 0):
        """
        returns the timestep of the current trajectory in seconds (stride and cfg-timestep included)
        """
        if isinstance(config, str):
            config = Config.from_toml(config)
            
        params = toml.load(open(get_path(config, Filetypes.Parameters), encoding="UTF-8"))
        mobility = np.array(params["mobilities"])
        
        mu = mobility[monomer_tag]      # mobility of the system in reduced units
        a = 1e-9*config.r0_nm           # m, reduced legth scale: PEG monomere radius
        r = 1*a                         # m, particle radius

        eta_w = 8.53e-4 # Pa*s
        kB = 1.380649e-23 # m^2 kg s^-2 K^-1
        T = 300 # K
        mu_0 = kB*T/(6*np.pi*eta_w*r*a**2) 

        dt_step = mu/mu_0
        # multiply stepwise timestep with the simulation stride 
        dt = config.stride*dt_step*config.timestep # s
        
        return dt
    

def plot_box(positions, l_box, centered=False, nchains=None, title=None):
    # todo add bond list as input argument
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # for i in range(len(positions)):
    #    ax.scatter(positions[i, 0],positions[i, 1], positions[i, 2], c="b")
    
    ax.scatter(*positions.T, c="b")
    
    
    xy = np.array(((0,0),
                    (l_box, 0),
                    (l_box, l_box),
                    (0, l_box),
                    (0,0)), dtype="float64")
    
    if nchains is not None:
        bpc = int(len(positions)/nchains)
        for i in range(nchains):
            ax.plot(positions[i*bpc:(i+1)*bpc, 0], positions[i*bpc:(i+1)*bpc, 1], positions[i*bpc:(i+1)*bpc, 2])
    
    if centered==True:
        xy -= l_box/2

        ax.plot(xy[:, 0], xy[:, 1], zs=-l_box/2, zdir='z', c="r")
        ax.plot(xy[:, 0], xy[:, 1], zs=l_box/2, zdir='z', c="r")
        ax.plot(xy[:, 0], xy[:, 1], zs=-l_box/2, zdir='y', c="r")
        ax.plot(xy[:, 0], xy[:, 1], zs=l_box/2, zdir='y', c="r")
        
    if centered == False:
        ax.plot(xy[:, 0], xy[:, 1], zs=0, zdir='z', c="r")
        ax.plot(xy[:, 0], xy[:, 1], zs=l_box, zdir='z', c="r")
        ax.plot(xy[:, 0], xy[:, 1], zs=0, zdir='y', c="r")
        ax.plot(xy[:, 0], xy[:, 1], zs=l_box, zdir='y', c="r")
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    if title is not None:
        ax.set_title(title)
    
    plt.show()