import os
import toml
import h5py
import gsd.fl
import numpy as np
# import cupy as cp
import matplotlib.pyplot as plt

from .config import Config
from typing import Optional
from pathlib import Path
from enum import Enum

# TODO add load_data(Config, Filetype) function

class Filetypes(Enum):
    # TODO: add dtype property?
    #                     subdir                 prefix          suffix      key        
    TrajectoryGro      = ("snapshots",           "snap",         ".gro" ,    "trajectory_gro")          
    Config             = ("configs",             "cfg",          ".toml",    "config")                  
    Parameters         = ("parameters",          "param",        ".toml",    "parameters")              
    Tags               = ("parameters",          "tags",         ".npy",     "tags")
    Bonds              = ("parameters",          "bonds",        ".npy",     "bonds")
    InitPos            = ("initial_positions",   "xyz",          ".npy" ,    "init_pos")                
    Rdf                = ("results",             "rdf",          ".hdf5",    "rdf")                     
    StructureFactor    = ("results",             "Sq",           ".hdf5",    "structure_factor")        
    StructureFactorRdf = ("results",             "Sq_rdf",       ".hdf5",    "structure_factor_rdf")    
    Virial             = ("results",             "virial",       ".hdf5",    "virial")
    StressTensor       = ("results",             "sigma",        ".hdf5",    "stress_tensor")
    StressTensorDirect = ("results",             "sigma_direct", ".npy",     "stress_tensor_direct")
    StressAcf          = ("results",             "sigma_acf",    ".npy",     "stress_tensor_acf")           
    Msd                = ("results",             "msd",          ".hdf5",    "msd")                     
    Trajectory         = ("results",             "traj",         ".hdf5",    "trajectory")              
    Forces             = ("results",             "forces",       ".hdf5",    "forces")                  
    Distances          = ("results",             "distances",    ".hdf5",    "distances")               
    TrajectoryGsd      = ("results",             "traj",         ".gsd",     "trajectory_gsd")          
    Results            = ("results",             "results",      ".hdf5",    "results")
    Filenames          = ("results",             "filenames",    ".toml",    "filenames")                 

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
    CalcVirial          = "calc_virial"
    StrideVirial        = "stride_virial"

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
    
def create_fname(dir_sys, name_sys, filetype: Filetypes, overwrite=False, comment=None):
    base = Path(dir_sys) / Path(filetype.subdir)
    if comment is None:
        fname = base / Path(f"{filetype.prefix}_{name_sys}{filetype.suffix}")
    else:
        fname = base / Path(f"{filetype.prefix}_{name_sys}_{comment}_{filetype.suffix}")
    
    if not overwrite and fname.exists():
        raise FileExistsError(f"File {fname} already")
    
    if not os.path.exists(fname.parent):
        os.makedirs(fname.parent)
    
    return fname

def get_path(Config: Config, filetype: Filetypes, comment=None):
    
    dir_sys = Config.dir_sys
    name_sys = Config.name_sys
    
    fname = create_fname(dir_sys, name_sys, filetype, overwrite=True, comment=comment)
    
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
            fname_traj = str(fname_traj).split(".gro")[0] + f"_frame_{frame_range[0]}_to_{frame_range[1]}_stride{stride}.gro"       
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
    return int(np.ceil(config.steps/config.stride))
    

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
    
def gsd_to_hdf5(cfg, filetype="all", overwrite=False, wrap=True, shift=[0.0, 0.0, 0.0]):
    
    if filetype == "all":
        calculate_positions = True
        calculate_forces = True
    elif filetype == Filetypes.Trajectory:
        calculate_positions = True
        calculate_forces = False
    elif filetype == Filetypes.Forces:
        calculate_positions = False
        calculate_forces = True
    
    shift = np.array(shift)
    
    gsd_chunkname_positions = "particles/position"
    gsd_chunkname_forces    = "particles/force"
    
    path_gsd = get_path(cfg, filetype=Filetypes.TrajectoryGsd)
    traj = gsd.fl.open(path_gsd, 'r')
    n_frames = traj.nframes
    
    if calculate_positions:
        positions_exist = traj.chunk_exists(frame=0, name=gsd_chunkname_positions)
        if not positions_exist:
            print("Warning: No position data in gsd trajectory!")
            # return
            
    if calculate_forces:
        forces_exist = traj.chunk_exists(frame=0, name=gsd_chunkname_forces)
        if not forces_exist:
            print("Warning: No force data in gsd trajectory!")
            # return
    
    if calculate_positions and positions_exist:
        fname_traj_out = get_path(cfg, filetype=Filetypes.Trajectory)
        
        # remove file first
        if overwrite and os.path.exists(fname_traj_out):
            os.remove(fname_traj_out)

        if not os.path.exists(fname_traj_out):
            # create trajectory hdf5
            #! TODO: check if n_frames is correct
            fh5_traj = h5py.File(fname_traj_out, 'w-')
            fh5_traj.create_dataset(Filetypes.Trajectory.key, 
                                    shape=(n_frames, cfg.n_particles, 3), 
                                    dtype="float32")

            # save gsd trajectory in hdf5, wrap positions for proper function of analysis scripts
            for i in range(n_frames):
                if wrap:
                    fh5_traj[Filetypes.Trajectory.key][i, :, :] = np.mod(traj.read_chunk(frame=i, name=gsd_chunkname_positions), cfg.lbox)
                else:
                    fh5_traj[Filetypes.Trajectory.key][i, :, :] = traj.read_chunk(frame=i, name=gsd_chunkname_positions) + shift
        else:
            print("Info: h5 file containing position data already exists and will not be converted")

        fh5_traj.close()
    
    
    if calculate_forces and forces_exist:
        fname_forces_out  = get_path(cfg, filetype=Filetypes.Forces)
        
        # remove file first
        if os.path.exists(fname_forces_out):
            if overwrite:
                os.remove(fname_forces_out)
        
        if not os.path.exists(fname_forces_out):
            # create trajectory hdf5 for forces
            fh5_forces = h5py.File(fname_forces_out, 'w-')
            fh5_forces.create_dataset(Filetypes.Forces.key, 
                                      shape=(n_frames, cfg.n_particles, 3), 
                                      dtype="float32")

            for i in range(n_frames):
                fh5_forces[Filetypes.Forces.key][i, :, :] = traj.read_chunk(frame=i, name=gsd_chunkname_forces)
        
        else:
            print("Info: h5 file containing force data already exists and will not be converted")
                
        fh5_forces.close()
        
    traj.close()
    

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
    
    if centered == True:
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
    

# """
# forces_cuda.py
# ==============
# GPU-accelerated drop-in replacement for get_forces_cell_linked_virial()
# using CuPy + a raw CUDA kernel.

# Requirements:
#     pip install cupy-cuda12x   # (or cupy-cuda11x depending on your CUDA version)

# Drop-in usage
# -------------
# Replace calls to the Rust function with:

#     from forces_cuda import get_forces_cell_linked_virial_cuda
#     get_forces_cell_linked_virial_cuda(
#         positions, tags, bond_offsets, bond_neighbors,
#         force_constants, bond_lengths, sigmas_lj, epsilons_lj,
#         charges, lB_debye, B_debye,
#         force_total, distances,
#         l_box, cutoff2,
#         n_particles, n_dim,
#         write_distances, use_force_bonded, use_force_lj, use_force_deb,
#         neighbour_cells_idx, head_array, list_array,
#         n_cells, n_neighbour_cells,
#         calc_virial, virial
#     )

# Bond list format change
# -----------------------
# The Rust function accepted a Python list-of-lists for bond_list.
# CUDA kernels require flat, fixed-length arrays. Convert once with:

#     bond_offsets, bond_neighbors = flatten_bond_list(bond_list)

# and reuse across time steps (the connectivity does not change).
# """

# import numpy as np
# import cupy as cp

# # ---------------------------------------------------------------------------
# # CUDA kernel source
# # ---------------------------------------------------------------------------
# # Parallelisation strategy
# # ------------------------
# # One CUDA thread per particle *i*.  Each thread iterates over neighbour
# # cells of the cell that contains i and accumulates forces/virial for i
# # (f_i) and atomically accumulates the Newton-3rd-law reaction onto j
# # (f_j).  Atomic adds on double-precision floats require compute capability
# # ≥ 6.0 (Pascal and newer); on older cards you can fall back to float32.
# #
# # This maps the original "loop over cells → loop over particles in cell →
# # loop over neighbour cells → loop over particles in neighbour cell" to
# # "one thread per particle → loop over neighbour cells → loop over
# # particles in neighbour cell", which gives good occupancy for large N.

# _KERNEL_SOURCE = r"""

# // atomicAdd for double is natively supported on CC ≥ 6.0.
# // For older hardware replace double -> float throughout.

# extern "C" __global__
# void forces_virial_kernel(
#     // inputs
#     const double* __restrict__ positions,        // (n_particles, n_dim)
#     const int*    __restrict__ tags,             // (n_particles,)
#     const int*    __restrict__ bond_offsets,     // (n_particles+1,)  CSR row pointers
#     const int*    __restrict__ bond_neighbors,   // (nnz,)            CSR column indices
#     const double* __restrict__ force_constants,  // (n_tags, n_tags)
#     const double* __restrict__ bond_lengths,     // (n_tags, n_tags)
#     const double* __restrict__ sigmas_lj,        // (n_tags, n_tags)
#     const double* __restrict__ epsilons_lj,      // (n_tags, n_tags)
#     const double* __restrict__ charges,          // (n_tags,)
#     double lB_debye,
#     double B_debye,
#     // outputs (pre-zeroed by caller)
#     double* __restrict__ force_total,            // (n_particles, n_dim)
#     double* __restrict__ distances,              // (n_particles, n_particles)
#     double* __restrict__ virial,                 // (n_dim, n_dim)
#     // cell-linked-list
#     const int*    __restrict__ head_array,       // (n_cells^3,)  flattened
#     const int*    __restrict__ list_array,       // (n_particles,)
#     const int*    __restrict__ neighbour_cells_idx, // (n_neighbour_cells, 3)
#     // scalars
#     double l_box,
#     double cutoff2,
#     int n_particles,
#     int n_dim,
#     int n_cells,
#     int n_neighbour_cells,
#     int n_tags,
#     // flags
#     int write_distances,
#     int use_force_bonded,
#     int use_force_lj,
#     int use_force_deb,
#     int calc_virial
# )
# {
#     // Each thread handles one particle i
#     int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i >= n_particles) return;

#     // ---- cell coordinates of particle i ----
#     int ci = (int)(positions[i * n_dim + 0] / l_box * n_cells);
#     int cj = (int)(positions[i * n_dim + 1] / l_box * n_cells);
#     int ck = (n_dim == 3) ? (int)(positions[i * n_dim + 2] / l_box * n_cells) : 0;
#     // clamp (floating-point rounding at boundary)
#     ci = min(max(ci, 0), n_cells - 1);
#     cj = min(max(cj, 0), n_cells - 1);
#     ck = min(max(ck, 0), n_cells - 1);

#     int ti = tags[i];

#     // local accumulators for force on i (avoid repeated global writes)
#     double fi[3] = {0.0, 0.0, 0.0};

#     // ---- iterate over neighbour cells ----
#     for (int n = 0; n < n_neighbour_cells; n++) {
#         int dni = neighbour_cells_idx[n * 3 + 0];
#         int dnj = neighbour_cells_idx[n * 3 + 1];
#         int dnk = neighbour_cells_idx[n * 3 + 2];

#         int nci = ((ci + dni) % n_cells + n_cells) % n_cells;
#         int ncj = ((cj + dnj) % n_cells + n_cells) % n_cells;
#         int nck = ((ck + dnk) % n_cells + n_cells) % n_cells;

#         int cell_flat = nci * n_cells * n_cells + ncj * n_cells + nck;
#         int j = head_array[cell_flat];

#         while (j != -1) {
#             if (i > j) {   // Newton's 3rd law: only compute each pair once
#                 double dir[3] = {0.0, 0.0, 0.0};
#                 double dist2  = 0.0;

#                 for (int d = 0; d < n_dim; d++) {
#                     double dd = positions[j * n_dim + d] - positions[i * n_dim + d];
#                     // minimum image convention
#                     dd -= l_box * round(dd / l_box);
#                     dir[d]  = dd;
#                     dist2  += dd * dd;
#                 }

#                 double dist = sqrt(dist2);

#                 if (write_distances) {
#                     distances[i * n_particles + j] = dist;
#                     distances[j * n_particles + i] = dist;
#                 }

#                 int tj = tags[j];

#                 // ---- check bond ----
#                 int bonded = 0;
#                 int bstart = bond_offsets[i];
#                 int bend   = bond_offsets[i + 1];
#                 for (int b = bstart; b < bend; b++) {
#                     if (bond_neighbors[b] == j) { bonded = 1; break; }
#                 }

#                 double force = 0.0;

#                 if (bonded && use_force_bonded) {
#                     double fc = force_constants[ti * n_tags + tj];
#                     double bl = bond_lengths   [ti * n_tags + tj];
#                     force += 2.0 * fc * (1.0 - bl / dist);
#                 }

#                 if (dist2 < cutoff2) {
#                     if (!bonded && use_force_lj) {
#                         double sig = sigmas_lj [ti * n_tags + tj];
#                         double eps = epsilons_lj[ti * n_tags + tj];
#                         double s6  = pow(sig, 6);
#                         double s12 = s6 * s6;
#                         double d8  = pow(dist, 8);
#                         double d14 = d8 * pow(dist, 6);
#                         force += -24.0 * eps * (2.0 * s12 / d14 - s6 / d8);
#                     }

#                     if (use_force_deb) {
#                         double qi = charges[ti];
#                         double qj = charges[tj];
#                         double ex = exp(-B_debye * dist);
#                         force += -qi * qj * lB_debye
#                                  * (1.0 + B_debye * dist)
#                                  * ex / (dist * dist * dist);
#                     }
#                 }

#                 // ---- accumulate forces ----
#                 for (int d = 0; d < n_dim; d++) {
#                     double fd = force * dir[d];
#                     fi[d] += fd;
#                     // Newton 3rd law reaction on j  (atomic because multiple
#                     // threads may update j simultaneously)
#                     atomicAdd(&force_total[j * n_dim + d], -fd);
#                 }

#                 // ---- virial  (atomic for same reason) ----
#                 if (calc_virial) {
#                     for (int a = 0; a < n_dim; a++) {
#                         for (int b = 0; b < n_dim; b++) {
#                             atomicAdd(&virial[a * n_dim + b],
#                                       force * dir[b] * dir[a]);
#                         }
#                     }
#                 }
#             }
#             j = list_array[j];
#         }
#     }

#     // ---- write local fi accumulator to global memory (no race for i) ----
#     for (int d = 0; d < n_dim; d++) {
#         force_total[i * n_dim + d] += fi[d];
#     }
# }
# """

# # Compile once at import time
# _module = cp.RawModule(code=_KERNEL_SOURCE, options=("--use_fast_math",))
# _kernel = _module.get_function("forces_virial_kernel")


# # ---------------------------------------------------------------------------
# # Helper: convert Python list-of-lists → CSR flat arrays (call once)
# # ---------------------------------------------------------------------------
# def flatten_bond_list(bond_list: list[list[int]]):
#     """
#     Convert the Rust-style list-of-lists bond representation to two flat
#     numpy arrays in CSR (compressed sparse row) format, ready for the GPU.

#     Parameters
#     ----------
#     bond_list : list of lists of int
#         bond_list[i] contains the indices of all particles bonded to i.

#     Returns
#     -------
#     bond_offsets   : np.ndarray, shape (n_particles+1,), dtype=int32
#     bond_neighbors : np.ndarray, shape (nnz,),           dtype=int32
#     """
#     offsets   = [0]
#     neighbors = []
#     for row in bond_list:
#         neighbors.extend(row)
#         offsets.append(offsets[-1] + len(row))
#     return (np.array(offsets,   dtype=np.int32),
#             np.array(neighbors, dtype=np.int32))


# # ---------------------------------------------------------------------------
# # Public API – drop-in replacement for the Rust pyfunction
# # ---------------------------------------------------------------------------
# def get_forces_cell_linked_virial_cuda(
#     positions,              # np.ndarray (n_particles, n_dim), float64
#     tags,                   # np.ndarray (n_particles,),       int/uint → cast to int32
#     bond_offsets,           # np.ndarray (n_particles+1,)  from flatten_bond_list()
#     bond_neighbors,         # np.ndarray (nnz,)            from flatten_bond_list()
#     force_constants,        # np.ndarray (n_tags, n_tags),     float64
#     bond_lengths,           # np.ndarray (n_tags, n_tags),     float64
#     sigmas_lj,              # np.ndarray (n_tags, n_tags),     float64
#     epsilons_lj,            # np.ndarray (n_tags, n_tags),     float64
#     charges,                # np.ndarray (n_tags,),            float64
#     lB_debye:   float,
#     B_debye:    float,
#     force_total,            # np.ndarray (n_particles, n_dim), float64  [OUT]
#     distances,              # np.ndarray (n_particles, n_particles), float64 [OUT]
#     l_box:      float,
#     cutoff2:    float,
#     n_particles: int,
#     n_dim:       int,
#     write_distances:  bool,
#     use_force_bonded: bool,
#     use_force_lj:     bool,
#     use_force_deb:    bool,
#     neighbour_cells_idx,    # np.ndarray (n_neighbour_cells, 3), int
#     head_array,             # np.ndarray (n_cells, n_cells, n_cells), int
#     list_array,             # np.ndarray (n_particles,), int
#     n_cells:          int,
#     n_neighbour_cells: int,
#     calc_virial: bool,
#     virial,                 # np.ndarray (n_dim, n_dim), float64  [OUT]
#     threads_per_block: int = 128,
# ):
#     """
#     GPU-accelerated force + virial calculation.

#     All numpy arrays are uploaded to the GPU automatically.
#     The output arrays (force_total, distances, virial) are written back to
#     their original numpy arrays in-place after the kernel finishes.

#     Note
#     ----
#     bond_list (list-of-lists) from the Rust version must be converted once:
#         bond_offsets, bond_neighbors = flatten_bond_list(bond_list)
#     """
#     n_tags = charges.shape[0]

#     # ---- upload inputs to GPU ----
#     d_pos   = cp.asarray(positions.astype(np.float64,   copy=False), order='C')
#     d_tags  = cp.asarray(tags.astype(np.int32,          copy=False))
#     d_boff  = cp.asarray(bond_offsets.astype(np.int32,  copy=False))
#     d_bnbr  = cp.asarray(bond_neighbors.astype(np.int32,copy=False))
#     d_fc    = cp.asarray(force_constants.astype(np.float64, copy=False), order='C')
#     d_bl    = cp.asarray(bond_lengths.astype(np.float64,    copy=False), order='C')
#     d_sig   = cp.asarray(sigmas_lj.astype(np.float64,       copy=False), order='C')
#     d_eps   = cp.asarray(epsilons_lj.astype(np.float64,     copy=False), order='C')
#     d_chg   = cp.asarray(charges.astype(np.float64,         copy=False))
#     d_head  = cp.asarray(head_array.astype(np.int32,        copy=False).ravel(), order='C')
#     d_list  = cp.asarray(list_array.astype(np.int32,        copy=False))
#     d_nidx  = cp.asarray(neighbour_cells_idx.astype(np.int32, copy=False), order='C')

#     # ---- output arrays (zero on GPU) ----
#     d_ft  = cp.zeros((n_particles, n_dim), dtype=np.float64)
#     d_dst = cp.zeros((n_particles, n_particles), dtype=np.float64) if write_distances \
#             else cp.empty((1, 1), dtype=np.float64)
#     d_vir = cp.zeros((n_dim, n_dim), dtype=np.float64)

#     # ---- launch ----
#     blocks = (n_particles + threads_per_block - 1) // threads_per_block
#     _kernel(
#         (blocks,), (threads_per_block,),
#         (
#             d_pos, d_tags, d_boff, d_bnbr,
#             d_fc, d_bl, d_sig, d_eps, d_chg,
#             np.float64(lB_debye), np.float64(B_debye),
#             d_ft, d_dst, d_vir,
#             d_head, d_list, d_nidx,
#             np.float64(l_box), np.float64(cutoff2),
#             np.int32(n_particles), np.int32(n_dim),
#             np.int32(n_cells), np.int32(n_neighbour_cells), np.int32(n_tags),
#             np.int32(write_distances),
#             np.int32(use_force_bonded), np.int32(use_force_lj), np.int32(use_force_deb),
#             np.int32(calc_virial),
#         )
#     )

#     # ---- copy results back to numpy in-place ----
#     cp.asnumpy(d_ft,  out=force_total)
#     if write_distances:
#         cp.asnumpy(d_dst, out=distances)
#     if calc_virial:
#         cp.asnumpy(d_vir, out=virial)