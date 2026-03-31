# 运行维护逻辑：速度限制、退激、寿命初始化与基态总量补偿

from pywarpx import warpx, callbacks
import numpy as np
import os
import time

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# 判定是否有GPU支持
try:
    import cupy as cp
except ImportError:
    cp = None

# WarpX/AMReX uses this exact idcpu value to mark particles invalid.
INVALID_PARTICLE_IDCPU = 16777216
GROUND_CESIUM_TARGET_WEIGHT = None

ELECTRON_MASS = 9.109e-31
BOLTZMANN_J = 1.380649e-23
LIGHT_SPEED = 3.0e8
TARGET_THETA = 3.322633948469325e-07
ELECTRON_SPEED_LIMIT_J = 3.5 * 1.602e-19
ELECTRON_LIMIT_ACTIVE_UNTIL = 10e-9
TARGET_TEMPERATURE = TARGET_THETA * ELECTRON_MASS * LIGHT_SPEED * LIGHT_SPEED / BOLTZMANN_J
TARGET_THERMAL_SPEED = np.sqrt(2.0 * TARGET_TEMPERATURE * BOLTZMANN_J / ELECTRON_MASS)
PROFILE_INTERVAL = 100
DEXCITE_HEARTBEAT_INTERVAL = 2.0
DEXCITE_LOG_THRESHOLD = 5.0

RANK = int(
    os.environ.get(
        "OMPI_COMM_WORLD_RANK",
        os.environ.get("PMI_RANK", os.environ.get("MV2_COMM_WORLD_RANK", "0")),
    )
)
MPI_COMM = MPI.COMM_WORLD if MPI is not None else None
MPI_SIZE = MPI_COMM.Get_size() if MPI_COMM is not None else 1
# STATUS_PATH = f"callback_status.rank{RANK}.txt"
# PROFILE_PATH = f"callback_profile.rank{RANK}.log"
CALLBACK_TIMERS = {
    "before_total": 0.0,
    "before_limit_velocity": 0.0,
    "before_dexcite": 0.0,
    "before_dexcite_scan": 0.0,
    "before_dexcite_buffer": 0.0,
    "before_dexcite_add": 0.0,
    "after_collisions_total": 0.0,
    "after_collisions_set_lifetime": 0.0,
    "after_step_total": 0.0,
    "after_step_increment_age": 0.0,
    "after_step_supplement_ground": 0.0,
}


def write_status(stage_name, step=None, detail=""):
    if step is None:
        step = "init"
    # with open(STATUS_PATH, "w", encoding="utf-8") as status_file:
    #     status_file.write(f"rank={RANK}\n")
    #     status_file.write(f"step={step}\n")
    #     status_file.write(f"stage={stage_name}\n")
    #     if detail:
    #         status_file.write(f"detail={detail}\n")
    return


def append_profile_line(step, stage_name, elapsed, extra=""):
    # with open(PROFILE_PATH, "a", encoding="utf-8") as profile_file:
    #     profile_file.write(
    #         f"step={step} rank={RANK} stage={stage_name} elapsed={elapsed:.6f}s {extra}\n"
    #     )
    return


def get_local_species_macro_count(species_name, level=0):
    macro_count = 0
    species_wrapper = sim.particles.get(species_name)
    for pti in species_wrapper.iterator(level=level):
        macro_count += len(pti["w"])
    return macro_count


def get_local_excited_macro_count(level=0):
    excited_count = 0
    for specie_name in specie_name_list:
        excited_count += get_local_species_macro_count(specie_name, level=level)
    return excited_count


def flush_profile_snapshot(step):
    # electrons_local = get_local_species_macro_count("electrons")
    # ions_local = get_local_species_macro_count("cesium_ion")
    # ground_local = get_local_species_macro_count("cesium_6s")
    # excited_local = get_local_excited_macro_count()
    # with open(PROFILE_PATH, "a", encoding="utf-8") as profile_file:
    #     profile_file.write(
    #         " ".join(
    #             [
    #                 f"step={step}",
    #                 f"rank={RANK}",
    #                 "snapshot",
    #                 f"before_total={CALLBACK_TIMERS['before_total']:.6f}s",
    #                 f"limit_velocity={CALLBACK_TIMERS['before_limit_velocity']:.6f}s",
    #                 f"dexcite={CALLBACK_TIMERS['before_dexcite']:.6f}s",
    #                 f"dexcite_scan={CALLBACK_TIMERS['before_dexcite_scan']:.6f}s",
    #                 f"dexcite_buffer={CALLBACK_TIMERS['before_dexcite_buffer']:.6f}s",
    #                 f"dexcite_add={CALLBACK_TIMERS['before_dexcite_add']:.6f}s",
    #                 f"after_collisions_total={CALLBACK_TIMERS['after_collisions_total']:.6f}s",
    #                 f"set_lifetime={CALLBACK_TIMERS['after_collisions_set_lifetime']:.6f}s",
    #                 f"after_step_total={CALLBACK_TIMERS['after_step_total']:.6f}s",
    #                 f"increment_age={CALLBACK_TIMERS['after_step_increment_age']:.6f}s",
    #                 f"supplement_ground={CALLBACK_TIMERS['after_step_supplement_ground']:.6f}s",
    #                 f"electrons_local={electrons_local}",
    #                 f"ions_local={ions_local}",
    #                 f"ground_local={ground_local}",
    #                 f"excited_local={excited_local}",
    #             ]
    #         )
    #         + "\n"
    #     )
    for key in CALLBACK_TIMERS:
        CALLBACK_TIMERS[key] = 0.0


def get_array_module(array):
    if cp is not None and hasattr(array, "__cuda_array_interface__"):
        return cp
    return np


def sample_exponential_lifetimes(count, tau, xp):
    if count <= 0:
        return xp.empty(0)
    random_values = 1.0 - xp.random.random(count)
    return -xp.log(random_values) * tau


def concatenate_chunks(chunks, xp):
    if len(chunks) == 1:
        return chunks[0]
    return xp.concatenate(chunks)


def to_numpy_array(array):
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def empty_particle_payload():
    return {
        "z": np.empty(0),
        "ux": np.empty(0),
        "uy": np.empty(0),
        "uz": np.empty(0),
        "w": np.empty(0),
        "lifetime": np.empty(0),
        "age": np.empty(0),
    }


def finalize_particle_payload(data):
    if data is None or len(data["w"]) == 0:
        return empty_particle_payload()

    xp = data["xp"]
    return {
        "z": to_numpy_array(concatenate_chunks(data["z"], xp)),
        "ux": to_numpy_array(concatenate_chunks(data["ux"], xp)),
        "uy": to_numpy_array(concatenate_chunks(data["uy"], xp)),
        "uz": to_numpy_array(concatenate_chunks(data["uz"], xp)),
        "w": to_numpy_array(concatenate_chunks(data["w"], xp)),
        "lifetime": to_numpy_array(concatenate_chunks(data["lifetime"], xp)),
        "age": to_numpy_array(concatenate_chunks(data["age"], xp)),
    }


def ensure_real_comp(species_wrapper, comp_name):
    try:
        species_wrapper.get_real_comp_index(comp_name)
        return True
    except RuntimeError:
        species_wrapper.add_real_comp(comp_name)
        return False


def get_species_total_weight(species_name, level=0):
    total_weight = 0.0
    species_wrapper = sim.particles.get(species_name)
    for pti in species_wrapper.iterator(level=level):
        total_weight += float(pti["w"].sum())
    return total_weight


def limit_electron_velocity():
    """仅在启动初期限制电子高速尾部，防止初始化阶段的极端粒子拉高代价。"""
    if sim.extension.warpx.gett_new(0) > ELECTRON_LIMIT_ACTIVE_UNTIL:
        return

    electron_wrapper = sim.particles.get("electrons")
    for pti in electron_wrapper.iterator(level=0):
        ux_arr = pti["ux"]
        uy_arr = pti["uy"]
        uz_arr = pti["uz"]
        xp = get_array_module(ux_arr)

        v_sq = ux_arr**2 + uy_arr**2 + uz_arr**2
        mask = 0.5 * ELECTRON_MASS * v_sq > ELECTRON_SPEED_LIMIT_J
        if not xp.any(mask):
            continue

        v_mag = xp.sqrt(v_sq[mask])
        ux_unit = ux_arr[mask] / v_mag
        uy_unit = uy_arr[mask] / v_mag
        uz_unit = uz_arr[mask] / v_mag

        n_reset = int(mask.sum())
        std_dev = TARGET_THERMAL_SPEED / xp.sqrt(3.0)
        vx_random = xp.random.normal(0.0, std_dev, n_reset)
        vy_random = xp.random.normal(0.0, std_dev, n_reset)
        vz_random = xp.random.normal(0.0, std_dev, n_reset)
        new_speeds = xp.sqrt(vx_random**2 + vy_random**2 + vz_random**2)

        v_max = 2.0 * TARGET_THERMAL_SPEED
        over_max = new_speeds > v_max
        if xp.any(over_max):
            new_speeds[over_max] = v_max

        ux_arr[mask] = ux_unit * new_speeds
        uy_arr[mask] = uy_unit * new_speeds
        uz_arr[mask] = uz_unit * new_speeds


def supplement_ground_cesium():
    """定期修正基态铯总权重，使其维持在初始化目标值附近。"""
    step = sim.extension.warpx.getistep(0)
    if step % 1000 != 0:
        return

    cesium_ground = sim.particles.get("cesium_6s")
    total_particles = 0.0
    macro_particles = 0

    for pti in cesium_ground.iterator(level=0):
        w = pti["w"]
        total_particles += float(w.sum())
        macro_particles += len(w)

    if macro_particles == 0 or total_particles >= GROUND_CESIUM_TARGET_WEIGHT:
        return

    supplement_particles = (GROUND_CESIUM_TARGET_WEIGHT - total_particles) / macro_particles
    for pti in cesium_ground.iterator(level=0):
        pti["w"][...] += supplement_particles

# 创建粒子名列表
specie_name_list = [
                        'cesium_6p', 
                        'cesium_5d', 
                        'cesium_7s', 
                        'cesium_7p', 
                        'cesium_6d', 
                        'cesium_8s', 
                        'cesium_4f', 
                        'cesium_8p', 
                        'cesium_7d', 
                        'cesium_9s', 
                        'cesium_5g', 
                        'cesium_9p', 
                        'cesium_8d', 
                        'cesium_10s', 
                        'cesium_10p', 
                        'cesium_9d', 
                        'cesium_11s', 
                        'cesium_11p', 
                        'cesium_12s', 
                        'cesium_12p', 
                        'cesium_13s', 
                        'cesium_13p', 
                        'cesium_14s', 
                        'cesium_14p', 
                        'cesium_15s', 
                        'cesium_14d', 
                        'cesium_16p', 
                        'cesium_17p', 
                        'cesium_18p', 
                        'cesium_20s', 
                        'cesium_22s', 
                        'cesium_25s'
                    ]

# 创建粒子名对应的寿命
specie_lifetime_list = [
    0.0318e-6,
    1.1549e-6,
    0.0484e-6,
    0.1371e-6, # 正常应该是 0.1371e-6
    0.0611e-6,
    0.0898e-6,
    0.0494e-6,
    0.3412e-6,
    0.0922e-6,
    0.1624e-6,
    0.2279e-6,
    0.5674e-6,
    0.1425e-6,
    0.2710e-6,
    0.9500e-6,
    0.2182e-6,
    0.4225e-6,
    1.4460e-6,
    0.6232e-6,
    2.0674e-6,
    0.8806e-6,
    2.9854e-6,
    1.2024e-6,
    4.0020e-6,
    1.5960e-6,
    1.0659e-6,
    6.6696e-6,
    8.3591e-6,
    10.3115e-6,
    4.8881e-6,
    6.9575e-6,
    11.0480e-6
]

# 构建字典
specie_dexcited_dic = dict(zip(specie_name_list, specie_lifetime_list))

# 创建粒子发生跃迁的目标类型以及对应的CDF
specie_decay_data = {
    'cesium_6p': {
        'targets': ['cesium_6s'],   # 正常为：['cesium_6s'],
        'cdf':     [1.000000]
    },
    'cesium_5d': {
        'targets': ['cesium_6p'],
        'cdf':     [1.000000]
    },
    'cesium_7s': {
        'targets': ['cesium_6p'],
        'cdf':     [1.000000]
    },
    'cesium_7p': {
        'targets': ['cesium_7s', 'cesium_5d', 'cesium_6s'],
        'cdf':     [0.542513, 0.795964, 1.000000]
    },
    'cesium_6d': {
        'targets': ['cesium_6p', 'cesium_7p'],
        'cdf':     [0.995213, 1.000000]
    },
    'cesium_8s': {
        'targets': ['cesium_6p', 'cesium_7p'],
        'cdf':     [0.628844, 1.000000]
    },
    'cesium_4f': {
        'targets': ['cesium_5d', 'cesium_6d'],
        'cdf':     [0.930278, 1.000000]
    },
    'cesium_8p': {
        'targets': ['cesium_8s', 'cesium_6d', 'cesium_5d', 'cesium_7s', 'cesium_6s'],
        'cdf':     [0.320950, 0.563035, 0.742136, 0.893198, 1.000000]
    },
    'cesium_7d': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_4f', 'cesium_8p'],
        'cdf':     [0.769047, 0.967342, 0.998449, 1.000000]
    },
    'cesium_9s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p'],
        'cdf':     [0.553426, 0.791987, 1.000000]
    },
    'cesium_5g': {
        'targets': ['cesium_4f'],
        'cdf':     [1.000000]
    },
    'cesium_9p': {
        'targets': ['cesium_9s', 'cesium_6s', 'cesium_7d', 'cesium_6d', 'cesium_5d', 'cesium_8s', 'cesium_7s'],
        'cdf':     [0.176396, 0.350051, 0.522172, 0.688115, 0.826320, 0.914609, 1.000000]
    },
    'cesium_8d': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_4f', 'cesium_9p'],
        'cdf':     [0.703140, 0.907163, 0.981860, 0.999225, 1.000000]
    },
    'cesium_10s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_9p', 'cesium_8p'],
        'cdf':     [0.523833, 0.734815, 0.869963, 1.000000]
    },
    'cesium_10p': {
        'targets': ['cesium_6s', 'cesium_6d', 'cesium_7d', 'cesium_8d', 'cesium_5d', 'cesium_10s', 'cesium_7s', 'cesium_9s', 'cesium_8s'],
        'cdf':     [0.151893, 0.298781, 0.438287, 0.574540, 0.698321, 0.818127, 0.884193, 0.942296, 1.000000]
    },
    'cesium_9d': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_4f', 'cesium_10p'],
        'cdf':     [0.662050, 0.863201, 0.947922, 0.985766, 0.999516, 1.000000]
    },
    'cesium_11s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_10p', 'cesium_9p'],
        'cdf':     [0.508876, 0.708105, 0.822073, 0.918101, 1.000000]
    },
    'cesium_11p': {
        'targets': ['cesium_6s', 'cesium_6d', 'cesium_7d', 'cesium_5d', 'cesium_8d', 'cesium_9d', 'cesium_11s', 'cesium_7s', 'cesium_8s', 'cesium_10s', 'cesium_9s'],
        'cdf':     [0.137351, 0.271075, 0.394455, 0.515365, 0.627815, 0.735746, 0.820488, 0.876193, 0.921410, 0.961897, 1.000000]
    },
    'cesium_12s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_11p', 'cesium_9p', 'cesium_10p'],
        'cdf':     [0.499699, 0.692492, 0.799476, 0.871255, 0.942749, 1.000000]
    },
    'cesium_12p': {
        'targets': ['cesium_6d', 'cesium_6s', 'cesium_7d', 'cesium_5d', 'cesium_8d', 'cesium_9d', 'cesium_12s', 'cesium_7s', 'cesium_8s', 'cesium_9s', 'cesium_11s', 'cesium_10s'],
        'cdf':     [0.140037, 0.279019, 0.402446, 0.524855, 0.633438, 0.731271, 0.799543, 0.863625, 0.905694, 0.938390, 0.970623, 1.000000]
    },
    'cesium_13s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_12p', 'cesium_10p', 'cesium_11p'],
        'cdf':     [0.493801, 0.683085, 0.786369, 0.853080, 0.908860, 0.957987, 1.000000]
    },
    'cesium_13p': {
        'targets': ['cesium_6s', 'cesium_6d', 'cesium_7d', 'cesium_8d', 'cesium_9d', 'cesium_5d', 'cesium_7s', 'cesium_13s', 'cesium_8s', 'cesium_9s', 'cesium_12s', 'cesium_10s', 'cesium_11s'],
        'cdf':     [0.148037, 0.290277, 0.420747, 0.533207, 0.631315, 0.722027, 0.788748, 0.847698, 0.890140, 0.921656, 0.949224, 0.975480, 1.000000]
    },
    'cesium_14s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_10p', 'cesium_13p', 'cesium_11p', 'cesium_12p'],
        'cdf':     [0.490034, 0.677004, 0.777931, 0.841901, 0.887407, 0.932067, 0.967832, 1.000000]
    },
    'cesium_14p': {
        'targets': ['cesium_6s', 'cesium_6d', 'cesium_7d', 'cesium_8d', 'cesium_9d', 'cesium_5d', 'cesium_7s', 'cesium_14s', 'cesium_8s', 'cesium_9s', 'cesium_10s', 'cesium_13s', 'cesium_11s', 'cesium_12s'],
        'cdf':     [0.150032, 0.294474, 0.425956, 0.537855, 0.633597, 0.725503, 0.792046, 0.841285, 0.882718, 0.912611, 0.936479, 0.959326, 0.980062, 1.000000]
    },
    'cesium_15s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_10p', 'cesium_14p', 'cesium_11p', 'cesium_12p', 'cesium_13p'],
        'cdf':     [0.487567, 0.672799, 0.772104, 0.834354, 0.877815, 0.914418, 0.947377, 0.974566, 1.000000]
    },
    'cesium_14d': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_10p', 'cesium_11p', 'cesium_12p', 'cesium_4f', 'cesium_13p', 'cesium_14p'],
        'cdf':     [0.594979, 0.784346, 0.870498, 0.917220, 0.945398, 0.963674, 0.976106, 0.985799, 0.994421, 1.000000]
    },
    'cesium_16p': {
        'targets': ['cesium_6s', 'cesium_6d', 'cesium_7d', 'cesium_8d', 'cesium_5d', 'cesium_9d', 'cesium_7s', 'cesium_14d', 'cesium_8s', 'cesium_9s', 'cesium_10s', 'cesium_11s', 'cesium_15s', 'cesium_12s', 'cesium_14s', 'cesium_13s'],
        'cdf':     [0.149396, 0.293579, 0.423594, 0.532576, 0.623991, 0.715226, 0.780129, 0.837589, 0.876939, 0.904387, 0.925334, 0.942366, 0.958296, 0.972983, 0.986502, 1.000000]
    },
    'cesium_17p': {
        'targets': ['cesium_6s', 'cesium_6d', 'cesium_7d', 'cesium_8d', 'cesium_5d', 'cesium_9d', 'cesium_7s', 'cesium_14d', 'cesium_8s', 'cesium_9s', 'cesium_10s', 'cesium_11s', 'cesium_12s', 'cesium_13s', 'cesium_15s', 'cesium_14s'],
        'cdf':     [0.152836, 0.300456, 0.433166, 0.543875, 0.637342, 0.729414, 0.795354, 0.846692, 0.886326, 0.913681, 0.934284, 0.950744, 0.964592, 0.976843, 0.988518, 1.000000]
    },
    'cesium_18p': {
        'targets': ['cesium_6s', 'cesium_6d', 'cesium_7d', 'cesium_8d', 'cesium_5d', 'cesium_9d', 'cesium_7s', 'cesium_14d', 'cesium_8s', 'cesium_9s', 'cesium_10s', 'cesium_11s', 'cesium_12s', 'cesium_13s', 'cesium_14s', 'cesium_15s'],
        'cdf':     [0.154956, 0.304715, 0.439029, 0.550661, 0.645375, 0.737752, 0.804240, 0.852093, 0.891787, 0.918963, 0.939229, 0.955216, 0.968439, 0.979861, 0.990176, 1.000000]
    },
    'cesium_20s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_10p', 'cesium_11p', 'cesium_12p', 'cesium_13p', 'cesium_14p', 'cesium_16p', 'cesium_18p', 'cesium_17p'],
        'cdf':     [0.496636, 0.683441, 0.782103, 0.842614, 0.883521, 0.913039, 0.935441, 0.953170, 0.967738, 0.978769, 0.989630, 1.000000]
    },
    'cesium_22s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_10p', 'cesium_11p', 'cesium_12p', 'cesium_13p', 'cesium_14p', 'cesium_16p', 'cesium_17p', 'cesium_18p'],
        'cdf':     [0.501543, 0.689861, 0.789075, 0.849719, 0.890531, 0.919805, 0.941846, 0.959103, 0.973069, 0.983061, 0.991897, 1.000000]
    },
    'cesium_25s': {
        'targets': ['cesium_6p', 'cesium_7p', 'cesium_8p', 'cesium_9p', 'cesium_10p', 'cesium_11p', 'cesium_12p', 'cesium_13p', 'cesium_14p', 'cesium_16p', 'cesium_17p', 'cesium_18p'],
        'cdf':     [0.504877, 0.694122, 0.793584, 0.854185, 0.894800, 0.923778, 0.945449, 0.962269, 0.975729, 0.985021, 0.993002, 1.000000]
    },
}


def build_collective_target_names():
    target_candidates = set()
    for decay_info in specie_decay_data.values():
        target_candidates.update(decay_info["targets"])
    return [name for name in specie_name_list if name in target_candidates]


COLLECTIVE_TARGET_NAMES = build_collective_target_names()
COLLECTIVE_TARGET_INDEX = {
    target_name: idx for idx, target_name in enumerate(COLLECTIVE_TARGET_NAMES)
}


def get_global_active_mask(local_mask):
    if MPI_COMM is None or MPI_SIZE == 1:
        return local_mask

    global_mask = np.zeros_like(local_mask)
    MPI_COMM.Allreduce(local_mask, global_mask, op=MPI.MAX)
    return global_mask

def initialize_excited_species_metadata():
    for specie_name in specie_name_list:
        cesium_excited_atom = sim.particles.get(specie_name)
        has_lifetime = ensure_real_comp(cesium_excited_atom, "lifetime")
        has_age = ensure_real_comp(cesium_excited_atom, "age")

        # On restart, both components already exist and may contain valid particle state.
        # In that case do not overwrite them here.
        if has_lifetime and has_age:
            continue

        lifetime_nature = specie_dexcited_dic[specie_name]
        for pti in cesium_excited_atom.iterator(level=0):
            lifetime = pti["lifetime"]
            age = pti["age"]
            xp = get_array_module(lifetime)

            count = int(lifetime.size)
            if count > 0:
                lifetime[...] = sample_exponential_lifetimes(count, lifetime_nature, xp)
            age[...] = 0.0


def increment_excited_species_age():
    dt_now = sim.extension.warpx.getdt(0)
    for specie_name in specie_name_list:
        cesium_excited_atom = sim.particles.get(specie_name)
        for pti in cesium_excited_atom.iterator(level=0):
            pti["age"][...] += dt_now


def set_lifetime_and_age():
    for specie_name in specie_name_list:
        cesium_excited_atom = sim.particles.get(specie_name)
        tau_natural = specie_dexcited_dic[specie_name]

        for pti in cesium_excited_atom.iterator(level=0):
            lifetime = pti["lifetime"]
            age = pti["age"]
            xp = get_array_module(lifetime)
            mask = lifetime < 1.0e-15

            if not xp.any(mask):
                continue

            num_new = int(mask.sum())
            age[mask] = 0.0
            lifetime[mask] = sample_exponential_lifetimes(num_new, tau_natural, xp)


def maybe_write_dexcite_heartbeat(step, last_heartbeat_time, detail):
    now = time.perf_counter()
    if now - last_heartbeat_time >= DEXCITE_HEARTBEAT_INTERVAL:
        write_status("before_collisions:dexcite_progress", step, detail)
        return now
    return last_heartbeat_time


def dexcite_high_excited_atoms(step):
    particle_buffer = {}
    total_decay_particles = 0
    total_decay_tiles = 0
    active_species = 0
    scan_elapsed = 0.0
    buffer_elapsed = 0.0
    add_elapsed = 0.0
    last_heartbeat_time = time.perf_counter()

    for specie_name in specie_name_list:
        cesium_excited_atom = sim.particles.get(specie_name)
        decay_info = specie_decay_data.get(specie_name)
        if not decay_info or not decay_info["targets"]:
            continue

        targets_list = decay_info["targets"]
        cdf_list = decay_info["cdf"]
        specie_decay_particles = 0
        specie_decay_tiles = 0
        specie_start = time.perf_counter()
        write_status("before_collisions:dexcite_species", step, f"species={specie_name}")

        for pti in cesium_excited_atom.iterator(level=0):
            age = pti["age"]
            lifetime = pti["lifetime"]
            xp = get_array_module(age)

            t_scan = time.perf_counter()
            decay_indices = xp.where(age >= lifetime)[0]
            scan_elapsed += time.perf_counter() - t_scan
            if int(decay_indices.size) == 0:
                last_heartbeat_time = maybe_write_dexcite_heartbeat(
                    step,
                    last_heartbeat_time,
                    f"species={specie_name} tiles={specie_decay_tiles} decays={specie_decay_particles}",
                )
                continue

            t_buffer = time.perf_counter()
            decay_count = int(decay_indices.size)
            specie_decay_particles += decay_count
            total_decay_particles += decay_count
            specie_decay_tiles += 1
            total_decay_tiles += 1

            pz = pti["z"][decay_indices]
            pux = pti["ux"][decay_indices]
            puy = pti["uy"][decay_indices]
            puz = pti["uz"][decay_indices]
            pw = pti["w"][decay_indices]

            cdf_array = xp.asarray(cdf_list)
            random_channels = xp.random.random(int(decay_indices.size))
            target_indices = xp.searchsorted(cdf_array, random_channels, side="left")

            for t_idx in xp.unique(target_indices).tolist():
                target_name = targets_list[int(t_idx)]
                if target_name not in specie_name_list:
                    continue

                mask = target_indices == t_idx
                subset_count = int(mask.sum())
                if subset_count == 0:
                    continue

                target_tau = specie_dexcited_dic[target_name]
                buf = particle_buffer.setdefault(
                    target_name,
                    {
                        "xp": xp,
                        "z": [],
                        "ux": [],
                        "uy": [],
                        "uz": [],
                        "w": [],
                        "lifetime": [],
                        "age": [],
                    },
                )
                buf["z"].append(pz[mask])
                buf["ux"].append(pux[mask])
                buf["uy"].append(puy[mask])
                buf["uz"].append(puz[mask])
                buf["w"].append(pw[mask])
                buf["lifetime"].append(sample_exponential_lifetimes(subset_count, target_tau, xp))
                buf["age"].append(xp.zeros(subset_count))

            idcpu = xp.array(pti.soa().get_idcpu_data(), copy=False)
            idcpu[decay_indices] = INVALID_PARTICLE_IDCPU
            buffer_elapsed += time.perf_counter() - t_buffer
            last_heartbeat_time = maybe_write_dexcite_heartbeat(
                step,
                last_heartbeat_time,
                "species="
                f"{specie_name} tiles={specie_decay_tiles} decays={specie_decay_particles} "
                f"total_decays={total_decay_particles}",
            )

        specie_elapsed = time.perf_counter() - specie_start
        if specie_decay_tiles > 0:
            active_species += 1
        if specie_elapsed >= DEXCITE_LOG_THRESHOLD or specie_decay_particles > 0:
            append_profile_line(
                step,
                "dexcite_species",
                specie_elapsed,
                (
                    f"species={specie_name} decay_tiles={specie_decay_tiles} "
                    f"decay_particles={specie_decay_particles}"
                ),
            )

    local_active_mask = np.zeros(len(COLLECTIVE_TARGET_NAMES), dtype=np.int32)
    for target_name in particle_buffer:
        target_index = COLLECTIVE_TARGET_INDEX.get(target_name)
        if target_index is not None:
            local_active_mask[target_index] = 1

    global_active_mask = get_global_active_mask(local_active_mask)
    global_active_targets = [
        target_name
        for target_name, is_active in zip(COLLECTIVE_TARGET_NAMES, global_active_mask)
        if is_active
    ]

    write_status(
        "before_collisions:dexcite_before_add",
        step,
        "local_targets="
        f"{int(local_active_mask.sum())} global_targets={len(global_active_targets)} "
        f"total_tiles={total_decay_tiles} total_decays={total_decay_particles}",
    )
    append_profile_line(
        step,
        "dexcite_collective_targets",
        0.0,
        (
            f"local_targets={int(local_active_mask.sum())} "
            f"global_targets={len(global_active_targets)} "
            f"global_list={','.join(global_active_targets) if global_active_targets else 'none'}"
        ),
    )

    for target_name in global_active_targets:
        t_add = time.perf_counter()
        data = particle_buffer.get(target_name)
        payload = finalize_particle_payload(data)
        pc_target = sim.particles.get(target_name)
        particle_count = int(payload["w"].size)
        write_status(
            "before_collisions:dexcite_add_particles",
            step,
            f"target={target_name} particles={particle_count}",
        )
        pc_target.add_particles(
            z=payload["z"],
            ux=payload["ux"],
            uy=payload["uy"],
            uz=payload["uz"],
            w=payload["w"],
            lifetime=payload["lifetime"],
            age=payload["age"],
        )
        add_elapsed += time.perf_counter() - t_add

    CALLBACK_TIMERS["before_dexcite_scan"] += scan_elapsed
    CALLBACK_TIMERS["before_dexcite_buffer"] += buffer_elapsed
    CALLBACK_TIMERS["before_dexcite_add"] += add_elapsed

    append_profile_line(
        step,
        "dexcite_summary",
        scan_elapsed + buffer_elapsed + add_elapsed,
        (
            f"species_with_decay={active_species} total_decay_tiles={total_decay_tiles} "
            f"total_decay_particles={total_decay_particles} "
            f"scan={scan_elapsed:.6f}s buffer={buffer_elapsed:.6f}s add={add_elapsed:.6f}s"
        ),
    )


def before_collisions_maintenance():
    step = sim.extension.warpx.getistep(0)
    write_status("before_collisions:start", step)
    t0 = time.perf_counter()

    t_stage = time.perf_counter()
    limit_electron_velocity()
    CALLBACK_TIMERS["before_limit_velocity"] += time.perf_counter() - t_stage
    write_status("before_collisions:after_limit_velocity", step)

    t_stage = time.perf_counter()
    dexcite_high_excited_atoms(step)
    CALLBACK_TIMERS["before_dexcite"] += time.perf_counter() - t_stage

    elapsed = time.perf_counter() - t0
    CALLBACK_TIMERS["before_total"] += elapsed
    if elapsed > 5.0:
        append_profile_line(step, "before_collisions_slow", elapsed)
    write_status("before_collisions:done", step)


def after_collisions_maintenance():
    step = sim.extension.warpx.getistep(0)
    write_status("after_collisions:start", step)
    t0 = time.perf_counter()

    t_stage = time.perf_counter()
    set_lifetime_and_age()
    CALLBACK_TIMERS["after_collisions_set_lifetime"] += time.perf_counter() - t_stage

    elapsed = time.perf_counter() - t0
    CALLBACK_TIMERS["after_collisions_total"] += elapsed
    if elapsed > 5.0:
        append_profile_line(step, "after_collisions_slow", elapsed)
    write_status("after_collisions:done", step)


def after_step_maintenance():
    step = sim.extension.warpx.getistep(0)
    write_status("after_step:start", step)
    t0 = time.perf_counter()

    t_stage = time.perf_counter()
    increment_excited_species_age()
    CALLBACK_TIMERS["after_step_increment_age"] += time.perf_counter() - t_stage
    write_status("after_step:after_increment_age", step)

    t_stage = time.perf_counter()
    supplement_ground_cesium()
    CALLBACK_TIMERS["after_step_supplement_ground"] += time.perf_counter() - t_stage

    elapsed = time.perf_counter() - t0
    CALLBACK_TIMERS["after_step_total"] += elapsed
    if elapsed > 5.0:
        append_profile_line(step, "after_step_slow", elapsed)
    if step % PROFILE_INTERVAL == 0:
        flush_profile_snapshot(step)
    write_status("idle", step)


sim = warpx
sim.load_inputs_file("./thermionicCesium-dsmc-Lietz-fixed")
GROUND_CESIUM_TARGET_WEIGHT = get_species_total_weight("cesium_6s")
write_status("loaded_inputs", detail="starting_metadata_initialization")

initialize_excited_species_metadata()

# 先运行一步确保初始化与初始粒子创建完成
sim.step(1)
set_lifetime_and_age()
write_status("post_init_step", sim.extension.warpx.getistep(0))

callbacks.installbeforecollisions(before_collisions_maintenance)
callbacks.installaftercollisions(after_collisions_maintenance)
callbacks.installafterstep(after_step_maintenance)

# 真正的运行
sim.step()
