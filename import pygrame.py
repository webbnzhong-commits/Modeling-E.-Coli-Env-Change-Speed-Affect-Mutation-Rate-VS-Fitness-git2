import random
import math
import os
import json
import shutil
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False
#import pyautogui
import time
import statistics
import csv
from collections import Counter
from pathlib import Path
from data_tracking import RunDataTracker
from simulatino_parser import parse_run
from settings_manager import load_settings, save_settings
from multiprocessing import Pool, cpu_count
from fps_tracker import FPSTracker
fast_update = None
if HAS_NUMPY:
    try:
        from fast_loop import fast_update
    except Exception:
        fast_update = None

def work(x):
    return x * x



#from pathlib import Path


#git debug



# initialize per-run logging
tracker = RunDataTracker()
run_num = tracker.run_num
RUN_META_PATH = tracker.results_dir / str(run_num) / "run_meta.json"
MASTER_RUN_META_PATH = None
if os.environ.get("SIM_MASTER_DIR"):
    MASTER_RUN_META_PATH = Path(os.environ["SIM_MASTER_DIR"]) / str(run_num) / "run_meta.json"

def _load_existing_run_meta(path: Path) -> dict:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


# Window settings
settings = load_settings()
WIDTH = int(settings["screen"]["width"])
HEIGHT = int(settings["screen"]["height"])
WIDTH = max(200, WIDTH)
HEIGHT = max(200, HEIGHT)

display_mode = int(settings["display_mode"])
draw = bool(settings["draw"])
drawSometimes = bool(settings["drawSometimes"])
drawAmnt = int(settings["drawAmnt"])
population_cap = float(settings["population_cap"])

def _env_change_rate_from_settings(cfg):
    try:
        return float(os.environ.get("SIM_ENV_CHANGE_RATE", cfg["enviormentChangeRate"]))
    except Exception:
        return float(cfg["enviormentChangeRate"])


enviormentChangeRate = _env_change_rate_from_settings(settings)
PH_EFFECT_SCALE = float(settings["ph_effect"]["scale"])
PH_EFFECT_DIVISOR = float(settings["ph_effect"]["divisor"])
TEMP_EFFECT_SCALE = float(settings["temp_effect"]["scale"])
TEMP_EFFECT_DIVISOR = float(settings["temp_effect"]["divisor"])
REPRO_DEBUF_MIN = float(settings["reproduction_debuf_min"])
QUAN = float(settings.get("quan", 2.0))
IMMUNE_SYSTEM_QUAN_FACTOR = float(settings.get("immune_system_quan_factor", 0.15))

evo_speed_range = [0.05, 0.4]

SIM_CONTROL_FILE = os.environ.get("SIM_CONTROL_FILE")
ALL_ACTIVE = os.environ.get("SIM_ALL_ACTIVE") == "1"
SIM_FPS_PATH = os.environ.get("SIM_FPS_PATH")
FAST_MODE = os.environ.get("FAST_MODE") == "1" and HAS_NUMPY and fast_update is not None
SIMRUN_LAUNCH = str(os.environ.get("HUB_RUNNER_FROM_SIMRUN", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TIMELINE_SNAPSHOT_LOGGING_ENABLED = not SIMRUN_LAUNCH
try:
    SIM_INDEX = int(os.environ.get("SIM_INDEX", "0"))
except ValueError:
    SIM_INDEX = 0
try:
    SIM_TOTAL = int(os.environ.get("SIM_TOTAL", "0"))
except ValueError:
    SIM_TOTAL = 0


def _read_control_state():
    if not SIM_CONTROL_FILE:
        return None, None, None, None, None, None
    try:
        data = Path(SIM_CONTROL_FILE).read_text().strip()
    except Exception:
        return None, None, None, None, None, None
    if not data:
        return None, None, None, None, None, None
    if data.lstrip().startswith("{"):
        try:
            payload = json.loads(data)
        except Exception:
            return None, None, None, None, None, None
        active = payload.get("active")
        enabled = payload.get("enabled")
        draw_modes = payload.get("draw_mode")
        draw_every = payload.get("draw_every")
        mode_values = payload.get("mode")
        update_tokens = payload.get("update_tokens")
        try:
            active = int(active) if active is not None else None
        except (TypeError, ValueError):
            active = None
        enabled_flags = None
        if isinstance(enabled, list):
            enabled_flags = [bool(item) for item in enabled]
        draw_mode_list = None
        if isinstance(draw_modes, list):
            draw_mode_list = []
            for item in draw_modes:
                try:
                    draw_mode_list.append(int(item))
                except (TypeError, ValueError):
                    draw_mode_list.append(0)
        draw_every_list = None
        if isinstance(draw_every, list):
            draw_every_list = []
            for item in draw_every:
                try:
                    draw_every_list.append(int(item))
                except (TypeError, ValueError):
                    draw_every_list.append(1)
        mode_list = None
        if isinstance(mode_values, list):
            mode_list = []
            for item in mode_values:
                try:
                    mode_list.append(int(item))
                except (TypeError, ValueError):
                    mode_list.append(0)
        update_list = None
        if isinstance(update_tokens, list):
            update_list = []
            for item in update_tokens:
                try:
                    update_list.append(int(item))
                except (TypeError, ValueError):
                    update_list.append(0)
        return active, enabled_flags, draw_mode_list, draw_every_list, mode_list, update_list
    try:
        return int(data), None, None, None, None, None
    except ValueError:
        return None, None, None, None, None, None


def apply_settings(new_settings):
    global settings, WIDTH, HEIGHT
    global display_mode, draw, drawSometimes, drawAmnt
    global population_cap, enviormentChangeRate
    global PH_EFFECT_SCALE, PH_EFFECT_DIVISOR, TEMP_EFFECT_SCALE, TEMP_EFFECT_DIVISOR, REPRO_DEBUF_MIN, QUAN

    settings = new_settings
    new_width = int(settings["screen"]["width"])
    new_height = int(settings["screen"]["height"])
    new_width = max(200, new_width)
    new_height = max(200, new_height)

    if new_width != WIDTH or new_height != HEIGHT:
        WIDTH, HEIGHT = new_width, new_height
        pass

    display_mode = int(settings["display_mode"])
    draw = bool(settings["draw"])
    drawSometimes = bool(settings["drawSometimes"])
    drawAmnt = int(settings["drawAmnt"])
    population_cap = float(settings["population_cap"])
    enviormentChangeRate = _env_change_rate_from_settings(settings)
    PH_EFFECT_SCALE = float(settings["ph_effect"]["scale"])
    PH_EFFECT_DIVISOR = float(settings["ph_effect"]["divisor"])
    TEMP_EFFECT_SCALE = float(settings["temp_effect"]["scale"])
    TEMP_EFFECT_DIVISOR = float(settings["temp_effect"]["divisor"])
    REPRO_DEBUF_MIN = float(settings["reproduction_debuf_min"])
    QUAN = float(settings.get("quan", 2.0))


def reload_settings():
    apply_settings(load_settings())


'''
Three traits:
reproduction rate: what resource amount is needed to reproduce. How much p or c or n is needed to reproduce, however they need 10 of each resource to live

speed of evolution: how fast traits change over generations
Charizma: when to dots encounter, how like one is to steal resources from the other, or if diffrent which one will kill the other
After ten cycles they will die.
Size - chance of surviving attacks from other dots
Chance to fight - how aggressive they are when encountering other dots
amount per reproduction - how much they reproduce each time (the more the more food they need to eat)
Offspring size


Resource taken: three resources, p, c, and n
Each cycle the amount of resources added to the environment will be similar to last cycle but a bit diffrent, however p + c + n will be 100

Each cycle there will be a new amount of resources added to the environment.
The resources will be distributed to each dot. They collect resources until they have enough to reproduce. (reproduction rate)

Ability to breed: get the squared diffrence of the traits above. If the those numbers are low enough they can breed.
'''

'''
Thesis: How does evoloution speed allow species to adapt to a changing environment?
Does evoloution speed affect species fitness
'''

def _load_arithmetic_mean_points(results_dir: Path, run_num: int):
    run_dir = results_dir / str(run_num)
    parsed_path = run_dir / f"parsedArithmeticMeanSimulatino{run_num}_Log.csv"
    if not parsed_path.exists():
        return []
    points = []
    with open(parsed_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                x = float(row["evolution rate"])
                y = float(row["arithmetic mean length lived"])
            except (ValueError, KeyError, TypeError):
                continue
            points.append((x, y))
    return points


# Create a Dot class
class Dot:
    def __init__(self, x, y, radius=5, color=(255, 255, 255)):
        self.x = x
        self.realx = x
        self.y = y
        self.realy = y
        self.radius = radius
        self.color = color
        self.speed = [random.uniform(-0.002, 0.002), random.uniform(-0.002, 0.002)]
        self.cancerous = False
        self._cancer_surface = None
        self._cancer_surface_size = None
        self._cancer_surface_color = None

        self.favored_resource = random.choice(["p", "c", "n"])
        self.immune_system = random.randint(0, 5)  # resistance to antiBiotics
        self.optimal_ph = random.uniform(7, 8)  # preferred pH level, works best in 7.2
        self.optimal_temp = random.uniform(36, 38)  # preferred temperature, works best at 37



        self.color = [random.randint(100, 255) for _ in range(3)]
        num = math.sqrt(self.speed[0]**2 + self.speed[1]**2)
        num /= 4.0
        if num != 0:
            self.speed = [self.speed[0]/num, self.speed[1]/num]

        # Evolution traits
        self.reproduction_resource = {
            "p": random.randint(5, 1000),
            "c": random.randint(5, 1000),
            "n": random.randint(5, 1000)
        }  # resources required to reproduce
        factor = 10/(total := sum(self.reproduction_resource.values()))
        for r in ["p", "c", "n"]:
            self.reproduction_resource[r] = max(1, int(self.reproduction_resource[r] * factor))   

        


        self.evolution_speed = random.uniform(evo_speed_range[0], evo_speed_range[1])  # how quickly traits evolve
        #self.evolution_speed = 0.05 + (random.randint(0, 100)/1000)    

        
        
        # Resources held
        self.resources = {"p": 0, "c": 0, "n": 0}
        
        # Life cycle count
        self.life_cycles = 0
        self.max_cycles = 40 + random.randint(0, 20)  # lifespan in cycles

        self.size = 1 + random.randint(0, 5)  # size affects survival chance



    def update(self):
        global dots
        self.collect_resources()
        # Move the dot
        self.realx += self.speed[0]
        self.realy += self.speed[1]
        self.x = int(self.realx)
        self.y = int(self.realy)

        # Bounce off edges
        if self.x <= 0 or self.x >= WIDTH:
            self.speed[0] *= -1
        if self.y <= 0 or self.y >= HEIGHT:
            self.speed[1] *= -1

        self.life_cycles += 1
        if self.life_cycles >= self.max_cycles:
            self.death()
            return
        
        reproduce = True
        ph_diff = abs(self.optimal_ph - enviorment_state.ph)
        temp_diff = abs(self.optimal_temp - enviorment_state.temp)
        ph_effect = max(1.0, (ph_diff) * PH_EFFECT_SCALE)
        temp_effect = max(1.0, (temp_diff) * TEMP_EFFECT_SCALE)
        debuf = max(REPRO_DEBUF_MIN, ph_effect * temp_effect)
        for r in ["p", "c", "n"]:
            if self.resources[r] / debuf < self.reproduction_resource[r]:
                reproduce = False

        if reproduce:
            self.resources["p"] -= self.reproduction_resource["p"]
            self.resources["c"] -= self.reproduction_resource["c"]
            self.resources["n"] -= self.reproduction_resource["n"]
            child = _spawn_child_from_parent(self)
            dots.append(child)

    def death (self):

        enviorment_state.dead_cell(self.resources)
        dots.remove(self)
        return
    
    def collect_resources(self):
        resource_pool = enviorment_state.get_resources()
        total = 0
        for r in ["p", "c", "n"]:
            if resource_pool[r] > 0:
                
                self.resources[r] += resource_pool[r]
                

    def draw(self, surface):
        return
        



# Initial resource pool and changing conditions
class enviorment():
    def __init__ (self):
        self.resource_pool = {"p": 34, "c": 33, "n": 33}
        self.deadNutreints = {"p": 0, "c": 0, "n": 0}
        self.antiBiotic = 0
        self.ph = 7.0
        self.temp = 37.0
        self.tempDirUp = True

        self.goingToAmnt = random.randint(10, 30)
        self.foodAmnt = 100
        
        



    def dead_cell(self, resources):
        for r in ["p", "c", "n"]:
            self.deadNutreints[r] += resources[r]
    

    def regenerate_resources(self):
        # Slightly vary each resource value
        for r in ["p", "c", "n"]:
            self.resource_pool[r] += random.uniform(-0.1, 0.1)

        # Prevent negative values
        for r in ["p", "c", "n"]:
            if self.resource_pool[r] < 0:
                self.resource_pool[r] = 0
            if self.resource_pool[r] > 1:
                self.resource_pool[r] = 1
            
            
            

        

        # Normalize so that p + c + n ≈ 100
        total = self.resource_pool["p"] + self.resource_pool["c"] + self.resource_pool["n"]
        for r in ["p", "c", "n"]:
            self.resource_pool[r] *= 1/ total
        '''
        if len(dots) > 300:
            self.foodAmnt -= 1
        else:
            self.foodAmnt += 1
        '''
        
        if self.foodAmnt > self.goingToAmnt - enviormentChangeRate:
            self.foodAmnt -= enviormentChangeRate / 2 + 0.5
        elif self.foodAmnt < self.goingToAmnt + enviormentChangeRate:
            self.foodAmnt += enviormentChangeRate / 2 + 0.5
        else:
            # Choose a new target amount based on population.
            pop = max(1, len(dots))
            scale = 300 / pop

            low = int(7 * scale) * (1 - enviormentChangeRate/10)
            
            high = low * ((1 + enviormentChangeRate/10) / (1 - enviormentChangeRate/10))

            # Clamp to keep targets reasonable
            low = int(max(1, min(low, 200)))
            high = int(max(low + 1, min(high, 200)))

            self.goingToAmnt = random.randint(low, high)


        

    def get_resources(self):
        const = len(dots)
        return {"p": (self.resource_pool["p"] * self.foodAmnt + self.deadNutreints["p"])/const,
                "c": (self.resource_pool["c"] * self.foodAmnt + self.deadNutreints["c"])/const,
                "n": (self.resource_pool["n"] * self.foodAmnt + self.deadNutreints["n"])/const} 
    
    def update(self):
        self.deadNutreints = {"p": 0, "c": 0, "n": 0}

    def _change_interval(self):
        return max(1, round(500 / max(enviormentChangeRate, 1e-6)))

    def update_antibiotic(self, dots, frame_count):
        if frame_count % self._change_interval() != 0:
            return
        immune_counts = Counter(dot.immune_system for dot in dots)
        if not immune_counts:
            return
        self.antiBiotic = immune_counts.most_common(1)[0][0]
        for dot in list(dots):
            if self.antiBiotic != dot.immune_system:
                dot.death()

    def update_climate(self, frame_count):
        if frame_count % self._change_interval() != 0:
            return
        self.ph += random.uniform(-enviormentChangeRate / 2 + 0.5, enviormentChangeRate / 2 + 0.5) * 2
        if self.ph < 4:
            self.ph = 4
        if self.ph > 10:
            self.ph = 10

        if self.tempDirUp:
            self.temp += random.uniform(0, enviormentChangeRate / 2 + 0.5)
        else:
            self.temp -= random.uniform(0, enviormentChangeRate / 2 + 0.5)
        if self.temp > 40:
            self.temp = 40
            self.tempDirUp = False
        if self.temp < 34:
            self.temp = 34
            self.tempDirUp = True

def reset_simulation():
    
    #dots = [Dot(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for x in range(30)]
    dots = []
    for x in range(30):
        dots.append(Dot(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
        #dots[-1].evolution_speed = (evo_speed_range[1] - evo_speed_range[0]) / 30 * x + evo_speed_range[0]
    enviorment_state = enviorment()
    #totalSim += 1
    frame_count = 0
    return dots, enviorment_state

enviorment_state = enviorment()

# Create dots
dots = []
dots, enviorment_state = reset_simulation()



running = True
should_parse = False
frame_count = 0
fps_tracker = FPSTracker(sample_interval=1000, log_path=SIM_FPS_PATH)
avgSpecies = []
run_start_time = time.perf_counter()
run_start_wall = time.time()
META_WRITE_INTERVAL = 1000
SNAPSHOT_INTERVAL = 10000
SNAPSHOT_DIR_NAME = "snapshots"
existing_meta = _load_existing_run_meta(RUN_META_PATH)
if not existing_meta and MASTER_RUN_META_PATH is not None:
    existing_meta = _load_existing_run_meta(MASTER_RUN_META_PATH)
if existing_meta:
    try:
        prev_frames = int(existing_meta.get("frame_count", 0))
    except Exception:
        prev_frames = 0
    try:
        prev_elapsed = float(existing_meta.get("elapsed_seconds", 0))
    except Exception:
        prev_elapsed = 0.0
    try:
        prev_start_wall = float(existing_meta.get("start_time", 0))
    except Exception:
        prev_start_wall = 0.0
    if prev_frames > 0:
        frame_count = prev_frames
    if prev_elapsed > 0:
        run_start_time = time.perf_counter() - prev_elapsed
    if prev_start_wall > 0:
        run_start_wall = prev_start_wall
    elif prev_elapsed > 0:
        run_start_wall = time.time() - prev_elapsed






#time.sleep(10)  # Give user time to switch to pygame window
totalSim = 0
pause = False
info = ""
info2 = ""
info3 = ""
info4 = ""
info5 = ""
info6 = ""
info7 = ""
info8 = ""
info9 = ""
last_avg_evo_speed          = 0
species_period              = 0
current_species_populatino  = 0
species_trackers            = {}
arithmetic_points           = []
arithmetic_graph_error       = ""

fightChance                 = 5

def _update_stats_snapshot():
    global info, info2, info3, info4, info5, info6, info7, info8, info9
    global arithmetic_points, arithmetic_graph_error
    evo_speeds = [round(dot.evolution_speed, 2) for dot in dots] if len(dots) > 0 else []
    try:
        speed_counts = Counter(evo_speeds)
        common = speed_counts.most_common()
        infos = []
        total_dots = len(dots)

        qualified = []
        for speed, _ in common:
            mode_dots = [dot for dot in dots if round(dot.evolution_speed, 2) == speed]
            count = len(mode_dots)
            if total_dots > 0 and count / total_dots >= 0.10:
                qualified.append((speed, mode_dots, count))

        if len(qualified) == 0 and common:
            qualified = [
                (
                    common[0][0],
                    [dot for dot in dots if round(dot.evolution_speed, 2) == common[0][0]],
                    speed_counts[common[0][0]],
                )
            ]
        while len(qualified) < 3 and qualified:
            qualified.append(qualified[0])

        for idx in range(3):
            if idx >= len(qualified):
                infos.append("No dots")
                infos.append("")
                continue
            speed, mode_dots, count = qualified[idx]
            pct = round((count / total_dots) * 100, 2) if total_dots > 0 else 0
            chosen = random.choice(mode_dots) if mode_dots else None
            if chosen:
                infos.append(
                    f"{pct}% Mode {str(speed)[0:5]} Size:{str(chosen.size)[0:5]} "
                    f" Imm:{str(chosen.immune_system)[0:5]} "
                    f"Fav:{chosen.favored_resource}  Cyc:{str(chosen.max_cycles)[0:5]} Evo Speed :{str(chosen.evolution_speed)[0:5]} "
                )
                infos.append(("color_square", chosen.color))
                infos.append(
                    f"Needs p:{str(chosen.reproduction_resource['p'])[0:5]} "
                    f"c:{str(chosen.reproduction_resource['c'])[0:5]} "
                    f"n:{str(chosen.reproduction_resource['n'])[0:5]}"
                )
            else:
                infos.append("No dots")
                infos.append("")

        while len(infos) < 9:
            infos.append("")

        info, info2, info3, info4, info5, info6, info7, info8, info9 = infos[:9]
    except Exception:
        info = "No unique mode"
        info2 = ""

    try:
        tracker.csv_file.flush()
    except Exception:
        pass

    arithmetic_points = []
    arithmetic_graph_error = ""
    try:
        parse_run(tracker.results_dir, tracker.run_num, quiet=True)
        arithmetic_points = _load_arithmetic_mean_points(
            tracker.results_dir,
            tracker.run_num,
        )
    except Exception as e:
        arithmetic_graph_error = f"Graph error: {e}"

    return

def _update_all_stats_snapshot():
    _update_stats_snapshot()
    _write_run_meta()
    if tracker.master_csv_file is not None:
        try:
            tracker.master_csv_file.flush()
        except Exception:
            pass
    if tracker.master_dir:
        try:
            parse_run(Path(tracker.master_dir), tracker.run_num, quiet=True)
        except Exception as e:
            print(f"Master parse error: {e}")


def _snapshot_arithmetic_mean(frame_count: int) -> None:
    if frame_count <= 0 or not TIMELINE_SNAPSHOT_LOGGING_ENABLED:
        return
    _update_all_stats_snapshot()
    run_dir = tracker.results_dir / str(tracker.run_num)
    parsed_path = run_dir / f"parsedArithmeticMeanSimulatino{tracker.run_num}_Log.csv"
    if not parsed_path.exists():
        return
    snapshot_dir = run_dir / SNAPSHOT_DIR_NAME
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"arith_mean_{int(frame_count)}.csv"
        shutil.copy2(parsed_path, snapshot_path)
    except Exception:
        pass


def _spawn_child_from_parent(parent):
    child = Dot(parent.x, parent.y)
    child.evolution_speed = max(0.001, parent.evolution_speed)
    child.size = max(1, parent.size + (random.uniform(-child.evolution_speed, child.evolution_speed)))
    child.favored_resource = parent.favored_resource
    maxAmnt = child.evolution_speed * QUAN + (0.5 - IMMUNE_SYSTEM_QUAN_FACTOR * QUAN)
    changeAmnt = int(round(random.uniform(-maxAmnt, maxAmnt)))
    child.immune_system = parent.immune_system + changeAmnt
    
    child.immune_system = child.immune_system % 5


    child.optimal_ph = parent.optimal_ph + random.uniform(-child.evolution_speed * 1.5, child.evolution_speed * 1.5)
    child.optimal_temp = parent.optimal_temp + random.uniform(-child.evolution_speed * 1.5, child.evolution_speed * 1.5)
    child.color = parent.color.copy()
    
    
    total = 0
    total2 = 0
    for r in ["p", "c", "n"]:
        mutation = int(random.uniform(-child.evolution_speed, child.evolution_speed) / 4)
        child.reproduction_resource[r] = max(1, parent.reproduction_resource[r] + mutation)
        if child.favored_resource != r:
            total2 += child.reproduction_resource[r]
        total += child.reproduction_resource[r]

    if total > 0:
        factor = child.size / total * 2
        for r in ["p", "c", "n"]:
            child.reproduction_resource[r] = max(1, int(child.reproduction_resource[r] * factor))

    child.reproduction_resource[child.favored_resource] = max(
        1 * child.size / 2, child.reproduction_resource[child.favored_resource]
    )

    if random.uniform(0, child.evolution_speed/3 + 0.0367) < 0.11:
        return child
    for r in ["p", "c", "n"]:
        child.reproduction_resource[r] = float("inf")
    child.cancerous = True
    return child


def _log_species(evo_val, data):
    tracker.write_species_info(
        evo_val,
        data,
        frame_count=frame_count,
        min_frame_gap=SPECIES_LOG_GAP,
    )


def _write_run_meta(final=False):
    try:
        elapsed = time.perf_counter() - run_start_time
        small_species = max(
            0, int(tracker.amntOfSpecies) - int(tracker.amntOfMediumSpecies) - int(tracker.amntOfBigSpecies)
        )
        payload = {
            "frame_count": int(frame_count),
            "start_time": float(run_start_wall),
            "elapsed_seconds": float(elapsed),
            "amnt_of_species": int(tracker.amntOfSpecies),
            "amnt_of_small_species": int(small_species),
            "amnt_of_medium_species": int(tracker.amntOfMediumSpecies),
            "amnt_of_big_species": int(tracker.amntOfBigSpecies),
            "final": bool(final),
        }
        RUN_META_PATH.write_text(json.dumps(payload))
        if MASTER_RUN_META_PATH is not None:
            MASTER_RUN_META_PATH.parent.mkdir(parents=True, exist_ok=True)
            MASTER_RUN_META_PATH.write_text(json.dumps(payload))
    except Exception:
        pass

CONTROL_CHECK_INTERVAL = 50
SPECIES_LOG_GAP = 1000
PRINT_INTERVAL = 5000
LINEAGE_RESET_LIFESPAN = 2_100
LINEAGE_RECORD_LIFESPAN = 3_500
control_active_index = None
enabled_flags = None
draw_modes = None
draw_every_list = None
mode_values = None
update_tokens = None
enabled = True
draw_mode = 0
draw_every = 500
last_update_token = None
external_active = True
caption_active = False
if SIM_CONTROL_FILE:
    (
        control_active_index,
        enabled_flags,
        draw_modes,
        draw_every_list,
        mode_values,
        update_tokens,
    ) = _read_control_state()
    if enabled_flags is not None and SIM_INDEX < len(enabled_flags):
        enabled = bool(enabled_flags[SIM_INDEX])
    if draw_modes is not None and SIM_INDEX < len(draw_modes):
        draw_mode = int(draw_modes[SIM_INDEX])
    if draw_every_list is not None and SIM_INDEX < len(draw_every_list):
        draw_every = max(1, int(draw_every_list[SIM_INDEX]))
    if mode_values is not None and SIM_INDEX < len(mode_values):
        try:
            display_mode = int(mode_values[SIM_INDEX])
        except (TypeError, ValueError):
            display_mode = display_mode
    if control_active_index is not None:
        caption_active = (control_active_index == SIM_INDEX)
    if ALL_ACTIVE:
        external_active = enabled
    else:
        external_active = enabled if control_active_index is None else enabled and caption_active
    if update_tokens is not None and SIM_INDEX < len(update_tokens):
        last_update_token = update_tokens[SIM_INDEX]
