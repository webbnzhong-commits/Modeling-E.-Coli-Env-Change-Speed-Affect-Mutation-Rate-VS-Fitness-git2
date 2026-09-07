from pathlib import Path
import pygame

#debug thing for github
#debug 2.0

core_path = Path(__file__).with_name("import pygrame.py")
exec(core_path.read_text(), globals())

FPS_GRAPH_X = 400
FPS_GRAPH_Y = 25
FPS_GRAPH_H = 80
CANCER_ALPHA = 128
GRAPH_DOT_ALPHA = 110
GRAPH_DOT_RADIUS = 3


def _recalc_graph_width():
    global FPS_GRAPH_W
    FPS_GRAPH_W = WIDTH - FPS_GRAPH_X - 150


_recalc_graph_width()
ARITH_OVERLAY_H = 140

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dots Example")
font = pygame.font.SysFont("Consolas", 20)
clock = pygame.time.Clock()


def _build_caption(active, enabled):
    if SIM_TOTAL > 0:
        label = f"Simulation {SIM_INDEX + 1}/{SIM_TOTAL}"
    else:
        label = f"Simulation {SIM_INDEX + 1}"
    if not enabled:
        status = "OFF"
    else:
        status = "ACTIVE" if active else "INACTIVE"
    return f"Dots Example - {label} - {status}"


_base_apply_settings = apply_settings

def apply_settings(new_settings, update_screen=False):
    global settings, WIDTH, HEIGHT, screen
    old_w, old_h = WIDTH, HEIGHT
    _base_apply_settings(new_settings)
    if WIDTH != old_w or HEIGHT != old_h:
        _recalc_graph_width()
        if update_screen and pygame.get_init():
            screen = pygame.display.set_mode((WIDTH, HEIGHT))


def reload_settings():
    apply_settings(load_settings(), update_screen=True)


_base_update_stats_snapshot = _update_stats_snapshot

def _update_stats_snapshot():
    global show_arithmetic_graph, arithmetic_points, arithmetic_graph_error, arithmetic_surface
    _base_update_stats_snapshot()
    show_arithmetic_graph = True
    overlay_rect = pygame.Rect(
        FPS_GRAPH_X,
        FPS_GRAPH_Y + FPS_GRAPH_H + 20,
        FPS_GRAPH_W,
        ARITH_OVERLAY_H,
    )
    arithmetic_surface = pygame.Surface(
        (overlay_rect.width, overlay_rect.height), pygame.SRCALPHA
    )
    _draw_arithmetic_mean_overlay(
        arithmetic_surface,
        font,
        arithmetic_points,
        pygame.Rect(0, 0, overlay_rect.width, overlay_rect.height),
        arithmetic_graph_error,
    )


def _draw_arithmetic_mean_overlay(surface, font, points, rect, error_text=""):
    surface.fill((0, 0, 0, 0))
    pygame.draw.rect(surface, (180, 180, 180), rect, 1)
    title = font.render("Arithmetic Mean Length Lived", True, (220, 220, 0))
    
    surface.blit(title, (rect.x + 8, rect.y + 6))

    if error_text:
        err = font.render(error_text, True, (255, 160, 160))
        surface.blit(err, (rect.x + 8, rect.y + 28))
        return

    if not points:
        msg = font.render("No arithmetic-mean data yet.", True, (200, 200, 200))
        surface.blit(msg, (rect.x + 8, rect.y + 28))
        return

    points_sorted = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in points_sorted]
    ys = [p[1] for p in points_sorted]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    x_pad = (max_x - min_x) * 0.05
    y_pad = (max_y - min_y) * 0.05
    min_x -= x_pad
    max_x += x_pad
    min_y -= y_pad
    max_y += y_pad

    plot_left = rect.x + 8
    plot_top = rect.y + 28
    plot_width = rect.width - 16
    plot_height = rect.height - 36

    def _scale_point(x, y):
        px = plot_left + int(((x - min_x) / (max_x - min_x)) * plot_width)
        py = plot_top + plot_height - int(((y - min_y) / (max_y - min_y)) * plot_height)
        return px, py

    last_pt = None
    for x, y in points_sorted:
        px, py = _scale_point(x, y)
        if last_pt is not None:
            pygame.draw.line(surface, (0, 200, 255), last_pt, (px, py), 2)
        pygame.draw.circle(
            surface,
            (0, 220, 255, GRAPH_DOT_ALPHA),
            (px, py),
            GRAPH_DOT_RADIUS,
        )
        last_pt = (px, py)



def _dot_draw(self, surface):
    if self.cancerous:
        color_tuple = tuple(self.color)
        if (
            self._cancer_surface is None
            or self._cancer_surface_size != self.size
            or self._cancer_surface_color != color_tuple
        ):
            size = max(1, int(self.size))
            diameter = size * 2 + 2
            self._cancer_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
            pygame.draw.circle(
                self._cancer_surface,
                (color_tuple[0], color_tuple[1], color_tuple[2], CANCER_ALPHA),
                (diameter // 2, diameter // 2),
                size,
            )
            self._cancer_surface_size = self.size
            self._cancer_surface_color = color_tuple
        surface.blit(self._cancer_surface, (int(self.x) - self.size - 1, int(self.y) - self.size - 1))
    else:
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)


Dot.draw = _dot_draw

show_arithmetic_graph = False
arithmetic_surface = None

last_caption_state = None
pygame.display.set_caption(_build_caption(caption_active, enabled))
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            for evo_val, data in list(species_trackers.items()):
                    if data["alive"]:
                        data["alive"] = False
                        if data["lifespan"] != data["pop_time"]: #they reproduced at least once
                            _log_species(evo_val, data)
            should_parse = True
            running = False
        elif event.type == pygame.KEYDOWN:  # key pressed
            if event.key == pygame.K_SPACE:
                for evo_val, data in list(species_trackers.items()):
                    if data["alive"]:
                        data["alive"] = False
                        if data["lifespan"] != data["pop_time"]: #they reproduced at least once
                            _log_species(evo_val, data)
                dots, nutrient = reset_simulation()
                species_trackers = {}
                totalSim += 1

                # Put your code here to handle space press
            if event.key == pygame.K_ESCAPE:
                should_parse = True
                running = False
            if event.key == pygame.K_p:
                pause = not pause  
            if event.key == pygame.K_q:
                for evo_val, data in list(species_trackers.items()):
                    if data["alive"]:
                        data["alive"] = False
                        if data["lifespan"] != data["pop_time"]: #they reproduced at least once
                            _log_species(evo_val, data)
                should_parse = True
                running = False
            if event.key == pygame.K_d and not SIM_CONTROL_FILE:
                draw = not draw
                settings["draw"] = draw
                save_settings(settings)
            if event.key == pygame.K_h and not SIM_CONTROL_FILE:
                display_mode = (display_mode + 1) % 3
                settings["display_mode"] = display_mode
                save_settings(settings)
            if event.key == pygame.K_u:
                reload_settings()
            if event.key == pygame.K_s:
                _update_all_stats_snapshot()
    if SIM_CONTROL_FILE and frame_count % CONTROL_CHECK_INTERVAL == 0:
        (
            control_active_index,
            enabled_flags,
            draw_modes,
            draw_every_list,
            mode_values,
            update_tokens,
        ) = _read_control_state()
        if control_active_index == -1:
            should_parse = True
            running = False
            break
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
        else:
            caption_active = False
        if ALL_ACTIVE:
            external_active = enabled
        else:
            external_active = enabled if control_active_index is None else enabled and caption_active
        if update_tokens is not None and SIM_INDEX < len(update_tokens):
            current_token = update_tokens[SIM_INDEX]
            if current_token != last_update_token:
                _update_stats_snapshot()
                last_update_token = current_token

    current_caption_state = (caption_active, enabled)
    if current_caption_state != last_caption_state:
        pygame.display.set_caption(_build_caption(caption_active, enabled))
        last_caption_state = current_caption_state
    if not external_active:
        clock.tick(30)
        continue
    if pause:
        continue
    
    if SIM_CONTROL_FILE:
        if draw_mode == 1:
            should_draw = external_active and (frame_count % draw_every == 0)
        elif draw_mode == 2:
            should_draw = False
        else:
            should_draw = external_active
    else:
        should_draw = external_active and (draw or (drawSometimes and frame_count % drawAmnt == 0))

    if should_draw:
        screen.fill((0, 0, 0))

    # Display frame count
    
    
    
    # (Text labels moved to after drawing dots)

    # Calculate mode evolution speed and pick a random dot with that speed
    

    
    if len(dots) <= 5:
        
        dots.append(Dot(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
            
    
    

    # (Info text rendering moved to after drawing dots)
    
    

    

        

    
    nutrient.regenerate_resources()
    resource_pool = nutrient.get_resources()
    nutrient.update()
    

    # --- Multi-species tracking system ---
    evo_groups = {}
    for dot in dots:
        key = round(dot.evolution_speed, 6)
        evo_groups.setdefault(key, []).append(dot)

    total_population = len(dots)
    active_species = set(evo_groups.keys())

    # If there are 2 or fewer distinct species types, inject a new random dot
    # (species types are defined by evolution_speed rounded to 6 decimals, consistent with tracking)
    if len(active_species) <= 2:
        dots.append(Dot(random.randint(0, WIDTH), random.randint(0, HEIGHT)))

    # Update trackers for active species
    longestAlive = 0
    for evo_val, group in evo_groups.items():
        if evo_val not in species_trackers:
            species_trackers[evo_val] = {"lifespan": 0, "pop_time": 0, "alive": True}
        species_trackers[evo_val]["lifespan"] += 1
        species_trackers[evo_val]["pop_time"] += len(group)
        longestAlive = max(longestAlive, species_trackers[evo_val]["lifespan"])

    # Check for extinct species
    extinct_species = []
    for evo_val, data in list(species_trackers.items()):
        if data["lifespan"] >= LINEAGE_RESET_LIFESPAN:
            #reset simulation
            for evo_val, data in list(species_trackers.items()):
                if data["alive"]:
                    data["alive"] = False
                    should_log = (
                        data["lifespan"] != data["pop_time"]
                        or data["lifespan"] >= LINEAGE_RESET_LIFESPAN
                    )
                    if should_log: #they reproduced at least once, or hit lineage reset threshold
                        row = data.copy()
                        if row["lifespan"] >= LINEAGE_RESET_LIFESPAN:
                            row["lifespan"] = LINEAGE_RECORD_LIFESPAN
                        _log_species(evo_val, row)
            dots, nutrient = reset_simulation()
            species_trackers = {}
            totalSim += 1
            break
            
        if evo_val not in active_species and data["alive"]:
            
            data["alive"] = False
            if data["lifespan"] != data["pop_time"]: #they reproduced at least once
                _log_species(evo_val, data)
            
            extinct_species.append(evo_val)

    # Remove extinct species after recording
    for evo_val in extinct_species:
        del species_trackers[evo_val]
    


    if frame_count % max(1, round(500 / enviormentChangeRate)) == 0:
        # Find the most common immune_system value among all dots
        immune_counts = Counter(dot.immune_system for dot in dots)
        if immune_counts:
            # Choose the most common immune_system type as the antiBiotic target
            antiBiotic = immune_counts.most_common(1)[0][0]
            # Pick a random new immune_system different from the most common one for the antiBiotic type
            
            # Infect all dots that do not match the new antiBiotic type
            for dot in list(dots):
                if abs(antiBiotic - dot.immune_system) <= 1:
                    dot.death()

    # --- Population Cap by Species ---
    if frame_count % 5 == 0:
        
        # Group dots by rounded evolution speed
        evo_groups = {}
        for dot in dots:
            key = round(dot.evolution_speed, 2)
            evo_groups.setdefault(key, []).append(dot)

        total_pop = len(dots)
        for evo, group in evo_groups.items():
            # If species population exceeds cap * total population, cull randomly
            max_allowed = int(total_pop * population_cap)
            if len(group) > max_allowed:
                excess = len(group) - max_allowed
                # Randomly kill the excess dots
                for dot in random.sample(group, excess):
                    if dot in dots:
                        dot.death()

    


    # Update and draw each dot
    if FAST_MODE:
        n = len(dots)
        if n > 0:
            realx = np.array([dot.realx for dot in dots], dtype=np.float32)
            realy = np.array([dot.realy for dot in dots], dtype=np.float32)
            speedx = np.array([dot.speed[0] for dot in dots], dtype=np.float32)
            speedy = np.array([dot.speed[1] for dot in dots], dtype=np.float32)
            life_cycles = np.array([dot.life_cycles for dot in dots], dtype=np.int32)
            max_cycles = np.array([dot.max_cycles for dot in dots], dtype=np.int32)
            res_p = np.array([dot.resources["p"] for dot in dots], dtype=np.float32)
            res_c = np.array([dot.resources["c"] for dot in dots], dtype=np.float32)
            res_n = np.array([dot.resources["n"] for dot in dots], dtype=np.float32)
            repro_p = np.array([dot.reproduction_resource["p"] for dot in dots], dtype=np.float32)
            repro_c = np.array([dot.reproduction_resource["c"] for dot in dots], dtype=np.float32)
            repro_n = np.array([dot.reproduction_resource["n"] for dot in dots], dtype=np.float32)
            opt_ph = np.array([dot.optimal_ph for dot in dots], dtype=np.float32)
            opt_temp = np.array([dot.optimal_temp for dot in dots], dtype=np.float32)

            dead_mask, reproduce_mask = fast_update(
                realx,
                realy,
                speedx,
                speedy,
                life_cycles,
                max_cycles,
                res_p,
                res_c,
                res_n,
                repro_p,
                repro_c,
                repro_n,
                opt_ph,
                opt_temp,
                phLevel,
                temp,
                PH_EFFECT_SCALE,
                PH_EFFECT_DIVISOR,
                TEMP_EFFECT_SCALE,
                TEMP_EFFECT_DIVISOR,
                REPRO_DEBUF_MIN,
                WIDTH,
                HEIGHT,
                resource_pool["p"],
                resource_pool["c"],
                resource_pool["n"],
            )

            if np.any(dead_mask):
                dead_p = float(res_p[dead_mask].sum())
                dead_c = float(res_c[dead_mask].sum())
                dead_n = float(res_n[dead_mask].sum())
                nutrient.deadNutreints["p"] += dead_p
                nutrient.deadNutreints["c"] += dead_c
                nutrient.deadNutreints["n"] += dead_n

            alive_mask = ~dead_mask
            if not np.all(alive_mask):
                dots = [dot for i, dot in enumerate(dots) if alive_mask[i]]
                realx = realx[alive_mask]
                realy = realy[alive_mask]
                speedx = speedx[alive_mask]
                speedy = speedy[alive_mask]
                life_cycles = life_cycles[alive_mask]
                res_p = res_p[alive_mask]
                res_c = res_c[alive_mask]
                res_n = res_n[alive_mask]
                reproduce_mask = reproduce_mask[alive_mask]

            for idx, dot in enumerate(dots):
                dot.realx = float(realx[idx])
                dot.realy = float(realy[idx])
                dot.x = int(dot.realx)
                dot.y = int(dot.realy)
                dot.speed[0] = float(speedx[idx])
                dot.speed[1] = float(speedy[idx])
                dot.life_cycles = int(life_cycles[idx])
                dot.resources["p"] = float(res_p[idx])
                dot.resources["c"] = float(res_c[idx])
                dot.resources["n"] = float(res_n[idx])

            if reproduce_mask is not None and len(dots) > 0:
                for idx, flag in enumerate(reproduce_mask):
                    if flag:
                        child = _spawn_child_from_parent(dots[idx])
                        dots.append(child)

            if should_draw:
                for dot in dots:
                    dot.draw(screen)
    else:
        for dot in list(dots):
            dot.update()
            if should_draw:
                dot.draw(screen)

    # Draw text labels (frame count, population, evo avg, evo median, etc.) after all dots are drawn
    if should_draw:
        if display_mode >= 1:
            fps_enabled = display_mode == 2
            if len(dots) > 0:
                avg_evo_speed = sum(dot.evolution_speed for dot in dots) / len(dots)
            else:
                avg_evo_speed = 0
            text = font.render(f"Run: {run_num} | Frames: {frame_count}", True, (255, 255, 255))
            screen.blit(text, (10, 10))
            # Count how many distinct species exist (species defined by evolution_speed rounded to 6 decimals)
            species_count = len({round(dot.evolution_speed, 6) for dot in dots}) if len(dots) > 0 else 0
            text = font.render(f"Population: {len(dots)} | Species#: {species_count}", True, (255, 255, 255))
            screen.blit(text, (10, 50))
            if fps_enabled:
                fps_est = fps_tracker.fps_estimate()
                if fps_tracker.last_interval_time is not None and fps_tracker.last_interval_time > 0 and fps_est is not None:
                    time_1000_text = f"1000 iters: {fps_tracker.last_interval_time:.2f}s @ {fps_est:.1f} FPS"
                else:
                    time_1000_text = "1000 iters: --"
            else:
                time_1000_text = "1000 iters: --"
            text = font.render(time_1000_text, True, (255, 255, 255))
            screen.blit(text, (10, 80))
            text = font.render(f"evo avg: {avg_evo_speed}", True, (255, 255, 255))
            screen.blit(text, (10, 110))

            # Graph of 1000-iteration times (0-2s) to the right of the HUD
            graph_x, graph_y = FPS_GRAPH_X, FPS_GRAPH_Y
            graph_w, graph_h = FPS_GRAPH_W, FPS_GRAPH_H
            if fps_enabled:
                fps_tracker.draw_graph(screen, font, graph_x, graph_y, graph_w, graph_h)

            if display_mode == 2 and show_arithmetic_graph and arithmetic_surface is not None:
                overlay_rect = pygame.Rect(
                    graph_x,
                    graph_y + graph_h + 20,
                    graph_w,
                    ARITH_OVERLAY_H,
                )
                screen.blit(arithmetic_surface, overlay_rect.topleft)
    if frame_count % max(1, round(500 / enviormentChangeRate)) == 0:
        phLevel += random.uniform(-2, 2)
        if phLevel < 4:
            phLevel = 4
        if phLevel > 10:
            phLevel = 10
    if frame_count % max(1, round(500 / enviormentChangeRate)) == 0:
        if tempDirUp:
            temp += random.uniform(0, 1)
        else:
            temp -= random.uniform(0, 1)
        if temp > 40:
            temp = 40
            tempDirUp = False
        if temp < 34:
            temp = 34
            tempDirUp = True
    '''
    if frame_count % max(1, round(500 / enviormentChangeRate)) == 0:
        enviormentChangeRate += random.uniform(-0.5, 0.5)
        if enviormentChangeRate < 0.5:
            enviormentChangeRate = 0.5
        if enviormentChangeRate > 5:
            enviormentChangeRate = 5'''
    if should_draw:
        if display_mode >= 1:
            # Display median evolution speed below average
            if len(dots) > 0:
                median_evo_speed = statistics.median(dot.evolution_speed for dot in dots)
            else:
                median_evo_speed = 0

            text = font.render(
                f"evo median: {str(median_evo_speed)[:5]} | alive for: {longestAlive} | species #{tracker.amntOfSpecies} | medium Seepcies #{tracker.amntOfMediumSpecies} | big species #{tracker.amntOfBigSpecies}",
                True,
                (255, 255, 255),
            )
            screen.blit(text, (10, 130))

        if display_mode == 2:
            # Draw info text after all dots and labels are drawn so it overlays the dots
            # Draw info text and color squares for top performing dots
            y = 160
            for line in [info, info2, info3, info4, info5, info6, info7, info8, info9]:
                if isinstance(line, tuple) and line[0] == "color_square":
                    pygame.draw.rect(screen, line[1], (10, y, 20, 20))  # small 20x20 square
                else:
                    txt = font.render(line, True, (255, 255, 255))
                    screen.blit(txt, (40, y))  # shift text right so square doesn't overlap
                y += 30
    
        pygame.display.flip()
    clock.tick(0) # keep commented
    
    frame_count += 1
    if display_mode == 2:
        fps_tracker.update(frame_count, display_mode)
    if frame_count % META_WRITE_INTERVAL == 0:
        _write_run_meta()
    if frame_count % SNAPSHOT_INTERVAL == 0:
        _snapshot_arithmetic_mean(frame_count)
    if frame_count % PRINT_INTERVAL == 0 and (not SIM_CONTROL_FILE or caption_active):
        print(frame_count)
        print(totalSim)
        if display_mode == 2:
            fps_est = fps_tracker.fps_estimate()
            if fps_tracker.last_interval_time is not None and fps_tracker.last_interval_time > 0 and fps_est is not None:
                print(f"FPS (last 1000): {fps_est:.1f}, and 1000 iters: {fps_tracker.last_interval_time}")
        print(f"amount of species: {tracker.amntOfSpecies}")
        print(tracker.amntOfSpeciesEach)

        print(f"resource pool: {nutrient.get_resources()}")
        print(f"food amnt {nutrient.foodAmnt}")
        
    
    

    
pygame.quit()
_write_run_meta(final=True)
tracker.set_should_parse(should_parse)
if __name__ == "__main__":
    if os.environ.get("SIM_RUN_POOL") == "1":
        with Pool(cpu_count()) as p:
            results = p.map(work, range(10_000))
