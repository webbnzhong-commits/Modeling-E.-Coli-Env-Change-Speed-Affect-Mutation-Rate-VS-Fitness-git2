from pathlib import Path

core_path = Path(__file__).with_name("import pygrame.py")
exec(core_path.read_text(), globals())
while running:
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
    if not external_active:
        time.sleep(1 / 30)
        continue
    if len(dots) <= 5:
        
        dots.append(Dot(random.randint(0, WIDTH), random.randint(0, HEIGHT)))
            
    
    

    # (Info text rendering moved to after drawing dots)
    
    

    

        

    
    enviorment_state.regenerate_resources()
    resource_pool = enviorment_state.get_resources()
    enviorment_state.update()
    

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
            print(f"[{time.strftime('%H:%M:%S')}] Resetting simulation due to long-lived species")
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
            dots, enviorment_state = reset_simulation()
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
    


    enviorment_state.update_antibiotic(dots, frame_count)

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
                enviorment_state.ph,
                enviorment_state.temp,
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
                enviorment_state.deadNutreints["p"] += dead_p
                enviorment_state.deadNutreints["c"] += dead_c
                enviorment_state.deadNutreints["n"] += dead_n

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

    else:
        for dot in list(dots):
            dot.update()
    enviorment_state.update_climate(frame_count)
    '''
    if frame_count % max(1, round(500 / enviormentChangeRate)) == 0:
        enviormentChangeRate += random.uniform(-0.5, 0.5)
        if enviormentChangeRate < 0.5:
            enviormentChangeRate = 0.5
        if enviormentChangeRate > 5:
            enviormentChangeRate = 5'''
    
    

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

        print(f"resource pool: {enviorment_state.get_resources()}")
        print(f"food amnt {enviorment_state.foodAmnt}")
        
    
    

    
_write_run_meta(final=True)
tracker.set_should_parse(should_parse)
if __name__ == "__main__":
    if os.environ.get("SIM_RUN_POOL") == "1":
        with Pool(cpu_count()) as p:
            results = p.map(work, range(10_000))
